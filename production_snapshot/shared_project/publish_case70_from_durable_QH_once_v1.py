"""Finite add-only publication; caller must hold the original QH HF writer lock."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,'/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps')
from huggingface_hub import CommitOperationAdd, HfApi

R = Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
REPO = 'FrontRear/VLMdataset'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--case', required=True); ap.add_argument('--release-relative',required=True)
    args=ap.parse_args(); case = args.case; assert case.isdigit()
    assert os.environ.get('ROOT_ORIGINAL_HF_LOCK_HELD') == 'HF_ORIGINAL_STRICT500_v1/HF_UPLOAD_OWNER.lock'
    original_lock=(R.parent/'HF_ORIGINAL_STRICT500_v1/HF_UPLOAD_OWNER.lock').open('a')
    fcntl.flock(original_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock = (R / 'REALCASE_QH_ARCHIVE_HF_WRITER.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rel=Path(args.release_relative); assert not rel.is_absolute() and '..' not in rel.parts
    allowed=Path('releases')/case/'revisions/fixed_world_yellow_depth_v5/RELEASE.json'
    assert rel==allowed
    release_path = R / rel
    release = json.loads(release_path.read_text())
    assert release['quality_accepted'] and release['case'] == case
    policy=json.loads((R/'HUMAN_DEPTH_AWARE_YELLOW_OVERLAP_CURRENT.json').read_text())
    assert release['joint_revision']==policy['required_mode'] and release['renderer_SHA256']==policy['renderer_SHA256']
    assert sha(R/policy['renderer_copy'])==policy['renderer_SHA256'] and release['opaque_depth_faces'] and not release['PCF_background_alpha_blend']
    assert case=='70' and release['archive_host']=='cscg-qh00'
    durable=json.loads((R/'receipts/CASE70_DURABLE_ARCHIVE_RECEIPT_v1.json').read_text())
    assert durable['all_members_SHA256_verified'] and durable['persistent_disk']
    for key in ['archive','archive_size','archive_SHA256','package_manifest_SHA256']:assert durable[key]==release[key]
    archive = Path(release['archive'])
    assert archive.parent==R/'archives' and archive.name=='case_70_fixed_world_yellow_depth_v5.tar'
    assert archive.stat().st_size == release['archive_size'] and sha(archive) == release['archive_SHA256']
    prefix = release['HF_prefix']
    routing = json.loads((R/'HUMAN_STARREAL_DELIVERY_ROOT_CURRENT.json').read_text())
    assert routing['HF_prefix'] == 'StarReal'
    assert prefix == f"{routing['HF_prefix']}/{case}/revisions/fixed_world_yellow_depth_v5"
    note = release_path.parent / 'README.txt'
    note.write_text(
        f"Reviewed realcase{case} conditioning reconstruction. Five edit operations, {release['actual_library_assets']} actual library lamps,\n"
        'eight requested recolors, source-appearance pure move and two mask families.\n'
        '1280x704,49frames,10FPS; old original clips and corrected shared-depth PCF+bbox comparisons.\n'
        f'Extract tar and read case_{case}/MANIFEST.json, QUALITY_RELEASE.json and reader/.\n'
        'No observed edited-B RGB GT, calibrated metre scale or measured source photometry.\n'
        'Opaque fixed-world shared ray depth; visible projected red/blue overlap yellow; original background blend is not used. CPU/reference numeric precision and compiler notes included.\n'
        'This is a versioned conditioning release, not a new supervised observed training group.\n'
        'Original realcase paths, v2/v3 and historical data are preserved unchanged.\n')
    files = [archive, release_path, note]
    paths = [prefix + '/' + p.name for p in files]
    # Caller passes the existing private writer credential over encrypted SSH
    # stdin. It is never stored in this project, package, environment or argv.
    token = Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip()
    assert token, 'Existing private writer credential unavailable'
    api = HfApi(token=token)
    api.whoami()
    existing = api.get_paths_info(REPO, paths=paths, repo_type='dataset')
    assert not existing, 'Destination already exists: inspect receipt/remote SHA; never auto-reupload'
    parent = api.repo_info(REPO, repo_type='dataset').sha
    commit = api.create_commit(
        repo_id=REPO, repo_type='dataset',
        operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(p)) for p, path in zip(files, paths)],
        commit_message=f'Add reviewed realcase{case} conditioning reconstruction with unchanged historical comparisons',
        parent_commit=parent)
    commit_id = commit.oid
    remote = api.get_paths_info(REPO, paths=paths, repo_type='dataset', revision=commit_id)
    by_path = {p.path: p for p in remote}
    checked = []
    for p, path in zip(files, paths):
        row = by_path[path]
        assert row.size == p.stat().st_size
        local_sha = sha(p)
        lfs = row.lfs
        if lfs is not None:
            server_sha = lfs.sha256
            assert server_sha == local_sha
            proof = {'kind': 'server_LFS_SHA256', 'sha256': server_sha}
        else:
            data = p.read_bytes()
            oid = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            assert row.blob_id == oid
            proof = {'kind': 'server_Git_blob', 'blob_id': oid, 'local_SHA256': local_sha}
        checked.append({'path': path, 'size': row.size, **proof})
    receipt = {'state': 'published_and_server_verified', 'case': case,
               'revision': release['revision'], 'repo': REPO,
               'commit': commit_id, 'parent_commit': parent,
               'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'add_only_paths': paths, 'server_verified': checked,
               'original_shared_writer_lock_held': True, 'archive_backend':'QH_persistent_StarReal', '6000D_writer_lock_held_by_finite_caller':False, 'QH_only_execution':True,
               'new_supervised_observed_training_groups': 0,
               'new_conditioning_release_revisions': 1,
               'new_distinct_case_increment':release['new_distinct_case_increment'],
               'prior_history_modified': False}
    (release_path.parent / 'HF_RECEIPT.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

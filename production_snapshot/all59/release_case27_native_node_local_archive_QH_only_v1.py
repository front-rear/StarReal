"""Specific revised portable package release after actual full49 image review."""
import argparse, datetime, fcntl, hashlib, json, tarfile
from pathlib import Path

R=Path('/tmp/starreal_uid1007_case27_cpu_v2')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def put(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();case=a.case;assert case=='27'
    import os
    assert os.environ.get('STARREAL_QH_ONLY_COMPONENT_OWNER')=='1'
    Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
    owner=(Q/'logs/case27_release.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=(R/'logs'/f'case{case}_release.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    package=R/'packages'/case/'fixed_world_yellow_depth_v5_staging'
    policy=json.loads((R/'HUMAN_DEPTH_AWARE_YELLOW_OVERLAP_CURRENT.json').read_text())
    m=json.loads((package/'MANIFEST.json').read_text());audit=json.loads((package/'PORTABLE_FINAL_AUDIT.json').read_text());unchanged=json.loads((package/'UNCHANGED_INPUT_SHA_READBACK.json').read_text())
    assert m['case']==case and m['frames']==49 and m['resolution']==[1280,704] and m['fps']==10
    assert m['joint_revision']==audit['joint_revision']==policy['required_mode']
    assert m['renderer_SHA256']==audit['renderer_SHA256']==policy['renderer_SHA256']
    assert sha(R/policy['renderer_copy'])==policy['renderer_SHA256']
    assert audit['state']=='portable_audit_passed_pending_release_record' and audit['packaged_joints_exact_readback_frames']==len(m['variants'])*98
    assert audit['observed_source_RGB_exact_readback_frames']==49 and audit['Native_depth_K_E_checked_all49'] and audit['Original_reader_code_and_controls_unchanged']
    assert audit['all_explicit_videos_decoded49']==len(m['variants'])*3
    visual=json.loads((package/'audits'/m['audited_visual_revision']).read_text());assert visual['verdict']=='PASS_RENDERED_DEPTH_COLOR_VIDEO_COMPONENT_ONLY' and visual['renderer_SHA256']==policy['renderer_SHA256']
    for row in visual['actual_all49_new_joint_sheets']:
        assert row['frames_viewed']==list(range(49))
        assert sha(package/'audits/visual_contact_sheets'/Path(row['all49_actual_contact_sheet']).name)==row['sha256']
    assert len(visual['actual_all49_new_joint_sheets'])==len(m['variants'])
    for rel,digest in unchanged['all_condition_file_SHA256'].items():assert sha(package/'conditions'/rel)==digest
    assert sha(package/m['source_RGB']['path'])==m['source_RGB']['sha256']==unchanged['source_RGB_SHA256']
    assert sha(package/'reader/realcase_condition_reader.py')==unchanged['reader_SHA256']
    for asset in m['assets']:assert sha(package/asset['package_path'])==asset['sha256']
    idx=json.loads((package/'VIDEO_INDEX.json').read_text())
    for name in m['variants']:
        for key,rel in [('ordinary',f'regular_pcf_bbox/{name}.mp4'),('approved_depth_aware',f'depth_aware_pcf_bbox/{name}.mp4'),('comparison',f'comparison/{name}_regular_vs_depth_aware.mp4')]:assert sha(package/rel)==idx['checks'][name][key]['sha256']
    assert m['edited_B_RGB'] is None and m['training_GT_available'] is False
    assert unchanged['raw_Native_K_E_all49_exact'] and unchanged['Native_full_K_exact_scaled_raw49'];assert audit['full_K_exact_scaled_raw49'];assert not m['quality_accepted'];assert len(m['assets'])==3 and audit['condition_files_fully_read']==490 and audit['all_historical_clips_and_cameras_SHA_preserved']==2
    for suffix in ['SIGNED','SOURCE_HINT']:
        d=json.loads((package/'provenance'/f'CASE27_NATIVE_FAST5_{suffix}_ORIGINAL_ENTRY_READBACK_v1.json').read_text())
        if suffix=='SIGNED':assert d['tests']==26 and d['normalization_count']==1 and not d['body_changed']
        else:assert len(d['records'])==2 and all(x['frames']==49 and x['Original_entry_passed'] for x in d['records'])
    aux=json.loads((package/'AUXILIARY_GEOMETRY_SAME_CASE_SHA_CHECK.json').read_text());assert aux['case']==case and aux['all_same_case_and_SHA_verified'] and len(aux['geometries'])==3
    for row in aux['geometries']:
        (R/row['source']).resolve().relative_to((R/'outputs'/case).resolve())
        assert sha(package/row['package_path'])==row['SHA256']==sha(R/row['source'])
    revision='fixed_world_yellow_depth_v5';dest=R/'releases'/case/'revisions'/revision;assert not dest.exists();dest.mkdir(parents=True)
    release=dict(case=case,revision=revision,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision='ACCEPTED_RECONSTRUCTED_CONDITIONING_REVISION',reviewer='Root actual five-operation all49 depth-color review plus independent portable reader and unchanged source/conditions audit',source_SHA256=m['source_RGB']['sha256'],resolution=[1280,704],frames=49,fps=10,operations=['insert','remove','move','replace','move_replace'],variants=len(m['variants']),actual_library_assets=len(m['assets']),mask_families=2,requested_recolors=8,pure_move_source_appearance_hint=True,joint_revision=policy['required_mode'],renderer_SHA256=policy['renderer_SHA256'],yellow='Visible projected redP/blueQ intersection after independent shared PCF depth tests',opaque_depth_faces=True,PCF_background_alpha_blend=False,source_geometry_contact_and_original_controls_review='actual_case27_full49_native_geometry_and_Original_entry',actual_case27_pcf_foreground_AA_samples=sum(v['pcf_in_front_of_box_AA_samples'] for f in json.loads((R/'outputs/27/two_pcf49_fixed_world_yellow_QH_v1/RESULT.json').read_text())['frame_depth_checks'] for v in f['variants'].values()),shared_depth_kernel_has_separate_bidirectional_native_validation=True,source_original_support_contact_observed=m['source_original_support_contact_observed'],hidden_source_lower_completion_explicit=m['source_lower_completion_added'],source_shade_completion_added=m['source_shade_completion_added'],source_base_contact_visual_reviewed=m['source_base_contact_visual_reviewed'],metric_support_not_claimed=True,full_native_archive_not_claimed_in_package=True,scene_visibility_uncertainty_revision=m['scene_visibility_uncertainty_revision'],condition_revision='geometry_masks49_cached_QH_v1',fixed_geometry_and_PCF_reused_unchanged=True,all_condition_files_SHA_unchanged=audit['all_condition_files_SHA_unchanged'],all_videos_decoded49=len(m['variants'])*3,HF_prefix=f"{json.loads((R/'HUMAN_STARREAL_DELIVERY_ROOT_CURRENT.json').read_text())['HF_prefix']}/{case}/revisions/{revision}",quality_accepted=True,uploaded=False,archive_host='cscg-qh00',archive_creation_backend='QH_node_local_complete_package_pending_durable_copy',new_distinct_case_increment=1,new_distinct_case_definition='First server-visible reconstructed primary conditioning delivery, not a new original source scene or observed supervised GT',new_supervised_observed_training_groups=0,scope_limits=['Conditioning-only; no observed editedB RGB ground truth','Relative monocular units and hidden geometry/photometry are estimates','Original fixed-world depth kernel retained, visible projected overlap marker coloring changed by explicit HUMAN request','Specific primary49 revision, not completion of all59/98 source histories'],prior_history_modified=False)
    release.update(quality_scope='Core conditioning archive after actual49 review; latest mandatory Remove source supplement and strict combined reader still required before queue/upload',latest_contract_ready=False,required_before_latest_contract_upload=['Remove observedA and estimatedB sourceColor and signedIntensity for both lamp/bbox families','actual Remove49 and source/reference review','strict combined588 condition/490 joint/5reference/15video reader','unique writer server file/size/SHA verification'],Native_full_K_exact_scaled_raw49=True,raw_Native_depth_K_E_unchanged=True)
    put(package/'QUALITY_RELEASE.json',release);m.update(state='quality_accepted_fixed_world_yellow_depth_conditioning_pending_HF',quality_accepted=True,quality_release='QUALITY_RELEASE.json',latest_contract_ready=False,required_remaining=['Mandatory Remove source supplement and actual visual review','Strict combined588 reader','HF server verification']);put(package/'MANIFEST.json',m)
    files=[]
    for p in sorted(package.rglob('*')):
        if p.is_file() and p!=package/'SHA256_FILES.json':
            assert not p.is_symlink();files.append(dict(path=str(p.relative_to(package)),size=p.stat().st_size,sha256=sha(p)))
    put(package/'SHA256_FILES.json',files);archive=R/'archives'/f'case_{case}_{revision}.tar';archive.parent.mkdir(exist_ok=True);assert not archive.exists()
    with tarfile.open(archive,'x') as tf:tf.add(package,arcname=f'case_{case}',recursive=True)
    with tarfile.open(archive) as tf:
        members=tf.getmembers();assert all(not x.issym() and not x.islnk() and '..' not in Path(x.name).parts and not x.name.startswith('/') for x in members)
        assert len([x for x in members if x.isfile()])==len(files)+1
        expected={f'case_{case}/'+x['path']:x for x in files}
        for member in members:
            if not member.isfile() or member.name==f'case_{case}/SHA256_FILES.json':continue
            item=expected.pop(member.name);assert member.size==item['size'];h=hashlib.sha256()
            with tf.extractfile(member) as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==item['sha256'],member.name
        assert not expected
        manifest=tf.extractfile(f'case_{case}/MANIFEST.json').read();assert hashlib.sha256(manifest).hexdigest()==sha(package/'MANIFEST.json')
    release.update(archive=str(archive),archive_size=archive.stat().st_size,archive_SHA256=sha(archive),files_in_archive=len(files)+1,package_manifest_SHA256=sha(package/'MANIFEST.json'));put(dest/'RELEASE.json',release)
    proof=dict(state='node_local_archive_all_members_SHA_verified',case=case,archive=release['archive'],archive_size=release['archive_size'],archive_SHA256=release['archive_SHA256'],files_in_archive=release['files_in_archive'],package_manifest_SHA256=release['package_manifest_SHA256'],all_members_SHA256_verified=True,persistent_disk=False,source_files_preserved=True,archive_creation_backend='QH_node_local_complete_package_pending_durable_copy')
    proofpath=R/'receipts/CASE27_NODE_LOCAL_ARCHIVE_RECEIPT_v1.json';assert not proofpath.exists();put(proofpath,proof)
    release.update(node_local_archive_receipt='receipts/CASE27_NODE_LOCAL_ARCHIVE_RECEIPT_v1.json',node_local_archive_receipt_SHA256=sha(proofpath));put(dest/'RELEASE.json',release)
    # Canonical QH/local books updated only after exact HF receipt validation; no6000D execution.

    print(json.dumps(release),flush=True)

if __name__=='__main__':main()

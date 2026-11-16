"""Finite preparation of missing latest-contract fields; never approve or upload."""
from pathlib import Path
import argparse, datetime, hashlib, importlib.util, io, json, os, shutil, sys, tarfile, time

Q = Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
U = Q / 'all59_reconstruction_20261115_v1'
N = Path('/tmp/starreal_uid1007_case14_cpu_v2')
sys.path.insert(0, str(Q / 'runtime/qh_cpu_py39_v1'))
import numpy as np
from PIL import Image, ImageDraw

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def put(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2))

def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--case', required=True); ap.add_argument('--revision',required=True); args = ap.parse_args()
    case = args.case; assert case=='14'; assert case in json.loads((U/'POLICY.json').read_text())['all59_case_ids']
    start = time.time(); dest = U/'prepared'/case/args.revision; dest.mkdir(parents=True, exist_ok=True)
    release = json.loads((Q/'releases'/case/'revisions'/args.revision/'RELEASE.json').read_text())
    assert release['quality_accepted'] and release['case'] == case
    archive = Path(release['archive']); assert archive.stat().st_size == release['archive_size']
    assert sha(archive) == release['archive_SHA256']
    row = dict(HF_path=release['HF_prefix']+'/'+archive.name, local_path=str(archive), bytes=archive.stat().st_size,
               archive_LFS_SHA256=release['archive_SHA256'], verified=True)
    with tarfile.open(archive) as tf:
        members={m.name:m for m in tf.getmembers() if m.isfile()}
        root=min([p for p in members if p.endswith('/MANIFEST.json')],key=lambda s:s.count('/')).rsplit('/',1)[0]
        get=lambda p:tf.extractfile(members[root+'/'+p]).read()
        manifest=json.loads(get('MANIFEST.json')); assert manifest['quality_accepted']
        controls=json.loads(get('conditions/MANIFEST.json'))
        assert manifest['frames']==49 and manifest['resolution']==[1280,704] and manifest['fps']==10
        assert len(manifest['variants'])==5
        assert {controls['operations'][v]['operation'] for v in manifest['variants']}=={'remove','move','insert','replace','move_replace'}
        for a in manifest['assets']: assert hashlib.sha256(get(a['package_path'])).hexdigest()==a['sha256']
        put(dest/'SOURCE_MANIFEST.json',manifest)
        camera_sha=hashlib.sha256(get('conditions/cameras49.npz')).hexdigest()
    remove=module('remove_'+case,Q/'reference_completion_20261115_v1/code/build_remove_source_conditions_QH_v2.py')
    remove.N=U; (U/'remove_outputs').mkdir(exist_ok=True)
    rroot=U/'remove_outputs'/row['archive_LFS_SHA256']
    rm=remove.build(row) if not rroot.exists() else json.loads((rroot/'MANIFEST.json').read_text())
    observed=module('observed_'+case,Q/'reference_completion_20261115_v1/code/add_remove_observed_A_conditions_QH_v1.py')
    if not (rroot/'A_observed').exists(): observed.one(rroot)
    assert (rroot/'A_observed').exists() and len(list(rroot.glob('**/Intensity49/*.npz')))==196
    # The prior preparation already finished all196 Remove fields; retain them unchanged.
    # A preparation archive is read only by the next numeric audit. It is not a release.
    rel='remove_packages/'+row['archive_LFS_SHA256']+'.tar.gz'
    rp=U/'remove_publish'/rel; rp.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(rp,'w:gz',compresslevel=1) as tf:
        for p in sorted(rroot.rglob('*')):
            if p.is_file():tf.add(p,arcname='remove_'+row['archive_LFS_SHA256']+'/'+str(p.relative_to(rroot)),recursive=False)
    binding=dict(HF_source_archive=row['HF_path'],archive_LFS_SHA256=row['archive_LFS_SHA256'],case=case,
        new_condition_archive=dict(path=rel,SHA256=sha(rp),bytes=rp.stat().st_size,extracted_root='remove_'+row['archive_LFS_SHA256']),
        frames=49,resolution=[1280,704],families=['lamp_silhouette','bbox_silhouette'],views=dict(A='A_observed/{family}',B='{family}'),
        raw_gamma=rm['raw_gamma'],model_relative_gamma=rm['raw_gamma']/1.1,normalization_count=1,
        source_color_linear_rgb=rm['source_color_linear_rgb'],source_reference=dict(path='source_references/'+row['archive_LFS_SHA256']+'.png',
        SHA256=sha(rroot/'source_reference.png'),kind=rm['source_reference']['kind'],frame=rm['source_reference']['frame'],crop_xyxy=rm['source_reference']['crop_xyxy']))
    sp=U/'remove_publish'/binding['source_reference']['path'];sp.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(rroot/'source_reference.png',sp)
    solid=module('solid_'+case,Q/'solid_mask_audit_20261115_v1/code/audit_and_export_solid_all35_v1.py')
    solid.U=U/'solid'/args.revision;solid.OUT=solid.U/'outputs';solid.U.mkdir(parents=True,exist_ok=True);solid.T=U
    sr=solid.case(dict(row=dict(case=case,HF_archive=row['HF_path'],archive_LFS_SHA256=row['archive_LFS_SHA256']),path=str(archive),remove_binding=binding))
    put(dest/'SOLID_PREPARED.json',sr);put(dest/'REMOVE_BINDING_PREPARED.json',binding)
    # Actual source and all three target choices at the beginning, middle and end.
    old=json.loads((N/'reviews'/f'CASE{case}_WHOLE_SOURCE_AND_THREE_TARGET_PILOT_ACTUAL_REVIEW_v1.json').read_text())
    def paths(v):
        if isinstance(v,dict):
            for w in v.values():yield from paths(w)
        elif isinstance(v,list):
            for w in v:yield from paths(w)
        elif isinstance(v,str) and '/whole_vs_visibility_A' in v and v.endswith('.jpg'):yield v
    source=Q/'component_cache/14/whole_source_rigid_edge_pilot3_v2'
    targets=N/'outputs/14/three_whole_supported_targets_pilot3_QH_v1'
    pp=[str(source/(prefix+str(f).zfill(2)+'.jpg')) for prefix in ['source_whole_vs_visibility_A','rigid_move_edge_whole_vs_visibility_A'] for f in [0,24,48]]
    pp += [str(targets/role/('whole_vs_visibility_A'+str(f).zfill(2)+'.jpg')) for role in ['insert_orig_1b8ac8be', 'replace_orig_f27b6bc0', 'move_replace_orig_7553414e'] for f in [0,24,48]]
    assert len(pp)==15 and all(Path(x).is_file() for x in pp)
    assert old['all_source_and_three_target_pilots15_actually_viewed']
    if case=='72': pp=[p.replace('outputs/72/five_wall_pilot3_QH_v1/replace_orig_e3849e99/','outputs/72/replacement_upper_rear_mount_pilot3_v2/replace_orig_e3849e99/') for p in pp]
    if case=='70': pp=[p.replace('outputs/70/three_whole_floor_targets_pilot3_QH_v3/move_replace_orig_c72770dd/','outputs/70/move_replace_repair_pilot3_QH_v5/move_replace_orig_0751b0e2/') for p in pp]
    page=Image.new('RGB',(1536,5*320+45),(24,24,24));draw=ImageDraw.Draw(page)
    draw.text((8,8),f'Case {case}: actual source / rigid move / insert / replace / move+replace. Geometry diagnostics, not edited RGB.',fill='white')
    seen=[]
    for j,p in enumerate(pp):
        path=N/p
        with Image.open(path) as im: im=im.convert('RGB');im.thumbnail((508,285));page.paste(im,((j%3)*512,(j//3)*320+45))
        draw.text(((j%3)*512+4,(j//3)*320+332),'/'.join(Path(p).parts[-2:]),fill='white')
        seen.append(dict(path=str(path),SHA256=sha(path)))
    review=U/'review';review.mkdir(exist_ok=True);page.save(review/f'CASE{case}_FIVE_ROLE_ACTUAL_PILOTS.jpg',quality=96)
    for p in (rroot/'review').glob('*.jpg'):shutil.copyfile(p,review/(f'CASE{case}_'+p.name))
    result=dict(case=case,state='prepared_latest_contract_awaiting_actual_visual_review',source=row,
        full_archive_SHA_verified=True,source_camera_SHA256=camera_sha,actual_pilots=seen,
        remove_root=str(rroot),solid=sr,remove_binding=binding,all_Original_controls_unchanged=True,
        GPU_used=False,quality_accepted=False,automatic_upload=False,elapsed_seconds=time.time()-start)
    put(dest/'PREPARED.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['solid','actual_pilots','remove_binding']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

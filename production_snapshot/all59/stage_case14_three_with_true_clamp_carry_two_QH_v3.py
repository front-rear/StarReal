from pathlib import Path
import sys,io,json,hashlib,zipfile,subprocess,fcntl
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1';raw=sys.stdin.buffer.read(32*1024**2+1);assert len(raw)<=32*1024**2;z=zipfile.ZipFile(io.BytesIO(raw));proof=json.loads(z.read('ORIGINAL_LIBRARY_RAW_CANDIDATES_PROOF.json'));c=proof['case'];assert c=='14';guard=(Q/f'logs/case{c}_assets_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);rows=proof['actual_exact_raw_SHA_matched'];assert len(rows)==3 and {r['role'] for r in rows}=={'insert','replace','move_replace'};used=json.loads((Q/'supplements/reference_and_remove_completion_v1/REFERENCE_BINDINGS.json').read_text())['assets'];used_hash={r['asset_SHA256'] for r in used};used_uid={r.get('uid') for r in used}
# Read canonical published raw meshes and UIDs plus all retained prepared inputs, no silent reuse of another case's model.
for d in (Q/'remote_package_metadata').iterdir():
 if not d.is_dir():continue
 manifests=[d/'MANIFEST.json']+list((d/'revisions').glob('*/MANIFEST.json'))
 for p in manifests:
  if not p.exists():continue
  a=json.loads(p.read_text())
  if a.get('quality_accepted'):
   for b in a.get('assets',[]):
    used_hash.add(b.get('sha256'));used_uid.add(b.get('selection',{}).get('original_uid'))
allowed={}
previous=Q/'inputs/case14_three_fresh_whole_models_v1/ORIGINAL_LIBRARY_RAW_CANDIDATES_PROOF.json'
for b in json.loads(previous.read_text())['actual_exact_raw_SHA_matched']:
 if b['role'] in ['insert','move_replace']:allowed[b['role']]=(b['source_index'],b['actual_raw_SHA256'],b['original_uid'])
for r in rows:
 if r['role'] in allowed:assert (r['source_index'],r['actual_raw_SHA256'],r['original_uid'])==allowed[r['role']]
for p in (Q/'inputs').glob('case*_three_fresh_whole_models_v*/ORIGINAL_LIBRARY_RAW_CANDIDATES_PROOF.json'):
 a=json.loads(p.read_text())
 for b in a.get('actual_exact_raw_SHA_matched',[]):
  if p==previous and b['role'] in allowed and (b['source_index'],b['actual_raw_SHA256'],b['original_uid'])==allowed[b['role']]:continue
  used_hash.add(b['actual_raw_SHA256']);used_uid.add(b.get('original_uid'))

for r in rows:assert r['actual_raw_SHA256'] not in used_hash and r['original_uid'] not in used_uid;assert hashlib.sha256(z.read('raw_assets/lamp_'+r['source_index']+'.glb')).hexdigest()==r['actual_raw_SHA256']
version=int(proof.get('private_version',1));assert version==2;out=Q/f'inputs/case{c}_three_fresh_whole_models_v{version}';out.mkdir(parents=True,exist_ok=False);parser=U/'export_whole_raw_mesh_views_QH_v2.py';r=subprocess.run(['/mnt/atlas/fangrui/repro_video_20261105/_envs/pisco/bin/python',str(parser)],input=raw,capture_output=True,timeout=300);assert r.returncode==0,r.stderr.decode(errors='replace')[-4000:];views=zipfile.ZipFile(io.BytesIO(r.stdout))
for archive in [z,views]:
 for info in archive.infolist():
  rel=Path(info.filename);assert not rel.is_absolute() and '..' not in rel.parts;dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(archive.read(info))
alias=N/f'inputs/case{c}_three_fresh_whole_models_v{version}';assert not alias.exists() and not alias.is_symlink();alias.symlink_to(out,target_is_directory=True);R=Path('/tmp')/f'starreal_uid1007_case{c}_cpu_v2';r_alias=R/f'inputs/case{c}_three_fresh_whole_models_v{version}';assert not r_alias.exists();r_alias.symlink_to(out,target_is_directory=True);print(json.dumps(dict(case=c,state='THREE_WHOLE_RAW_GLB_PARSED_PENDING_ACTUAL_MESH_AND_MOUNT_REVIEW',out=str(out),input_payload_SHA256=hashlib.sha256(raw).hexdigest(),assets=[dict(index=r['source_index'],role=r['role'],name=r['catalog_metadata']['name'],SHA256=r['actual_raw_SHA256']) for r in rows],source_T9_files_unchanged=True,GPU_used=False,quality_accepted=False)))

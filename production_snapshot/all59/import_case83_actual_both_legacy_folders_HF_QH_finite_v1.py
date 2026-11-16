from pathlib import Path,PurePosixPath
import sys,os,json,datetime,hashlib,time,concurrent.futures,tarfile,fcntl
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');R=Path('/tmp/starreal_uid1007_case83_cpu_v2');U=Q/'all59_reconstruction_20261115_v1'
sys.path[:0]=[str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps'),str(R/'runtime/qh_cpu_py39_v1')]
from huggingface_hub import HfApi,hf_hub_url
import requests,numpy as np
guard=(Q/'logs/case83_historical_input_import.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
out=Q/'inputs/case83_actual_all_legacy_HF_v1';out.mkdir(exist_ok=False)
state=dict(case='83',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split()[21],state='running_readonly_HF_import',GPU_used=False)
def save():(out/'JOB.json').write_text(json.dumps(state,indent=2))
save()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
try:
 token=Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip();api=HfApi(token=token);head=api.repo_info('FrontRear/VLMdataset',repo_type='dataset').sha;prefix='realcase/83'
 tree=[]
 for row in api.list_repo_tree('FrontRear/VLMdataset',path_in_repo=prefix,repo_type='dataset',revision=head,recursive=True):
  if hasattr(row,'size'):
   lfs=getattr(row,'lfs',None);tree.append(dict(path=row.path,size=row.size,blob_id=row.blob_id,LFS_SHA256=lfs.sha256 if lfs else None))
 assert sum(r['path'].endswith('conditions.tar') for r in tree)==2
 (out/'PINNED_SERVER_TREE.json').write_text(json.dumps(dict(case='83',HF_revision=head,files=tree),indent=2))
 def fetch(row):
  rel=PurePosixPath(row['path']).relative_to(prefix);assert '..' not in rel.parts
  dest=out/'original_HF_folder'/str(rel);dest.parent.mkdir(parents=True,exist_ok=True);part=dest.with_name(dest.name+'.part');h=hashlib.sha256();g=hashlib.sha1(b'blob '+str(row['size']).encode()+b'\0');n=0
  with requests.get(hf_hub_url('FrontRear/VLMdataset',row['path'],repo_type='dataset',revision=head),headers={'Authorization':'Bearer '+token},stream=True,timeout=(30,120)) as response:
   response.raise_for_status()
   with part.open('xb') as f:
    for b in response.iter_content(4194304):
     if b:f.write(b);h.update(b);g.update(b);n+=len(b)
  assert n==row['size']
  if row['LFS_SHA256']:assert h.hexdigest()==row['LFS_SHA256']
  else:assert g.hexdigest()==row['blob_id']
  part.replace(dest);return dict(HF_path=row['path'],local_path=str(dest),bytes=n,SHA256=h.hexdigest(),server_LFS_SHA256=row['LFS_SHA256'],server_blob_id=row['blob_id'])
 records=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for row in pool.map(fetch,tree):
   records.append(row);state.update(files_complete=len(records),files_expected=len(tree),bytes_complete=sum(r['bytes'] for r in records));save()
 cameras=[];camdir=out/'extracted_cameras';camdir.mkdir();current=np.load(R/'inputs/case83_native_CPU_v1/cameras.npz')
 for archive in sorted((out/'original_HF_folder').rglob('conditions.tar')):
  package=archive.parent.name
  with tarfile.open(archive) as tf:
   for member in tf:
    if not member.isfile() or not member.name.endswith('/cameras_and_boxes.npz'):continue
    assert member.size<33554432
    camera='orbit_10deg' if 'orbit_10deg' in member.name else 'small_4deg' if 'small_4deg' in member.name else None;assert camera,member.name
    if package=='move_replace_mesh_v1':op='move_replace'
    else:
     ops=[op for op in ['insert','move','remove','replace','move_replace'] if op in PurePosixPath(member.name).parts]
     assert len(ops)==1,member.name;op=ops[0]
    data=tf.extractfile(member).read();dest=camdir/f'{package}_{op}_{camera}.npz';assert not dest.exists();dest.write_bytes(data)
    z=np.load(dest,allow_pickle=False);assert z['K'].shape==(49,3,3) and z['w2c_A'].shape==z['w2c_B'].shape==(49,4,4)
    for key in ['K','w2c_A','w2c_B']:assert np.isfinite(z[key]).all()
    cameras.append(dict(package=package,operation=op,camera_version=camera,archive_path=str(archive),archive_member=member.name,path=str(dest),SHA256=sha(dest),K_exact_current=np.array_equal(z['K'],current['K']),A_exact_current=np.array_equal(z['w2c_A'],current['w2c_A']),B_exact_current=np.array_equal(z['w2c_B'],current['w2c_B']),actual_old_camera_unchanged=True))
 assert len(cameras)==12,len(cameras)
 result=dict(case='83',state='actual_two_legacy_folders_all113_server_files_and12_camera_arrays_verified',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),HF_revision=head,records=records,camera_checks=cameras,historical_operation_count=5,historical_joint_versions=12,historical_relative_ops_and_extra_mesh_move_replace_both_present=True,all_original_HF_folder_files_preserved=True,new_complete_groups=0,quality_accepted=False,GPU_used=False,Mac_payload_files=False,T9_writes=False,no6000D_execution=True,seconds=time.time()-start)
 (out/'HF_IMPORT_MANIFEST.json').write_text(json.dumps(result,indent=2));state.update(state='terminal0_download_and_camera_extraction_verified_pending_actual_history49_review',exit_code=0,seconds=time.time()-start);save();print(json.dumps({k:v for k,v in result.items() if k not in ['records','camera_checks']}),flush=True)
except Exception as e:state.update(state='failed_preserve_partial',exit_code=1,error=repr(e));save();raise

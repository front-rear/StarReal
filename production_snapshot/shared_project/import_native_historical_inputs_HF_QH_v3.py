"""Pinned read-only history import on Qinghai; no6000D execution/stand-ins."""
from pathlib import Path,PurePosixPath
import argparse,concurrent.futures,datetime,fcntl,hashlib,json,shutil,sys,tarfile,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');N=Path('/tmp/starreal_uid1007_case19_cpu_v1')
sys.path.insert(0,str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps'))
from huggingface_hub import HfApi,hf_hub_download
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();c=a.case;assert c.isdecimal();start=time.time()
 lock=(Q/'logs'/('case'+c+'_historical_input_import.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 native=json.loads((N/('inputs/case'+c+'_native_CPU_v1/INPUT_MANIFEST.json')).read_text());assert native['case']==c
 token=Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip();api=HfApi(token=token);head=api.repo_info('FrontRear/VLMdataset',repo_type='dataset').sha;tree=[]
 for row in api.list_repo_tree('FrontRear/VLMdataset',path_in_repo='realcase/'+c,repo_type='dataset',revision=head,recursive=True):
  if hasattr(row,'size'):
   lfs=getattr(row,'lfs',None);tree.append(dict(path=row.path,size=row.size,blob_id=row.blob_id,lfs_sha256=lfs.sha256 if lfs else None))
 by={r['path']:r for r in tree};prefix='realcase/'+c+'/relative_ops_v1';meshmanifest=str(PurePosixPath(native['source_video_HF_path']).parents[1]/'manifest.json')
 names=[f'media/{op}/{cam}/joint.mkv' for op in ['insert','remove','move','replace','move_replace'] for cam in ['orbit_10deg','small_4deg']]+['media/source_A.mkv','manifest.json']
 assert all(prefix+'/'+n in by for n in names) and meshmanifest in by and prefix+'/conditions.tar' in by,'Concrete history input missing; do not invent substitute'
 optional=['README.md','operation_inventory.json','relative_gamma.json','gamma_interface_verification.json','validation.json','hf_v2_relative_adapter.json'];present=[n for n in optional if prefix+'/'+n in by];missing=[n for n in optional if n not in present];names+=present
 treepath=Q/'receipts'/('CASE'+c+'_HF_HISTORICAL_TREE_v3.json');assert not treepath.exists();treepath.write_text(json.dumps(dict(case=c,head=head,files=tree,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),read_only=True),indent=2))
 out=N/('inputs/case'+c+'_historical_five_ops_HF_v2');out.mkdir(exist_ok=False);durable=Q/('inputs/case'+c+'_historical_five_ops_HF_v2');assert not durable.exists();durable.mkdir();download=Q/('inputs/case'+c+'_historical_HF_download_v3');assert not download.exists();download.mkdir()
 def get(path):
  row=by[path];p=Path(hf_hub_download('FrontRear/VLMdataset',filename=path,repo_type='dataset',revision=head,token=token,local_dir=download));h=sha(p);assert p.stat().st_size==row['size']
  if row['lfs_sha256']:assert h==row['lfs_sha256']
  else:assert hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()==row['blob_id']
  return p,h
 def clip(rel):
  path=prefix+'/'+rel;p,h=get(path);row=by[path]
  for root in [out,durable]:
   dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(dest)==h
  return dict(path=rel,HF_path=path,SHA256=h,bytes=p.stat().st_size,server_blob_id=row['blob_id'],server_LFS_SHA256=row['lfs_sha256'])
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(clip,names))
 p,h=get(meshmanifest)
 for root in [out,durable]:shutil.copy2(p,root/'mesh_revision_manifest.json');assert sha(root/'mesh_revision_manifest.json')==h
 records.append(dict(path='mesh_revision_manifest.json',HF_path=meshmanifest,SHA256=h,bytes=p.stat().st_size,private_rename_only_original_bytes_preserved=True))
 result=dict(case=c,state='all10_historical_joint_clips_and_source_metadata_pinned_server_SHA_imported',HF_revision=head,records=records,historical_joint_clips=10,optional_metadata_missing=missing,source_original_bytes_preserved=True,quality_accepted=False,GPU_used=False,elapsed_seconds=time.time()-start)
 for root in [out,durable]:(root/'HF_IMPORT_MANIFEST.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(stage='ten_historical_clips_imported',case=c,seconds=time.time()-start)),flush=True)
 archive,archive_sha=get(prefix+'/conditions.tar');camout=N/('inputs/case'+c+'_historical_cameras_v1');camdur=Q/('inputs/case'+c+'_historical_cameras_v1');assert not camout.exists() and not camdur.exists();camout.mkdir();camdur.mkdir();camrecords=[]
 with tarfile.open(archive) as tf:
  for member in tf.getmembers():
   if not member.isfile() or not member.name.endswith('/cameras_and_boxes.npz') or 'operations/' not in member.name:continue
   rel=member.name[member.name.index('operations/'):];assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
   assert rel not in {x['path'] for x in camrecords};data=tf.extractfile(member).read();digest=hashlib.sha256(data).hexdigest()
   for root in [camout,camdur]:p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert sha(p)==digest
   camrecords.append(dict(path=rel,archive_member=member.name,bytes=len(data),SHA256=digest))
 assert len(camrecords)==10
 provenance=dict(case=c,source_archive=str(archive),source_archive_SHA256=archive_sha,source_archive_bytes=archive.stat().st_size,HF_path=prefix+'/conditions.tar',HF_revision=head,records=camrecords,original_archive_unchanged=True,GPU_used=False,no6000D_execution=True)
 for root in [camout,camdur]:(root/'CAMERA_PROVENANCE.json').write_text(json.dumps(provenance,indent=2))
 receipt=dict(case=c,state='ten_original_HF_clips_and_exact_archive_camera_members_persistent',HF_revision=head,clips=10,camera_members=10,archive_SHA256=archive_sha,archive_bytes=archive.stat().st_size,quality_accepted=False,new_complete_groups=0,GPU_used=False,no6000D_execution=True,elapsed_seconds=time.time()-start)
 (Q/'receipts'/('CASE'+c+'_HISTORICAL_HF_IMPORT_QH_v3.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

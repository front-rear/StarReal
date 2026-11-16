"""Publish reviewed Remove supplements, verify public bytes, then add completion pointer."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,io,json,os,sys,tarfile,threading,time,urllib.request
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';P=T/'remove_publish';PREFIX='StarReal/supplements/reference_and_remove_completion_v1';REPO='FrontRear/VLMdataset'
sys.path[:0]=[str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps'),str(Q/'runtime/qh_cpu_py39_v1')]
from huggingface_hub import HfApi,CommitOperationAdd
import cv2,numpy as np

def sha(b):return hashlib.sha256(b).hexdigest()
def git(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def put(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def row_for(p,base):
 b=p.read_bytes();return dict(local=p,path=PREFIX+'/'+str(p.relative_to(base)),relative=str(p.relative_to(base)),size=len(b),SHA256=sha(b),git_blob=git(b))
def main():
 assert os.getuid()==1007;start=time.time();state=T/'logs/REMOVE_PUBLICATION_CURRENT.json';receipt=T/'inventory/HF_REMOVE_AND_COMPLETION_VERIFIED_v1.json';locks=[]
 for path in [Q.parent/'HF_ORIGINAL_STRICT500_v1/HF_UPLOAD_OWNER.lock',Q/'REALCASE_QH_ARCHIVE_HF_WRITER.lock',T/'logs/remove_publication.lock']:
  f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 portable=json.loads((T/'inventory/REMOVE_PORTABLE_PACKAGES_VERIFIED_v1.json').read_text());assert portable['archives']==48 and portable['NPZ_decoded']==9408 and portable['PNG_condition_files_decoded']==18816
 bindings=json.loads((P/'REMOVE_BINDINGS.json').read_text());ops=json.loads((P/'OPERATION_REFERENCE_BINDINGS.json').read_text());review=json.loads((P/'REMOVE_ACTUAL_REVIEW.json').read_text());reference=json.loads((T/'inventory/HF_REFERENCE_PUBLICATION_VERIFIED_v1.json').read_text());scope=json.loads((T/'publish/HF_SOURCE_SCOPE.json').read_text());assert bindings['archive_versions']==48 and bindings['fps']==10 and len(bindings['rows'])==48;assert ops['operations']==463 and not ops['unbound'];assert len(review['reviewed_groups'])==35 and reference['all_files_server_metadata_and_anonymous_GET_SHA_readable']
 assert not any('/70/' in x['HF_source_archive'] for x in bindings['rows']);bykey={x['archive_LFS_SHA256']:x for x in bindings['rows']}
 files=[row_for(p,P) for p in sorted(P.rglob('*')) if p.is_file() and not any(x.startswith('._') for x in p.parts)]
 assert len([r for r in files if r['relative'].endswith('.tar.gz')])==48 and all(r['size']<64*1024**2 for r in files)
 ref_files=[row_for(p,T/'publish') for p in sorted((T/'publish').rglob('*')) if p.is_file() and not any(x.startswith('._') for x in p.parts)];assert len(ref_files)==532
 token=Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip();assert token;api=HfApi(token=token)
 def meta(paths,revision):return {x.path:x for x in api.get_paths_info(REPO,paths=paths,repo_type='dataset',revision=revision)}
 def exact(x,row):return x.size==row['size'] and (x.lfs.sha256==row['SHA256'] if x.lfs is not None else x.blob_id==row['git_blob'])
 def check_source(revision):
  paths=[r['path'] for r in scope['archives']]+[scope['non_asset_supplement']['HF_path']];now=meta(paths,revision)
  for r in scope['archives']:assert now[r['path']].size==r['bytes'] and now[r['path']].lfs.sha256==r['LFS_SHA256']
  ordinary=scope['non_asset_supplement'];assert now[ordinary['HF_path']].lfs.sha256==ordinary['archive_LFS_SHA256']
  current=meta([x['path'] for x in ref_files],revision);assert len(current)==532 and all(exact(current[r['path']],r) for r in ref_files)
  current_numeric={x.path for x in api.list_repo_tree(REPO,path_in_repo='StarReal',recursive=True,repo_type='dataset',revision=revision) if hasattr(x,'size') and len(x.path.split('/'))>2 and x.path.split('/')[1].isdigit() and x.path.endswith('.tar')};assert current_numeric=={x['path'] for x in scope['archives']},'Numeric source scope changed: audit new source assets instead of claiming complete'
 parent=api.repo_info(REPO,repo_type='dataset').sha;check_source(parent);existing=meta([r['path'] for r in files],parent)
 if existing:
  assert len(existing)==len(files) and all(exact(existing[r['path']],r) for r in files),'Partial/different destination requires exact recovery, never overwrite';commit=parent
 else:
  put(state,dict(state='uploading_reviewed_Remove_additions',pid=os.getpid(),files=len(files),bytes=sum(r['size'] for r in files),original_locks_held=True,parent=parent))
  c=api.create_commit(repo_id=REPO,repo_type='dataset',parent_commit=parent,commit_message='Complete original-source Color and signed Intensity for all48 StarReal Remove versions in lamp and3D-bbox masks',operations=[CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=str(r['local'])) for r in files]);commit=c.oid
 server=meta([r['path'] for r in files],commit);assert len(server)==len(files) and all(exact(server[r['path']],r) for r in files)
 guard=threading.Lock();checked=[]
 def one(row):
  error=None
  for attempt in range(3):
   try:
    url=f"https://huggingface.co/datasets/{REPO}/resolve/{commit}/{row['path']}?download=true"
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'StarReal-Remove-public-readback/1.0'}),timeout=90) as f:b=f.read(row['size']+1)
    assert len(b)==row['size'] and sha(b)==row['SHA256'];inner=None
    if row['relative'].endswith('.tar.gz'):
     key=Path(row['relative']).name.split('.')[0];expect=bykey[key];prefix=expect['new_condition_archive']['extracted_root'];got={}
     with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as tf:
      for member in tf:
       assert member.isfile() and member.name.startswith(prefix+'/') and '..' not in Path(member.name).parts;data=tf.extractfile(member).read();rel=member.name[len(prefix)+1:];got[rel]=dict(bytes=len(data),SHA256=sha(data))
       if rel=='FILE_MANIFEST.json':filelist=json.loads(data)
       if rel=='QUALITY_RELEASE.json':quality=json.loads(data)
     assert quality['source_archive_LFS_SHA256']==key and quality['frames']==49 and quality['normalization_count']==1
     assert len(got)==expect['new_condition_archive']['files'] and len(filelist['files'])==len(got)-1
     for item in filelist['files']:assert got[item['path']]==dict(bytes=item['bytes'],SHA256=item['SHA256'])
     inner=dict(all_members_SHA256_exact=True,members=len(got),source_archive_LFS_SHA256=key,frames=49,both_families_both_views=True)
    elif row['relative'].endswith(('.jpg','.png')):
     im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED);assert im is not None
    elif row['relative'].endswith('.json'):json.loads(b)
    elif row['relative'].endswith('.py'):compile(b.decode(),'verified_published_reader','exec')
    else:b.decode('utf-8')
    r=dict(path=row['path'],size=row['size'],SHA256=row['SHA256'],server_metadata_exact=True,anonymous_GET_SHA_exact=True,readable=True)
    if inner:r['portable_tar']=inner
    with guard:
     checked.append(r)
     if len(checked)%10==0:put(state,dict(state='verifying_public_Remove_bytes_and_portable_members',commit=commit,verified=len(checked),expected=len(files),elapsed_seconds=time.time()-start));print(json.dumps(dict(verified=len(checked),expected=len(files))),flush=True)
    return r
   except Exception as e:error=type(e).__name__+': '+str(e)[:120]
  raise RuntimeError('Bounded public readback failed: '+row['relative']+' '+str(error))
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(one,files))
 # Every operation, including Remove/Move, now has an actual corresponding reference.
 wanted={}
 for op in ops['bindings']:
  r=op['reference'];wanted[PREFIX+'/'+r['path']]=r['SHA256']
 refs=meta(list(wanted),commit)
 for path,h in wanted.items():
  x=refs[path]
  if x.lfs is not None:assert x.lfs.sha256==h
  else:
   local=(P/path[len(PREFIX)+1:]) if path.startswith(PREFIX+'/source_references/') else (T/'publish'/path[len(PREFIX)+1:]);assert x.blob_id==git(local.read_bytes())
 assert len(verified)==len(files) and sum('portable_tar' in r for r in verified)==48
 complete=dict(schema='StarReal.reference_and_Remove_completion.v1',state='COMPLETE_SERVER_VERIFIED',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),repo=REPO,HF_prefix=PREFIX,source_revision=scope['HF_source_revision'],reference_commit=reference['commit'],Remove_data_commit=commit,unique_raw_GLB_assets=129,library_reference_views=516,source_reference_crops=48,archive_versions=48,numeric_cases=35,operation_bindings=463,missing_asset_references=0,missing_operation_references=0,missing_Remove_source_Color_Intensity=0,Remove_frames=49,resolution=[1280,704],fps=10,mask_families=['lamp_silhouette','bbox_silhouette'],views=['observed_A','estimated_B'],raw_gamma=-.5,model_relative_gamma=-.5/1.1,model_normalization_count=1,independent_field_reader_checks=9408,portable_Intensity_NPZ_decoded_before_upload=9408,portable_Color_and_mask_PNG_decoded_before_upload=18816,all_new_files_server_metadata_and_anonymous_GET_SHA_readable=True,all48_public_tar_members_SHA_verified=True,all463_operation_references_bound=True,original_RGB_PCFS_cameras_other_edits_unchanged=True,original_archives_overwritten=False,case70_uploaded=False,original_full_reconstruction_resumed=False,loader='remove_condition_reader.py',reference_loader='operation_reference_reader.py',entry_manifests=['REFERENCE_BINDINGS.json','REMOVE_BINDINGS.json','OPERATION_REFERENCE_BINDINGS.json'])
 completion_path=PREFIX+'/COMPLETION.json';pointer_path='StarReal/CONDITION_COMPLETION_CURRENT.json';payload=json.dumps(complete,indent=2).encode();pointer=json.dumps(dict(state='COMPLETE_SERVER_VERIFIED',HF_prefix=PREFIX,completion=completion_path,reference_bindings=PREFIX+'/REFERENCE_BINDINGS.json',Remove_bindings=PREFIX+'/REMOVE_BINDINGS.json',operation_reference_bindings=PREFIX+'/OPERATION_REFERENCE_BINDINGS.json',original_packages='Preserved unchanged; apply this explicit supplement to replace old empty Remove target-only fields with source Color/Intensity',case70_uploaded=False),indent=2).encode();final_parent=api.repo_info(REPO,repo_type='dataset').sha;check_source(final_parent);new=meta([completion_path,pointer_path],final_parent);assert not new,'Completion pointer already exists: exact verify without overwrite'
 c=api.create_commit(repo_id=REPO,repo_type='dataset',parent_commit=final_parent,commit_message='Mark StarReal reference and Remove source-condition completion verified; add current binding pointer',operations=[CommitOperationAdd(path_in_repo=completion_path,path_or_fileobj=payload),CommitOperationAdd(path_in_repo=pointer_path,path_or_fileobj=pointer)]);final=c.oid
 for path,b in [(completion_path,payload),(pointer_path,pointer)]:
  info=meta([path],final)[path];assert info.size==len(b) and (info.lfs.sha256==sha(b) if info.lfs is not None else info.blob_id==git(b))
  with urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{final}/{path}?download=true',timeout=90) as f:assert f.read()==b
 r=dict(state='goal_complete_all_requirements_server_verified',repo=REPO,HF_prefix=PREFIX,completion_commit=final,Remove_data_commit=commit,reference_commit=reference['commit'],at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files_verified=len(files),bytes=sum(x['size'] for x in files),checks=verified,completion=complete,completion_file_SHA256=sha(payload),root_current_pointer_SHA256=sha(pointer),elapsed_seconds=time.time()-start,overall_goal_complete=True)
 put(receipt,r);put(state,{k:v for k,v in r.items() if k not in ['checks','completion']});print(json.dumps({k:v for k,v in r.items() if k not in ['checks','completion']}),flush=True)
if __name__=='__main__':main()

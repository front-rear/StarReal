"""Publish only the explicit reviewed static-reference supplement. No old archives."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,sys,threading,time,urllib.request
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';P=T/'publish';PREFIX='StarReal/supplements/reference_and_remove_completion_v1';REPO='FrontRear/VLMdataset'
sys.path.insert(0,str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps'))
from huggingface_hub import HfApi,CommitOperationAdd
import cv2,numpy as np

def sha(b):return hashlib.sha256(b).hexdigest()
def put(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
 assert os.getuid()==1007;start=time.time();state=T/'logs/REFERENCE_PUBLICATION_CURRENT.json';receipt=T/'inventory/HF_REFERENCE_PUBLICATION_VERIFIED_v1.json';receipt.parent.mkdir(exist_ok=True)
 locks=[]
 for p in [Q.parent/'HF_ORIGINAL_STRICT500_v1/HF_UPLOAD_OWNER.lock',Q/'REALCASE_QH_ARCHIVE_HF_WRITER.lock',T/'logs/reference_publication.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 bindings=json.loads((P/'REFERENCE_BINDINGS.json').read_text());review=json.loads((P/'REFERENCE_ACTUAL_REVIEW.json').read_text());scope=json.loads((P/'HF_SOURCE_SCOPE.json').read_text());assert bindings['unique_assets']==129 and len(bindings['assets'])==129 and review['images_viewed']==516 and scope['case_archives']==48
 assert bindings['missing_asset_references']==0 and all(a['actual_visual_review'].startswith('APPROVED_') for a in bindings['assets']);assert not any('/70/' in b['archive_path'] for a in bindings['assets'] for b in a['bound_assets'])
 paths=[]
 for p in sorted(P.rglob('*')):
  if not p.is_file() or any(x.startswith('._') for x in p.parts):continue
  rel=str(p.relative_to(P));assert p.suffix in ['.jpg','.json','.txt','.py'] and '..' not in p.parts and not p.is_symlink();b=p.read_bytes();assert len(b)<8*1024**2;paths.append(dict(local=p,relative=rel,path=PREFIX+'/'+rel,size=len(b),SHA256=sha(b),git_blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()))
 token=Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip();assert token;api=HfApi(token=token);parent=api.repo_info(REPO,repo_type='dataset').sha
 def metadata(pathset,revision):return {x.path:x for x in api.get_paths_info(REPO,paths=pathset,repo_type='dataset',revision=revision)}
 # Recheck all immutable source archive pointers at the live parent, not a stale ledger.
 current=metadata([x['path'] for x in scope['archives']],parent)
 for row in scope['archives']:
  x=current[row['path']];assert x.size==row['bytes'] and x.lfs.sha256==row['LFS_SHA256']
 dest=metadata([x['path'] for x in paths],parent)
 def exact(x,row):return x.size==row['size'] and (x.lfs.sha256==row['SHA256'] if x.lfs is not None else x.blob_id==row['git_blob'])
 if dest:
  assert len(dest)==len(paths) and all(exact(dest[r['path']],r) for r in paths),'Destination partially exists or differs; no overwrite permitted';commit=parent;new_commit=False
 else:
  put(state,dict(state='publishing_reviewed_reference_images',pid=os.getpid(),files=len(paths),bytes=sum(x['size'] for x in paths),original_writer_locks_held=True,parent=parent))
  c=api.create_commit(repo_id=REPO,repo_type='dataset',parent_commit=parent,commit_message='Add exact SHA-bound StarReal whole-lamp reference views and reference lookup manifest',operations=[CommitOperationAdd(path_in_repo=r['path'],path_or_fileobj=str(r['local'])) for r in paths]);commit=c.oid;new_commit=True
 remote=metadata([x['path'] for x in paths],commit);assert len(remote)==len(paths)
 for r in paths:assert exact(remote[r['path']],r)
 checks=[];guard=threading.Lock()
 def one(row):
  url=f"https://huggingface.co/datasets/{REPO}/resolve/{commit}/{row['path']}?download=true";error=None
  for trial in range(3):
   try:
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'StarReal-reference-verification/1.0'}),timeout=60) as f:b=f.read(row['size']+1)
    assert len(b)==row['size'] and sha(b)==row['SHA256']
    if row['relative'].endswith('.jpg'):
     image=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_COLOR);assert image is not None
     if row['relative'].startswith('assets/'):assert image.shape==(512,512,3)
    elif row['relative'].endswith('.json'):json.loads(b)
    else:b.decode('utf-8')
    result=dict(path=row['path'],size=row['size'],SHA256=row['SHA256'],server_metadata_exact=True,anonymous_GET_exact=True,readable=True)
    with guard:
     checks.append(result)
     if len(checks)%50==0:put(state,dict(state='public_GET_and_readback_verification',commit=commit,verified=len(checks),expected=len(paths),elapsed_seconds=time.time()-start));print(json.dumps(dict(verified=len(checks),expected=len(paths))),flush=True)
    return result
   except Exception as e:error=type(e).__name__
  raise RuntimeError('Public GET failed after bounded attempts: '+row['relative']+' '+str(error))
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:verified=list(pool.map(one,paths))
 r=dict(state='all129_asset_references_published_and_server_verified',repo=REPO,HF_prefix=PREFIX,commit=commit,parent=parent,new_commit=new_commit,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),unique_assets=129,reference_views=516,files=len(paths),bytes=sum(x['size'] for x in paths),all_files_server_metadata_and_anonymous_GET_SHA_readable=True,checks=verified,raw_asset_GLBSHA_bound=True,original_archives_modified=False,case70_uploaded=False,Remove_complete=False,overall_goal_complete=False,elapsed_seconds=time.time()-start)
 put(receipt,r);put(state,{k:v for k,v in r.items() if k!='checks'});print(json.dumps({k:v for k,v in r.items() if k!='checks'}),flush=True)
if __name__=='__main__':main()

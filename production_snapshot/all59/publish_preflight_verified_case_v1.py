"""One finite, locked publication after actual visual and full reader validation."""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, sys, time, urllib.request
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
U=Q/'all59_reconstruction_20261115_v1'
sys.path.insert(0,str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps'))
from huggingface_hub import HfApi, CommitOperationAdd
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();c=a.case;start=time.time()
 d=json.loads((U/f'CASE{c}_PUBLICATION_PAYLOAD.json').read_text());assert d['state']=='reviewed_publication_payload_ready'
 assert d['preflight']['actual_reader_frames']==588 and d['preflight']['actual_joint_frames']==490 and d['preflight']['operation_references_checked']==5
 assert not (U/f'CASE{c}_HF_DATA_VERIFIED.json').exists(),'Publication already has a receipt; never duplicate'
 locks=[]
 for path in [Q.parent/'HF_ORIGINAL_STRICT500_v1/HF_UPLOAD_OWNER.lock',Q/'REALCASE_QH_ARCHIVE_HF_WRITER.lock']:
  f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 api=HfApi(token=Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip());repo='FrontRear/VLMdataset'
 head=api.repo_info(repo,repo_type='dataset').sha;assert head==d['parent_HF_head'],'HF changed; reconcile current manifests before publishing'
 archive=d['preflight']['source_archive'];assert not api.get_paths_info(repo,[archive],repo_type='dataset'),'Case archive already exists; inspect rather than reupload'
 for r in d['files']:p=Path(r['local_path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['SHA256']
 commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,commit_message=f'Add reviewed StarReal case{c} with references, source Remove conditions and strict solid masks',operations=[CommitOperationAdd(path_in_repo=r['HF_path'],path_or_fileobj=r['local_path']) for r in d['files']]);oid=commit.oid
 (U/f'CASE{c}_HF_COMMIT_CREATED.json').write_text(json.dumps(dict(commit=oid,parent=head,case=c),indent=2))
 remote={r.path:r for r in api.get_paths_info(repo,[r['HF_path'] for r in d['files']],repo_type='dataset',revision=oid)};checks=[]
 for r in d['files']:
  z=remote[r['HF_path']];assert z.size==r['bytes']
  if z.lfs:assert z.lfs.sha256==r['SHA256'];server_kind='LFS_SHA256'
  else:
   b=Path(r['local_path']).read_bytes();assert z.blob_id==hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();server_kind='Git_blob'
  h=hashlib.sha256();num=0
  with urllib.request.urlopen('https://huggingface.co/datasets/'+repo+'/resolve/'+oid+'/'+r['HF_path'],timeout=150) as response:
   for b in iter(lambda:response.read(8388608),b''):h.update(b);num+=len(b)
  assert num==r['bytes'] and h.hexdigest()==r['SHA256']
  checks.append(dict(path=r['HF_path'],bytes=num,SHA256=h.hexdigest(),metadata=server_kind,public_GET_SHA256_exact=True))
  print(json.dumps(dict(verified_path=r['HF_path'],bytes=num)),flush=True)
 receipt=dict(case=c,state='current_case_latest_contract_published_and_server_verified',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=oid,parent=head,files=checks,preflight=d['preflight'],original_writer_locks_held=True,current_cases=d['current_cases'],required_cases=59,all59_complete=False,unchanged_prior_archives=True,GPU_used=False,elapsed_seconds=time.time()-start)
 local=U/f'CASE{c}_HF_DATA_VERIFIED.json';local.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
 final_path=f'StarReal/supplements/all59_completion_20261115_v1/CASE{c}_FINAL_VERIFIED.json'
 final=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=oid,commit_message=f'Record StarReal case{c} public download and server SHA verification',operations=[CommitOperationAdd(path_in_repo=final_path,path_or_fileobj=str(local))])
 with urllib.request.urlopen('https://huggingface.co/datasets/'+repo+'/resolve/'+final.oid+'/'+final_path,timeout=90) as r:b=r.read()
 assert hashlib.sha256(b).hexdigest()==sha(local)
 result=dict(case=c,state='final_public_receipt_verified',data_commit=oid,final_commit=final.oid,current_cases=d['current_cases'],required_cases=59,all_scope_complete=False,receipt_path=final_path,receipt_SHA256=sha(local))
 (U/f'CASE{c}_HF_FINAL_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 state=json.loads((U/'RUN_STATE.json').read_text());state.update(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='active_case'+c+'_published_remaining_work',current_completed=d['current_cases'],all_scope_complete=False,last_server_verified=result);state['remaining']=[x for x in state['remaining'] if x!=c];(U/'RUN_STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2))
 latest=json.loads((Q/'LATEST_FACTS.json').read_text());latest.update(all59_current_latest_contract_HF_cases=d['current_cases'],all59_remaining=59-d['current_cases'],all59_last_verified_case=c,all59_last_commit=final.oid,all59_goal_active=True,all_scope_complete=False);(Q/'LATEST_FACTS.json').write_text(json.dumps(latest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

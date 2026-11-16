from pathlib import Path
import os,json,hashlib,fcntl,datetime,sys,time,urllib.request
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1'
sys.path.insert(0,str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps'))
from huggingface_hub import HfApi,CommitOperationAdd
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
state_path=U/'CASE7505429_HF_500_POSTCOMMIT_VERIFICATION_FINITE_v2.json'
state=dict(case='7505429',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split()[21],state='running',GPU_used=False,reupload_existing_data=False)
def save():state_path.write_text(json.dumps(state,indent=2))
save();start=time.time()
try:
 locks=[]
 for p in [Q.parent/'HF_ORIGINAL_STRICT500_v1/HF_UPLOAD_OWNER.lock',Q/'REALCASE_QH_ARCHIVE_HF_WRITER.lock',U/'CASE7505429_HF_500_POSTCOMMIT_VERIFICATION_FINITE_v2.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 old_controller=json.loads((U/'CASE7505429_PREFLIGHT_PUBLISH_FINITE_20261116_v1.json').read_text());assert old_controller['state']=='failed'
 # Guard against accidental duplicate live controller; old PID/start identity must be gone.
 for oldpid,oldticks in [(old_controller['pid'],old_controller['startticks'])]:
  p=Path('/proc')/str(oldpid)/'stat'
  assert not p.exists() or p.read_text().split()[21]!=oldticks
 recon=json.loads((U/'CASE7505429_HF_500_RESPONSE_SERVER_COMMIT_RECONCILIATION_v1.json').read_text());assert recon['all35_metadata_exact']
 oid=recon['head'];assert oid=='a4efb95ba7e60e45e03db0aa6ae239bd96c25172'
 assert recon['recent_commits'][0]['id']==oid and 'case7505429' in recon['recent_commits'][0]['title']
 d=json.loads((U/'CASE7505429_PUBLICATION_PAYLOAD.json').read_text());assert d['preflight']['actual_reader_frames']==588 and d['preflight']['actual_joint_frames']==490 and d['preflight']['operation_references_checked']==5
 api=HfApi(token=Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip());repo='FrontRear/VLMdataset'
 remote={z.path:z for z in api.get_paths_info(repo,[r['HF_path'] for r in d['files']],repo_type='dataset',revision=oid)}
 checks=[]
 for r in d['files']:
  p=Path(r['local_path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['SHA256']
  z=remote[r['HF_path']];assert z.size==r['bytes']
  if z.lfs:assert z.lfs.sha256==r['SHA256'];kind='LFS_SHA256'
  else:
   b=p.read_bytes();assert z.blob_id==hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();kind='Git_blob'
  hh=hashlib.sha256();num=0
  with urllib.request.urlopen('https://huggingface.co/datasets/'+repo+'/resolve/'+oid+'/'+r['HF_path'],timeout=150) as response:
   for b in iter(lambda:response.read(8388608),b''):hh.update(b);num+=len(b)
  assert num==r['bytes'] and hh.hexdigest()==r['SHA256']
  checks.append(dict(path=r['HF_path'],bytes=num,SHA256=hh.hexdigest(),metadata=kind,public_GET_SHA256_exact=True))
  state.update(state='verifying_existing_server_commit',verified_files=len(checks),total_files=len(d['files']),last_verified=r['HF_path']);save();print(json.dumps(checks[-1]),flush=True)
 receipt=dict(case='7505429',state='current_case_latest_contract_published_and_server_verified',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=oid,parent=d['parent_HF_head'],files=checks,preflight=d['preflight'],original_writer_locks_held=True,current_cases=d['current_cases'],required_cases=59,all59_complete=False,unchanged_prior_archives=True,GPU_used=False,elapsed_seconds=time.time()-start,server_commit_HTTP500_response_reconciled=True,existing_commit_reused_without_archive_reupload=True,reconciliation=str(U/'CASE7505429_HF_500_RESPONSE_SERVER_COMMIT_RECONCILIATION_v1.json'))
 local=U/'CASE7505429_HF_DATA_VERIFIED.json';assert not local.exists();local.write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
 final_path='StarReal/supplements/all59_completion_20261115_v1/CASE7505429_FINAL_VERIFIED.json'
 head=api.repo_info(repo,repo_type='dataset').sha
 with urllib.request.urlopen('https://huggingface.co/datasets/'+repo+'/resolve/'+head+'/StarReal/DATASET_CURRENT.json',timeout=60) as r:pointer=json.load(r)
 assert pointer['current_cases']==53 and pointer['latest_verification']==final_path
 state.update(state='recording_verified_receipt_only',data_commit=oid);save()
 try:
  commit=api.create_commit(repo_id=repo,repo_type='dataset',parent_commit=head,commit_message='Record StarReal case7505429 public download and server SHA verification after successful commit returned HTTP500',operations=[CommitOperationAdd(path_in_repo=final_path,path_or_fileobj=str(local))]);final_oid=commit.oid
 except Exception as e:
  # A second server response error can also occur after success. Verify actual receipt, never blindly repeat.
  final_oid=api.repo_info(repo,repo_type='dataset').sha;state['receipt_commit_response_error']=type(e).__name__;save()
 with urllib.request.urlopen('https://huggingface.co/datasets/'+repo+'/resolve/'+final_oid+'/'+final_path,timeout=90) as r:b=r.read()
 assert hashlib.sha256(b).hexdigest()==sha(local)
 result=dict(case='7505429',state='final_public_receipt_verified',data_commit=oid,final_commit=final_oid,current_cases=53,required_cases=59,all_scope_complete=False,receipt_path=final_path,receipt_SHA256=sha(local),existing_data_commit_reused_after_HTTP500=True,GPU_used=False)
 (U/'CASE7505429_HF_FINAL_VERIFIED.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 s=json.loads((U/'RUN_STATE.json').read_text());s.update(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='active_case7505429_published_remaining6_CPU_ONLY',current_completed=53,all_scope_complete=False,last_server_verified=result);s['remaining']=[x for x in s['remaining'] if x!='7505429'];(U/'RUN_STATE.json').write_text(json.dumps(s,ensure_ascii=False,indent=2))
 f=Q/'LATEST_FACTS.json';s=json.loads(f.read_text());s.update(all59_current_latest_contract_HF_cases=53,all59_remaining=6,all59_last_verified_case='7505429',all59_last_commit=final_oid,all59_goal_active=True,all_scope_complete=False);f.write_text(json.dumps(s,ensure_ascii=False,indent=2))
 state.update(state='terminal0_existing_commit_full_GET_and_final_receipt_verified',exit_code=0,final=result,elapsed_seconds=time.time()-start);save()
except Exception as e:state.update(state='failed',exit_code=1,error=repr(e));save();raise

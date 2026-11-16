from pathlib import Path
import os,json,time,datetime,subprocess,fcntl
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1';R=Path('/tmp/starreal_uid1007_case6374204_true4K_cpu_v2');out=R/'packages/6374204/fixed_world_yellow_depth_v5_staging';pid=3029284
lock=(Q/'logs/case6374204_core_delivery_owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
st=dict(case='6374204',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],state='waiting_specific_reviewed_portable_stage',stage_pid=pid,steps=[],GPU_used=False,quality_HF_ready=False,HF_upload_started=False)
state=U/'CASE6374204_CORE_DURABLE_PREP_FINITE_v3.json'
def save():state.write_text(json.dumps(st,indent=2)+'\n')
save()
try:
 start=time.monotonic()
 while True:
  p=out/'BUILD_STATE.json'
  if p.exists():
   b=json.loads(p.read_text())
   if b['pid']!=pid:
    assert not Path('/proc/'+str(b['pid'])).exists() and Path('/proc/'+str(pid)).exists(),('unexpected live different stage owner',b['pid'],pid)
    assert time.monotonic()-start<1200,'bounded observation expired, healthy child not signalled'
    time.sleep(2);continue
   if b.get('exit_code')==0 and b['state'].startswith('terminal0'):break
   if b.get('exit_code')==1:raise RuntimeError('actual stage terminal1: '+str(b))
  assert Path('/proc/'+str(pid)).exists(),'specific stage handle missing before business completion'
  assert time.monotonic()-start<1200,'bounded stage observation expired, healthy child not signalled'
  time.sleep(2)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',STARREAL_QH_ONLY_COMPONENT_OWNER='1')
 for name,args in [
  ('core_release',['python3','-u',str(U/'release_case6374204_true4K_node_local_archive_QH_only_v1.py'),'--case','6374204']),
  ('copy_durable_independent_members_SHA',['python3','-u',str(U/'copy_case6374204_true4K_accepted_archive_to_durable_QH_v1.py')]),
  ('prepare_latest_Remove_source_and_reference_not_HF',['python3','-u',str(U/'prepare_case6374204_true4K_latest_contract_v1.py'),'--case','6374204','--revision','fixed_world_yellow_depth_v5'])]:
  st.update(state='running',step=name);save();start=time.monotonic()
  p=U/('CASE6374204_'+name+'_CORE_FINITE_v3.log')
  with p.open('xb') as f:
   child=subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT);st['child_pid']=child.pid;save();rc=child.wait(timeout=1200)
  st['steps'].append(dict(step=name,exit_code=rc,seconds=time.monotonic()-start));save()
  if rc:raise RuntimeError('actual step '+name+' exited '+str(rc))
 st.update(state='terminal0_core_durable_and_latest_Remove_prepared_NEEDS_ACTUAL_REMOVE_REVIEW_AND_STRICT588_AND_UNIQUE_HF',exit_code=0,quality_HF_ready=False);save()
except BaseException as e:st.update(state='terminal1_or_bounded_observation_exception_no_child_signalled',error=repr(e),exit_code=1);save();raise

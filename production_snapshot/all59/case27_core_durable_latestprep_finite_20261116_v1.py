from pathlib import Path
import json,subprocess,os,time,fcntl
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1';R=Path('/tmp/starreal_uid1007_case27_cpu_v2')
lock=(U/'CASE27_CORE_DURABLE_PREP_FINITE_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=U/'CASE27_CORE_DURABLE_PREP_FINITE_v1.json';d=dict(case='27',pid=os.getpid(),state='waiting_current_portable_stage',steps=[],GPU_used=False,HF_upload=False)
def save():state.write_text(json.dumps(d,indent=2))
save()
try:
 deadline=time.monotonic()+600
 while time.monotonic()<deadline:
  b=json.loads((R/'packages/27/fixed_world_yellow_depth_v5_staging/BUILD_STATE.json').read_text())
  if 'failed' in b['state'].lower() or 'terminal1' in b['state'].lower():raise RuntimeError(b)
  if (R/'packages/27/fixed_world_yellow_depth_v5_staging/PORTABLE_FINAL_AUDIT.json').exists():break
  pp=Path('/proc/2724278')
  if not pp.exists():raise RuntimeError('Stage exited without finalaudit, inspect actual log')
  st=(pp/'stat').read_text().split(') ',1)[1].split();assert pp.stat().st_uid==1007 and st[19]=='5033345' and st[0]!='Z'
  time.sleep(3)
 else:raise TimeoutError('Boundedwait600, do not cancel healthy stage')
 for name,args in [('release_case27_native_node_local_archive_QH_only_v1.py',['--case','27']),('copy_case27_native_accepted_archive_to_durable_QH_v1.py',[]),('prepare_case27_native_latest_contract_v1.py',['--case','27','--revision','fixed_world_yellow_depth_v5'])]:
  d.update(state='running',step=name);save();rc=subprocess.call(['python3','-u',str(U/name)]+args,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',STARREAL_QH_ONLY_COMPONENT_OWNER='1'));d['steps'].append(dict(step=name,exit_code=rc));save();assert rc==0
 d.update(state='terminal0_pending_actual_Remove49_and_reference_visual_review',exit_code=0);save()
except Exception as e:d.update(state='failed',exit_code=1,error=repr(e));save();raise

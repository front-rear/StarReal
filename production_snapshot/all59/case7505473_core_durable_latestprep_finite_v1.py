from pathlib import Path
import os,json,subprocess,time,datetime
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1');R=Path('/tmp/starreal_uid1007_case7505473_true4K_cpu_v2');CASE='7505473'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',STARREAL_QH_ONLY_COMPONENT_OWNER='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
state=U/('CASE'+CASE+'_CORE_DURABLE_LATESTPREP_FINITE_v1.json');d=dict(case=CASE,pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],steps=[],GPU_used=False,HF_upload=False)
def put():state.write_text(json.dumps(d,indent=2))
for name,args in [('stage',['python3','-u',str(U/('stage_case'+CASE+'_reviewed_true4K_portable_QH_v1.py'))]),('release_core',['python3','-u',str(U/('release_case'+CASE+'_true4K_node_local_archive_QH_only_v1.py')),'--case',CASE]),('durable_copy',['python3','-u',str(U/('copy_case'+CASE+'_true4K_accepted_archive_to_durable_QH_v1.py'))]),('latestprep',['python3','-u',str(U/('prepare_case'+CASE+'_true4K_latest_contract_v1.py')),'--case',CASE,'--revision','fixed_world_yellow_depth_v5'])]:
 d.update(state='running',step=name);put();t=time.time()
 with (U/('CASE'+CASE+'_'+name+'_CORE_CHAIN_v1.log')).open('xb') as out:
  p=subprocess.Popen(args,env=env,stdout=out,stderr=subprocess.STDOUT);d['child_pid']=p.pid;put()
  try:rc=p.wait(timeout=1200)
  except subprocess.TimeoutExpired:d.update(state='verified_wait_timeout_child_not_signalled',exit_code=None);put();raise
 d['steps'].append(dict(name=name,exit_code=rc,seconds=time.time()-t));put()
 if rc:d.update(state='terminal1_preserve_completed_no_automatic_retry',exit_code=rc);put();break
else:d.update(state='terminal0_pending_actual_Remove49_reference_review_and_strict588',exit_code=0);put()

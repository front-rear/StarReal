from pathlib import Path
import subprocess,json,time,os,datetime,fcntl
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1'); S=U/'CASE111_CORE_RELEASE_DURABLE_FINITE_20261116_v3.json'
lock=(U/'CASE111_CORE_RELEASE_DURABLE_FINITE_20261116_v3.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.environ.update(STARREAL_QH_ONLY_COMPONENT_OWNER='1',STARREAL_CPU_ROOT='/tmp/starreal_uid1007_case111_true4K_cpu_v2')
d=dict(case='111',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split()[21],state='running',GPU_used=False,HF_upload=False,steps=[])
def save(): S.write_text(json.dumps(d,indent=2))
save()
try:
 for name,args in [('stage_case111_reviewed_true4K_portable_QH_v3.py',[]),('release_case111_true4K_node_local_archive_QH_only_v2.py',['--case','111']),('copy_case111_true4K_accepted_archive_to_durable_QH_v2.py',[]),('prepare_case111_true4K_latest_contract_v2.py',['--case','111','--revision','fixed_world_yellow_depth_v5'])]:
  s=time.time();rc=subprocess.call(['python3',str(U/name)]+args);d['steps'].append(dict(script=name,exit_code=rc,seconds=time.time()-s));save()
  if rc: raise RuntimeError(name+' exit '+str(rc))
 d.update(state='terminal0_core_and_latest_fields_prepared_pending_actual_Remove_review',exit_code=0);save()
except Exception as e:
 d.update(state='failed',error=repr(e),exit_code=1);save();raise

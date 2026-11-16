from pathlib import Path
import subprocess,json,time,os,datetime,fcntl
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1'); S=U/'CASE109_CORE_RELEASE_DURABLE_FINITE_20261116_v1.json'
lock=(U/'CASE109_CORE_RELEASE_DURABLE_FINITE_20261116_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d=dict(case='109',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split()[21],state='running',GPU_used=False,HF_upload=False,steps=[])
def save(): S.write_text(json.dumps(d,indent=2))
save()
try:
 for name,args in [('release_case109_true4K_node_local_archive_QH_only_v2.py',['--case','109']),('copy_case109_true4K_accepted_archive_to_durable_QH_v2.py',[])]:
  s=time.time();rc=subprocess.call(['python3',str(U/name)]+args);d['steps'].append(dict(script=name,exit_code=rc,seconds=time.time()-s));save()
  if rc: raise RuntimeError(name+' exit '+str(rc))
 d.update(state='terminal0_core_release_and_durable_copy_verified_pending_latest_contract',exit_code=0);save()
except Exception as e:
 d.update(state='failed',error=repr(e),exit_code=1);save();raise

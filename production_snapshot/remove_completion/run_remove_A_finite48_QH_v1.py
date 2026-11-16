"""Finite observed-A addition consumes completed archive-specific B supplements."""
from pathlib import Path
import concurrent.futures,fcntl,json,os,subprocess,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';guard=(T/'logs/REMOVE_A_FINITE48_CPU.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);state=T/'logs/REMOVE_A_FINITE48_CURRENT.json';assert not state.exists();start=time.time();seen=set();done=[];errors=[];futures={};deadline=time.monotonic()+7200

def save(status):
 d=dict(state=status,pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),elapsed_seconds=time.time()-start,expected=48,admitted=len(seen),completed=len(done),failed=len(errors),jobs=done,errors=errors,workers=4,GPU_used=False,automatic_upload=False);p=state.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(state)
def one(row):
 root=Path(row['manifest']).parent;key=row['key'];out=root/'A_OBSERVED_RECORD.json'
 if out.exists():
  d=json.loads(out.read_text());assert len(d['frames_verified'])==49;return dict(case=row['case'],key=key,root=str(root),reused=True)
 log=T/'logs/remove_frame_jobs'/(key+'.A_observed.log');assert not log.exists()
 with log.open('xb') as f:
  e=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1');p=subprocess.run(['python3',str(T/'code/add_remove_observed_A_conditions_QH_v1.py'),'--root',str(root)],stdout=f,stderr=subprocess.STDOUT,env=e)
 if p.returncode:return dict(error=True,case=row['case'],key=key,exit_code=p.returncode,log=str(log))
 d=json.loads(out.read_text());assert len(d['frames_verified'])==49;return dict(case=row['case'],key=key,root=str(root),reused=False)
save('running_finite48_A_source_conditions')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 while len(seen)<48 or futures:
  p=T/'logs/REMOVE_FINITE48_CURRENT.json';current=json.loads(p.read_text())
  for row in current['jobs']:
   if row['key'] in seen or len(futures)>=4:continue
   seen.add(row['key']);futures[pool.submit(one,row)]=row['key']
  if futures:
   ready,_=concurrent.futures.wait(futures,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in ready:
    key=futures.pop(f)
    try:r=f.result()
    except Exception as e:r=dict(error=True,key=key,exception=type(e).__name__+': '+str(e)[:160])
    (errors if r.get('error') else done).append(r);print(json.dumps(r),flush=True)
   save('running_finite48_A_source_conditions')
  else:time.sleep(1)
  if current['state'].startswith('terminal1') and not futures:break
  if time.monotonic()>deadline:errors.append(dict(error='finite_admission_deadline_no_healthy_kill'));break
save('terminal0_all48_A_source_conditions_PENDING_actual_review' if len(done)==48 and not errors else 'terminal1_preserve_outputs')

"""Finite48 archive-specific Remove additions; actual final review remains mandatory."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,subprocess,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';N=Path('/tmp/starreal_uid1007_case19_cpu_v1/reference_completion_20261115_v1');lock=(T/'logs/REMOVE_FINITE48_CPU.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);review=json.loads((T/'reviews/REMOVE_PILOT64_ACTUAL_REVIEW_v1.json').read_text());assert review['may_generate_other_published_Remove_supplements'] and not review['overall_goal_complete'];state=T/'logs/REMOVE_FINITE48_CURRENT.json';assert not state.exists();start=time.time();seen=set();done=[];errors=[];inflight={};expected=48;deadline=time.monotonic()+7200;(T/'logs/remove_frame_jobs').mkdir(exist_ok=True)
def save(status):
 d=dict(state=status,pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],elapsed_seconds=time.time()-start,expected=48,admitted=len(seen),completed=len(done),failed=len(errors),inflight=len(inflight),jobs=done,errors=errors,workers=4,GPU_used=False,quality_auto_accept=False,automatic_upload=False);p=state.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(state)
def one(row):
 key=row['archive_LFS_SHA256'];out=N/'remove_outputs'/key;manifest=out/'MANIFEST.json'
 if manifest.exists():
  m=json.loads(manifest.read_text());assert m['HF_source_archive']==row['HF_path'] and m['frames']==49 and len(m['frames_verified'])==49;return dict(archive=row['HF_path'],key=key,case=m['case'],manifest=str(manifest),reused=True)
 cfg=T/'logs/remove_frame_jobs'/(key+'.row.json');assert not cfg.exists();cfg.write_text(json.dumps(row,indent=2));log=T/'logs/remove_frame_jobs'/(key+'.log');assert not log.exists()
 with log.open('xb') as f:
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');p=subprocess.run(['python3',str(T/'code/build_remove_source_conditions_QH_v2.py'),'--row-json',str(cfg)],stdout=f,stderr=subprocess.STDOUT,env=env)
 if p.returncode!=0:return dict(error=True,archive=row['HF_path'],key=key,exit_code=p.returncode,log=str(log))
 m=json.loads(manifest.read_text());assert m['frames']==49 and len(m['frames_verified'])==49;return dict(archive=row['HF_path'],key=key,case=m['case'],manifest=str(manifest),reused=False)
save('running_finite48_waiting_actual_verified_inputs')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 while len(seen)<48 or inflight:
  p=T/'logs/REMOVE_INPUT_ARCHIVES_CURRENT.json';rows=json.loads(p.read_text())['rows'] if p.exists() else []
  for row in rows:
   key=row.get('archive_LFS_SHA256')
   if not row.get('verified') or key in seen or len(inflight)>=4:continue
   seen.add(key);inflight[pool.submit(one,row)]=row['HF_path']
  if inflight:
   ready,_=concurrent.futures.wait(inflight,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
   for f in ready:
    archive=inflight.pop(f)
    try:r=f.result()
    except Exception as e:r=dict(error=True,archive=archive,exception=type(e).__name__+': '+str(e)[:160])
    if r.get('error'):errors.append(r)
    else:done.append(r)
    print(json.dumps(r),flush=True)
   save('running_finite48_generating_conditions_pending_actual_review')
  else:time.sleep(1)
  final=T/'inventory/REMOVE_ARCHIVE_INPUTS_v1.json'
  if final.exists() and json.loads(final.read_text())['verified']<48 and not inflight:break
  if time.monotonic()>deadline:
   errors.append(dict(error='finite_admission_deadline_elapsed_no_healthy_process_killed'));break
save('terminal0_all48_generated_PENDING_actual_review' if len(done)==48 and not errors else 'terminal1_preserve_all_outputs_exact_failed_archive_only')
print(json.dumps(dict(completed=len(done),errors=errors,elapsed_seconds=time.time()-start,overall_goal_complete=False)),flush=True)

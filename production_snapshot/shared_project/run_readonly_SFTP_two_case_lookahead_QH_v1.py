"""Finite two-case pipeline; wait only for exact incoming bytes, no AI polling/retry."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,subprocess,time,traceback
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');source=N/'inputs/StarReal_native_readonly_SFTP_66_69_v1';control=N/'ORIGINAL_SFTP_BATCH_CONTROL_66_69_v1.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 guard=(Q/'logs/READONLY_SFTP_TWO_CASE_LOOKAHEAD_FINITE.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();spec=json.loads(control.read_text());rows={};state=dict(state='finite_exact_file_lookahead_running',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],cases=rows,deadline_seconds=1800,GPU_used=False,remote_6000D_execution=False,remote_6000D_writes=False,automatic_retry=False,quality_accepted=False,new_complete_groups=0)
 def save():
  state['elapsed_seconds']=time.time()-start
  for base in [N,Q]:
   p=base/'logs/READONLY_SFTP_TWO_CASE_LOOKAHEAD_FINITE_v1.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(p)
 def run(case,name,args):
  env=dict(os.environ,STARREAL_CASE=case,STARREAL_CPU_ROOT=str(N),STARREAL_QH_ONLY_COMPONENT_OWNER='1',STARREAL_FRAME_WORKERS='32',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');log=N/'logs'/('CASE'+case+'_'+name+'.SFTP_lookahead.log');assert not log.exists()
  t=time.time()
  with log.open('x') as f:r=subprocess.run(['python3',str(N/name)]+args,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=600)
  rows[case][name]=dict(exit_code=r.returncode,seconds=time.time()-t,log=str(log));save();assert r.returncode==0,'Inspect finite stage log; no automatic retry: '+str(log)
 save()
 try:
  for case in ['66','69']:
   rows[case]=dict(state='waiting_existing_transfer_exact_bytes');save();expected=spec['expected_files'][case]
   while True:
    assert time.time()-start<1800,'Finite input-transfer deadline reached; no claim of readiness'
    ready=all((source/case/rel).is_file() and (source/case/rel).stat().st_size==row['size'] for rel,row in expected.items())
    if ready and all(sha(source/case/rel)==row['SHA256'] for rel,row in expected.items()):break
    time.sleep(10)
   rows[case]['state']='exact_transfer_all_file_SHA_matched';save()
   run(case,'prepare_native_from_readonly_SFTP_QH_v1.py',['--transfer',str(source),'--spec',str(control),'--cases',case]);rows[case]['state']='Native49_inputs_verified_persistent';save()
   run(case,'precompute_native_PCF_background49_QH_only_owner_cpu_v2.py',[])
   b=N/'outputs'/case/'pcf_background49_precomputed_QH_v1';job=json.loads((b/'JOB.json').read_text());r=json.loads((b/'RESULT.json').read_text());assert job['exit_code']==0 and job['frames_finished']==r['frames']==49 and r['camera_source_SHA256']==sha(N/f'inputs/case{case}_native_CPU_v1/cameras.npz')
   for x in r['frame_receipts']:assert sha(b/'shared_pcf'/f"{x['frame']:06d}.png")==x['PNG_SHA256']
   records=[]
   for p in sorted(b.rglob('*')):
    if not p.is_file():continue
    dst=Q/'component_cache'/case/'pcf_background49_precomputed_QH_v1'/p.relative_to(b);dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():assert sha(dst)==sha(p)
    else:shutil.copyfile(p,dst);assert sha(dst)==sha(p)
    records.append(dict(path=str(dst.relative_to(Q)),SHA256=sha(dst)))
   proof=dict(case=case,state='Native_inputs_and_exact_PCF_background49_ready_NOT_delivery',source_receipt=f'receipts/CASE{case}_READONLY_SFTP_NATIVE_INPUTS_v1.json',camera_SHA256=r['camera_source_SHA256'],background_seconds=r['elapsed_seconds'],background_frames=49,files=records,GPU_used=False,remote_6000D_execution=False,remote_6000D_writes=False,quality_accepted=False,new_complete_groups=0)
   p=Q/f'receipts/CASE{case}_READONLY_SFTP_LOOKAHEAD_READY_v1.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2));rows[case].update(state=proof['state'],background_seconds=r['elapsed_seconds'],receipt=str(p));save()
  state.update(state='terminal0_both_exact_inputs_and_background49_persistent_NOT_delivery',exit_code=0);save()
 except BaseException as e:state.update(state='terminal1_preserve_completed_components_no_automatic_retry',exit_code=1,error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()

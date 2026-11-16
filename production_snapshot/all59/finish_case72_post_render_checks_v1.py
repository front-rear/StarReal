from pathlib import Path
import concurrent.futures, datetime, fcntl, json, os, subprocess, time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1')
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
U=Q/'all59_reconstruction_20261115_v1'
def main():
 t=time.time();case='72';job=json.loads((N/'outputs/72/two_pcf49_fixed_world_yellow_QH_v1/JOB.json').read_text());assert job['exit_code']==0 and job['frames_finished']==49
 guard=(Q/'logs/CASE72_FINITE_REVIEWED_CHAIN.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=os.environ.copy();env.update(STARREAL_CASE=case,STARREAL_CPU_ROOT=str(N),STARREAL_QH_ONLY_COMPONENT_OWNER='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def run(spec):
  py,name=spec;log=U/('CASE72_POST_'+name+'.log');assert not log.exists();begin=time.time()
  with log.open('x') as f:r=subprocess.run([py,'-u',str(N/name)],stdout=f,stderr=subprocess.STDOUT,env=env,timeout=900)
  row=dict(script=name,exit_code=r.returncode,elapsed_seconds=time.time()-begin,log=str(log));print(json.dumps(row),flush=True);return row
 specs=[('python3','build_native_historical_comparison49_QH_cpu_v3.py'),('python3','review_native_conditions_and_joint_pixels_QH_v2.py'),('/mnt/atlas/fangrui/repro_video_20261105/_envs/pisco/bin/python','check_native_actual_Original_entry_QH_v2.py')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,specs))
 result=dict(case=case,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='all_post_render_checks_passed_awaiting_actual_final_visual_review' if all(r['exit_code']==0 for r in rows) else 'specific_post_render_stage_failed_preserve_successful_outputs',stages=rows,all_stages_passed=all(r['exit_code']==0 for r in rows),PCF_all49_generated_and_decoded=True,quality_accepted=False,automatic_upload=False,GPU_used=False,elapsed_seconds=time.time()-t)
 (U/'CASE72_POST_RENDER_ACTUAL_RESULTS_v1.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

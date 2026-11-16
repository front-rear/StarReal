from pathlib import Path
import os,subprocess,concurrent.futures,json,time,argparse,fcntl,hashlib,datetime,threading
ap=argparse.ArgumentParser();ap.add_argument("--case",required=True);ap.add_argument("--tag",required=True);args=ap.parse_args();CASE=args.case;TAG=args.tag;assert CASE.isdecimal() and TAG.replace("_","").isalnum()
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');os.environ.update(STARREAL_CASE=CASE,STARREAL_CPU_ROOT=str(N),STARREAL_QH_ONLY_COMPONENT_OWNER='1',STARREAL_GEOMETRY_REVIEW=f'CASE{CASE}_FIVE_ROLE_ALL49_GEOMETRY_ACTUAL_REVIEW_v1.json',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.time();state=N/'logs'/f'CASE{CASE}_{TAG}_FINITE_PARALLEL_PRODUCTION_v3.json';assert not state.exists();rows=[]
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');guard=(Q/'logs'/f'CASE{CASE}_FINITE_REVIEWED_CHAIN.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
review=N/'reviews'/os.environ['STARREAL_GEOMETRY_REVIEW'];v=json.loads(review.read_text());assert v['case']==CASE and v['may_generate_full49_conditions'] and not v['quality_accepted'] and len(v['actual_viewed_all49_sheets'])==10
for row in v['actual_viewed_all49_sheets']:assert hashlib.sha256((N/row['path']).read_bytes()).hexdigest()==row['SHA256']
for folder in ['geometry_masks49_cached_QH_v1','two_pcf49_fixed_world_yellow_QH_v1','historical_joint_comparison49_QH_v1','actual_condition_palette_review_QH_v1']:assert not (N/'outputs'/CASE/folder).exists(),'Existing output: resume exact failed stage separately, never replay healthy or complete work'
identity=dict(uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],geometry_review_SHA256=hashlib.sha256(review.read_bytes()).hexdigest())
status='running';write_guard=threading.RLock()
def save():
 with write_guard:
  data=dict(case=CASE,tag=TAG,pid=os.getpid(),identity=identity,state=status,finite=True,automatic_quality=False,automatic_upload=False,stages=list(rows),elapsed_seconds=time.time()-start)
  if status.startswith('terminal'):data['exit_code']=0 if status=='terminal0_awaiting_actual_final_review' else 1
  t=state.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(state)

def run(name,py='python3'):
 t=time.time();log=N/'logs'/(f'CASE{CASE}_{TAG}_'+name+'.finite.log');assert not log.exists()
 with log.open('x') as f:p=subprocess.run([py,str(N/name)],stdout=f,stderr=subprocess.STDOUT)
 row=dict(script=name,exit_code=p.returncode,elapsed_seconds=time.time()-t,log=str(log));rows.append(row);save();print(json.dumps(row),flush=True);assert p.returncode==0,row
try:
 save()
 for name in ['build_native_cached_conditions49_QH_v3.py','verify_native_conditions49_QH_v2.py','prepare_native_source_appearance_reference_QH_v2.py','build_native_case_two_pcf49_QH_cpu_v4.py']:run(name)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:
  fs=[p.submit(run,'build_native_historical_comparison49_QH_cpu_v3.py'),p.submit(run,'review_native_conditions_and_joint_pixels_QH_v2.py'),p.submit(run,'check_native_actual_Original_entry_QH_v2.py','/mnt/atlas/fangrui/repro_video_20261105/_envs/pisco/bin/python')]
  for f in fs:f.result()
 status='terminal0_awaiting_actual_final_review';save()
except BaseException:
 status='terminal1_preserve_outputs_no_retry';save();raise
print('TERMINAL0 awaiting actual all49 PCF/history/palette review, no automatic release',flush=True)

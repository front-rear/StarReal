from pathlib import Path
import os,subprocess,concurrent.futures,json,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');os.environ.update(STARREAL_CASE='64',STARREAL_CPU_ROOT=str(N),STARREAL_QH_ONLY_COMPONENT_OWNER='1',STARREAL_GEOMETRY_REVIEW='CASE64_FIVE_ROLE_ALL49_GEOMETRY_ACTUAL_REVIEW_v1.json',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.time();state=N/'logs/CASE64_FINITE_PARALLEL_PRODUCTION_v1.json';assert not state.exists();rows=[]
def save():state.write_text(json.dumps(dict(case='64',pid=os.getpid(),finite=True,automatic_quality=False,automatic_upload=False,stages=rows,elapsed_seconds=time.time()-start),indent=2))
def run(name,py='python3'):
 t=time.time();log=N/'logs'/('CASE64_'+name+'.finite.log');assert not log.exists()
 with log.open('x') as f:p=subprocess.run([py,str(N/name)],stdout=f,stderr=subprocess.STDOUT)
 row=dict(script=name,exit_code=p.returncode,elapsed_seconds=time.time()-t,log=str(log));rows.append(row);save();print(json.dumps(row),flush=True);assert p.returncode==0,row
save()
for name in ['build_native_cached_conditions49_QH_v3.py','verify_native_conditions49_QH_v2.py','prepare_native_source_appearance_reference_QH_v2.py','build_native_case_two_pcf49_QH_cpu_v4.py']:run(name)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:
 fs=[p.submit(run,'build_native_historical_comparison49_QH_cpu_v3.py'),p.submit(run,'review_native_conditions_and_joint_pixels_QH_v2.py'),p.submit(run,'check_native_actual_Original_entry_QH_v2.py','/mnt/atlas/fangrui/repro_video_20261105/_envs/pisco/bin/python')]
 for f in fs:f.result()
print('TERMINAL0 awaiting actual all49 PCF/history/palette review, no automatic release',flush=True)

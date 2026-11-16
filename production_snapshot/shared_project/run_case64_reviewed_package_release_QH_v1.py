from pathlib import Path
import os,subprocess,json,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');os.environ.update(STARREAL_CASE='64',STARREAL_CPU_ROOT=str(N),STARREAL_QH_ONLY_COMPONENT_OWNER='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');rows=[];t=time.time()
for name,args in [('stage_case64_reviewed_portable_QH_v1.py',[]),('release_case64_node_local_archive_QH_only_v1.py',['--case','64']),('copy_case64_accepted_archive_to_durable_QH_v1.py',[])]:
 p=N/'logs'/('CASE64_'+name+'.finite.log');assert not p.exists();start=time.time()
 with p.open('x') as log:r=subprocess.run(['python3',str(N/name)]+args,stdout=log,stderr=subprocess.STDOUT)
 row=dict(script=name,exit_code=r.returncode,seconds=time.time()-start,log=str(p));rows.append(row);print(json.dumps(row),flush=True);assert r.returncode==0,row
(N/'logs/CASE64_FINITE_REVIEWED_PACKAGE_RELEASE_v1.json').write_text(json.dumps(dict(case='64',steps=rows,elapsed_seconds=time.time()-t,automatic_upload=False),indent=2))

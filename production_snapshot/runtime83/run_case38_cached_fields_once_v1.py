from pathlib import Path
import os,subprocess,sys,time,json
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');assert os.environ.get('ROOT_ORIGINAL_CASE38_COMPONENT_LOCK_HELD')=='1'
for name in ['build_native_cached_conditions49_QH_v2.py','verify_case38_conditions49_QH_v1.py','prepare_case38_source_appearance_reference_QH_v1.py']:
 t=time.time();r=subprocess.run([sys.executable,str(N/name)]);print(json.dumps(dict(stage=name,exit_code=r.returncode,seconds=time.time()-t)),flush=True);assert r.returncode==0

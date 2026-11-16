from pathlib import Path
import subprocess,os,json
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1');N=Path('/tmp/starreal_uid1007_case19_cpu_v1')
env=dict(os.environ,STARREAL_CASE='84',STARREAL_FRAME_WORKERS='32',STARREAL_QH_ONLY_COMPONENT_OWNER='1',CUDA_VISIBLE_DEVICES='')
for argv in [['python3','-u',str(N/'precompute_native_PCF_background49_QH_only_owner_cpu_v2.py')],['python3','-u',str(U/'persist_completed_background_on_QH_direct_v2.py'),'84']]:
 r=subprocess.run(argv,env=env)
 if r.returncode:raise SystemExit(r.returncode)
print(json.dumps({'case':'84','terminal':0,'background_only_not_delivery':True}),flush=True)

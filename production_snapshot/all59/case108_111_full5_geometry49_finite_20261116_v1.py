from pathlib import Path
import subprocess,json,os
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
steps=[]
for c in ['108','111']:
 env=dict(os.environ,STARREAL_CASE=c,STARREAL_CPU_ROOT='/tmp/starreal_uid1007_case'+c+'_true4K_cpu_v2',**{'ROOT_ORIGINAL_CASE'+c+'_COMPONENT_LOCK_HELD':'1'})
 rc=subprocess.call(['python3',str(U/'build_case108_111_true4K_five_role_geometry49_QH_v4.py')],env=env);steps.append(dict(case=c,exit_code=rc));(U/'CASE108_111_FULL5_GEOMETRY49_FINITE_20261116_v1_RESULT.json').write_text(json.dumps(dict(pid=os.getpid(),steps=steps,GPU_used=False),indent=2));assert rc==0

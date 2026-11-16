from pathlib import Path
import subprocess,json,os
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
steps=[]
for p in ['/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1/import_case108_actual_two_camera_mesh_history_HF_QH_finite_20261116_v1.py', '/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1/import_case111_actual_two_camera_mesh_history_HF_QH_finite_20261116_v1.py']:
 r=subprocess.call(['python3',p]);steps.append(dict(script=p,exit_code=r));(U/'CASE108_111_HISTORY_IMPORT_FINITE_20261116_v1_RESULT.json').write_text(json.dumps(dict(pid=os.getpid(),steps=steps,GPU_used=False),indent=2));assert r==0

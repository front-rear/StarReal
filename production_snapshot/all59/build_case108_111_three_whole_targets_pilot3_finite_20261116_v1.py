from pathlib import Path
import subprocess,json,os
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
steps=[]
for p in ['/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1/build_case108_three_whole_supported_targets_true4K_pilot3_QH_v1.py', '/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1/build_case111_three_whole_supported_targets_true4K_pilot3_QH_v1.py']:
 rc=subprocess.call(['python3',p]);steps.append(dict(script=p,exit_code=rc));(U/'CASE108_111_THREE_TARGETS_PILOT3_FINITE_20261116_v1_RESULT.json').write_text(json.dumps(dict(pid=os.getpid(),steps=steps,GPU_used=False),indent=2));assert rc==0

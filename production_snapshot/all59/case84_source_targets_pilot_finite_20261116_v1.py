import subprocess,json
from pathlib import Path
u=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
for name in ['build_case84_source_visible_paving_estimated_foot_rigid_pilot3_QH_v1.py', 'build_case84_three_whole_paving_targets_pilot3_QH_v1.py']:
 print(json.dumps({'step':name}),flush=True)
 r=subprocess.run(['python3','-u',str(u/name)])
 if r.returncode:raise SystemExit(r.returncode)

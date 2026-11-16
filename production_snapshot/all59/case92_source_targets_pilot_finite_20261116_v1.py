from pathlib import Path
import subprocess
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
for name in ['build_case92_source_visible_table_rigid_pilot3_QH_v1.py', 'build_case92_three_whole_table_targets_pilot3_QH_v1.py']:
 r=subprocess.run(["python3","-u",str(U/name)])
 if r.returncode:raise SystemExit(r.returncode)

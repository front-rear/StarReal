import subprocess,os
steps=[['python3', '-u', '/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/code/import_native_historical_inputs_HF_QH_v3.py', '--case', '92'], ['python3', '-u', '/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/code/read_native_history_cameras_and_source49_QH_v2.py']]
for argv in steps:
 r=subprocess.run(argv)
 if r.returncode:raise SystemExit(r.returncode)

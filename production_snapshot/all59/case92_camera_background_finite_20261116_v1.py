import subprocess,os
steps=[['python3', '-u', '/tmp/starreal_uid1007_case19_cpu_v1/precompute_native_PCF_background49_QH_only_owner_cpu_v2.py'], ['python3', '-u', '/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1/persist_completed_background_on_QH_direct_v2.py', '92']]
for argv in steps:
 r=subprocess.run(argv)
 if r.returncode:raise SystemExit(r.returncode)

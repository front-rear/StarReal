import subprocess,json,os
from pathlib import Path
u=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1")
for name,args in [("prepare_new_case_publication_v5.py",["--case","92","--revision","fixed_world_yellow_depth_v5"]),("publish_preflight_verified_case_v1.py",["--case","92"])]:
 print(json.dumps({"step":name}),flush=True)
 r=subprocess.run(["python3","-u",str(u/name)]+args)
 if r.returncode:raise SystemExit(r.returncode)

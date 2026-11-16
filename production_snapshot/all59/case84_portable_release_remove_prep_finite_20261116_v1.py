from pathlib import Path
import subprocess,os,json
u=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1")
env=dict(os.environ,STARREAL_QH_ONLY_COMPONENT_OWNER="1",CUDA_VISIBLE_DEVICES="")
for name,args in [("stage_case84_reviewed_portable_QH_v1.py",[]),("release_case84_node_local_archive_QH_only_v1.py",["--case","84"]),("copy_case84_accepted_archive_to_durable_QH_v2.py",[]),("prepare_new_case_latest_contract_v5.py",["--case","84","--revision","fixed_world_yellow_depth_v5"])]:
 print(json.dumps({"step":name}),flush=True)
 r=subprocess.run(["python3","-u",str(u/name)]+args,env=env)
 if r.returncode:raise SystemExit(r.returncode)

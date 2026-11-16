from pathlib import Path
import subprocess,os,json
q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal")
for name,args in [("import_native_historical_inputs_HF_QH_v3.py",["--case","84"]),("read_native_history_cameras_and_source49_QH_v2.py",[])]:
 print(json.dumps({"step":name}),flush=True)
 r=subprocess.run(["python3","-u",str(q/"code"/name)]+args,env=dict(os.environ,STARREAL_CASE="84",HF_HUB_DISABLE_PROGRESS_BARS="1"))
 if r.returncode:raise SystemExit(r.returncode)

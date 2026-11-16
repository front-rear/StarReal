from pathlib import Path
import subprocess,os,json
n=Path("/tmp/starreal_uid1007_case19_cpu_v1")
env=dict(os.environ,STARREAL_CASE="84",STARREAL_CPU_ROOT=str(n),STARREAL_FRAME_WORKERS="32",STARREAL_GEOMETRY_REVIEW="CASE84_FIVE_ROLE_ALL49_GEOMETRY_ACTUAL_REVIEW_v1.json")
for name in ["build_native_cached_conditions49_QH_v3.py","verify_native_conditions49_QH_v2.py","prepare_native_source_appearance_reference_QH_v2.py","build_native_case_two_pcf49_QH_cpu_v5.py"]:
 print(json.dumps({"step":name}),flush=True)
 r=subprocess.run(["python3","-u",str(n/"code"/name)],env=env)
 if r.returncode:raise SystemExit(r.returncode)

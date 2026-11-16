from pathlib import Path
import subprocess,os,json
n=Path("/tmp/starreal_uid1007_case19_cpu_v1");u=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1")
r=subprocess.run(["/mnt/atlas/fangrui/repro_video_20261105/_envs/pisco/bin/python","-u",str(n/"code/check_native_actual_Original_entry_QH_v2.py")],env=dict(os.environ,STARREAL_CASE="84",STARREAL_CPU_ROOT=str(n),CUDA_VISIBLE_DEVICES=""))
assert r.returncode==0
p=u/"CASE84_POST_READBACKS_TERMINAL_20261116_v1.json";d=json.loads(p.read_text());(u/"CASE84_POST_READBACKS_INITIAL_ENV_GUARD_FAILURE_HISTORY.json").write_bytes(p.read_bytes())
for row in d["results"]:
 if row["script"]=="check_native_actual_Original_entry_QH_v2.py":row.update(returncode=0,initial_guard_failure="Missing CUDA_VISIBLE_DEVICES empty CPU-only declaration; assertion before tests, data untouched",verified_retry_log="CASE84_ORIGINAL_ENTRY_CPU_DECLARED_20261116_v2.log")
p.write_text(json.dumps(d,indent=2))

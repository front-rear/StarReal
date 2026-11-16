from pathlib import Path
import concurrent.futures,subprocess,os,json
n=Path("/tmp/starreal_uid1007_case19_cpu_v1")
u=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1")
env=dict(os.environ,STARREAL_CASE="84",STARREAL_CPU_ROOT=str(n))
tasks=[("python3","review_native_conditions_and_joint_pixels_QH_v2.py"),("/mnt/atlas/fangrui/repro_video_20261105/_envs/pisco/bin/python","check_native_actual_Original_entry_QH_v2.py"),("python3","build_native_historical_comparison49_QH_cpu_v3.py")]
def run(row):
 exe,name=row
 with (u/("CASE84_"+name+".postreadback.log")).open("xb") as log:r=subprocess.run([exe,"-u",str(n/"code"/name)],env=env,stdout=log,stderr=subprocess.STDOUT)
 return dict(script=name,returncode=r.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,tasks))
(u/"CASE84_POST_READBACKS_TERMINAL_20261116_v1.json").write_text(json.dumps(dict(results=rows,quality_or_HF=False),indent=2))
assert all(x["returncode"]==0 for x in rows),rows
print(json.dumps(rows),flush=True)

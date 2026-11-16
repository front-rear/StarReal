from pathlib import Path
import os,json,subprocess,datetime
n=Path("/tmp/starreal_uid1007_case19_cpu_v1");q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");u=q/"all59_reconstruction_20261115_v1"
rows=[]
for case in ["99","100","101","103"]:
 out=n/f"outputs/{case}/pcf_background49_precomputed_QH_v1"
 if out.exists():
  rows.append(dict(case=case,state="existing_owner_or_result_preserved_no_duplicate"));continue
 env=dict(os.environ,STARREAL_CASE=case,STARREAL_QH_ONLY_COMPONENT_OWNER="1",STARREAL_FRAME_WORKERS="32")
 with (u/f"CASE{case}_CAMERA_ONLY_BACKGROUND_FINITE_20261116_v1.log").open("xb") as log:r=subprocess.run(["python3","-u",str(n/"precompute_native_PCF_background49_QH_only_owner_cpu_v2.py")],env=env,stdout=log,stderr=subprocess.STDOUT)
 rows.append(dict(case=case,returncode=r.returncode));print(json.dumps(rows[-1]),flush=True)
 if r.returncode:break
 receipt=q/f"receipts/CASE{case}_PERSISTENT_CONFIDENCE_AND_BACKGROUND_v1.json"
 if not receipt.exists():
  with (u/f"CASE{case}_PERSIST_BACKGROUND_20261116_v1.log").open("xb") as log:r=subprocess.run(["python3",str(u/"persist_completed_background_on_QH_direct_v2.py"),case],stdout=log,stderr=subprocess.STDOUT)
  rows[-1]["persist_returncode"]=r.returncode
  if r.returncode:break
(u/"FOUR_NATIVE_CAMERA_BACKGROUND_FINITE_20261116_v1_RESULT.json").write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,GPU_used=False,quality_accepted=False),indent=2))

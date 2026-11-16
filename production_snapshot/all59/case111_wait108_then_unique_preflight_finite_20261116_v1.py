from pathlib import Path
import json,time,subprocess,os
U=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1");deadline=time.monotonic()+1800;status=U/"CASE111_WAIT108_THEN_UNIQUE_PREFLIGHT_FINITE_20261116_v1.json"
status.write_text(json.dumps(dict(pid=os.getpid(),state="bounded_wait_for_existing108_final_verified_no_second_writer",GPU_used=False),indent=2))
while True:
 p=U/"CASE108_PREFLIGHT_PUBLISH_FINITE_20261116_v1.json"
 if p.exists():
  d=json.loads(p.read_text())
  if d["state"]=="failed":
   status.write_text(json.dumps(dict(state="not_started_parent108_failed",error=d,exit_code=1),indent=2));raise RuntimeError("108 requires exact repair")
  if d["state"]=="terminal0_HF_final_receipt_verified":break
 assert time.monotonic()<deadline,"Bounded wait expired, no owner signalled"
 time.sleep(4)
assert json.loads((U/"CASE108_HF_FINAL_VERIFIED.json").read_text())["case"]=="108"
with (U/"CASE111_PREFLIGHT_PUBLISH_FINITE_20261116_v1.log").open("xb") as f:rc=subprocess.call(["python3","-u",str(U/"case111_preflight_publish_finite_20261116_v1.py")],stdout=f,stderr=subprocess.STDOUT)
status.write_text(json.dumps(dict(state="terminal0_child_finished" if rc==0 else "child_failed_no_automatic_retry",exit_code=rc),indent=2));assert rc==0

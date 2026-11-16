from pathlib import Path
import os,json,time,subprocess,fcntl
U=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1");R=Path("/tmp/starreal_uid1007_case7505429_true4K_cpu_v2")
env=dict(os.environ,STARREAL_CASE="7505429",STARREAL_CPU_ROOT=str(R),STARREAL_FRAME_WORKERS="32",STARREAL_GEOMETRY_REVIEW="CASE7505429_FIVE_ROLE_ALL49_GEOMETRY_ACTUAL_REVIEW_v1.json",CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="2")
d=dict(pid=os.getpid(),steps=[],GPU_used=False,quality_accepted=False,HF_upload=False)
state=U/"CASE7505429_CONDITIONS_PCF_ORIGINAL_FINITE_20261116_v4.json"
def save():state.write_text(json.dumps(d,indent=2))
save()
for name,p,python in [("two_PCF49",R/"build_native_case_two_pcf49_QH_cpu_v5.py","python3"),("Original_entry",R/"check_native_actual_Original_entry_QH_v2.py","/mnt/atlas/fangrui/repro_video_20261105/_envs/pisco/bin/python"),("palette490",R/"review_native_conditions_and_joint_pixels_QH_v2.py","python3")]:
 d.update(state="running",step=name);save();start=time.time()
 with (U/("CASE7505429_"+name+"_FINITE_20261116_v4.log")).open("xb") as f:
  p=subprocess.Popen([python,"-u",str(p)],env=env,stdout=f,stderr=subprocess.STDOUT)
  try:rc=p.wait(timeout=1200)
  except subprocess.TimeoutExpired:d.update(state="bounded_wait_expired_healthy_child_not_signalled",exit_code=None);save();raise
 d["steps"].append(dict(step=name,exit_code=rc,seconds=time.time()-start));save()
 if rc:d.update(state="failed_no_automatic_retry",exit_code=rc);save();break
else:d.update(state="terminal0_components_and_actual_Original_passed_pending_actual_PCF49_color_review",exit_code=0);save()

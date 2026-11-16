from pathlib import Path
import os,json,subprocess,time,datetime,fcntl
U=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1")
lock=(U/"CASE108_111_ORIGINAL_PALETTE_FINITE_20261116_v1.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=dict(pid=os.getpid(),state="waiting_bounded_existing_PCF",steps=[],GPU_used=False,quality_accepted=False,automatic_upload=False)
p=U/"CASE108_111_ORIGINAL_PALETTE_FINITE_20261116_v1_RESULT.json"
def save():p.write_text(json.dumps(state,indent=2))
save()
try:
 for case in ["108","111"]:
  R=Path("/tmp/starreal_uid1007_case"+case+"_true4K_cpu_v2");deadline=time.monotonic()+1800
  while True:
   job=R/("outputs/"+case+"/two_pcf49_fixed_world_yellow_QH_v1/JOB.json")
   if job.exists():
    try:d=json.loads(job.read_text())
    except json.JSONDecodeError:d={}
    if "exit_code" in d:
     assert d["exit_code"]==0,d
     break
   assert time.monotonic()<deadline,"Bounded wait ended; healthy child not signalled"
   time.sleep(3)
  env=dict(os.environ,STARREAL_CASE=case,STARREAL_CPU_ROOT=str(R),CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="2")
  for name,python in [("check_native_actual_Original_entry_QH_v2.py","/mnt/atlas/fangrui/repro_video_20261105/_envs/pisco/bin/python"),("review_native_conditions_and_joint_pixels_QH_v2.py","python3")]:
   state.update(state="running",current_case=case,current_script=name);save()
   log=U/("CASE"+case+"_"+name+"_FINITE_20261116_v1.log")
   with log.open("xb") as f:rc=subprocess.call([python,"-u",str(R/name)],env=env,stdout=f,stderr=subprocess.STDOUT)
   state["steps"].append(dict(case=case,script=name,exit_code=rc));save();assert rc==0,(case,name,rc)
 state.update(state="terminal0_actual_Original_entry_and_all490_joint_pixel_palette_read_pending_actual_visual_review",exit_code=0);save()
except BaseException as e:state.update(state="failed_no_automatic_retry",error=repr(e),exit_code=1);save();raise

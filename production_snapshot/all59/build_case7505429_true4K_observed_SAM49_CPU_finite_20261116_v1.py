import os,sys,json,time,hashlib,fcntl,traceback
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OMP_NUM_THREADS"]="8";os.environ["OPENBLAS_NUM_THREADS"]="1"
Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");N=Path("/tmp/starreal_uid1007_case19_cpu_v1")
rt=json.loads((N/"runtime/sam2_private_cpu_py312_v1/RESULT.json").read_text());sys.path[:0]=[rt["SAM2_code_root"],rt["dependencies_root"]]
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
lock=open(Q/"logs/case7505429_geometry_component.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
I=Q/"source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1/prepared49_from_actual_Original_crop_v1/7505429";prep=json.loads((I/"RESULT.json").read_text())
pilot=Q/"component_cache/7505429/observed_true4K_RGB_SAM_pilot3_CPU_v1";pr=json.loads((pilot/"RESULT.json").read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
review=Q/"reviews/CASE7505429_TRUE4K_SAM_PILOT3_ACTUAL_REVIEW_v1.json";rv=json.loads(review.read_text());assert rv["may_generate_full49_same_CPU_SAM_prompts"] and rv["selected_candidate"]==2 and rv["pilot_result_SHA256"]==sha(pilot/"RESULT.json")
out=Q/"component_cache/7505429/observed_true4K_RGB_SAM49_CPU_v1";out.mkdir(exist_ok=False);(out/"mask49").mkdir();(out/"frame_receipts").mkdir()
job=dict(PID=os.getpid(),uid=os.getuid(),boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),startticks=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],GPU_used=False,device="cpu",CUDA_VISIBLE_DEVICES="",state="loading_CPU_SAM_finite49",frames_finished=0)
def put(p,d):
 t=p.with_suffix(".tmp");t.write_text(json.dumps(d,indent=2));t.replace(p)
put(out/"JOB.json",job);start=time.time()
try:
 assert sha(rt["checkpoint"])=="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
 model=build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",rt["checkpoint"],device="cpu");predictor=SAM2ImagePredictor(model)
 anchors={r["frame"]:r["prompts"] for r in pr["rows"]};rows=[]
 for fi in range(49):
  left,right=(0,24) if fi<=24 else (24,48);t=(fi-left)/(right-left)
  p={k:np.array(anchors[left][k],np.float32)*(1-t)+np.array(anchors[right][k],np.float32)*t for k in ["pos","neg","box"]}
  rgb=np.array(Image.open(I/"RGB49"/f"{fi:06d}.png").convert("RGB"),copy=True);assert hashlib.sha256(rgb.tobytes()).hexdigest()==prep["frame_readbacks"][fi]["RGB_SHA256"]
  if fi in anchors:
   src=pilot/f"A{fi:02d}_candidate2.png";oldrow=next(r for r in pr["rows"] if r["frame"]==fi)["candidates"][2];assert sha(src)==oldrow["mask_SHA256"];mask=np.array(Image.open(src))>0;score=oldrow["score_not_acceptance"];reuse=True
  else:
   coords=np.r_[p["pos"],p["neg"]];labels=np.array([1]*len(p["pos"])+[0]*len(p["neg"]),np.int32)
   with torch.inference_mode():
    predictor.set_image(rgb);m,scores,_=predictor.predict(point_coords=coords,point_labels=labels,box=p["box"],multimask_output=True)
   mask=np.asarray(m[2]).astype(bool);score=float(scores[2]);reuse=False
  dst=out/"mask49"/f"{fi:06d}.png";Image.fromarray(mask.astype(np.uint8)*255).save(dst)
  ov=rgb.copy();ov[mask]=np.rint(.5*ov[mask]+.5*np.array([0,255,0])).astype(np.uint8)
  if fi%7==0:sheet=Image.new("RGB",(1575,227),(20,20,20));dd=ImageDraw.Draw(sheet)
  sheet.paste(Image.fromarray(ov).crop((275,270,500,475)),((fi%7)*225,22));dd.text(((fi%7)*225+4,4),f"A{fi:02d} actual4K SAM2 CPU",(255,255,255))
  if fi%7==6:sheet.save(out/f"actual_all49_frames{fi-6:02d}_{fi:02d}.jpg",quality=96)
  r=dict(frame=fi,RGB_SHA256=prep["frame_readbacks"][fi]["RGB_SHA256"],mask_SHA256=sha(dst),foreground_pixels=int(mask.sum()),positive=p["pos"].tolist(),negative=p["neg"].tolist(),box=p["box"].tolist(),candidate=2,score_not_acceptance=score,reviewed_pilot_reused=reuse,SAM_on_actual_new4K_RGB=True,GPU_used=False,quality_accepted=False)
  rows.append(r);put(out/"frame_receipts"/f"{fi:06d}.json",r);job.update(state="finite49_CPU_SAM_running",frames_finished=fi+1,seconds=time.time()-start);put(out/"JOB.json",job)
  if fi%7==6:print(json.dumps(dict(frames_finished=fi+1,seconds=time.time()-start)),flush=True)
 result=dict(case="7505429",state="all49_true4K_observed_SAM_candidates_pending_actual_all49_review",frames=49,rows=rows,source_preparation=str(I/"RESULT.json"),source_preparation_SHA256=sha(I/"RESULT.json"),pilot_review=str(review),pilot_review_SHA256=sha(review),checkpoint_SHA256=sha(rt["checkpoint"]),old_SAM_not_promoted=True,Native_depth_estimation_not_run=True,training_RGB_not_upscaled640=True,GPU_used=False,quality_accepted=False,new_complete_groups=0,seconds=time.time()-start)
 put(out/"RESULT.json",result);job.update(state="terminal0_all49_pending_actual_review",exit_code=0)
except BaseException as e:
 job.update(state="terminal1_preserve_outputs",exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
finally:put(out/"JOB.json",job)

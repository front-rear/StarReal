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
I=Q/"source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1/prepared49_from_actual_Original_crop_v1/7505429"
out=Q/"component_cache/7505429/observed_true4K_RGB_SAM_pilot3_CPU_v1";out.mkdir(exist_ok=False)
prep=json.loads((I/"RESULT.json").read_text());assert prep["all49_FFV1_decoded_exact"] and prep["actual_Original_function_executed"]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
job=dict(PID=os.getpid(),uid=os.getuid(),boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),startticks=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],GPU_used=False,device="cpu",CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],state="loading_CPU_SAM_finite3")
def put(p,d):
 t=p.with_suffix(".tmp");t.write_text(json.dumps(d,indent=2));t.replace(p)
put(out/"JOB.json",job);start=time.time()
try:
 assert sha(rt["checkpoint"])=="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
 model=build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",rt["checkpoint"],device="cpu");predictor=SAM2ImagePredictor(model)
 prompts={
 0:dict(pos=[[423,309],[395,344],[467,365],[428,340],[427,366],[424,389],[424,408]],neg=[[379,350],[485,360],[433,285],[429,429],[409,320]],box=[383,290,480,423]),
 24:dict(pos=[[388,318],[359,355],[431,375],[391,348],[390,373],[390,397],[390,416]],neg=[[342,362],[448,377],[396,296],[391,437],[370,332]],box=[347,302,444,431]),
 48:dict(pos=[[342,340],[309,375],[386,393],[346,369],[343,394],[344,420],[344,437]],neg=[[293,382],[402,400],[349,318],[343,458],[322,352]],box=[299,325,397,451])
 }
 rows=[]
 for fi,d in prompts.items():
  rgb=np.array(Image.open(I/"RGB49"/f"{fi:06d}.png").convert("RGB"),copy=True)
  assert hashlib.sha256(rgb.tobytes()).hexdigest()==prep["frame_readbacks"][fi]["RGB_SHA256"]
  coords=np.array(d["pos"]+d["neg"],np.float32);labels=np.array([1]*len(d["pos"])+[0]*len(d["neg"]),np.int32)
  with torch.inference_mode():
   predictor.set_image(rgb)
   a,scores,_=predictor.predict(point_coords=coords,point_labels=labels,box=np.array(d["box"],np.float32),multimask_output=True)
  files=[]
  for n,(mask,score) in enumerate(zip(a,scores)):
   mask=np.asarray(mask).astype(bool);dst=out/f"A{fi:02d}_candidate{n}.png";Image.fromarray(mask.astype(np.uint8)*255).save(dst)
   ov=rgb.copy();ov[mask]=np.rint(.5*ov[mask]+.5*np.array([0,255,0])).astype(np.uint8)
   crop=(275,270,500,475);p=Image.new("RGB",(900,432),(20,20,20))
   for j,im in enumerate([Image.fromarray(rgb),Image.fromarray(ov),Image.fromarray(np.repeat((mask.astype(np.uint8)*255)[...,None],3,2))]):
    p.paste(im.crop(crop).resize((300,410)),(j*300,22))
   ImageDraw.Draw(p).text((4,4),f"7505429 true4K A{fi:02d} / actual CPU SAM candidate{n} / observed RGB, overlay, silhouette",(255,255,255))
   jpg=out/f"A{fi:02d}_candidate{n}_actual_review.jpg";p.save(jpg,quality=95)
   yy,xx=np.where(mask);files.append(dict(candidate=n,mask=str(dst),mask_SHA256=sha(dst),review=str(jpg),review_SHA256=sha(jpg),score_not_acceptance=float(score),foreground=int(mask.sum()),bbox=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]))
  rows.append(dict(frame=fi,prompts=d,RGB_SHA256=prep["frame_readbacks"][fi]["RGB_SHA256"],candidates=files))
  job.update(state="finite3_CPU_SAM_candidates_pending_actual_review",frames_finished=len(rows),seconds=time.time()-start);put(out/"JOB.json",job);print(json.dumps(dict(frame=fi,seconds=time.time()-start)),flush=True)
 result=dict(case="7505429",state="actual_true4K_RGB_CPU_SAM_pilot3_pending_actual_review",rows=rows,source_preparation=str(I/"RESULT.json"),source_preparation_SHA256=sha(I/"RESULT.json"),checkpoint_SHA256=sha(rt["checkpoint"]),old_SAM_not_used_as_foreground_truth=True,GPU_used=False,quality_accepted=False,seconds=time.time()-start)
 put(out/"RESULT.json",result);job.update(state="terminal0_pending_actual_review",exit_code=0)
except BaseException as e:
 job.update(state="terminal1_preserve_outputs",exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
finally:put(out/"JOB.json",job)

import os,sys,pathlib,json,hashlib,time,fcntl
os.environ.update(CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="8")
Q=pathlib.Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");U=Q/"all59_reconstruction_20261115_v1";N=pathlib.Path("/tmp/starreal_uid1007_case19_cpu_v1")
rt=json.loads((N/"runtime/sam2_private_cpu_py312_v1/RESULT.json").read_text());sys.path[:0]=[rt["SAM2_code_root"],rt["dependencies_root"]]
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(rt["checkpoint"])=="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
locks=[]
for case in ["108","111"]:
 l=open(Q/"logs"/("case"+case+"_geometry_component.lock"),"a");fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(l)
model=build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",rt["checkpoint"],device="cpu");predictor=SAM2ImagePredictor(model);start=time.time()
for case,orig in [("108","10302763"),("111","7505473")]:
 I=Q/"source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1/prepared49_from_actual_Original_crop_v1"/orig
 meta=json.loads((I/"RESULT.json").read_text());assert meta["actual_Original_function_executed"] and meta["all49_FFV1_decoded_exact"]
 H=np.asarray(meta["H_old_prepared1280_to_new_Original1280"],float)
 out=Q/"component_cache"/case/"observed_true4K_RGB_SAM_ECC_pilot3_CPU_v3";out.mkdir(exist_ok=False);rows=[]
 job=dict(state="running_CPU6pilot_only",PID=os.getpid(),uid=os.getuid(),boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),startticks=pathlib.Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],GPU_used=False)
 (out/"JOB.json").write_text(json.dumps(job))
 for f in [0,24,48]:
  rgb=np.asarray(Image.open(I/"RGB49"/f"{f:06d}.png").convert("RGB")).copy()
  assert hashlib.sha256(rgb.tobytes()).hexdigest()==meta["frame_readbacks"][f]["RGB_SHA256"]
  old=cv2.imread(str(N/"inputs"/f"case{case}_native_CPU_v1/original_mask49"/f"{f:06d}.png"),0)>0
  prior=cv2.warpPerspective(old.astype(np.uint8),H,(1280,704),flags=cv2.INTER_NEAREST)>0
  yy,xx=np.where(prior);box=np.array([xx.min()-8,yy.min()-8,xx.max()+8,yy.max()+8],np.float32)
  old0=cv2.imread(str(N/"inputs"/f"case{case}_native_CPU_v1/original_mask49/000000.png"),0)>0
  y0,x0=np.where(old0);origin=np.array([x0.min(),y0.min()]);span=np.array([np.ptp(x0),np.ptp(y0)])
  oldy,oldx=np.where(old);loc=np.array([oldx.min(),oldy.min()]);scale=np.array([np.ptp(oldx),np.ptp(oldy)])/span
  if case=="108":
   pos0=np.array([[954,140],[960,250],[958,350],[953,434],[929,480],[952,489],[987,483],[1007,478],[984,442],[985,374],[996,351],[1030,480],[1016,458]],np.float32)
   neg0=np.array([[857,150],[1070,150],[903,440],[1048,440],[951,515],[1015,519]],np.float32)
   box0=np.array([[860,90],[1068,503]],float)
  else:
   pos0=np.array([[833,541],[791,554],[887,554],[838,610],[838,673],[801,652],[888,653],[837,574]],np.float32)
   neg0=np.array([[834,552],[814,553],[861,552],[776,630],[900,620],[839,689]],np.float32)
   box0=np.array([[768,531],[906,687]],float)
  roi=(840,80,1090,515) if case=="108" else (760,510,920,704)
  x0,y0,x1,y1=roi
  rgb0=np.asarray(Image.open(I/"RGB49/000000.png").convert("RGB")).copy()
  z=cv2.cvtColor(rgb0[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY);g=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY)
  cc,w=cv2.findTransformECC(z,g,np.eye(2,3,dtype=np.float32),cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,150,1e-6),None,5)
  W=np.eye(3);W[:2]=w;T=np.array([[1,0,x0],[0,1,y0],[0,0,1]],float);W=T@W@np.linalg.inv(T)
  def track(points):return cv2.perspectiveTransform(points[None].astype(np.float64),W)[0].astype(np.float32)
  pos=track(pos0);neg=track(neg0);box=track(box0).ravel()
  semantic0=np.asarray(Image.open(Q/"component_cache"/case/"observed_true4K_RGB_SAM_semantic_parts_pilot3_v3/A00_mask.png"))>0
  prior=cv2.warpPerspective(semantic0.astype(np.uint8),W,(1280,704),flags=cv2.INTER_NEAREST)>0
  pts=np.r_[pos,neg];labels=np.array([1]*len(pos)+[0]*len(neg),np.int32)
  pr=cv2.resize(prior.astype(np.float32),(256,256),interpolation=cv2.INTER_AREA);pr=np.clip(pr,.001,.999);logits=np.log(pr/(1-pr))[None]
  with torch.inference_mode():
   predictor.set_image(rgb)
   mask,score,_=predictor.predict(point_coords=pts,point_labels=labels,box=box,mask_input=logits,multimask_output=False)
  prior_mask=np.asarray(mask[0],bool)
  with torch.inference_mode():
   plain,plain_score,_=predictor.predict(point_coords=pts,point_labels=labels,box=box,multimask_output=False)
  mask=prior_mask
  Image.fromarray(prior_mask.astype(np.uint8)*255).save(out/f"A{f:02d}_with_prior.png")
  Image.fromarray(np.asarray(plain[0],bool).astype(np.uint8)*255).save(out/f"A{f:02d}_points_only.png")
  Image.fromarray(mask.astype(np.uint8)*255).save(out/f"A{f:02d}_priorSAM.png")
  x0=max(0,int(box[0])-20);y0=max(0,int(box[1])-20);x1=min(1280,int(box[2])+20);y1=min(704,int(box[3])+20)
  overlay=rgb.copy();overlay[mask]=np.rint(.6*overlay[mask]+.4*np.array([0,255,0])).astype(np.uint8)
  panels=[Image.fromarray(rgb).crop((x0,y0,x1,y1)),Image.fromarray(overlay).crop((x0,y0,x1,y1)),Image.fromarray(mask.astype(np.uint8)*255).convert("RGB").crop((x0,y0,x1,y1))]
  w,h=panels[0].size;sheet=Image.new("RGB",(w*3,h+25),(20,20,20))
  for j,im in enumerate(panels):sheet.paste(im,(j*w,25))
  ImageDraw.Draw(sheet).text((3,3),f"{case} A{f:02d} true4K / actual SAM candidate / silhouette; not acceptance",fill="white")
  sheet.save(out/f"A{f:02d}_actual_review.jpg",quality=95)
  rows.append(dict(frame=f,RGB_SHA256=meta["frame_readbacks"][f]["RGB_SHA256"],positive=pos.tolist(),negative=neg.tolist(),box=box.tolist(),mask_SHA256=sha(out/f"A{f:02d}_priorSAM.png"),score_not_acceptance=score.tolist(),actual_RGB_ECC=cc,A0_to_frame_from_actual_RGB=W.tolist()))
  print(json.dumps(dict(case=case,frame=f,seconds=time.time()-start)),flush=True)
 result=dict(case=case,state="terminal0_actual_SAM_pilots_pending_visual_review",rows=rows,source_input=str(I/"RESULT.json"),source_input_SHA256=sha(I/"RESULT.json"),GPU_used=False,quality_accepted=False)
 (out/"RESULT.json").write_text(json.dumps(result,indent=2));job.update(state=result["state"],exit_code=0);(out/"JOB.json").write_text(json.dumps(job))

import os,sys,pathlib,json,hashlib,time,fcntl,traceback
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="8"
N=pathlib.Path("/tmp/starreal_uid1007_case19_cpu_v1");Q=pathlib.Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");U=Q/"all59_reconstruction_20261115_v1"
rt=json.loads((N/"runtime/sam2_private_cpu_py312_v1/RESULT.json").read_text());sys.path[:0]=[rt["SAM2_code_root"],rt["dependencies_root"]]
import numpy as np,cv2,torch
from PIL import Image,ImageDraw
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
lock=open(Q/"logs/case109_geometry_component.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
I=Q/"source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1/prepared49_from_actual_Original_crop_v1/6374204"
prepared=json.loads((I/"RESULT.json").read_text());assert prepared["all49_FFV1_decoded_exact"] and prepared["actual_Original_function_executed"]
out=Q/"component_cache/109/observed_true4K_RGB_SAM_pilot3_CPU_v2";out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert sha(rt["checkpoint"])=="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
start=time.time();job=dict(PID=os.getpid(),uid=os.getuid(),boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),startticks=pathlib.Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],state="CPU_SAM_loading_finite3",GPU_used=False)
(out/"JOB.json").write_text(json.dumps(job))
model=build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",rt["checkpoint"],device="cpu");predictor=SAM2ImagePredictor(model)
H=np.asarray(prepared["H_old_prepared1280_to_new_Original1280"])
old0=cv2.imread(str(N/"inputs/case109_native_CPU_v1/original_mask49/000000.png"),0)>0
yy,xx=np.where(old0);lo=np.array([xx.min(),yy.min()]);span=np.array([np.ptp(xx),np.ptp(yy)])
pos=np.array([[629,85],[614,65],[552,132],[488,181],[489,267],[486,321],[488,500],[486,638],[451,647]],np.float32)
neg=np.array([[529,231],[619,431],[811,522],[405,403],[697,232],[565,603],[452,540]],np.float32)
rows=[]
for fi in [0,24,48]:
 rgb=np.array(Image.open(I/"RGB49"/f"{fi:06d}.png").convert("RGB"),copy=True);assert hashlib.sha256(rgb.tobytes()).hexdigest()==prepared["frame_readbacks"][fi]["RGB_SHA256"]
 old=cv2.imread(str(N/"inputs/case109_native_CPU_v1/original_mask49"/f"{fi:06d}.png"),0)>0
 yy,xx=np.where(old);loc=np.array([xx.min(),yy.min()]);scale=np.array([np.ptp(xx),np.ptp(yy)])/span
 prompts=(np.r_[pos,neg]-lo)*scale+loc;prompts=cv2.perspectiveTransform(prompts[None].astype(np.float64),H)[0].astype(np.float32)
 box=np.array([[xx.min()-12,yy.min()-8],[xx.max()+12,yy.max()+8]],float)
 box=cv2.perspectiveTransform(box[None],H)[0].ravel().astype(np.float32)
 warped=cv2.warpPerspective(old.astype(np.uint8),H,(1280,704),flags=cv2.INTER_NEAREST)>0
 prior=cv2.resize(warped.astype(np.float32),(256,256),interpolation=cv2.INTER_AREA);prior=np.clip(prior,.001,.999);logits=np.log(prior/(1-prior))[None]
 with torch.inference_mode():
  predictor.set_image(rgb)
  a,scorea,_=predictor.predict(point_coords=prompts,point_labels=np.array([1]*len(pos)+[0]*len(neg),np.int32),box=box,multimask_output=False)
  b,scoreb,_=predictor.predict(point_coords=prompts,point_labels=np.array([1]*len(pos)+[0]*len(neg),np.int32),box=box,mask_input=logits,multimask_output=False)
 for tag,mask in [("points",a[0]),("prior",b[0]),("old_transformed",warped)]:
  mask=np.asarray(mask).astype(bool)
  Image.fromarray(mask.astype(np.uint8)*255).save(out/f"A{fi:02d}_{tag}.png")
  ov=rgb.copy();ov[mask]=np.rint(.55*ov[mask]+.45*np.array([0,255,0])).astype(np.uint8)
  raw=Image.fromarray(rgb).crop((410,20,680,680));over=Image.fromarray(ov).crop((410,20,680,680));black=Image.fromarray(np.repeat((mask.astype(np.uint8)*255)[...,None],3,2)).crop((410,20,680,680))
  panel=Image.new("RGB",(810,682),(18,18,18));panel.paste(raw,(0,22));panel.paste(over,(270,22));panel.paste(black,(540,22));ImageDraw.Draw(panel).text((4,4),f"109 A{fi:02d} true4K RGB / {tag} actual SAM / silhouette",(255,255,255));panel.save(out/f"A{fi:02d}_{tag}_actual_review.jpg",quality=95)
 rows.append(dict(frame=fi,positive=prompts[:len(pos)].tolist(),negative=prompts[len(pos):].tolist(),box=box.tolist(),point_score_not_acceptance=scorea.tolist(),prior_score_not_acceptance=scoreb.tolist(),RGB_SHA256=prepared["frame_readbacks"][fi]["RGB_SHA256"],masks={tag:sha(out/f"A{fi:02d}_{tag}.png") for tag in ["points","prior","old_transformed"]}))
 job.update(state="finite3_CPU_SAM_candidates_pending_actual_review",frames_finished=len(rows),seconds=time.time()-start);(out/"JOB.json").write_text(json.dumps(job));print(json.dumps(dict(frame=fi,seconds=time.time()-start)),flush=True)
result=dict(state="true4K_Source_pilot3_actual_SAM_candidates_not_accepted",rows=rows,source_input=str(I/"RESULT.json"),source_input_SHA256=sha(I/"RESULT.json"),checkpoint_SHA256=sha(rt["checkpoint"]),seconds=time.time()-start,GPU_used=False,quality_accepted=False)
(out/"RESULT.json").write_text(json.dumps(result,indent=2));job.update(state="terminal0_pending_actual_review",exit_code=0);(out/"JOB.json").write_text(json.dumps(job))

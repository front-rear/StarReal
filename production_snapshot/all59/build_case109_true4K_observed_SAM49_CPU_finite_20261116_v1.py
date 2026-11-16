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
lk=open(Q/"logs/case109_geometry_component.lock","a");fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
I=Q/"source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1/prepared49_from_actual_Original_crop_v1/6374204"
prepared=json.loads((I/"RESULT.json").read_text());assert prepared["all49_FFV1_decoded_exact"] and prepared["actual_Original_function_executed"]
pilot=Q/"component_cache/109/observed_true4K_RGB_SAM_pilot3_CPU_v2"
pilot_result=json.loads((pilot/"RESULT.json").read_text());review=json.loads((Q/"reviews/CASE109_TRUE4K_SAM_PILOT3_ACTUAL_REVIEW_v1.json").read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert review["may_generate_full49_same_CPU_SAM_prompts"] and review["selected_variant"]=="prior" and review["pilot_result_SHA256"]==sha(pilot/"RESULT.json") and review["source_result_SHA256"]==sha(I/"RESULT.json")
out=Q/"component_cache/109/observed_true4K_RGB_SAM49_CPU_v1";out.mkdir(exist_ok=False);(out/"mask49").mkdir();(out/"frame_receipts").mkdir()
start=time.time();job=dict(PID=os.getpid(),uid=os.getuid(),boot=pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip(),startticks=pathlib.Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],state="loading_same_CPU_SAM_finite49",frames_finished=0,GPU_used=False,quality_accepted=False)
def put(p,d):
 tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
put(out/"JOB.json",job)
try:
 assert sha(rt["checkpoint"])=="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318"
 model=build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",rt["checkpoint"],device="cpu");predictor=SAM2ImagePredictor(model)
 H=np.asarray(prepared["H_old_prepared1280_to_new_Original1280"])
 old0=cv2.imread(str(N/"inputs/case109_native_CPU_v1/original_mask49/000000.png"),0)>0
 yy,xx=np.where(old0);lo=np.array([xx.min(),yy.min()]);span=np.array([np.ptp(xx),np.ptp(yy)])
 pos=np.array([[629,85],[614,65],[552,132],[488,181],[489,267],[486,321],[488,500],[486,638],[451,647]],np.float32)
 neg=np.array([[529,231],[619,431],[811,522],[405,403],[697,232],[565,603],[452,540]],np.float32)
 rows=[];sheet=None;dd=None
 for fi in range(49):
  rgb=np.array(Image.open(I/"RGB49"/f"{fi:06d}.png").convert("RGB"),copy=True);assert hashlib.sha256(rgb.tobytes()).hexdigest()==prepared["frame_readbacks"][fi]["RGB_SHA256"]
  oldp=N/"inputs/case109_native_CPU_v1/original_mask49"/f"{fi:06d}.png";old=cv2.imread(str(oldp),0)>0
  yy,xx=np.where(old);loc=np.array([xx.min(),yy.min()]);scale=np.array([np.ptp(xx),np.ptp(yy)])/span
  prompts=(np.r_[pos,neg]-lo)*scale+loc;prompts=cv2.perspectiveTransform(prompts[None].astype(np.float64),H)[0].astype(np.float32)
  box=np.array([[xx.min()-12,yy.min()-8],[xx.max()+12,yy.max()+8]],float);box=cv2.perspectiveTransform(box[None],H)[0].ravel().astype(np.float32)
  warped=cv2.warpPerspective(old.astype(np.uint8),H,(1280,704),flags=cv2.INTER_NEAREST)>0
  prior=cv2.resize(warped.astype(np.float32),(256,256),interpolation=cv2.INTER_AREA);prior=np.clip(prior,.001,.999);logits=np.log(prior/(1-prior))[None]
  if fi in [0,24,48]:
   mask=np.array(Image.open(pilot/f"A{fi:02d}_prior.png"))>0;oldrow=next(x for x in pilot_result["rows"] if x["frame"]==fi);assert sha(pilot/f"A{fi:02d}_prior.png")==oldrow["masks"]["prior"];score=oldrow["prior_score_not_acceptance"];from_pilot=True
  else:
   with torch.inference_mode():
    predictor.set_image(rgb)
    masks,scores,_=predictor.predict(point_coords=prompts,point_labels=np.array([1]*len(pos)+[0]*len(neg),np.int32),box=box,mask_input=logits,multimask_output=False)
   mask=np.asarray(masks[0]).astype(bool);score=scores.tolist();from_pilot=False
  dst=out/"mask49"/f"{fi:06d}.png";Image.fromarray(mask.astype(np.uint8)*255).save(dst)
  if fi%7==0:sheet=Image.new("RGB",(1890,682),(18,18,18));dd=ImageDraw.Draw(sheet)
  ov=rgb.copy();ov[mask]=np.rint(.5*ov[mask]+.5*np.array([0,255,0])).astype(np.uint8)
  sheet.paste(Image.fromarray(ov).crop((410,20,680,680)),((fi%7)*270,22));dd.text(((fi%7)*270+4,4),f"A{fi:02d} TRUE4K / SAM",(255,255,255))
  if fi%7==6:sheet.save(out/f"actual_all49_frames{fi-6:02d}_{fi:02d}.jpg",quality=94)
  rec=dict(frame=fi,RGB_SHA256=prepared["frame_readbacks"][fi]["RGB_SHA256"],mask_SHA256=sha(dst),original_SAM_SHA256=sha(oldp),positive=prompts[:len(pos)].tolist(),negative=prompts[len(pos):].tolist(),box=box.tolist(),score_not_acceptance=score,reviewed_pilot_reused=from_pilot,foreground_pixels=int(mask.sum()),SAM_on_actual_new4K_RGB=True,quality_accepted=False)
  rows.append(rec);put(out/"frame_receipts"/f"{fi:06d}.json",rec);job.update(state="finite49_CPU_SAM_running",frames_finished=fi+1,seconds=time.time()-start);put(out/"JOB.json",job)
  if fi%7==6:print(json.dumps(dict(frames_finished=fi+1,seconds=time.time()-start)),flush=True)
 result=dict(state="all49_true4K_observed_SAM_candidates_ready_pending_actual_all49_review",case="109",source_recovered_id="6374204",frames=49,rows=rows,source_preparation=str(I/"RESULT.json"),source_preparation_SHA256=sha(I/"RESULT.json"),pilot_review=str(Q/"reviews/CASE109_TRUE4K_SAM_PILOT3_ACTUAL_REVIEW_v1.json"),checkpoint_SHA256=sha(rt["checkpoint"]),old_source_inputs_preserved=True,new_Native_depth_inference_run=False,GPU_used=False,quality_accepted=False,formal_group_increment=0,seconds=time.time()-start)
 put(out/"RESULT.json",result);job.update(state="terminal0_all49_pending_actual_review",exit_code=0)
except BaseException as e:
 job.update(state="terminal1_preserve_outputs",exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
finally:put(out/"JOB.json",job)

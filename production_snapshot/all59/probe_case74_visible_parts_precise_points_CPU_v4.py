from pathlib import Path
import hashlib,json,sys,os,time,fcntl
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');rt=json.loads((N/'runtime/sam2_private_cpu_py312_v1/RESULT.json').read_text());sys.path[:0]=[rt['SAM2_code_root'],rt['dependencies_root']]
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1);assert sha(rt['checkpoint'])=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318';model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',rt['checkpoint'],device='cpu');predictor=SAM2ImagePredictor(model)
 for c,positive,negative,box,crop in [('74',[[942,523],[941,551],[941,579],[940,601],[938,611],[920,623],[929,625],[929,515],[917,518],[907,522]],[[925,525],[954,549],[926,582],[955,596],[914,612],[933,638],[908,537],[916,528],[907,513],[930,524]],[900,499,958,641],[865,490,975,647])]:
  guard=(Q/f'logs/case{c}_geometry_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);t=time.time();out=N/f'outputs/{c}/observed_missing_parts_SAM2_A00_probe_v4';out.mkdir(parents=True,exist_ok=False);I=N/f'inputs/case{c}_native_CPU_v1';m=json.loads((I/'INPUT_MANIFEST.json').read_text());source=N/f'inputs/case{c}_source_video_HF_v1/source_A.mkv';assert sha(source)==m['source_video_SHA256'];cap=cv2.VideoCapture(str(source));ok,bgr=cap.read();cap.release();assert ok and bgr.shape==(704,1280,3) and hashlib.sha256(bgr.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][0];old=cv2.imread(str(I/'original_mask49/000000.png'),0)>0
  with torch.inference_mode():
   predictor.set_image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));mm,ss,_=predictor.predict(point_coords=np.asarray(positive+negative,np.float32),point_labels=np.asarray([1]*len(positive)+[0]*len(negative)),box=np.asarray(box,np.float32),multimask_output=False)
  parts=mm[0].astype(bool);union=old|parts;panels=[]
  for label,mask in [('actual source RGB',None),('unchanged original SAM',old),('new visible-parts SAM2 candidate',parts),('private candidate union',union)]:
   im=bgr.copy()
   if mask is not None:im[mask]=np.rint(.55*im[mask]+.45*np.array([0,255,0])).astype(np.uint8)
   cv2.putText(im,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1);panels.append(im)
  cv2.imwrite(str(out/'actual_A00_four_versions.jpg'),np.concatenate(panels,1),[cv2.IMWRITE_JPEG_QUALITY,97]);x,y,x1,y1=crop;detail=np.concatenate([im[y:y1,x:x1] for im in panels],1);cv2.imwrite(str(out/'actual_A00_closeup.jpg'),cv2.resize(detail,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST),[cv2.IMWRITE_JPEG_QUALITY,98]);cv2.imwrite(str(out/'candidate_parts.png'),parts.astype(np.uint8)*255);cv2.imwrite(str(out/'candidate_union.png'),union.astype(np.uint8)*255)
  result=dict(case=c,state='single_actual_observed_parts_probe_pending_visual_review',frame=0,positive=positive,negative=negative,box=box,source_SHA256=sha(source),old_SAM_SHA256=sha(I/'original_mask49/000000.png'),checkpoint_SHA256=sha(rt['checkpoint']),added_pixels=int((parts&~old).sum()),model_score_not_acceptance=ss.tolist(),RGB_depth_K_E_SAM_originals_unchanged=True,no_hidden_geometry_SAM_claim=True,quality_accepted=False,GPU_used=False,seconds=time.time()-t);(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);fcntl.flock(guard,fcntl.LOCK_UN)
if __name__=='__main__':main()

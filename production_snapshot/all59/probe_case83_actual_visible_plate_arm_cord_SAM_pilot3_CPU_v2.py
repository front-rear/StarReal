from pathlib import Path
import sys,json,hashlib,time,fcntl,os
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');rt=json.loads((N/'runtime/sam2_private_cpu_py312_v1/RESULT.json').read_text());sys.path[:0]=[rt['SAM2_code_root'],rt['dependencies_root']]
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=(Q/'logs/case83_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1);assert sha(rt['checkpoint'])=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
 model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',rt['checkpoint'],device='cpu');pred=SAM2ImagePredictor(model);I=N/'inputs/case83_native_CPU_v1';manifest=json.loads((I/'INPUT_MANIFEST.json').read_text());source=N/'inputs/case83_source_video_HF_v1/source_A.mkv';assert sha(source)==manifest['source_video_SHA256'];out=Q/'component_cache/83/observed_visible_parts_SAM_pilot3_v2';out.mkdir(parents=True,exist_ok=False)
 positive=[[1137,70],[1084,49],[1105,42],[1126,39],[1082,77],[1075,98],[1073,110],[1070,120],[1064,146],[1058,173],[1051,199],[1046,214]];negative=[[1040,233],[1088,155],[1045,155],[1101,120],[1150,135],[1068,60],[1060,111],[1086,110],[1099,60],[1104,72],[1110,55],[1075,150],[1049,190]];box=[1035,15,1180,230];cap=cv2.VideoCapture(str(source));records=[]
 for i in range(49):
  ok,bgr=cap.read();assert ok and hashlib.sha256(bgr.tobytes()).hexdigest()==manifest['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  oldp=I/'original_mask49'/f'{i:06d}.png';old=cv2.imread(str(oldp),0)>0;oldsha=sha(oldp)
  with torch.inference_mode():
   crop=[1028,5,1190,248];x0,y0,x1,y1=crop;pred.set_image(cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));points=np.array(positive+negative,np.float32)-[x0,y0];bb=np.array(box,np.float32)-[x0,y0,x0,y0];small,scores,_=pred.predict(point_coords=points,point_labels=np.array([1]*len(positive)+[0]*len(negative),np.int32),box=bb,multimask_output=False);masks=np.zeros((1,704,1280),bool);masks[0,y0:y1,x0:x1]=small[0]
  parts=masks[0].astype(bool);union=old|parts;added=parts&~old
  for name,a in [('parts',parts),('union',union),('additions',added)]:cv2.imwrite(str(out/f'{i:06d}_{name}.png'),a.astype(np.uint8)*255)
  images=[]
  for mask,color in [(np.zeros_like(old),[0,0,0]),(union,[0,255,0]),(added,[255,0,255])]:
   im=bgr.copy();im[mask]=np.rint(.5*im[mask]+.5*np.array(color)).astype(np.uint8);images.append(im[10:245,1025:1185])
  detail=np.concatenate(images,1);detail=cv2.resize(detail,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST);cv2.putText(detail,f'Case83 A{i:02d} actualRGB / union / observed additions; NOT acceptance',(3,14),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1);cv2.imwrite(str(out/f'A{i:02d}_actual_RGB_union_additions.jpg'),detail,[cv2.IMWRITE_JPEG_QUALITY,97]);assert sha(oldp)==oldsha;records.append(dict(frame=i,original_mask_SHA256=oldsha,union_SHA256=sha(out/f'{i:06d}_union.png'),original_mask_pixels_retained=True,added_pixels=int(added.sum()),score_not_acceptance=scores.tolist()))
 cap.release();r=dict(case='83',state='PRIVATE_ACTUAL_VISIBLE_PARTS_PILOT_PENDING_REVIEW',positive=positive,negative=negative,box=box,checkpoint_SHA256=sha(rt['checkpoint']),source_SHA256=sha(source),frames=records,Original_RGB_SAM_depth_K_E_unchanged=True,CPU_only=True,quality_accepted=False,seconds=time.time()-start);(out/'RESULT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()

from pathlib import Path
import json,hashlib,sys,time,fcntl
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');rt=json.loads((N/'runtime/sam2_private_cpu_py312_v1/RESULT.json').read_text());sys.path[:0]=[rt['SAM2_code_root'],rt['dependencies_root']]
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 guard=(Q/'logs/case80_geometry_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1);assert sha(rt['checkpoint'])=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318';model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',rt['checkpoint'],device='cpu');pred=SAM2ImagePredictor(model);I=N/'inputs/case80_native_CPU_v1';m=json.loads((I/'INPUT_MANIFEST.json').read_text());source=N/'inputs/case80_source_video_HF_v1/source_A.mkv';assert sha(source)==m['source_video_SHA256'];out=Q/'component_cache/80/observed_visible_parts_actual_anchors_v2';out.mkdir(parents=True,exist_ok=False)
 anchors={24:dict(positive=[[343,69],[339,54],[314,67],[301,66],[290,64],[285,49],[284,29],[277,9],[262,8],[251,23]],negative=[[310,38],[302,85],[370,70],[319,8],[270,42]],box=[240,0,365,108]),48:dict(positive=[[337,96],[334,83],[314,93],[302,92],[289,89],[285,76],[283,50],[277,29],[262,28],[250,31]],negative=[[315,57],[305,111],[370,98],[320,27],[271,67]],box=[233,12,360,129])}
 for i,a in anchors.items():
  cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=cap.read();cap.release();assert ok and hashlib.sha256(bgr.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i];old=cv2.imread(str(I/'original_mask49'/f'{i:06d}.png'),0)>0
  with torch.inference_mode():
   pred.set_image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));mm,ss,_=pred.predict(point_coords=np.array(a['positive']+a['negative'],np.float32),point_labels=np.array([1]*len(a['positive'])+[0]*len(a['negative']),np.int32),box=np.array(a['box'],np.float32),multimask_output=False)
  parts=mm[0].astype(bool);union=old|parts;panels=[]
  for label,mask in [('actual RGB',None),('unchanged old SAM',old),('new actual-frame parts candidate',parts),('candidate union',union)]:
   im=bgr.copy()
   if mask is not None:im[mask]=np.rint(.55*im[mask]+.45*np.array([0,255,0])).astype(np.uint8)
   cv2.putText(im,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1);panels.append(im)
  detail=np.concatenate([x[:160,140:430] for x in panels],1);cv2.imwrite(str(out/f'actual_A{i:02d}_four_way_closeup.jpg'),cv2.resize(detail,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST),[cv2.IMWRITE_JPEG_QUALITY,98]);cv2.imwrite(str(out/f'candidate_parts_A{i:02d}.png'),parts.astype(np.uint8)*255);cv2.imwrite(str(out/f'candidate_union_A{i:02d}.png'),union.astype(np.uint8)*255);r=dict(case='80',frame=i,actual_native_RGB_SHA256=hashlib.sha256(bgr.tobytes()).hexdigest(),source_SHA256=m['source_video_SHA256'],checkpoint_SHA256=sha(rt['checkpoint']),quality_accepted=False,score_not_acceptance=ss.tolist(),added_pixels=int((parts&~old).sum()),**a);(out/f'RESULT_A{i:02d}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()

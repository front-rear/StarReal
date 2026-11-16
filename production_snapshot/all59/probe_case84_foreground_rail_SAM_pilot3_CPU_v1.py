from pathlib import Path
import sys,json,hashlib,time,fcntl,os
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');rt=json.loads((N/'runtime/sam2_private_cpu_py312_v1/RESULT.json').read_text());sys.path[:0]=[rt['SAM2_code_root'],rt['dependencies_root']]
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 guard=(Q/'logs/case84_geometry_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
 assert sha(rt['checkpoint'])=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
 model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',rt['checkpoint'],device='cpu');pred=SAM2ImagePredictor(model)
 I=N/'inputs/case84_native_CPU_v1';m=json.loads((I/'INPUT_MANIFEST.json').read_text());source=N/'inputs/case84_source_video_HF_v1/source_A.mkv';assert sha(source)==m['source_video_SHA256'];out=Q/'component_cache/84/observed_foreground_rail_SAM_pilot3_v1';out.mkdir(parents=True,exist_ok=False)
 specs=[dict(name='direct_lamp',crop=[855,235,980,610],box=[889,254,943,575],positive=[[914,293],[916,413],[916,481],[919,553]],negative=[[914,506],[939,531],[899,526],[886,320],[956,416]]),dict(name='actual_foreground_rail',crop=[780,475,1050,620],box=[785,496,1045,603],positive=[[817,505],[970,506],[915,506],[875,541],[943,541]],negative=[[915,479],[916,552],[1004,484],[824,480],[1005,615]])]
 records=[];cap=cv2.VideoCapture(str(source))
 for i in range(49):
  ok,bgr=cap.read();assert ok and hashlib.sha256(bgr.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  oldp=I/'original_mask49'/f'{i:06d}.png';oldsha=sha(oldp);old=cv2.imread(str(oldp),0)>0
  for spec in specs:
   x0,y0,x1,y1=spec['crop'];origin=np.array([x0,y0]);points=np.array(spec['positive']+spec['negative'],np.float32)-origin;labels=np.array([1]*len(spec['positive'])+[0]*len(spec['negative']),np.int32)
   with torch.inference_mode():
    pred.set_image(cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));masks,scores,_=pred.predict(point_coords=points,point_labels=labels,box=np.array(spec['box'],np.float32)-np.tile(origin,2),multimask_output=True)
   panels=[]
   for title,support in [('actual RGB',np.zeros_like(old)),('original candidate',old)]+[(spec['name']+str(k),None) for k in range(len(masks))]:
    if support is None:
     k=int(title[-1]);support=np.zeros_like(old);support[y0:y1,x0:x1]=masks[k].astype(bool);cv2.imwrite(str(out/f'{i:06d}_{spec["name"]}_{k}.png'),support.astype(np.uint8)*255)
     if spec['name']=='actual_foreground_rail':support=old&~support;cv2.imwrite(str(out/f'{i:06d}_source_minus_rail_{k}.png'),support.astype(np.uint8)*255)
    im=bgr.copy();im[support]=np.rint(im[support]*.4+np.array([0,255,0])*.6).astype(np.uint8);im=im[235:610,855:980];im=cv2.resize(im,(250,750),interpolation=cv2.INTER_NEAREST);cv2.putText(im,title,(4,18),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,255,255),1);panels.append(im)
   cv2.imwrite(str(out/f'A{i:02d}_{spec["name"]}_all_candidates.jpg'),np.concatenate(panels,1),[cv2.IMWRITE_JPEG_QUALITY,97]);records.append(dict(frame=i,mode=spec['name'],scores=scores.tolist(),original_mask_SHA256=oldsha,candidate_paths=[str(out/f'{i:06d}_{spec["name"]}_{k}.png') for k in range(len(masks))],actual_review_pending=True))
  assert sha(oldp)==oldsha
 cap.release();result=dict(case='84',state='private_SAM_rail_occlusion_candidates_pending_actual_review',specs=specs,records=records,original_RGB_SAM_depth_K_E_unchanged=True,checkpoint_SHA256=sha(rt['checkpoint']),GPU_used=False,quality_accepted=False,seconds=time.time()-start);(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()

from pathlib import Path
import sys,os,json,hashlib,time,fcntl,traceback
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1';N=Path('/tmp/starreal_uid1007_case19_cpu_v1')
rt=json.loads((N/'runtime/sam2_private_cpu_py312_v1/RESULT.json').read_text());sys.path[:0]=[rt['SAM2_code_root'],rt['dependencies_root']]
import numpy as np,cv2,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
out=Q/'component_cache/83/observed_native_metal_negative_holes_SAM_pilot3_v9';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
job=out/'JOB.json';state=dict(state='running',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split()[21],GPU_used=False)
def save():job.write_text(json.dumps(state,indent=2))
save()
try:
 lock=(Q/'logs/case83_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 start=time.time();torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
 assert sha(rt['checkpoint'])=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
 pred=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',rt['checkpoint'],device='cpu'))
 pos=[[1084,44],[1085,58],[1084,66],[1097,53],[1104,51],[1111,51],[1116,51],[1117,39],[1119,35],[1121,33],[1125,32],[1127,37],[1138,38],[1126,44]]
 neg=[[1099,43],[1104,45],[1108,47],[1112,45],[1115,43],[1118,42],[1117,47],[1100,61],[1071,35],[1089,31],[1116,34],[1139,60],[1098,35]]
 I=N/'inputs/case83_native_CPU_v1';manifest=json.loads((I/'INPUT_MANIFEST.json').read_text());source=N/'inputs/case83_source_video_HF_v1/source_A.mkv';assert sha(source)==manifest['source_video_SHA256']
 cap=cv2.VideoCapture(str(source));records=[]
 for i in range(49):
  ok,bgr=cap.read();assert ok and hashlib.sha256(bgr.tobytes()).hexdigest()==manifest['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  # Native-pixel crop enlarges tiny metal context. No synthetic foreground is drawn.
  origin=np.array([1060,15]);end=np.array([1154,83]);scale=6
  crop=cv2.resize(bgr[origin[1]:end[1],origin[0]:end[0]],None,fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)
  pts=(np.array(pos+neg,np.float32)-origin)*scale
  with torch.inference_mode():
   pred.set_image(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
   masks,scores,_=pred.predict(point_coords=pts,point_labels=np.array([1]*len(pos)+[0]*len(neg),np.int32),multimask_output=True)
  original=cv2.imread(str(I/'original_mask49'/f'{i:06d}.png'),0)>0
  previous=cv2.imread(str(Q/'component_cache/83/observed_visible_parts_SAM_pilot3_v6'/f'{i:06d}_union.png'),0)>0
  previous[:83]=False
  candidates=[]
  for k,m in enumerate(masks):
   small=cv2.resize(m.astype(np.uint8),(int(end[0]-origin[0]),int(end[1]-origin[1])),interpolation=cv2.INTER_NEAREST).astype(bool)
   body=np.zeros_like(original);body[origin[1]:end[1],origin[0]:end[0]]=small
   union=original|previous|body
   cv2.imwrite(str(out/f'{i:06d}_candidate{k}_body.png'),body.astype(np.uint8)*255);cv2.imwrite(str(out/f'{i:06d}_candidate{k}_union.png'),union.astype(np.uint8)*255)
   seen=[]
   for mask,color in [(np.zeros_like(original),(0,0,0)),(union,(0,255,0)),(body&~original,(255,0,255))]:
    im=bgr.copy();im[mask]=np.rint(im[mask]*.5+.5*np.array(color)).astype(np.uint8);seen.append(im[10:245,1025:1185])
   detail=cv2.resize(np.concatenate(seen,1),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST);cv2.putText(detail,f'83 A{i:02d} native RGB / union / SAM candidate{k} addition; pending actual review',(2,12),0,.3,(0,255,255),1);cv2.imwrite(str(out/f'A{i:02d}_candidate{k}_actual_RGB.jpg'),detail,[cv2.IMWRITE_JPEG_QUALITY,96])
   candidates.append(dict(candidate=k,score_not_acceptance=float(scores[k]),positive_hits=sum(bool(body[y,x]) for x,y in pos),negative_false_hits=sum(bool(body[y,x]) for x,y in neg),body_pixels=int(body.sum()),union_SHA256=sha(out/f'{i:06d}_candidate{k}_union.png')))
  records.append(dict(frame=i,candidates=candidates));state['completed_pilots']=len(records);save()
 cap.release()
 result=dict(case='83',state='private_corrected_native_metal_positive_and_wallpaper_hole_negatives_SAM_CPU_pilot3_pending_actual_review',source_SHA256=sha(source),checkpoint_SHA256=sha(rt['checkpoint']),actual_native_RGB_coordinate_review=str(Q/'component_cache/83/observed_RGB_pixel_inspection_v8/A00_METAL_NATIVE_PIXEL_COORDINATES.jpg'),positive=pos,negative=neg,earlier_ambiguous_wallpaper_positive_coordinates_not_reused=True,records=records,quality_accepted=False,original_RGB_SAM_K_E_depth_unchanged=True,GPU_used=False,seconds=time.time()-start)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2));state.update(state='terminal0_pending_actual_candidate_review',exit_code=0,seconds=time.time()-start);save();print(json.dumps(result),flush=True)
except Exception as e:
 state.update(state='failed',error=repr(e),exit_code=1);save();raise

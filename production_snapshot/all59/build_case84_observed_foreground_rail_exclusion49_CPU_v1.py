from pathlib import Path
import sys,json,hashlib,time,fcntl,os
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');rt=json.loads((N/'runtime/sam2_private_cpu_py312_v1/RESULT.json').read_text());sys.path[:0]=[rt['SAM2_code_root'],rt['dependencies_root']]
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,d):t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
 guard=(Q/'logs/case84_geometry_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
 gate=Q/'component_cache/84/observed_foreground_rail_union_selected_pilot3_v1/RESULT.json';g=json.loads(gate.read_text());assert g['may_compute_private_full49'] and sha(g['actual_image'])==g['actual_image_SHA256'];assert sha(rt['checkpoint'])=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
 model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',rt['checkpoint'],device='cpu');pred=SAM2ImagePredictor(model);I=N/'inputs/case84_native_CPU_v1';m=json.loads((I/'INPUT_MANIFEST.json').read_text());source=N/'inputs/case84_source_video_HF_v1/source_A.mkv';assert sha(source)==m['source_video_SHA256'];out=Q/'component_cache/84/observed_foreground_rail_exclusion49_CPU_v1';out.mkdir(parents=True,exist_ok=False)
 for d in ['mask49','rail_exclusions49','actual_foot_review49','frame_receipts']:(out/d).mkdir()
 specs=[dict(name='horizontal_rail',crop=[780,475,1050,620],box=[785,496,1045,603],positive=[[817,505],[970,506],[915,506],[875,541],[943,541]],negative=[[915,479],[916,552],[1004,484],[824,480],[1005,615]],selected=0),dict(name='rail_left',crop=[890,494,918,586],box=[897,508,907,579],positive=[[902,515],[902,538],[902,560]],negative=[[911,520],[913,551],[894,520],[910,574]],selected=1),dict(name='rail_right',crop=[918,494,944,588],box=[927,508,936,581],positive=[[931,515],[931,542],[931,566]],negative=[[921,524],[922,552],[939,519],[926,550]],selected=1)]
 state=dict(case='84',state='running_private_observed_rail_exclusion49',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],frames_finished=0,quality_accepted=False,GPU_used=False);put(out/'JOB.json',state);cap=cv2.VideoCapture(str(source));records=[]
 for i in range(49):
  ok,bgr=cap.read();assert ok and hashlib.sha256(bgr.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i];oldp=I/'original_mask49'/f'{i:06d}.png';oldsha=sha(oldp);old=cv2.imread(str(oldp),0)>0;rail=np.zeros_like(old);stats=[]
  for sp in specs:
   x0,y0,x1,y1=sp['crop'];origin=np.array([x0,y0]);pts=np.array(sp['positive']+sp['negative'],np.float32)-origin;lab=np.array([1]*len(sp['positive'])+[0]*len(sp['negative']),np.int32)
   with torch.inference_mode():
    pred.set_image(cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));ms,sc,_=pred.predict(point_coords=pts,point_labels=lab,box=np.array(sp['box'],np.float32)-np.tile(origin,2),multimask_output=True)
   a=np.zeros_like(old);a[y0:y1,x0:x1]=ms[sp['selected']].astype(bool);rail|=a;stats.append(dict(part=sp['name'],selected=sp['selected'],scores=sc.tolist(),actual_foreground_candidates_not_new_lamp_pixels=True))
  corrected=old&~rail;removed=old&rail;assert not np.any(corrected&~old);cv2.imwrite(str(out/'mask49'/f'{i:06d}.png'),corrected.astype('uint8')*255);cv2.imwrite(str(out/'rail_exclusions49'/f'{i:06d}.png'),removed.astype('uint8')*255)
  panels=[]
  for title,a,color in [('native RGB',np.zeros_like(old),[0,0,0]),('observed correction',corrected,[0,255,0]),('rail removed',removed,[0,0,255])]:
   v=bgr.copy();v[a]=np.rint(v[a]*.4+np.array(color)*.6).astype('uint8');v=v[485:590,880:960];v=cv2.resize(v,(160,210),interpolation=cv2.INTER_NEAREST);cv2.putText(v,f'{i:02d} '+title,(2,12),cv2.FONT_HERSHEY_SIMPLEX,.23,(0,255,255),1);panels.append(v)
  cv2.imwrite(str(out/'actual_foot_review49'/f'{i:06d}.jpg'),np.concatenate(panels,1),[cv2.IMWRITE_JPEG_QUALITY,95]);assert sha(oldp)==oldsha;r=dict(frame=i,original_mask_SHA256=oldsha,corrected_mask_SHA256=sha(out/'mask49'/f'{i:06d}.png'),removed_actual_rail_pixels=int(removed.sum()),source_RGB_exact=True,SAM_parts=stats);records.append(r);put(out/'frame_receipts'/f'{i:06d}.json',r);state.update(frames_finished=i+1,elapsed_seconds=time.time()-start);put(out/'JOB.json',state)
 assert not cap.read()[0];cap.release()
 for page,ids in enumerate([list(range(25)),list(range(25,49))]):
  canvas=np.full((5*230,5*480,3),22,np.uint8)
  for j,i in enumerate(ids):canvas[(j//5)*230:(j//5)*230+210,(j%5)*480:(j%5+1)*480]=cv2.imread(str(out/'actual_foot_review49'/f'{i:06d}.jpg'))
  cv2.imwrite(str(out/f'actual_native_foreground_rail_review_page{page+1}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
 result=dict(case='84',state='private49_SAM_observed_rail_correction_READY_FOR_ACTUAL_REVIEW',frames=49,source_RGB_depth_K_E_and_original_SAM_unchanged=True,checkpoint_SHA256=sha(rt['checkpoint']),correction='Original observed lamp candidate minus three SAM foreground rail parts. No new lamp foreground painted. Human three-frame chosen SAM components, full49 actual review still required.',specs=specs,records=records,quality_accepted=False,automatic_HF=False,GPU_used=False,elapsed_seconds=time.time()-start);put(out/'RESULT.json',result);state.update(state='terminal0_all49_correction_PENDING_ACTUAL_REVIEW',exit_code=0,elapsed_seconds=time.time()-start);put(out/'JOB.json',state);print(json.dumps({k:v for k,v in result.items() if k not in ['records','specs']}),flush=True)
if __name__=='__main__':main()

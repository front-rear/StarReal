"""Full49 observed cable repair, private CPU SAM2; preserve all originals."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,sys,time,traceback
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');I=N/'inputs/case56_native_CPU_v1';runtime=N/'runtime/sam2_private_cpu_py312_v1'
runtime_proof=json.loads((runtime/'RESULT.json').read_text());sys.path[:0]=[runtime_proof['SAM2_code_root'],runtime_proof['dependencies_root']]
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main():
 lock=(Q/'logs/case56_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1);start=time.time()
 probe=N/'outputs/56/sam2_observed_cord_A24_QH_v1';p=json.loads((probe/'RESULT.json').read_text());review=json.loads((Q/'reviews/CASE56_OBSERVED_CORD_SAM2_A24_ACTUAL_REVIEW_v1.json').read_text());assert review['may_generate_full49_same_CPU_model_prompts'] and review['probe_result_SHA256']==sha(probe/'RESULT.json')
 source=N/'inputs/case56_source_video_HF_v1/source_A.mkv';m=json.loads((I/'INPUT_MANIFEST.json').read_text());assert p['case']=='56' and sha(source)==p['source_SHA256']==m['source_video_SHA256'];assert sha(p['model_checkpoint'])==p['model_SHA256']==runtime_proof['model_SHA256']
 out=N/'outputs/56/observed_SAM49_cord_QH_v1';out.mkdir(exist_ok=False)
 for name in ['mask49','parts49','added_actual49','frame_receipts']:(out/name).mkdir()
 olddir=I/'original_mask49';oldsha={f'{i:06d}.png':sha(olddir/f'{i:06d}.png') for i in range(49)}
 ident=dict(case='56',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],GPU_used=False,CPU_threads=8,quality_accepted=False)
 job=dict(ident,state='loading_original_SAM2_CPU_finite49',frames_finished=0);put(out/'JOB.json',job)
 try:
  model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',p['model_checkpoint'],device='cpu');predictor=SAM2ImagePredictor(model);old24=cv2.imread(str(olddir/'000024.png'),0)>0;yy,xx=np.where(old24);lo=np.array([xx.min(),yy.min()],float);span=np.array([np.ptp(xx),np.ptp(yy)],float);cap=cv2.VideoCapture(str(source));full=[];details=[];records=[]
  for i in range(49):
   ok,bgr=cap.read();assert ok and bgr.shape==(704,1280,3) and hashlib.sha256(bgr.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i]
   old=cv2.imread(str(olddir/f'{i:06d}.png'),0)>0;yy,xx=np.where(old);l=np.array([xx.min(),yy.min()],float);scale=np.array([np.ptp(xx),np.ptp(yy)],float)/span
   prompts=(np.array(p['positive']+p['negative'],np.float32)-lo)*scale+l;box=(np.array(p['box'],np.float32).reshape(2,2)-lo)*scale+l
   with torch.inference_mode():
    predictor.set_image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));masks,scores,_=predictor.predict(point_coords=prompts.astype(np.float32),point_labels=np.array([1]*len(p['positive'])+[0]*len(p['negative']),np.int32),box=box.ravel().astype(np.float32),multimask_output=False)
   parts=masks[0].astype(bool);union=old|parts;added=parts&~old;assert not (old&~union).any()
   for folder,a in [('mask49',union),('parts49',parts),('added_actual49',added)]:assert cv2.imwrite(str(out/folder/f'{i:06d}.png'),a.astype(np.uint8)*255)
   if i==24:np.testing.assert_array_equal(union,cv2.imread(str(probe/'SAM2_union_A24.png'),0)>0)
   im=bgr.copy();im[union]=np.rint(.55*im[union]+.45*np.array([0,255,0])).astype(np.uint8);ad=bgr.copy();ad[added]=np.rint(.4*ad[added]+.6*np.array([255,0,255])).astype(np.uint8);full.append(im)
   detail=np.concatenate([bgr[565:704,925:995],im[565:704,925:995],ad[565:704,925:995]],1);detail=cv2.resize(detail,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST);cv2.putText(detail,f'A{i:02d} RGB / private union / additions',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1);details.append(detail)
   if i in [0,24,48]:cv2.imwrite(str(out/f'actual_refined_A{i:02d}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,98])
   row=dict(frame=i,source_BGR_array_SHA256=hashlib.sha256(bgr.tobytes()).hexdigest(),mask_SHA256=sha(out/'mask49'/f'{i:06d}.png'),original_mask_SHA256=oldsha[f'{i:06d}.png'],added_mask_SHA256=sha(out/'added_actual49'/f'{i:06d}.png'),added_pixels=int(added.sum()),all_original_pixels_retained=True,positive=prompts[:len(p['positive'])].tolist(),negative=prompts[len(p['positive']):].tolist(),box=box.ravel().tolist(),score_not_acceptance=scores.tolist());records.append(row);put(out/'frame_receipts'/f'{i:06d}.json',row);job.update(state='CPU_observed_cord49_running',frames_finished=i+1,elapsed_seconds=time.time()-start);put(out/'JOB.json',job)
   if i%7==0:print(json.dumps(dict(frames_finished=i+1,elapsed_seconds=time.time()-start)),flush=True)
  assert not cap.read()[0];cap.release();canvas=np.zeros((7*264,7*480,3),np.uint8)
  for i,im in enumerate(full):y,x=(i//7)*264,(i%7)*480;canvas[y:y+264,x:x+480]=cv2.resize(im,(480,264));cv2.putText(canvas,str(i),(x+4,y+16),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
  cv2.imwrite(str(out/'actual_refined_all49.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,97])
  for j in range(7):cv2.imwrite(str(out/f'actual_cord_frames{j*7:02d}_{j*7+6:02d}.jpg'),np.concatenate(details[j*7:j*7+7],axis=0),[cv2.IMWRITE_JPEG_QUALITY,98])
  for name,h in oldsha.items():assert sha(olddir/name)==h
  assert sha(source)==p['source_SHA256']
  result=dict(case='56',state='all49_actual_observed_cord_private_union_PENDING_actual_review',frames=49,source_SHA256=p['source_SHA256'],model_SHA256=p['model_SHA256'],SAM2_code_git_commit=p['SAM2_code_git_commit'],frame_receipts=records,probe24_exact=True,all_original_pixels_and_mask_files_unchanged=True,Native_RGB_depth_K_E_unchanged=True,no_hidden_continuation_added=True,GPU_used=False,quality_accepted=False,elapsed_seconds=time.time()-start);put(out/'RESULT.json',result);job.update(state='terminal0_all49_masks_pending_actual_review',exit_code=0);print(json.dumps({k:v for k,v in result.items() if k!='frame_receipts'}),flush=True)
 except BaseException as e:job.update(state='terminal1_preserve_no_auto_retry',exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
 finally:job['elapsed_seconds']=time.time()-start;put(out/'JOB.json',job)
if __name__=='__main__':main()

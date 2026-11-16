from pathlib import Path
import argparse,fcntl,hashlib,json,os,sys,time,traceback,datetime
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1';rt=json.loads((N/'runtime/sam2_private_cpu_py312_v1/RESULT.json').read_text());sys.path[:0]=[rt['SAM2_code_root'],rt['dependencies_root']]
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def main(c,probe_version):
 assert c=='80';guard=(Q/f'logs/case{c}_geometry_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
 probe=N/f'outputs/{c}/observed_missing_parts_SAM2_A00_probe_v{probe_version}';p=json.loads((probe/'RESULT.json').read_text());review=json.loads((U/'CASE80_OBSERVED_PARTS_ACTUAL_ANCHORS24_48_REVIEW_v2.json').read_text());assert review['may_build_repaired_frames21_to48'];assert p['case']==c and p['frame']==0;assert sha(rt['checkpoint'])==p['checkpoint_SHA256']=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
 I=N/f'inputs/case{c}_native_CPU_v1';manifest=json.loads((I/'INPUT_MANIFEST.json').read_text());source=N/f'inputs/case{c}_source_video_HF_v1/source_A.mkv';assert sha(source)==p['source_SHA256']==manifest['source_video_SHA256'];olddir=I/'original_mask49';oldsha={f'{i:06d}.png':sha(olddir/f'{i:06d}.png') for i in range(49)}
 out=Q/f'component_cache/{c}/observed_parts_actual_anchors49_v2';out.mkdir(parents=True,exist_ok=False)
 for n in ['mask49','parts49','added_actual49','frame_receipts']:(out/n).mkdir()
 alias=N/f'outputs/{c}/observed_parts_actual_anchors49_v2';assert not alias.exists() and not alias.is_symlink();alias.symlink_to(out,target_is_directory=True)
 ident=dict(case=c,pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],GPU_used=False,CPU_threads=8,quality_accepted=False);job=dict(ident,state='loading_CPU_SAM2_for_finite49_observed_parts',frames_finished=0);put(out/'JOB.json',job)
 try:
  model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',rt['checkpoint'],device='cpu');predictor=SAM2ImagePredictor(model);base=cv2.imread(str(olddir/'000000.png'),0)>0;yy,xx=np.where(base);lo=np.array([xx.min(),yy.min()],float);span=np.array([np.ptp(xx),np.ptp(yy)],float);assert (span>0).all();cap=cv2.VideoCapture(str(source));full=[];details=[];records=[];crop=[150,0,380,105] if c=='80' else [865,490,975,647];x0,y0,x1,y1=crop
  for i in range(49):
   ok,bgr=cap.read();assert ok and bgr.shape==(704,1280,3) and hashlib.sha256(bgr.tobytes()).hexdigest()==manifest['RGB49_BGR_array_SHA256'][i];old=cv2.imread(str(olddir/f'{i:06d}.png'),0)>0;yy,xx=np.where(old);loc=np.array([xx.min(),yy.min()],float);scale=np.array([np.ptp(xx),np.ptp(yy)],float)/span;prompts=(np.array(p['positive']+p['negative'],np.float32)-lo)*scale+loc;box=(np.array(p['box'],np.float32).reshape(2,2)-lo)*scale+loc
   previous=Q/'component_cache/80/observed_missing_parts_SAM49_v1'
   if i<21:
    parts=cv2.imread(str(previous/'parts49'/f'{i:06d}.png'),0)>0;scores=np.array([])
   else:
    anchors={0:dict(positive=[[339,51],[335,37],[313,49],[301,48],[288,47],[284,39],[285,20],[277,-18],[262,-19],[251,-5]],negative=[[310,20],[302,65],[370,51],[319,-15],[270,20]],box=[235,-20,365,98])}
    for j in [24,48]:anchors[j]=json.loads((Q/f'component_cache/80/observed_visible_parts_actual_anchors_v2/RESULT_A{j:02d}.json').read_text())
    a,b=(0,24) if i<=24 else (24,48);t=(i-a)/(b-a);positive=(1-t)*np.array(anchors[a]['positive'],np.float32)+t*np.array(anchors[b]['positive'],np.float32);negative=(1-t)*np.array(anchors[a]['negative'],np.float32)+t*np.array(anchors[b]['negative'],np.float32);prompts=np.concatenate([positive,negative]);box=(1-t)*np.array(anchors[a]['box'],np.float32)+t*np.array(anchors[b]['box'],np.float32);box[1]=max(0,box[1]);box[3]=min(704,box[3])
    with torch.inference_mode():
     predictor.set_image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));mm,scores,_=predictor.predict(point_coords=prompts.astype(np.float32),point_labels=np.array([1]*len(positive)+[0]*len(negative),np.int32),box=box.ravel().astype(np.float32),multimask_output=False)
    parts=mm[0].astype(bool)
   union=old|parts;added=parts&~old;assert not (old&~union).any()
   if i<21:np.testing.assert_array_equal(union,cv2.imread(str(previous/'mask49'/f'{i:06d}.png'),0)>0)
   if i in [24,48]:np.testing.assert_array_equal(union,cv2.imread(str(Q/f'component_cache/80/observed_visible_parts_actual_anchors_v2/candidate_union_A{i:02d}.png'),0)>0)
   if i==0:np.testing.assert_array_equal(union,cv2.imread(str(probe/'candidate_union.png'),0)>0)
   for name,a in [('mask49',union),('parts49',parts),('added_actual49',added)]:assert cv2.imwrite(str(out/name/f'{i:06d}.png'),a.astype(np.uint8)*255)
   overlay=bgr.copy();overlay[union]=np.rint(.55*overlay[union]+.45*np.array([0,255,0])).astype(np.uint8);ad=bgr.copy();ad[added]=np.rint(.4*ad[added]+.6*np.array([255,0,255])).astype(np.uint8);full.append(overlay);detail=np.concatenate([bgr[y0:y1,x0:x1],overlay[y0:y1,x0:x1],ad[y0:y1,x0:x1]],1);detail=cv2.resize(detail,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST);cv2.putText(detail,f'A{i:02d} actual RGB / union / additions',(4,14),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1);details.append(detail)
   row=dict(frame=i,RGB_array_SHA256=hashlib.sha256(bgr.tobytes()).hexdigest(),mask_SHA256=sha(out/'mask49'/f'{i:06d}.png'),original_mask_SHA256=oldsha[f'{i:06d}.png'],added_pixels=int(added.sum()),all_original_pixels_retained=True,positive=(positive.tolist() if i>=21 else prompts[:len(p['positive'])].tolist()),negative=(negative.tolist() if i>=21 else prompts[len(p['positive']):].tolist()),parts_method=('exact_copy_previously_viewed_0_to20' if i<21 else 'actual_native_frame_0_24_48_anchored_prompts_no_old_bbox_affine'),box=box.ravel().tolist(),score_not_acceptance=scores.tolist());records.append(row);put(out/'frame_receipts'/f'{i:06d}.json',row);job.update(state='CPU_observed_parts49_running',frames_finished=i+1,elapsed_seconds=time.time()-start);put(out/'JOB.json',job)
   if i%7==0:print(json.dumps(dict(case=c,frames_finished=i+1,seconds=time.time()-start)),flush=True)
  assert not cap.read()[0];cap.release();canvas=np.zeros((7*264,7*480,3),np.uint8)
  for i,im in enumerate(full):
   y,x=(i//7)*264,(i%7)*480;canvas[y:y+264,x:x+480]=cv2.resize(im,(480,264));cv2.putText(canvas,str(i),(x+4,y+16),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
  cv2.imwrite(str(out/'actual_refined_all49.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,97])
  for j in range(7):cv2.imwrite(str(out/f'actual_visible_parts_frames{j*7:02d}_{j*7+6:02d}.jpg'),np.concatenate(details[j*7:j*7+7],axis=0),[cv2.IMWRITE_JPEG_QUALITY,98])
  for name,h in oldsha.items():assert sha(olddir/name)==h
  assert sha(source)==p['source_SHA256'];result=dict(ident,state='ALL49_OBSERVED_PARTS_CANDIDATE_PENDING_ACTUAL_REVIEW',frames=49,source_SHA256=p['source_SHA256'],checkpoint_SHA256=p['checkpoint_SHA256'],source_mask_files_unchanged=True,RGB_depth_K_E_unchanged=True,probe0_exact=True,frame_receipts=records,no_hidden_geometry_claim=True,seconds=time.time()-start);put(out/'RESULT.json',result);job.update(state='terminal0_all49_pending_actual_review',exit_code=0);print(json.dumps({k:v for k,v in result.items() if k!='frame_receipts'}),flush=True)
 except BaseException as e:job.update(state='terminal1_preserve_outputs_no_auto_retry',exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
 finally:put(out/'JOB.json',dict(job,elapsed_seconds=time.time()-start))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--case',required=True);a.add_argument('--probe-version',required=True,type=int);x=a.parse_args();main(x.case,x.probe_version)

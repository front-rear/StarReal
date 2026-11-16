from pathlib import Path
import sys,json,hashlib,time,fcntl,os
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');rt=json.loads((N/'runtime/sam2_private_cpu_py312_v1/RESULT.json').read_text());sys.path[:0]=[rt['SAM2_code_root'],rt['dependencies_root']]
import cv2,numpy as np,torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=(Q/'logs/case83_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();torch.set_num_threads(8);torch.set_num_interop_threads(1);cv2.setNumThreads(1);assert sha(rt['checkpoint'])=='2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318'
 model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',rt['checkpoint'],device='cpu');pred=SAM2ImagePredictor(model);I=N/'inputs/case83_native_CPU_v1';manifest=json.loads((I/'INPUT_MANIFEST.json').read_text());source=N/'inputs/case83_source_video_HF_v1/source_A.mkv';assert sha(source)==manifest['source_video_SHA256'];out=Q/'component_cache/83/observed_visible_parts_SAM_pilot3_v4';out.mkdir(parents=True,exist_ok=False)
 partspec=[dict(name='plate_arm',positive=[[1084,49],[1084,61],[1105,48],[1127,38]],negative=[[1073,38],[1072,74],[1103,63],[1103,30],[1117,59]],box=[1067,28,1143,78]),dict(name='upper_wire_switch',positive=[[1086,78],[1080,95],[1075,106],[1071,119]],negative=[[1095,85],[1069,88],[1087,114],[1062,107]],box=[1061,69,1095,126]),dict(name='lower_wire',positive=[[1067,134],[1061,156],[1055,177],[1048,201],[1043,214]],negative=[[1074,141],[1054,143],[1064,188],[1038,192],[1053,218]],box=[1038,122,1074,220])];positive=[];negative=[];box=[];cap=cv2.VideoCapture(str(source));records=[]
 for i in range(49):
  ok,bgr=cap.read();assert ok and hashlib.sha256(bgr.tobytes()).hexdigest()==manifest['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  oldp=I/'original_mask49'/f'{i:06d}.png';old=cv2.imread(str(oldp),0)>0;oldsha=sha(oldp)
  parts=np.zeros_like(old);partstats=[]
  with torch.inference_mode():
   for spec in partspec:
    pp,nn=spec['positive'],spec['negative'];x0,y0,x1,y1=spec['box'];origin=np.array([max(0,x0-10),max(0,y0-10)]);end=np.array([min(1280,x1+11),min(704,y1+11)]);crop=bgr[origin[1]:end[1],origin[0]:end[0]];pred.set_image(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));masks,scores,_=pred.predict(point_coords=np.array(pp+nn,np.float32)-origin,point_labels=np.array([1]*len(pp)+[0]*len(nn),np.int32),box=np.array(spec['box'],np.float32)-np.tile(origin,2),multimask_output=True)
    votes=[]
    for k,mask in enumerate(masks):
     votes.append(sum(bool(mask[int(y-origin[1]),int(x-origin[0])]) for x,y in pp))
    j=max(range(len(masks)),key=lambda k:(votes[k],float(scores[k])));candidate=np.zeros_like(old);candidate[origin[1]:end[1],origin[0]:end[0]]=masks[j].astype(bool);roi=np.zeros_like(old);roi[y0:y1+1,x0:x1+1]=True;candidate&=roi
    count,lab,stats,centroids=cv2.connectedComponentsWithStats(candidate.astype(np.uint8),8);keep=[]
    for x,y in pp:
     k=int(lab[int(y),int(x)])
     if k>0:keep.append(k)
    filtered=np.isin(lab,keep);parts|=filtered;partstats.append(dict(name=spec['name'],scores=scores.tolist(),selected_candidate=j,prompt_cover_votes=votes,internal_part_native_crop=[origin.tolist(),end.tolist()],pixels=int(filtered.sum()),selected_connected_components=sorted(set(keep))))
  union=old|parts;added=parts&~old;scores=np.array([s for z in partstats for s in z['scores']])
  for name,a in [('parts',parts),('union',union),('additions',added)]:cv2.imwrite(str(out/f'{i:06d}_{name}.png'),a.astype(np.uint8)*255)
  images=[]
  for mask,color in [(np.zeros_like(old),[0,0,0]),(union,[0,255,0]),(added,[255,0,255])]:
   im=bgr.copy();im[mask]=np.rint(.5*im[mask]+.5*np.array(color)).astype(np.uint8);images.append(im[10:245,1025:1185])
  detail=np.concatenate(images,1);detail=cv2.resize(detail,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST);cv2.putText(detail,f'Case83 A{i:02d} actualRGB / union / observed additions; NOT acceptance',(3,14),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1);cv2.imwrite(str(out/f'A{i:02d}_actual_RGB_union_additions.jpg'),detail,[cv2.IMWRITE_JPEG_QUALITY,97]);assert sha(oldp)==oldsha;records.append(dict(frame=i,original_mask_SHA256=oldsha,union_SHA256=sha(out/f'{i:06d}_union.png'),original_mask_pixels_retained=True,added_pixels=int(added.sum()),score_not_acceptance=scores.tolist(),parts=partstats))
 cap.release();r=dict(case='83',state='PRIVATE_ACTUAL_VISIBLE_PARTS_PILOT_PENDING_REVIEW',partspec=partspec,component_cleanup="Retain connected SAM output intersecting actual visible part prompts; exact local observed part ROI. Original mask retained.",checkpoint_SHA256=sha(rt['checkpoint']),source_SHA256=sha(source),frames=records,Original_RGB_SAM_depth_K_E_unchanged=True,CPU_only=True,quality_accepted=False,seconds=time.time()-start);(out/'RESULT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()

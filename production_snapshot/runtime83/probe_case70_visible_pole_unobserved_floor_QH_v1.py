"""Read-only visible pole registration; do not invent observed floor or hidden feet."""
from pathlib import Path
import os,sys,json,hashlib,time,fcntl
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');I=N/'inputs/case70_native_CPU_v1';Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
import cv2,numpy as np
from real_geometry import unproject,project,plane_intersection
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 f=(Q/'logs/case70_geometry_component.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);cv2.setNumThreads(1);start=time.time();out=N/'outputs/70/visible_pole_unobserved_floor_probe_QH_v1';out.mkdir(parents=True,exist_ok=False)
 cam=np.load(I/'cameras.npz');raw=np.load(I/'native_depth_K_E_subset.npz');K,A=cam['K'][0],cam['w2c_A'][0];mask=cv2.imread(str(I/'original_mask49/000000.png'),0)>0;depth=cv2.resize(raw['depth'][0],(1280,704),interpolation=cv2.INTER_LINEAR);m=json.loads((I/'INPUT_MANIFEST.json').read_text());anchors={}
 seeds={'shade_top_left':[330,85],'shade_top_right':[440,85],'shade_left':[330,147],'shade_right':[443,148],'shade_bottom_left':[336,211],'shade_bottom_right':[438,211],'shade_center_bottom':[387,212],'stem_top':[389,232],'stem_upper':[385,290],'stem_mid':[383,407],'stem_lower':[381,500],'stem_bottom':[380,558]}
 for label,(x,y) in seeds.items():
  x0,x1=max(0,x-20),min(1280,x+21);y0,y1=max(0,y-20),min(704,y+21);yy,xx=np.where(mask[y0:y1,x0:x1]);assert len(xx),label;pix=np.c_[xx+x0,yy+y0];pix=pix[np.argmin(((pix-[x,y])**2).sum(1))];x,y=map(int,pix);a,b=max(0,y-2),min(704,y+3);c,d=max(0,x-2),min(1280,x+3);z=float(np.median(depth[a:b,c:d][mask[a:b,c:d]]));world=unproject(pix[None],np.array([z]),K,A)[0];anchors[label]=dict(pixel=pix.tolist(),Native_camera_z=z,Native_world=world.tolist())
 upper=np.array(anchors['stem_upper']['Native_world']);lower=np.array(anchors['stem_bottom']['Native_world']);up=(upper-lower)/np.linalg.norm(upper-lower);direction=lower-upper;rayrow=K[1]-680*K[2];t=-(rayrow@(A@np.r_[lower,1])[:3])/(rayrow@(A[:3,:3]@direction));foot=lower+t*direction;P=lower;pd=-float(up@P);moves={}
 for x in [1030.,1070.]:
  w=plane_intersection([x,float(anchors['stem_bottom']['pixel'][1])],K,A,up,pd);moves[str(int(x))]=dict(registration_A00=[x,float(anchors['stem_bottom']['pixel'][1])],world=w.tolist(),estimated_foot_world=(foot+w-P).tolist(),estimated_foot_A00=project((foot+w-P)[None],K,A)[0][0].tolist())
 refs=dict(source_visible_pole_registration_A00=anchors['stem_bottom']['pixel'],source_visible_pole_registration_world=P.tolist(),estimated_up_world=up.tolist(),camera_up_alignment=float(up@(-A[:3,:3].T[:,1])),estimated_hidden_floor_reference_world=foot.tolist(),estimated_hidden_floor_reference_A00=project(foot[None],K,A)[0][0].tolist(),estimated_floor_plane_d=-float(up@foot),source_lower_foot_chair_occluded=True,shade_observed_complete=True,floor_contact_observed=False,hidden_foot_mesh_added=False,moves=moves)
 cap=cv2.VideoCapture(str(N/'inputs/case70_source_video_HF_v1/source_A.mkv'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);imgs=[]
 for i in range(49):
  ok,im=cap.read();assert ok and hashlib.sha256(im.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  if i==0:
   for label,p in anchors.items():x,y=p['pixel'];cv2.circle(im,(x,y),2,(0,255,0),-1);cv2.putText(im,label,(x-85,max(12,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.32,(0,255,255),1)
   for x in [1030,1070]:cv2.line(im,(x,180),(x,700),(255,0,255),1);cv2.putText(im,'candidate '+str(x),(x-35,180),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,0,255),1)
  cv2.putText(im,'70 actual source: chair occludes base; floor estimate NOT measured; no RGB edit',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.44,(0,255,255),1);p=out/f'actual_source_reference_A{i:02d}.jpg';cv2.imwrite(str(p),im,[cv2.IMWRITE_JPEG_QUALITY,98]);imgs.append(dict(path=str(p.relative_to(N)),SHA256=sha(p)))
 cap.release();r=dict(case='70',state='visible_pole_and_unobserved_floor_reference_pending_actual_pilot_review',source_anchors=anchors,references=refs,actual_images=imgs,source_SHA256=m['source_video_SHA256'],Original_inputs_unchanged=True,GPU_used=False,quality_accepted=False,elapsed_seconds=time.time()-start);(out/'RESULT.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_anchors','actual_images']}),flush=True)
if __name__=='__main__':main()

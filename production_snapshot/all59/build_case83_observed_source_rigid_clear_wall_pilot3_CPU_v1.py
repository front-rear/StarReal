from pathlib import Path
import sys,json,hashlib,time,os,fcntl
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');I=N/'inputs/case83_native_CPU_v1'
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
import cv2,numpy as np
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,plane_intersection,project,visibility
from case19_source_shell_reference_QH_v1 import regularized_shell,overlay
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 guard=(Q/'logs/case83_geometry_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);cv2.setNumThreads(1);start=time.time()
 out=Q/'component_cache/83/source_observed_fixture_rigid_clear_wall_pilot3_v1';out.mkdir(exist_ok=False)
 state=dict(case='83',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split()[21],state='running',GPU_used=False)
 def save():(out/'JOB.json').write_text(json.dumps(state,indent=2))
 save()
 try:
  rpath=Q/'reviews/CASE83_OBSERVED_SOURCE_SAM49_CURRENT.json';review=json.loads(rpath.read_text());assert review['may_compute_private_source_geometry']
  probe=json.loads((Q/'component_cache/83/native_actual_same_wall_support_probe_v1/RESULT.json').read_text())
  cams=np.load(I/'cameras.npz');raw=np.load(I/'native_depth_K_E_subset.npz');confidence=np.load(N/'inputs/CASE83_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf']
  K,A=cams['K'][0],cams['w2c_A'][0];n=np.array(probe['normal_world']);d=probe['d'];Ppix=[1084.,55.];Qpix=[625.,170.];P=plane_intersection(Ppix,K,A,n,d);mount=plane_intersection(Qpix,K,A,n,d)
  specs={'plate_top':([1084,41],.006),'plate_center':([1084,55],.006),'plate_bottom':([1081,69],.006),'arm_base':([1092,53],.009),'arm_elbow':([1115,43],.02),'head_top':([1128,27],.028),'neck':([1130,51],.04),'shade_top':([1131,64],.045),'shade_left':([1114,97],.048),'shade_right':([1165,94],.048),'shade_bottom':([1142,119],.05),'wire_top':([1092,58],.002),'switch_top':([1084,100],.004),'switch_bottom':([1078,126],.004),'wire_last_observed':([1051,220],.002)}
  anchors={}
  for k,(pix,offset) in specs.items():
   w=plane_intersection(pix,K,A,n,d-offset);anchors[k]=np.r_[pix,float(project(w[None],K,A)[1][0])]
  segments=[('plate_top','plate_center',6),('plate_center','plate_bottom',6),('arm_base','arm_elbow',3),('arm_elbow','head_top',3),('head_top','neck',16),('neck','shade_top',25),('shade_top','shade_left',55),('shade_top','shade_right',55),('shade_left','shade_bottom',55),('shade_right','shade_bottom',55),('wire_top','switch_top',1.5),('switch_top','switch_bottom',5),('switch_bottom','wire_last_observed',1.5)]
  mask=cv2.imread(str(Path(review['source_mask49'])/'000000.png'),0)>0;v,f,_=regularized_shell(mask,K,A,anchors,segments)
  behind=v@n+d<0
  if behind.any():
   uv,_=project(v[behind],K,A);v[behind]=np.array([plane_intersection(q,K,A,n,d) for q in uv])
  up=-A[:3,:3].T[:,1];up-=n*(up@n);up/=np.linalg.norm(up);rot=np.stack([np.cross(n,up),n,up],1);center,size,basis=bbox_for_mesh(v,rot);moved=v+mount-P
  geom=dict(source_vertices=v,source_faces=f,source_center=center,source_size=size,source_rotation=basis,source_mount=P,source_unobserved_completion_vertex=np.zeros(len(v),bool),move_vertices=moved,move_faces=f,move_mount=mount,K=cams['K'],w2c_A=cams['w2c_A'],w2c_B=cams['w2c_B'])
  np.savez_compressed(out/'geometry.npz',**geom)
  cap=cv2.VideoCapture(str(N/'inputs/case83_source_video_HF_v1/source_A.mkv'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);rast=Rasterizer();proof=json.loads((I/'INPUT_MANIFEST.json').read_text());rows=[]
  for i in range(49):
   ok,bgr=cap.read();assert ok and hashlib.sha256(bgr.tobytes()).hexdigest()==proof['RGB49_BGR_array_SHA256'][i]
   if i not in [0,24,48]:continue
   observed=cv2.imread(str(Path(review['source_mask49'])/f'{i:06d}.png'),0)>0;small=cv2.resize(observed.astype(np.float32),(raw['depth'].shape[2],raw['depth'].shape[1]),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0
   eligible=np.isfinite(confidence[i])&(confidence[i]>1.)&~small
   sv,sf=depth_surface(raw['depth'][i],raw['K'][i],cams['w2c_A'][i],valid_pixels=eligible,max_neighbor_ratio=1.15);scene=rast.depth(sv,sf,cams['K'][i],cams['w2c_A'][i])
   for role,verts in [('source',v),('rigid_move',moved)]:
    dep=rast.depth(verts,f,cams['K'][i],cams['w2c_A'][i]);hit=np.isfinite(dep);layers=visibility(dep,scene);left=bgr.copy();left[hit]=np.rint(.4*left[hit]+.6*np.array([255,0,255])).astype(np.uint8);right=overlay(bgr,layers)
    for im,label in [(left,'fixed world full observed silhouette shell'),(right,'Native visibility estimate; cyan UNKNOWN')]:cv2.putText(im,f'83 {role} A{i:02d}: {label}; NOT editedRGB/GT',(8,21),0,.4,(0,255,255),1)
    p=out/f'{role}_A{i:02d}_whole_vs_visibility.jpg';cv2.imwrite(str(p),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,96])
    rows.append(dict(role=role,frame=i,path=str(p),SHA256=sha(p),pixels=int(hit.sum()),visibility={k:int(a.sum()) for k,a in layers.items()}))
  cap.release()
  result=dict(case='83',state='source_whole_observed_fixture_and_rigid_move_pending_actual_review',source_review=str(rpath),source_review_SHA256=sha(rpath),geometry=str(out/'geometry.npz'),geometry_SHA256=sha(out/'geometry.npz'),source_mount_A00=Ppix,target_mount_A00=Qpix,source_support_plane=probe,source_anchors={k:v.tolist() for k,v in anchors.items()},source_front_back_thickness_are_estimates=True,source_no_hidden_cord_completion=True,wire_ends_at_actual_source_foreground_occlusion=True,source_original_wall_contact_observed=True,source_original_mount_is_not_floor=True,source_original_size_orientation_and_appearance_preserved_in_pure_move=True,world_translation=(mount-P).tolist(),actual_pilots=rows,Native_RGB_depth_K_E_and_original_SAM_preserved=True,quality_accepted=False,GPU_used=False,seconds=time.time()-start)
  (out/'RESULT.json').write_text(json.dumps(result,indent=2));state.update(state='terminal0_pending_actual_pilots_review',exit_code=0,seconds=time.time()-start);save();print(json.dumps({k:v for k,v in result.items() if k not in ['actual_pilots','source_anchors','source_support_plane']}))
 except Exception as e:state.update(state='failed',exit_code=1,error=repr(e));save();raise
if __name__=='__main__':main()

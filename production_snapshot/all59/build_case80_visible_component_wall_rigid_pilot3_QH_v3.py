from pathlib import Path
import sys,json,hashlib,os,time,fcntl
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');I=N/'inputs/case80_native_CPU_v1';sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
import cv2,numpy as np
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,plane_intersection,project,visibility,unproject
from case19_source_shell_reference_QH_v1 import regularized_shell,overlay
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=(Q/'logs/case80_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cv2.setNumThreads(1);start=time.time();review=json.loads((Q/'reviews/CASE80_OBSERVED_SOURCE_SAM49_CURRENT.json').read_text());assert review['may_compute_private_source_geometry'] and review['reference_frame']==48;p=json.loads((Q/'component_cache/80/native_actual_support_probe_v1/RESULT.json').read_text());cam=np.load(I/'cameras.npz');raw=np.load(I/'native_depth_K_E_subset.npz');K,A=cam['K'][48],cam['w2c_A'][48];row=p['planes']['clear_wall_right_of_plate'];n=np.array(row['normal_world']);d=row['d'];Ppix=[337.,96.];P=plane_intersection(Ppix,K,A,n,d);anchors={}
 for key,r in p['source_anchors'].items():
  pix=r['pixel_in_reference'];w=plane_intersection(pix,K,A,n,d) if key.startswith('plate') else np.array(r['world']);z=float(project(w[None],K,A)[1][0]);anchors[key]=np.r_[pix,z]
 # Reconstruct visible plate/arm/shade components, avoiding a pixel extrusion's severe side-view distortion.
 up_fit=-A[:3,:3].T[:,1];up_fit-=n*(up_fit@n);up_fit/=np.linalg.norm(up_fit);C=-A[:3,:3].T@A[:3,3]
 def fit_on_axis(origin,direction,pixel):
  ray=unproject(np.array([pixel],float),np.ones(1),K,A)[0]-C;co=np.linalg.lstsq(np.stack([direction,-ray],1),C-origin,rcond=None)[0];return origin+co[0]*direction,float(co[0])
 elbow,t_arm=fit_on_axis(P,n,[285.,90.]);top,t_up=fit_on_axis(elbow,up_fit,[281.,29.]);tip,t_tip=fit_on_axis(top,n,[253.,27.]);tip_z=float(project(tip[None],K,A)[1][0]);neck=unproject(np.array([[239.,58.]]),np.array([tip_z]),K,A)[0];length_image=46.*tip_z/K[1,1];aperture=unproject(np.array([[232.,104.]]),np.array([tip_z-.20*length_image]),K,A)[0];assert min(t_arm,t_up,t_tip)>0
 pieces=[]
 def frustum(a,b,ra,rb,sections=48):
  axis=b-a;axis/=np.linalg.norm(axis);helper=np.array([1.,0.,0.]) if abs(axis[0])<.8 else np.array([0.,1.,0.]);e1=np.cross(axis,helper);e1/=np.linalg.norm(e1);e2=np.cross(axis,e1);theta=np.arange(sections)*2*np.pi/sections;ring=np.cos(theta)[:,None]*e1+np.sin(theta)[:,None]*e2;vs=np.concatenate([a+ra*ring,b+rb*ring,a[None],b[None]]);fs=[]
  for j in range(sections):
   k=(j+1)%sections;fs.extend([[j,k,sections+k],[j,sections+k,sections+j],[2*sections,k,j],[2*sections+1,sections+j,sections+k]])
  pieces.append((vs,np.asarray(fs,np.int32)))
 zP=float(project(P[None],K,A)[1][0]);radius_plate=26.*zP/K[1,1];plate_thickness=3.*zP/K[1,1];frustum(P,P+n*plate_thickness,radius_plate,radius_plate,64)
 tube_radius=2.5*tip_z/K[0,0]
 for a,b in [(P+n*plate_thickness,elbow),(elbow,top),(top,tip),(tip,neck)]:frustum(a,b,tube_radius,tube_radius,24)
 frustum(neck,aperture,10.*tip_z/K[0,0],45.*tip_z/K[0,0],64)
 arrays=[];faces=[];offset=0
 for vv,ff in pieces:arrays.append(vv);faces.append(ff+offset);offset+=len(vv)
 v=np.concatenate(arrays);f=np.concatenate(faces);behind=v@n+d < -1e-8
 assert not behind.any(),'Visible component geometry must stay on the room side of its source wall'
 component_fit=dict(method='visible-component relative reconstruction: source wall plate disk, fitted arm/post axes, and recognizable three-dimensional tapered shade',reference_frame=48,source_axes_use_actual_native_K_E_and_wall_plane=True,source_visible_mask49_unchanged=True,source_appearance_RGBA_unchanged=True,source_backside_material_not_recovered=True,not_observed_edited_RGB_or_metric_mesh=True,arm_extension=t_arm,post_length=t_up,upper_extension=t_tip,plate_radius=radius_plate,shade_depth_tilt_fraction=.20,shade_pixel_controls=dict(neck=[239,58],aperture_center=[232,104],neck_radius=10,aperture_radius=45),physical_scale_is_relative=True)

 up=-A[:3,:3].T[:,1];up-=n*(up@n);up/=np.linalg.norm(up);x=np.cross(n,up);x/=np.linalg.norm(x);rot=np.stack([x,np.cross(up,x),up],1);assert np.linalg.det(rot)>0;center,size,basis=bbox_for_mesh(v,rot);out=N/'outputs/80/source_visible_components_wall_rigid_pilot3_QH_v3';out.mkdir(exist_ok=False);folder=out/'upper_wall650_120';folder.mkdir();pix=[650.,120.];target=plane_intersection(pix,cam['K'][0],cam['w2c_A'][0],n,d);moved=v+target-P;np.savez_compressed(folder/'geometry.npz',source_vertices=v,source_faces=f,source_center=center,source_size=size,source_rotation=basis,source_mount=P,source_reference_frame=np.array(48),source_unobserved_completion_vertex=np.zeros(len(v),bool),move_vertices=moved,move_faces=f,move_mount=target,K=cam['K'],w2c_A=cam['w2c_A'],w2c_B=cam['w2c_B']);jobs={'source':(v,f),'upper_wall650_120':(moved,f)};meta=json.loads((I/'INPUT_MANIFEST.json').read_text());confidence=np.load(N/'inputs/CASE80_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'];cap=cv2.VideoCapture(str(N/'inputs/case80_source_video_HF_v1/source_A.mkv'));rast=Rasterizer();records=[]
 for i in range(49):
  ok,rgb=cap.read();assert ok and hashlib.sha256(rgb.tobytes()).hexdigest()==meta['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  observed=cv2.imread(str(Path(review['source_mask49'])/f'{i:06d}.png'),0)>0;small=cv2.resize(observed.astype(np.float32),(raw['depth'].shape[2],raw['depth'].shape[1]),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0;valid=np.isfinite(confidence[i])&(confidence[i]>1.)&~small;sv,sf=depth_surface(raw['depth'][i],raw['K'][i],cam['w2c_A'][i],valid_pixels=valid,max_neighbor_ratio=1.15);scene=rast.depth(sv,sf,cam['K'][i],cam['w2c_A'][i])
  for name,(vertices,faces) in jobs.items():
   z=rast.depth(vertices,faces,cam['K'][i],cam['w2c_A'][i]);hit=np.isfinite(z);layers=visibility(z,scene);left=rgb.copy();left[hit]=np.rint(.4*left[hit]+.6*np.array([255,0,255])).astype(np.uint8);right=overlay(rgb,layers);contours,_=cv2.findContours(hit.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(right,contours,-1,(0,255,255),1)
   for im,label in [(left,'amodal fixed-world source shell'),(right,'Native estimated visibility/cyan unknown')]:cv2.putText(im,f'Case80 {name} A{i:02d} {label}; NOT editedRGB',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
   dest=out/name;dest.mkdir(exist_ok=True);image=dest/f'whole_vs_visibility_A{i:02d}.jpg';cv2.imwrite(str(image),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,97]);records.append(dict(role=name,frame=i,path=str(image),SHA256=sha(image),amodal_pixels=int(hit.sum()),visibility={k:int(a.sum()) for k,a in layers.items()}))
 cap.release();result=dict(case='80',state='PRIVATE_SOURCE_AND_RIGID_WALL_MOVE_PENDING_ACTUAL_REVIEW',source_reference_frame=48,source_mask49=review['source_mask49'],source_mount_A48=Ppix,source_mount_A00=project(P[None],cam['K'][0],cam['w2c_A'][0])[0][0].tolist(),source_support_plane=dict(row,support_type='wall',source_plate_visually_observed=True,relative_not_metric=True),models={'upper_wall650_120':dict(target_mount_A00=pix,geometry_SHA256=sha(folder/'geometry.npz'),world_translation=(target-P).tolist(),whole_source_size_and_orientation_preserved=True)},actual_pilot_images=records,source_back_thickness_estimated=True,source_component_fit=component_fit,source_visible_parts_reconstructed_from_actual_full_A48=True,source_unobserved_completion_added=False,behind_wall_estimate_vertices_projected_to_wall=int(behind.sum()),Original_RGB_depth_K_E_unchanged=True,quality_accepted=False,GPU_used=False,pid=os.getpid(),seconds=time.time()-start);(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['actual_pilot_images','source_support_plane']}),flush=True)
if __name__=='__main__':main()

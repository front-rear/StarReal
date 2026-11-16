"""One finite five-role actual RGB geometry pilot, Qinghai CPU only."""
from pathlib import Path
import fcntl,hashlib,json,os,sys,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');I=N/'inputs/case72_native_CPU_v1';M=N/'inputs/case72_three_fresh_whole_models_v1';sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
import cv2,numpy as np
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,plane_intersection,project,unproject,visibility,place_asset
from case19_source_shell_reference_QH_v1 import regularized_shell,overlay
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rotation(rear,n,up):
 rear=np.asarray(rear,float);rear/=np.linalg.norm(rear);ru=np.array([0.,0.,1.]);ru-=rear*(rear@ru);ru/=np.linalg.norm(ru);B=np.stack([np.cross(rear,ru),rear,ru],1);T=np.stack([np.cross(n,up),n,up],1);R=T@B.T;assert np.allclose(R.T@R,np.eye(3)) and np.linalg.det(R)>0;return R

def main():
 lock=(Q/'logs/case72_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);t=time.time();cv2.setNumThreads(1);out=N/'outputs/72/replacement_upper_rear_mount_pilot3_v2';out.mkdir(exist_ok=False);identity=dict(case='72',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]);(out/'JOB.json').write_text(json.dumps(dict(identity,state='running_finite'),indent=2))
 cam=np.load(I/'cameras.npz');raw=np.load(I/'native_depth_K_E_subset.npz');conf=np.load(N/'inputs/CASE72_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'];p=json.loads((N/'outputs/72/native_wall_source_A00_probe_v1/RESULT.json').read_text());K,A=cam['K'][0],cam['w2c_A'][0];obsdir=I/'original_mask49';obs=cv2.imread(str(obsdir/'000000.png'),0)>0
 n=np.array(p['planes']['source_bare_wall']['normal_world']);dP=p['planes']['source_bare_wall']['d'];dQ=dP;dI=-float(n@np.array(p['planes']['right_clear_wall']['queries']['insert_right_blank']['world']));P=plane_intersection([960,50],K,A,n,dP);Qmove=plane_intersection([855,65],K,A,n,dQ);up=-A[:3,:3].T[:,1];up-=n*(n@up);up/=np.linalg.norm(up);basis=np.stack([np.cross(n,up),n,up],1)
 anchors={}
 for key,row in p['source_anchors'].items():
  pix=row['pixel'];z=row['Native_camera_z']
  if key in ['backplate','upper_rod_clipped','lower_rod']:
   # Fine observed rods leak wall depth; register them at a connected wall stand-off estimate.
   gap={'backplate':.008,'upper_rod_clipped':.006,'lower_rod':.006}[key];z=float(project(plane_intersection(pix,K,A,n,dP+gap)[None],K,A)[1][0])
  anchors[key]=np.r_[pix,z]
 # An additional actually observed globe-interior Native anchor, not a B observation.
 H,W=raw['depth'].shape[1:];x,y=int(round(977*W/1280)),int(round(274*H/704));patch=raw['depth'][0,y-2:y+3,x-2:x+3];valid=np.isfinite(patch)&(patch>0);assert valid.any();anchors['globe_center']=np.r_[977,274,float(np.median(patch[valid]))]
 segments=[('upper_rod_clipped','backplate',5),('backplate','lower_rod',5),('lower_rod','globe_top',7),('globe_top','globe_center',50),('globe_left','globe_center',50),('globe_right','globe_center',50),('globe_bottom','globe_center',50),('globe_bottom','lower_stub',3)]
 v,f,_=regularized_shell(obs,K,A,anchors,segments)
 # Keep estimated closed shell behind its observed front and in front of its mounting wall.
 behind=v@n+dP<0
 if behind.any():
  uv,_=project(v[behind],K,A);v[behind]=np.array([plane_intersection(x,K,A,n,dP) for x in uv])
 moved=v+Qmove-P;np.testing.assert_allclose(np.ptp(v,0),np.ptp(moved,0),atol=1e-13);center,size,b=bbox_for_mesh(v,basis);folder=out/'move';folder.mkdir();geom=folder/'geometry.npz';np.savez_compressed(geom,source_vertices=v,source_faces=f,source_center=center,source_size=size,source_rotation=b,source_mount=P,source_unobserved_completion_vertex=np.zeros(len(v),bool),move_vertices=moved,move_faces=f,move_mount=Qmove,K=cam['K'],w2c_A=cam['w2c_A'],w2c_B=cam['w2c_B']);jobs={'source':dict(vertices=v,faces=f,op='source'),'move':dict(vertices=moved,faces=f,op='move')};models={};refs={'source':project(P[None],cam['K'][24],cam['w2c_A'][24])[0][0].tolist(),'move':project(Qmove[None],cam['K'][24],cam['w2c_A'][24])[0][0].tolist()}
 contacts=json.loads((M/'ACTUAL_REAR_CONTACTS_v1.json').read_text())['models']
 jobs={}
 for idx,op,pix,h,d in [('orig_e3849e99','replace',[960,50],75.,dP)]:
  z=np.load(M/'mesh_arrays'/f'{idx}.npz');rv,rf=z['vertices'],z['faces'];r=dict(contacts[idx]);rear_points=rv[np.isclose(rv[:,0],r['mount'][0],atol=1e-5)];assert len(rear_points)>0;mount=rear_points[np.argmax(rear_points[:,2])].copy();r['previous_mid_rail_mount']=r['mount'];r['mount']=mount.tolist();r['method']='Actual upper vertex of the same original rear mounting rail, at minimumX. Attach the upper rail to unchanged source wall contact so the whole lantern hangs below it; no source/camera/scale/orientation/mesh change. Relative support remains an estimate.';rear=np.array(r['rear_axis_local']);R=rotation(rear,n,up);target=plane_intersection(pix,K,A,n,d);scale=h*float(project(target[None],K,A)[1][0])/K[1,1]/np.ptp(rv[:,2]);world=place_asset(rv,mount,target,R,scale);c,s,b=bbox_for_mesh(world,R);name=op+'_'+idx;folder=out/name;folder.mkdir();np.savez_compressed(folder/'geometry.npz',vertices=world,faces=rf,raw_vertices=rv,raw_faces=rf,actual_raw_mount=mount,world_mount=target,uniform_scale=scale,center=c,size=s,rotation=b,K=cam['K'],w2c_A=cam['w2c_A'],w2c_B=cam['w2c_B']);jobs[name]=dict(vertices=world,faces=rf,op=op);refs[name]=project(target[None],cam['K'][24],cam['w2c_A'][24])[0][0].tolist();models[name]=dict(index=idx,raw_SHA256=sha(M/'raw_assets'/f'lamp_{idx}.glb'),whole_raw_V_F_nodes_preserved=True,contact=r,raw_mount=mount.tolist(),mount_world=target.tolist(),mount_A00=pix,height_ref_px=h,uniform_scale=scale,rotation=R.tolist(),wall_signed_range=[float((world@n+d).min()),float((world@n+d).max())],geometry_SHA256=sha(folder/'geometry.npz'),quality_accepted=False)
 manifest=json.loads((I/'INPUT_MANIFEST.json').read_text());cap=cv2.VideoCapture(str(N/'inputs/case72_source_video_HF_v1/source_A.mkv'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);rast=Rasterizer();records=[]
 for i in range(49):
  ok,rgb=cap.read();assert ok and hashlib.sha256(rgb.tobytes()).hexdigest()==manifest['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  mask=cv2.imread(str(obsdir/f'{i:06d}.png'),0)>0;small=cv2.resize(mask.astype(np.float32),(raw['depth'].shape[2],raw['depth'].shape[1]),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0;scenes={}
  for name,valid in [('full',np.ones_like(small,bool)),('removed',~small)]:
   valid=valid&np.isfinite(conf[i])&(conf[i]>1.);sv,sf=depth_surface(raw['depth'][i],raw['K'][i],cam['w2c_A'][i],valid_pixels=valid,max_neighbor_ratio=1.15);scenes[name]=rast.depth(sv,sf,cam['K'][i],cam['w2c_A'][i])
  for name,z in jobs.items():
   dep=rast.depth(z['vertices'],z['faces'],cam['K'][i],cam['w2c_A'][i]);hit=np.isfinite(dep);layers=visibility(dep,scenes['full' if z['op']=='insert' else 'removed']);left=rgb.copy();left[hit]=np.rint(.4*left[hit]+.6*np.array([255,0,255])).astype(np.uint8);right=rgb.copy();show=layers['visible_with_unknown_estimate'];right[show]=np.rint(.4*right[show]+.6*np.array([255,0,255])).astype(np.uint8);cv2.imwrite(str(out/name/f'uncertainty_A{i:02d}.png'),layers['unobserved_scene_region'].astype(np.uint8)*255);cv2.drawContours(right,cv2.findContours(hit.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,255),1)
   for im,label in [(left,'whole amodal geometry'),(right,'solid visibility; separate uncertainty')]:cv2.putText(im,f'Case72 {name} A{i:02d}: {label}; NOT editedRGB',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
   folder=out/name;folder.mkdir(exist_ok=True);path=folder/f'whole_vs_visibility_A{i:02d}.jpg';cv2.imwrite(str(path),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,97]);records.append(dict(role=name,frame=i,path=str(path.relative_to(N)),SHA256=sha(path),amodal_pixels=int(hit.sum()),visibility={k:int(x.sum()) for k,x in layers.items()}))
 cap.release();r=dict(case='72',state='five_role_actual_wall_pilots_pending_review',source_reference_frame=0,source_mount=P.tolist(),move_mount=Qmove.tolist(),source_wall_normal=n.tolist(),dP=dP,dQ=dQ,dI=dI,source_anchors={k:x.tolist() for k,x in anchors.items()},source_shell_hidden_back_thickness_estimated=True,source_front_mask_original_SAM_unchanged=True,source_partial_upper_rod_cut_in_observed_frame_preserved=True,source_globe_back_thickness_explicit_estimate=True,left_mixed_surface_plane_not_adopted=True,source_and_target_common_observed_wall_plane_estimate=True,source_observed_mask_root=str(obsdir.relative_to(N)),original_SAM_preserved=True,source_original_wall_position_retained=True,pure_move_translation_only=True,reviewed_operation_references_A24=refs,models=models,actual_pilot_images=records,quality_accepted=False,new_supervised_GT=0,GPU_used=False,elapsed_seconds=time.time()-t);(out/'RESULT.json').write_text(json.dumps(r,indent=2));(out/'JOB.json').write_text(json.dumps(dict(identity,state='terminal0_pending_actual_review',exit_code=0,elapsed_seconds=time.time()-t),indent=2));print(json.dumps(dict(case='72',elapsed_seconds=time.time()-t,roles=list(jobs),quality_accepted=False)),flush=True)
if __name__=='__main__':main()

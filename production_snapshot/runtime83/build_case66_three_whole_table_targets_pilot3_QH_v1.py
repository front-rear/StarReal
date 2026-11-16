"""Three fresh whole fixtures on visually reviewed floor locations; no crop."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,sys,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');I=N/'inputs/case66_native_CPU_v1';M=N/'inputs/case66_three_fresh_whole_models_v1';OBS=I
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
import cv2,numpy as np
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,place_asset,plane_intersection,project,visibility
from case19_source_shell_reference_QH_v1 import rotation_on_plane,overlay
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 f=(Q/'logs/case66_geometry_component.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);cv2.setNumThreads(1);t=time.time()
 out=N/'outputs/66/three_whole_table_targets_pilot3_QH_v1';out.mkdir(exist_ok=False);identity=dict(case='66',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]);(out/'JOB.json').write_text(json.dumps(dict(identity,state='running_finite_whole_targets_pilot'),indent=2))
 cam=np.load(I/'cameras.npz');raw=np.load(I/'native_depth_K_E_subset.npz');confidence=np.load(N/'inputs/CASE66_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'];p=json.loads((N/'outputs/66/source_A00_table_rigid_pilot3_QH_v1/RESULT.json').read_text());K,A=cam['K'][0],cam['w2c_A'][0];jobs={};models={}
 for name,op,idx,pix,h,pl,yaw in [('insert_orig_12c9c39b','insert','orig_12c9c39b',[1120.,542.],170.,'actual_clear_right_table_relative_plane',0.),('replace_orig_89f71a03','replace','orig_89f71a03',[970.,545.],360.,'actual_original_visible_base_table_relative_plane',0.),('move_replace_orig_107ae0f2','move_replace','orig_107ae0f2',[1100.,544.],220.,'actual_clear_right_table_relative_plane',0.)]:
  row=p['source_support_plane'];n=np.array(row['normal_world']);d=row['d'];target=plane_intersection(pix,K,A,n,d);rot=rotation_on_plane(n,A[:3,:3].T);ang=np.radians(yaw);rot=rot@np.array([[np.cos(ang),-np.sin(ang),0],[np.sin(ang),np.cos(ang),0],[0,0,1]])
  z=np.load(M/'mesh_arrays'/f'{idx}.npz');rv,rf=z['vertices'],z['faces'];tri=rv[rf];level=float(rv[:,2].min());eps=max(float(np.ptp(rv[:,2]))*1e-6,1e-8)
  flat=np.max(tri[:,:,2],axis=1)<=level+eps;foot=rv[rv[:,2]<=level+eps];assert len(foot)>0;mount=(foot.min(0)+foot.max(0))/2;mount[2]=level

  scale=h*float(project(target[None],K,A)[1][0])/K[1,1]/np.ptp(rv[:,2]);world=place_asset(rv,mount,target,rot,scale);c,s,basis=bbox_for_mesh(world,rot);folder=out/name;folder.mkdir();np.savez_compressed(folder/'geometry.npz',vertices=world,faces=rf,raw_vertices=rv,raw_faces=rf,actual_raw_mount=mount,world_mount=target,uniform_scale=scale,center=c,size=s,rotation=basis,K=cam['K'],w2c_A=cam['w2c_A'],w2c_B=cam['w2c_B']);foot_world=place_asset(foot,mount,target,rot,scale);uv,_=project(foot_world,K,A)
  jobs[name]=dict(vertices=world,faces=rf,op=op);models[name]=dict(operation=op,index=idx,raw_SHA256=sha(M/'raw_assets'/f'lamp_{idx}.glb'),whole_raw_V_F_retained=True,vertices=len(rv),faces=len(rf),actual_low_contact_triangles=int(flat.sum()),actual_low_contact_vertices=len(foot),support_contact_policy="actual_lowest_whole_fixture_base_or_leg_vertices_footprint_midpoint_full_geometry_unchanged",base_contact_is_plug_minimum=False,whole_nodes_retained=True,physical_base_contact_level=float(level),whole_accessory_zmin=float(rv[:,2].min()),whole_geometry_uncropped=True,actual_raw_mount=mount.tolist(),selection_reference_frame=0,target_mount_A00=pix,target_mount_A24=project(target[None],cam["K"][24],cam["w2c_A"][24])[0][0].tolist(),height_reference_px_A00=h,yaw_degrees=yaw,support_plane_relative_prior=pl,normal=n.tolist(),plane_d=d,whole_bottom_foot_projection_bounds_A00=[uv.min(0).tolist(),uv.max(0).tolist()],uniform_scale=scale,geometry_SHA256=sha(folder/'geometry.npz'),quality_accepted=False)
 cap=cv2.VideoCapture(str(N/'inputs/case66_source_video_HF_v1/source_A.mkv'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);rast=Rasterizer();manifest=json.loads((I/'INPUT_MANIFEST.json').read_text());records=[]
 for i in range(49):
  ok,rgb=cap.read();assert ok and hashlib.sha256(rgb.tobytes()).hexdigest()==manifest['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  obs=cv2.imread(str(OBS/'original_mask49'/f'{i:06d}.png'),0)>0;small=cv2.resize(obs.astype(np.float32),(raw['depth'].shape[2],raw['depth'].shape[1]),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0;scenes={}
  for name,valid in [('full',np.ones_like(small,bool)),('removed',~small)]:
   valid=valid&np.isfinite(confidence[i])&(confidence[i]>1.);sv,sf=depth_surface(raw['depth'][i],raw['K'][i],cam['w2c_A'][i],valid_pixels=valid,max_neighbor_ratio=1.15);scenes[name]=rast.depth(sv,sf,cam['K'][i],cam['w2c_A'][i])
  for name,z in jobs.items():
   dep=rast.depth(z['vertices'],z['faces'],cam['K'][i],cam['w2c_A'][i]);hit=np.isfinite(dep);layers=visibility(dep,scenes['full' if z['op']=='insert' else 'removed']);left=rgb.copy();left[hit]=np.rint(.4*left[hit]+.6*np.array([255,0,255])).astype(np.uint8);right=overlay(rgb,layers);contours,_=cv2.findContours(hit.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(right,contours,-1,(0,255,255),1)
   for im,label in [(left,'whole amodal geometry'),(right,'Native visibility/cyan UNKNOWN')]:cv2.putText(im,f'Case66 {name} A{i:02d}: {label}; NOT editedRGB',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
   path=out/name/f'whole_vs_visibility_A{i:02d}.jpg';cv2.imwrite(str(path),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,97]);records.append(dict(role=name,frame=i,path=str(path.relative_to(N)),SHA256=sha(path),amodal_pixels=int(hit.sum()),visibility={k:int(a.sum()) for k,a in layers.items()}))
 cap.release();result=dict(case='66',state='three_actual_whole_targets_pending_support_and_visibility_review',models=models,actual_pilot_images=records,not_editedRGB=True,Native_RGB_SAM_depth_K_E_unchanged=True,all_raw_nodes_retained=True,quality_accepted=False,GPU_used=False,elapsed_seconds=time.time()-t);(out/'RESULT.json').write_text(json.dumps(result,indent=2));(out/'JOB.json').write_text(json.dumps(dict(identity,state='terminal0_pending_actual_review',exit_code=0,elapsed_seconds=time.time()-t),indent=2));print(json.dumps({k:v for k,v in result.items() if k!='actual_pilot_images'}),flush=True)
if __name__=='__main__':main()

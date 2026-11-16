"""Finite source-shell and rigid whole-source moves; Native arrays unchanged."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,sys,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');I=N/'inputs/case66_native_CPU_v1'
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
import cv2,numpy as np
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,plane_intersection,project,visibility
from case19_source_shell_reference_QH_v1 import regularized_shell,rotation_on_plane,overlay
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 lock=(Q/'logs/case66_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cv2.setNumThreads(1)
 out=N/'outputs/66/source_A00_table_rigid_pilot3_QH_v1';out.mkdir(exist_ok=False);t=time.time();identity=dict(case='66',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]);(out/'JOB.json').write_text(json.dumps(dict(identity,state='running_finite_source_move_pilot',GPU_used=False),indent=2))
 cam=np.load(I/'cameras.npz');raw=np.load(I/'native_depth_K_E_subset.npz');p=json.loads((N/'outputs/66/native_table_source_A00_probe_v1/RESULT.json').read_text());K,A=cam['K'][0],cam['w2c_A'][0];row=p['planes']['right_clear_surface'];n=np.array(row['normal_world']);d=row['d'];Ppix=[970.,545.];P=plane_intersection(Ppix,K,A,n,d);Pz=float(project(P[None],K,A)[1][0])
 support_prior=dict(row,source_anchor_native_depth=p['source_anchors']['base_center']['Native_camera_z'],normal_source='Relative plane fit from274 actual clear tabletop Native points inA00, not calibrated gravity or metric support',source_reference_frame=0,source_mount_visual_base_seen=True)
 anchors={key:np.r_[row['pixel'],row['Native_camera_z']] for key,row in p['source_anchors'].items()}
 for key,pix in [('base_left',[930,534]),('base_right',[1006,533]),('base_center',Ppix)]:anchors[key]=np.r_[pix,float(project(plane_intersection(pix,K,A,n,d)[None],K,A)[1][0])]
 segments=[('finial_tip','finial_stem',8),('finial_stem','shade_top',8),('shade_top','shade_left',80),('shade_top','shade_right',80),('shade_left','shade_bottom',80),('shade_right','shade_bottom',80),('shade_bottom','candle_left',12),('shade_bottom','candle_center',12),('shade_bottom','candle_right',12),('candle_left','branch_left',12),('candle_right','branch_right',12),('candle_center','stem',12),('branch_left','stem',12),('branch_right','stem',12),('stem','base_center',18),('base_left','base_center',20),('base_center','base_right',20)]
 mask=cv2.imread(str(I/'original_mask49/000000.png'),0)>0;v,f,_=regularized_shell(mask,K,A,anchors,segments)
 below=v@n+d<0
 if below.any():
  uv,_=project(v[below],K,A);v[below]=np.array([plane_intersection(q,K,A,n,d) for q in uv])
 rot=rotation_on_plane(n,A[:3,:3].T);center,size,basis=bbox_for_mesh(v,rot);jobs={'source':dict(vertices=v,faces=f,op='source')};models={}
 for name,pix,pl in [('clear_right1100',[1100.,544.],'actual_clear_right_table_relative_plane')]:
  mount=plane_intersection(pix,K,A,n,d);world=v+mount-P;folder=out/name;folder.mkdir();z=dict(source_vertices=v,source_faces=f,source_center=center,source_size=size,source_rotation=basis,source_mount=P,source_unobserved_completion_vertex=np.zeros(len(v),bool),move_vertices=world,move_faces=f,move_mount=mount,K=cam['K'],w2c_A=cam['w2c_A'],w2c_B=cam['w2c_B']);np.savez_compressed(folder/'geometry.npz',**z)
  jobs[name]=dict(vertices=world,faces=f,op='move');models[name]=dict(source_mount_A00=Ppix,target_mount_A00=pix,source_mount_A24=project(P[None],cam["K"][24],cam["w2c_A"][24])[0][0].tolist(),target_mount_A24=project(mount[None],cam["K"][24],cam["w2c_A"][24])[0][0].tolist(),relative_support_plane=pl,whole_source_size_orientation_appearance_unchanged=True,source_back_and_thickness_estimated=True,world_translation=(mount-P).tolist(),geometry_SHA256=sha(folder/'geometry.npz'),not_quality_acceptance=True)
 cap=cv2.VideoCapture(str(N/'inputs/case66_source_video_HF_v1/source_A.mkv'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);rast=Rasterizer();confidence=np.load(N/'inputs/CASE66_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'];records=[];manifest=json.loads((I/'INPUT_MANIFEST.json').read_text())
 for i in range(49):
  ok,rgb=cap.read();assert ok and hashlib.sha256(rgb.tobytes()).hexdigest()==manifest['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  obs=cv2.imread(str(I/'original_mask49'/f'{i:06d}.png'),0)>0;small=cv2.resize(obs.astype(np.float32),(raw['depth'].shape[2],raw['depth'].shape[1]),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0;valid=np.isfinite(confidence[i])&(confidence[i]>1.)&~small
  sv,sf=depth_surface(raw['depth'][i],raw['K'][i],cam['w2c_A'][i],valid_pixels=valid,max_neighbor_ratio=1.15);scene=rast.depth(sv,sf,cam['K'][i],cam['w2c_A'][i])
  for name,z in jobs.items():
   dep=rast.depth(z['vertices'],z['faces'],cam['K'][i],cam['w2c_A'][i]);hit=np.isfinite(dep);layers=visibility(dep,scene);left=rgb.copy();left[hit]=np.rint(.4*left[hit]+.6*np.array([255,0,255])).astype(np.uint8);right=overlay(rgb,layers);contours,_=cv2.findContours(hit.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(right,contours,-1,(0,255,255),1)
   for im,label in [(left,'whole amodal geometry'),(right,'Native visibility estimate/cyan unknown')]:cv2.putText(im,f'Case66 {name} A{i:02d}: {label}; NOT editedRGB',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
   folder=out/name;folder.mkdir(exist_ok=True);path=folder/f'whole_vs_visibility_A{i:02d}.jpg';cv2.imwrite(str(path),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,97]);records.append(dict(role=name,frame=i,path=str(path.relative_to(N)),SHA256=sha(path),amodal_pixels=int(hit.sum()),visibility={k:int(a.sum()) for k,a in layers.items()}))
 cap.release();result=dict(case='66',state='source_and_rigid_moves_pending_actual_review',models=models,source_reference_frame=0,source_mount_A00=Ppix,source_mount_A24=project(P[None],cam["K"][24],cam["w2c_A"][24])[0][0].tolist(),source_support_plane=support_prior,source_anchors={k:v.tolist() for k,v in anchors.items()},source_shell_vertices=len(v),source_shell_faces=len(f),source_back_thickness_estimate=True,source_lower_completion_added=False,source_A00_observed_complete_lower_used_not_glove_shape=True,actual_hand_occlusion49_preserved=True,source_original_support_contact_observed=False,source_base_contact_visual_reviewed=True,support_normal_relative_Native_plane_not_calibrated=True,source_original_SAM49_preserved=True,Native_RGB49_depth_K_E_unchanged=True,actual_pilot_images=records,quality_accepted=False,GPU_used=False,elapsed_seconds=time.time()-t);(out/'RESULT.json').write_text(json.dumps(result,indent=2));(out/'JOB.json').write_text(json.dumps(dict(identity,state='terminal0_pending_actual_review',exit_code=0,GPU_used=False,elapsed_seconds=time.time()-t),indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['actual_pilot_images','models','source_anchors','source_support_plane']}),flush=True)
if __name__=='__main__':main()

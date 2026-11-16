from pathlib import Path
import sys,json,hashlib,time,os,fcntl
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');I=N/'inputs/case83_native_CPU_v1';M=Q/'inputs/case83_three_fresh_whole_models_v1'
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
import cv2,numpy as np
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,place_asset,plane_intersection,project,visibility
from case19_source_shell_reference_QH_v1 import overlay
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 guard=(Q/'logs/case83_geometry_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);cv2.setNumThreads(1);start=time.time()
 out=Q/'component_cache/83/three_whole_wall_targets_pilot3_v1';out.mkdir(exist_ok=False)
 ident=dict(case='83',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split()[21],state='running_finite9pilots',GPU_used=False)
 (out/'JOB.json').write_text(json.dumps(ident,indent=2))
 cam=np.load(I/'cameras.npz');raw=np.load(I/'native_depth_K_E_subset.npz');conf=np.load(N/'inputs/CASE83_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'];review=json.loads((Q/'reviews/CASE83_OBSERVED_SOURCE_SAM49_CURRENT.json').read_text());p=json.loads((Q/'component_cache/83/native_actual_same_wall_support_probe_v1/RESULT.json').read_text());n=np.array(p['normal_world']);d=p['d'];K,A=cam['K'][0],cam['w2c_A'][0];up=-A[:3,:3].T[:,1];up-=n*(up@n);up/=np.linalg.norm(up);rot=np.stack([np.cross(n,up),n,up],1)
 used={r['asset_SHA256'] for r in json.loads((Q/'supplements/reference_and_remove_completion_v1/REFERENCE_BINDINGS.json').read_text())['assets']};refs=json.loads((M/'MATERIAL_REFERENCE_SOURCE_BINDINGS.json').read_text());jobs={};models={}
 for op,ix,pix,height,axes in [('insert','orig_66663ad5',[625.,170.],120.,'native_Y_front'),('replace','orig_7095c535',[1084.,55.],110.,'raw_maxX_rear'),('move_replace','orig_65cbbefb',[625.,170.],105.,'raw_maxX_rear')]:
  z=np.load(M/'mesh_arrays'/f'{ix}.npz');original=z['vertices'];faces=z['faces'];rawsha=sha(M/'raw_assets'/f'lamp_{ix}.glb');assert rawsha not in used
  ref=next(x for x in refs['records'] if x['index']==ix);assert ref['asset_SHA256']==rawsha
  transform=np.eye(3) if axes=='native_Y_front' else np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
  rv=original@transform.T;level=float(rv[:,1].min());band=max(np.ptp(rv[:,1])*.008,1e-8);rear=rv[rv[:,1]<=level+band];tri=rv[faces];flat=np.all(tri[:,:,1]<=level+band,axis=1);mount=(rear.min(0)+rear.max(0))/2;mount[1]=level
  target=plane_intersection(pix,K,A,n,d);scale=height*float(project(target[None],K,A)[1][0])/K[1,1]/np.ptp(rv[:,2]);world=place_asset(rv,mount,target,rot,scale);assert float((world@n+d).min())>-1e-6
  center,size,basis=bbox_for_mesh(world,rot);folder=out/(op+'_'+ix);folder.mkdir()
  np.savez_compressed(folder/'geometry.npz',vertices=world,faces=faces,raw_vertices=original,raw_faces=faces,raw_to_canonical_rotation=transform,canonical_raw_vertices=rv,canonical_rear_mount=mount,world_mount=target,uniform_scale=scale,center=center,size=size,rotation=basis,K=cam['K'],w2c_A=cam['w2c_A'],w2c_B=cam['w2c_B'])
  policy='Reviewed actual broad rear minY mounting face on whole LED bar' if axes=='native_Y_front' else 'Reviewed actual rear maxX open wallplate rim, canonicalized by rigid90degree axis rotation; rim contact, not fabricated rear triangles'
  name=op+'_'+ix;jobs[name]=dict(vertices=world,faces=faces,operation=op);models[name]=dict(operation=op,index=ix,raw_SHA256=rawsha,whole_raw_V_F_retained=True,whole_all_nodes_retained=True,raw_to_canonical_rigid_rotation=transform.tolist(),actual_rear_contact_vertices=len(rear),actual_rear_contact_triangles=int(flat.sum()),rear_mount_policy=policy,raw_geometry_uncropped=True,canonical_rear_mount=mount.tolist(),world_mount=target.tolist(),target_mount_A00=pix,height_reference_px_A00=height,uniform_scale=scale,geometry=str(folder/'geometry.npz'),geometry_SHA256=sha(folder/'geometry.npz'),support_contact_visual_review_pending=True,quality_accepted=False)
 cap=cv2.VideoCapture(str(N/'inputs/case83_source_video_HF_v1/source_A.mkv'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);proof=json.loads((I/'INPUT_MANIFEST.json').read_text());rast=Rasterizer();rows=[]
 for i in range(49):
  ok,bgr=cap.read();assert ok and hashlib.sha256(bgr.tobytes()).hexdigest()==proof['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  observed=cv2.imread(str(Path(review['source_mask49'])/f'{i:06d}.png'),0)>0;small=cv2.resize(observed.astype(np.float32),(raw['depth'].shape[2],raw['depth'].shape[1]),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0;scenes={}
  for key,keep in [('full',np.ones_like(small,bool)),('removed',~small)]:
   valid=np.isfinite(conf[i])&(conf[i]>1.)&keep;sv,sf=depth_surface(raw['depth'][i],raw['K'][i],cam['w2c_A'][i],valid_pixels=valid,max_neighbor_ratio=1.15);scenes[key]=rast.depth(sv,sf,cam['K'][i],cam['w2c_A'][i])
  for name,row in jobs.items():
   dep=rast.depth(row['vertices'],row['faces'],cam['K'][i],cam['w2c_A'][i]);hit=np.isfinite(dep);layer=visibility(dep,scenes['full' if row['operation']=='insert' else 'removed']);left=bgr.copy();left[hit]=np.rint(left[hit]*.4+.6*np.array([255,0,255])).astype(np.uint8);right=overlay(bgr,layer)
   for im,label in [(left,'whole model fixed world'),(right,'Native visibility; cyan UNKNOWN')]:cv2.putText(im,f'83 {name} A{i:02d} {label}; NOT editedRGB/GT',(8,21),0,.4,(0,255,255),1)
   path=out/name/f'A{i:02d}_whole_vs_visibility.jpg';cv2.imwrite(str(path),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,96]);rows.append(dict(role=name,frame=i,path=str(path),SHA256=sha(path),pixels=int(hit.sum()),visibility={k:int(a.sum()) for k,a in layer.items()}))
 cap.release()
 result=dict(case='83',state='three_whole_wall_targets_9pilots_pending_actual_review',models=models,actual_pilot_images=rows,whole_nodes_and_material_reference_asset_SHA_exact=True,raw_geometry_uncropped=True,Native_RGB_SAM_depth_K_E_unchanged=True,source_corrected_visible_RGB_mask_separate=True,support_plane=p,quality_accepted=False,GPU_used=False,seconds=time.time()-start)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2));ident.update(state='terminal0_pending_actual9pilots',exit_code=0,seconds=time.time()-start);(out/'JOB.json').write_text(json.dumps(ident,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['actual_pilot_images','support_plane']}))
if __name__=='__main__':main()

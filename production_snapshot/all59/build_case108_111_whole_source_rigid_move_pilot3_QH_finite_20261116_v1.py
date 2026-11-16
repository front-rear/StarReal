from pathlib import Path
import sys,json,hashlib,fcntl,os,time,datetime
N=Path("/tmp/starreal_uid1007_case19_cpu_v1");Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal")
sys.path[:0]=[str(N/"runtime/qh_cpu_py39_v1"),str(Q/"inputs/case108_native_CPU_v1"),str(N/"code")]
import cv2,numpy as np
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,plane_intersection,project,visibility,unproject
from case19_source_shell_reference_QH_v1 import regularized_shell,rotation_on_plane,overlay
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.time()
for case in ["108","111"]:
 guard=open(Q/"logs"/("case"+case+"_geometry_component.lock"),"a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 I=Q/"inputs"/f"case{case}_native_CPU_v1";C=Q/"component_cache"/case/"true4K_RGB_camera_component_v2"
 reviewpath=Q/"reviews"/f"CASE{case}_TRUE4K_OBSERVED_SOURCE_SAM49_CURRENT.json";review=json.loads(reviewpath.read_text());assert review["all49_actually_viewed"] and review["may_compute_private_source_geometry"]
 support=Q/"component_cache"/case/"true4K_actual_support_probe_v1/RESULT.json";p=json.loads(support.read_text());cam=np.load(C/"cameras.npz");raw=np.load(I/"native_depth_K_E_subset.npz");conf=np.load(N/"inputs"/f"CASE{case}_NATIVE_POINT_CONFIDENCE49_v1.npz")["world_points_conf"]
 H=np.asarray(json.loads((C/"INPUT_MANIFEST.json").read_text())["H_old_prepared1280_to_new_Original1280"]);Hi=np.linalg.inv(H)
 rawsha=sha(I/"native_depth_K_E_subset.npz");camsha=sha(C/"cameras.npz");K=cam["K"][0];A=cam["w2c_A"][0]
 out=Q/"component_cache"/case/"whole_observed_source_rigid_move_pilot3_v1";out.mkdir(exist_ok=False)
 if case=="108":
  # Refit only the actual clear black tabletop strip below the fixture; the first broad patch included an open book and is not used.
  h,w=raw["depth"].shape[1:];yy,xx=np.mgrid[:h,:w];world=unproject(np.c_[xx.ravel(),yy.ravel()],raw["depth"][0].ravel(),raw["K"][0],A);uv,z=project(world,K,A)
  keep=np.isfinite(world).all(1)&np.isfinite(conf[0].ravel())&(conf[0].ravel()>1)&(z>0)&(uv[:,0]>=930)&(uv[:,0]<=1005)&(uv[:,1]>=515)&(uv[:,1]<=524)
  pts=world[keep];assert len(pts)>=3;ctr=pts.mean(0);_,_,vv=np.linalg.svd(pts-ctr,full_matrices=False);n=vv[-1]
  if n@(-A[:3,:3].T[:,1])<0:n=-n
  d=float(-n@ctr);plane=dict(normal_world=n.tolist(),d=d,ROI_A00=[930,515,1005,524],eligible_points=len(pts),source="Actual black tabletop strip, open-book patch rejected",not_metric_or_placement_acceptance=True)
  target=p["planes"]["empty_left_table"];targetpix=[210.,474.]
  points={"shade_top":[960,100],"shade_left":[873,208],"shade_right":[1053,208],"shade_lower":[960,214],"candle_top":[960,220],"candle_bottom":[961,285],"upper_joint":[961,316],"stem_bottom":[959,438],"phone_top":[989,352],"phone_side":[1012,372],"phone_bottom":[982,468],"base_left":[916,489],"base_right":[1015,493],"base_center":[969,498]}
  segments=[("shade_top","shade_left",150),("shade_top","shade_right",150),("shade_left","shade_lower",150),("shade_right","shade_lower",150),("candle_top","candle_bottom",18),("candle_bottom","upper_joint",22),("upper_joint","stem_bottom",26),("phone_top","phone_side",45),("phone_side","phone_bottom",45),("stem_bottom","base_center",38),("phone_bottom","base_center",65),("base_left","base_center",110),("base_right","base_center",110)]
 else:
  plane=p["planes"]["observed_original_lantern_bed_contact"];n=np.array(plane["normal_world"]);d=plane["d"];target=p["planes"]["empty_right_bed"];targetpix=[1110.,640.]
  points={"handle_top":[837,540],"handle_left":[786,553],"handle_right":[893,553],"handle_lower":[839,579],"body_top":[837,580],"body_bottom":[837,665],"base_left":[790,675],"base_right":[892,675],"base_center":[841,680]}
  segments=[("handle_top","handle_left",9),("handle_top","handle_right",9),("handle_left","handle_lower",9),("handle_right","handle_lower",9),("body_top","body_bottom",110),("body_bottom","base_center",110),("base_left","base_center",110),("base_right","base_center",110)]
 Ppix=p["source_observed_base_A00"];P=plane_intersection(Ppix,K,A,n,d)
 forward=A[:3,:3].T[:,2];cn=forward-(forward@n)*n;cn/=np.linalg.norm(cn);cd=float(-cn@P)
 anchors={}
 for key,uv in points.items():
  world=plane_intersection(uv,K,A,cn,cd);anchors[key]=np.r_[uv,float(project(world[None],K,A)[1][0])]
 mask=cv2.imread(str(Path(review["source_mask49"])/"000000.png"),0)>0
 v,f,_=regularized_shell(mask,K,A,anchors,segments)
 below=v@n+d<0
 if below.any():
  uv,_=project(v[below],K,A);v[below]=np.array([plane_intersection(a,K,A,n,d) for a in uv])
 tn=np.array(target["normal_world"]);td=float(target["d"]);mount=plane_intersection(targetpix,K,A,tn,td);shift=mount-P;mv=v+shift
 penetration=float(np.min(mv@tn+td));correction=max(0.,-penetration)*tn;mv+=correction;mount+=correction
 rotation=rotation_on_plane(n,A[:3,:3].T);center,size,basis=bbox_for_mesh(v,rotation)
 np.savez_compressed(out/"geometry.npz",source_vertices=v,source_faces=f,source_center=center,source_size=size,source_rotation=basis,source_mount=P,source_unobserved_completion_vertex=np.zeros(len(v),bool),move_vertices=mv,move_faces=f,move_mount=mount,K=cam["K"],w2c_A=cam["w2c_A"],w2c_B=cam["w2c_B"])
 raster=Rasterizer();records=[]
 for i in [0,24,48]:
  rgb=cv2.imread(str(C/"RGB49"/f"{i:06d}.png"));observed=cv2.imread(str(Path(review["source_mask49"])/f"{i:06d}.png"),0)>0
  oldmask=cv2.warpPerspective(observed.astype(np.uint8),Hi,(1280,704),flags=cv2.INTER_NEAREST);small=cv2.resize(oldmask.astype(np.float32),(raw["depth"].shape[2],raw["depth"].shape[1]),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0
  valid=np.isfinite(conf[i])&(conf[i]>1)&~small;sv,sf=depth_surface(raw["depth"][i],raw["K"][i],cam["w2c_A"][i],valid_pixels=valid,max_neighbor_ratio=1.15);scene=raster.depth(sv,sf,cam["K"][i],cam["w2c_A"][i])
  for role,mesh in [("source",v),("rigid_move",mv)]:
   dep=raster.depth(mesh,f,cam["K"][i],cam["w2c_A"][i]);hit=np.isfinite(dep);layers=visibility(dep,scene);left=rgb.copy();left[hit]=np.rint(.4*left[hit]+.6*np.array([255,0,255])).astype(np.uint8);right=overlay(rgb,layers)
   contours,_=cv2.findContours(hit.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(right,contours,-1,(0,255,255),1)
   for im,label in [(left,"whole inferred3D"),(right,"scene visibility estimate")]:cv2.putText(im,f"{case} {role} A{i:02d} {label}; NOT edited RGB or observedB GT",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
   dst=out/f"{role}_whole_vs_visibility_A{i:02d}.jpg";cv2.imwrite(str(dst),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,96]);records.append(dict(role=role,frame=i,path=str(dst),SHA256=sha(dst),amodal_pixels=int(hit.sum()),visibility={k:int(a.sum()) for k,a in layers.items()}))
 assert rawsha==sha(I/"native_depth_K_E_subset.npz") and camsha==sha(C/"cameras.npz")
 result=dict(case=case,state="terminal0_whole_observed_source_rigid_move_pilot3_pending_actual_review",source_review=str(reviewpath),source_review_SHA256=sha(reviewpath),source_plane=plane,target_plane=target,source_contact_candidate_A00=Ppix,target_candidate_A00=targetpix,target_ground_correction_world=correction.tolist(),source_vertices=len(v),source_faces=len(f),geometry=str(out/"geometry.npz"),geometry_SHA256=sha(out/"geometry.npz"),all_original_RGB_depth_K_E_unchanged=True,derived_full_K_crop_H_retained=True,native_source_cell_inverse_H_applied=True,source_front_plane_back_and_thickness_are_estimates=True,pure_move_size_orientation_appearance_preserved=True,source_observed_shape_not_amodally_painted=True,actual_pilots=records,GPU_used=False,quality_accepted=False,seconds=time.time()-start)
 (out/"RESULT.json").write_text(json.dumps(result,indent=2));print(json.dumps(dict(case=case,result=str(out/"RESULT.json"),seconds=time.time()-start)),flush=True);guard.close()

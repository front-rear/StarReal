from pathlib import Path
import sys,json,hashlib,fcntl,os,datetime,time
N=Path("/tmp/starreal_uid1007_case19_cpu_v1");Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal")
sys.path[:0]=[str(N/"runtime/qh_cpu_py39_v1"),str(Q/"inputs/case108_native_CPU_v1")]
import cv2,numpy as np
from PIL import Image
from real_geometry import unproject,project,plane_intersection
cv2.setNumThreads(1);start=time.time()
for case in ["108","111"]:
 guard=open(Q/"logs"/("case"+case+"_geometry_component.lock"),"a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 reviewpath=Q/"reviews"/f"CASE{case}_TRUE4K_OBSERVED_SOURCE_SAM49_CURRENT.json";review=json.loads(reviewpath.read_text());assert review["may_compute_private_source_geometry"] and review["all49_actually_viewed"]
 I=Q/"inputs"/f"case{case}_native_CPU_v1";C=Q/"component_cache"/case/"true4K_RGB_camera_component_v2"
 root=Q/"component_cache"/case/"true4K_actual_support_probe_v1";root.mkdir(exist_ok=False)
 cam=np.load(C/"cameras.npz");raw=np.load(I/"native_depth_K_E_subset.npz");conf=np.load(N/"inputs"/f"CASE{case}_NATIVE_POINT_CONFIDENCE49_v1.npz")["world_points_conf"]
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 rawsha=sha(I/"native_depth_K_E_subset.npz");camsha=sha(C/"cameras.npz")
 h,w=raw["depth"].shape[1:];yy,xx=np.mgrid[:h,:w];uv=np.c_[xx.ravel(),yy.ravel()];A=cam["w2c_A"][0];K=cam["K"][0]
 world=unproject(uv,raw["depth"][0].ravel(),raw["K"][0],A);image,z=project(world,K,A)
 valid=np.isfinite(world).all(1)&np.isfinite(conf[0].ravel())&(conf[0].ravel()>1)&(z>0)
 if case=="108":
  rois={"original_right_nightstand":[1020,495,1130,534],"empty_left_table":[65,454,360,495]}
  candidate_pixels={"new_left_table_center":[210,474],"prior_left_table_edge":[330,500]}
 else:
  rois={"observed_original_lantern_bed_contact":[775,678,970,702],"empty_right_bed":[1020,595,1240,685]}
  candidate_pixels={"new_right_bed_clear":[1110,640],"prior_left_headboard_location":[120,400]}
 m=np.asarray(Image.open(Path(review["source_mask49"])/"000000.png"))>0;my,mx=np.where(m)
 maxy=int(my.max());baseX=float(np.median(mx[my>=maxy-2]));basepix=[baseX,float(maxy)]
 rng=np.random.default_rng(int(case));planes={};rgb=cv2.imread(str(C/"RGB49/000000.png"));assert rgb.shape==(704,1280,3)
 for name,(x0,y0,x1,y1) in rois.items():
  hit=valid&(image[:,0]>=x0)&(image[:,0]<=x1)&(image[:,1]>=y0)&(image[:,1]<=y1);pts=world[hit];assert len(pts)>=3
  tol=float(np.median(raw["depth"][0].ravel()[hit])*.01);best=None
  for _ in range(240):
   a,b,c=pts[rng.choice(len(pts),3,replace=False)];normal=np.cross(b-a,c-a);length=np.linalg.norm(normal)
   if length<1e-12:continue
   normal/=length;d=-normal@a;keep=np.abs(pts@normal+d)<tol
   if best is None or keep.sum()>best[0]:best=(int(keep.sum()),keep)
  fit=pts[best[1]];center=fit.mean(0);_,_,vv=np.linalg.svd(fit-center,full_matrices=False);normal=vv[-1]
  if normal@(-A[:3,:3].T[:,1])<0:normal=-normal
  dist=float(-normal@center);res=np.abs(pts@normal+dist)
  planes[name]=dict(normal_world=normal.tolist(),d=dist,ROI_A00=[x0,y0,x1,y1],eligible_points=len(pts),fit_inliers=best[0],residual_all_p50=float(np.median(res)),residual_all_p90=float(np.percentile(res,90)),fitting_tolerance_not_acceptance_gate=tol,not_metric_or_placement_acceptance=True)
  cv2.rectangle(rgb,(x0,y0),(x1,y1),(0,255,0),1);cv2.putText(rgb,name,(x0,y0-8),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
 plan=json.loads((I/"HISTORICAL_GEOMETRY_PLAN.json").read_text());old=np.array(plan["support"]["mount_world"]);oldUV,_=project(old[None],K,A);candidate_pixels["actual_prior_world_mount_A00"]=oldUV[0].tolist();candidate_pixels["observed_source_contact"]=basepix
 for name,point in candidate_pixels.items():
  uvp=tuple(np.rint(point).astype(int));cv2.circle(rgb,uvp,6,(0,0,255),2);cv2.putText(rgb,name,(uvp[0]+7,uvp[1]-8),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1)
 cv2.putText(rgb,f"{case} actual RGB support ROI/candidate probe ONLY; neither geometry nor placement accepted",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
 cv2.imwrite(str(root/"ACTUAL_SUPPORT_ROIS_AND_DESTINATIONS_A00.jpg"),rgb,[cv2.IMWRITE_JPEG_QUALITY,96])
 assert rawsha==sha(I/"native_depth_K_E_subset.npz") and camsha==sha(C/"cameras.npz")
 result=dict(case=case,state="terminal0_actual_support_ROI_plane_candidates_pending_visual_geometry_review",planes=planes,source_observed_base_A00=basepix,candidate_pixels=candidate_pixels,old_target_world=old.tolist(),source_review=str(reviewpath),source_review_SHA256=sha(reviewpath),raw_depth_K_E_unchanged=True,derived_full_K_crop_H_verified=True,no_new_depth_inference=True,relative_monocular_units=True,quality_accepted=False,GPU_used=False,seconds=time.time()-start)
 (root/"RESULT.json").write_text(json.dumps(result,indent=2));print(json.dumps(dict(case=case,path=str(root/"RESULT.json"),eligible={k:v["eligible_points"] for k,v in planes.items()})),flush=True)
 guard.close()

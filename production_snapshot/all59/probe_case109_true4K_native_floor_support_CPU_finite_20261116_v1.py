from pathlib import Path
import sys,json,hashlib,fcntl,os,time,datetime
N=Path("/tmp/starreal_uid1007_case19_cpu_v1");Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal")
I=N/"inputs/case109_native_CPU_v1";C=Q/"component_cache/109/true4K_RGB_camera_component_v2"
sys.path[:0]=[str(N/"runtime/qh_cpu_py39_v1"),str(I),str(N/"code")]
import numpy as np,cv2
from real_geometry import unproject,project,plane_intersection
guard=open(Q/"logs/case109_geometry_component.lock","a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
review=json.loads((Q/"reviews/CASE109_TRUE4K_OBSERVED_SOURCE_SAM49_CURRENT.json").read_text());assert review["may_compute_private_source_geometry"]
root=Q/"component_cache/109/true4K_native_actual_support_probe_v1";root.mkdir(exist_ok=False)
cam=np.load(C/"cameras.npz");raw=np.load(I/"native_depth_K_E_subset.npz");conf=np.load(N/"inputs/CASE109_NATIVE_POINT_CONFIDENCE49_v1.npz")["world_points_conf"]
h,w=raw["depth"].shape[1:];yy,xx=np.mgrid[:h,:w];uv=np.c_[xx.ravel(),yy.ravel()]
A=cam["w2c_A"][0];K=cam["K"][0];world=unproject(uv,raw["depth"][0].ravel(),raw["K"][0],A);image,z=project(world,K,A)
valid=np.isfinite(world).all(1)&np.isfinite(conf[0].ravel())&(conf[0].ravel()>1.)&(z>0)
rng=np.random.default_rng(109);planes={};rois={"clear_left_floor":[100,650,410,703],"clear_right_floor":[1030,650,1260,703]}
for name,(x0,y0,x1,y1) in rois.items():
 hit=valid&(image[:,0]>=x0)&(image[:,0]<=x1)&(image[:,1]>=y0)&(image[:,1]<=y1)
 p=world[hit];assert len(p)>=3
 tol=float(np.median(raw["depth"][0].ravel()[hit])*.01);best=None
 for _ in range(240):
  a,b,c=p[rng.choice(len(p),3,replace=False)];n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
  if norm<1e-12:continue
  n/=norm;d=-n@a;res=np.abs(p@n+d);keep=res<tol
  if best is None or int(keep.sum())>best[0]:best=(int(keep.sum()),keep,n,d)
 keep=best[1];center=p[keep].mean(0);_,_,vv=np.linalg.svd(p[keep]-center,full_matrices=False);n=vv[-1];up=-A[:3,:3].T[:,1]
 if n@up<0:n=-n
 d=float(-n@center);res=np.abs(p@n+d)
 P=plane_intersection([488.,655.],K,A,n,d)
 planes[name]=dict(normal_world=n.tolist(),d=d,ROI_A00=[x0,y0,x1,y1],eligible_points=int(len(p)),fit_inliers=int(best[0]),fit_distance_tolerance_is_estimation_not_acceptance=tol,residual_all_p50=float(np.median(res)),residual_all_p90=float(np.percentile(res,90)),source_observed_base_A00=[488.,655.],source_base_world_estimate=P.tolist(),not_metric_or_geometry_acceptance=True)
rgb=cv2.imread(str(C/"RGB49/000000.png"));assert rgb.shape==(704,1280,3)
for name,(x0,y0,x1,y1) in rois.items():cv2.rectangle(rgb,(x0,y0),(x1,y1),(0,255,0),1);cv2.putText(rgb,name,(x0,y0-8),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
anchors={"shade_top":[613,52],"shade_left":[600,84],"shade_right":[655,76],"shade_lower":[625,108],"head_joint":[584,76],"elbow":[488,166],"lower_joint":[486,305],"stem_lower":[487,615],"base_center":[488,642],"base_left":[426,650],"base_right":[550,650]}
for name,p in anchors.items():cv2.circle(rgb,tuple(p),3,(0,0,255),-1)
for name,p in {"original_base":[488,655],"old_good_left400640":[400,640],"alternate_left300650":[300,650],"far_right_clear1140665":[1140,665]}.items():cv2.circle(rgb,tuple(p),5,(0,255,255),1);cv2.putText(rgb,name,(int(p[0])+6,int(p[1])-6),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1)
cv2.putText(rgb,"109 true4K actual RGB / floor fit ROIs + source anchors / candidate supports; NOT edited RGB or approval",(8,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1);cv2.imwrite(str(root/"actual_RGB_support_ROIs_and_anchor_candidates_A00.jpg"),rgb,[cv2.IMWRITE_JPEG_QUALITY,95])
r=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),case="109",state="private_true4K_image_K_native_depth_floor_support_probe_pending_actual_review",planes=planes,source_anchors={k:dict(pixel_in_reference=v) for k,v in anchors.items()},source_original_SAM_promotion_review=review["result"],RGB_source=str(C/"RGB49/000000.png"),RGB_source_SHA256=sha(C/"RGB49/000000.png"),new_full_K_SHA256=sha(C/"cameras.npz"),original_raw_depth_K_E_preserved=True,no_new_Native_inference=True,full_K_homography_derived=True,observed_base_contact_has_RGB_support=True,all_world_depth_is_monocular_estimate=True,quality_accepted=False,GPU_used=False)
(root/"RESULT.json").write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(dict(path=str(root/"RESULT.json"),planes=planes)))

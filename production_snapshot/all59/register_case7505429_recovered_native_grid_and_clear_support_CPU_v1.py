from pathlib import Path
import os,sys,json,time,hashlib,datetime,fcntl
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OPENBLAS_NUM_THREADS"]="1";os.environ["OMP_NUM_THREADS"]="2"
Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");N=Path("/tmp/starreal_uid1007_case19_cpu_v1")
sys.path[:0]=[str(N/"runtime/qh_cpu_py39_v1"),str(Q/"inputs/case111_native_CPU_v1")]
import numpy as np,cv2
from real_geometry import unproject,project,plane_intersection
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
C=Q/"inputs/original_native_readonly_SFTP_7505429_EXTRA640_CACHE_ONLY_stream_v3/7505429";I=Q/"source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1/prepared49_from_actual_Original_crop_v1/7505429"
guard=open(Q/"logs/case7505429_native_registration.lock","a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=Q/"component_cache/7505429/recovered_native_grid_and_clear_support_CPU_v1";out.mkdir(exist_ok=False)
start=time.time();assert sha(C/"vggt_raw.npz")=="1765615e5eb3d6bf2cf7c3d2042dad27b73b7a2e1d1d0e93c3503e6aa77d7321"
audit=Q/"component_cache/7505429/recovered_extra640_Native_INPUT_AUDIT_CPU_v1/RESULT.json";ad=json.loads(audit.read_text());assert ad["all_best_temporal_indices_identity"] and ad["Native_input_unique_RGB"]==49 and ad["true4K_input_unique_RGB"]==49
raw=np.load(C/"vggt_raw.npz");gh,gw=raw["depth"].shape[1:];assert (gh,gw)==(280,518)
producer=Q/"all59_reconstruction_20261115_v1/provenance/CASE7505429_NATIVE_PREPROCESS_AND_CAMERA_PRODUCER_READONLY_20261116_v1.json";pr=json.loads(producer.read_text());loadtext=next(x["text"] for x in pr["rows"] if x["source_path"].endswith("load_fn.py"))
assert 'new_height = round(height * (new_width / width) / 14) * 14' in loadtext
assert round(704*(518/1280)/14)*14==280
K=raw["K"].astype(np.float64).copy();K[:,0]*=1280/gw;K[:,1]*=704/gh
A=np.broadcast_to(np.eye(4),(49,4,4)).copy();A[:,:3]=raw["extrinsic_3x4"]
np.savez_compressed(out/"native_depth_K_E_subset.npz",**{k:raw[k] for k in ["depth","K","extrinsic_3x4"]})
np.savez_compressed(out/"world_points_conf.npz",world_points_conf=raw["world_points_conf"])
np.savez_compressed(out/"K_and_actual_Native_A_only.npz",K=K,w2c_A=A)
registration=[];orb=cv2.ORB_create(4500)
for fi in range(49):
 old=cv2.imread(str(C/"frames"/f"{fi:03d}.png"));new=cv2.imread(str(I/"RGB49"/f"{fi:06d}.png"));assert old.shape==new.shape==(704,1280,3)
 ka,da=orb.detectAndCompute(cv2.cvtColor(old,cv2.COLOR_BGR2GRAY),None);kb,db=orb.detectAndCompute(cv2.cvtColor(new,cv2.COLOR_BGR2GRAY),None);matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2);good=[m for m,n in matches if m.distance<.75*n.distance]
 pa=np.array([ka[m.queryIdx].pt for m in good],np.float64);pb=np.array([kb[m.trainIdx].pt for m in good],np.float64);H,inl=cv2.findHomography(pa,pb,cv2.RANSAC,3.0);assert H is not None
 inl=inl.ravel()>0;delta=np.linalg.norm(pa[inl]-pb[inl],axis=1);corn=np.array([[0,0],[1279,0],[0,703],[1279,703],[640,352]],np.float64);moved=cv2.perspectiveTransform(corn[None],H)[0]
 registration.append(dict(frame=fi,old_prepared_SHA256=sha(C/"frames"/f"{fi:03d}.png"),new_true4K_SHA256=sha(I/"RGB49"/f"{fi:06d}.png"),ORB_inlier_matches=int(inl.sum()),median_correspondence_distance_pixel=float(np.median(delta)),estimated_H=H.tolist(),H_corner_displacement_pixels=np.linalg.norm(moved-corn,axis=1).tolist(),diagnostic_only_not_replacing_analytical_pixel_grid=True))
yy,xx=np.indices((gh,gw));world=unproject(np.c_[xx.ravel(),yy.ravel()],raw["depth"][0].ravel(),raw["K"][0],A[0]);uv,z=project(world,K[0],A[0]);conf=raw["world_points_conf"][0].ravel()
image=cv2.imread(str(I/"RGB49/000000.png"));planes={}
for name,roi,pix,col in [("observed_lantern_bed_contact",[380,414,470,441],[431,414],(0,255,0)),("clear_left_bed_away_book_and_people",[165,465,355,650],[285,590],(255,150,0))]:
 x0,y0,x1,y1=roi;ok=np.isfinite(world).all(1)&np.isfinite(conf)&(conf>1)&(z>0)&(uv[:,0]>=x0)&(uv[:,0]<=x1)&(uv[:,1]>=y0)&(uv[:,1]<=y1);pts=world[ok];assert len(pts)>=3
 ctr=pts.mean(0);_,_,vv=np.linalg.svd(pts-ctr,full_matrices=False);n=vv[-1]
 if n@(-A[0,:3,:3].T[:,1])<0:n=-n
 d=float(-n@ctr);res=np.abs(pts@n+d);mount=plane_intersection(pix,K[0],A[0],n,d)
 planes[name]=dict(ROI_A00=roi,candidate_pixel_A00=pix,normal_world=n.tolist(),d=d,eligible_points=len(pts),absolute_residual_median=float(np.median(res)),absolute_residual_max=float(np.max(res)),mount_world=mount.tolist(),world_unit="relative monocular, metric scale unknown",support_is_estimate_not_scene_ground_truth=True,placement_accepted=False)
 cv2.rectangle(image,(x0,y0),(x1,y1),col,2);cv2.circle(image,tuple(pix),5,col,2)
 for p in uv[ok][::5].astype(int):cv2.circle(image,tuple(p),1,col,-1)
 cv2.putText(image,name,(x0,min(695,y1+18)),cv2.FONT_HERSHEY_SIMPLEX,.42,col,1)
cv2.putText(image,"7505429 actual true4K RGB; Native depth support candidates ONLY, not edited RGB or final placement approval",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
cv2.imwrite(str(out/"ACTUAL_CLEAR_SUPPORT_CANDIDATES_A00.jpg"),image,[cv2.IMWRITE_JPEG_QUALITY,95])
result=dict(case="7505429",at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state="Native_grid_and_true4K_RGB49_registration_support_candidates_pending_actual_review",source_cache=str(C),raw_native_SHA256=sha(C/"vggt_raw.npz"),Native_input_true4K_RGB_byte_identical=False,Native_input_is_historical_upscaled640=True,final_training_RGB_is_actual_Original4K_crop=True,actual_temporal_indices_identity_all49=True,Native_preprocess_code_provenance=str(producer),Native_preprocess_code_provenance_SHA256=sha(producer),Native_preprocessing_actual_hw=[280,518],analytical_raw_grid_to_prepared1280_scale=[1280/gw,704/gh],model_preprocess_real_anisotropic_rounding_preserved=True,H_old_prepared1280_to_new_Original1280=np.eye(3).tolist(),K_derivation="Kfull rows x1280/518 and y704/280; actual Original/native crop grid same; new depth inference not run",actual49_ORB_registration=registration,planes=planes,actual_A_world_poses_from_matching_cache=True,novel_B_not_yet_derived=True,old_wrong_timing_cache_not_used=True,metric_geometry_not_claimed=True,source_SAM_separate_CPU_actual4K=True,files=[dict(path=p.name,bytes=p.stat().st_size,SHA256=sha(p)) for p in out.iterdir() if p.is_file()],GPU_used=False,quality_accepted=False,new_complete_groups=0,seconds=time.time()-start)
(out/"RESULT.json").write_text(json.dumps(result,indent=2));print(json.dumps(dict(state=result["state"],planes=planes,seconds=time.time()-start,median_registration49=float(np.median([x["median_correspondence_distance_pixel"] for x in registration])))),flush=True)

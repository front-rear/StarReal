from pathlib import Path
import sys,json,hashlib,fcntl,os,time,datetime
N=Path("/tmp/starreal_uid1007_case19_cpu_v1");Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal")
I=N/"inputs/case109_native_CPU_v1";C=Q/"component_cache/109/true4K_RGB_camera_component_v2"
sys.path[:0]=[str(N/"runtime/qh_cpu_py39_v1"),str(I),str(N/"code")]
import numpy as np,cv2
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,plane_intersection,project,visibility
from case19_source_shell_reference_QH_v1 import regularized_shell,rotation_on_plane,overlay
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
guard=open(Q/"logs/case109_geometry_component.lock","a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);cv2.setNumThreads(1)
reviewpath=Q/"reviews/CASE109_TRUE4K_OBSERVED_SOURCE_SAM49_CURRENT.json";review=json.loads(reviewpath.read_text());assert review["may_compute_private_source_geometry"] and review["all49_actually_viewed"]
support=Q/"component_cache/109/true4K_native_actual_support_probe_v1/RESULT.json";p=json.loads(support.read_text());cam=np.load(C/"cameras.npz");raw=np.load(I/"native_depth_K_E_subset.npz");H=np.array(json.loads((C/"INPUT_MANIFEST.json").read_text())["H_old_prepared1280_to_new_Original1280"]);Hinv=np.linalg.inv(H)
rawsha=sha(I/"native_depth_K_E_subset.npz");camsha=sha(C/"cameras.npz")
out=Q/"component_cache/109/source_true4K_visible_floor_rigid_old_good_move_pilot3_v1";out.mkdir(exist_ok=False);t=time.time()
ident=dict(PID=os.getpid(),uid=os.getuid(),boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),startticks=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],case="109",GPU_used=False,quality_accepted=False)
(out/"JOB.json").write_text(json.dumps(dict(ident,state="finite_source_and_pure_move3"),indent=2))
K,A=cam["K"][0],cam["w2c_A"][0];plane=p["planes"]["clear_left_floor"];n=np.array(plane["normal_world"]);d=plane["d"];Ppix=[488.,655.];P=plane_intersection(Ppix,K,A,n,d)
forward=A[:3,:3].T[:,2];cn=forward-(forward@n)*n;cn/=np.linalg.norm(cn);cd=float(-cn@P)
anchors={}
for key,row in p["source_anchors"].items():
 uv=row["pixel_in_reference"];world=plane_intersection(uv,K,A,cn,cd);anchors[key]=np.r_[uv,float(project(world[None],K,A)[1][0])]
segments=[("shade_top","shade_left",24),("shade_top","shade_right",24),("shade_left","shade_lower",24),("shade_right","shade_lower",24),("head_joint","shade_top",14),("head_joint","elbow",6),("elbow","lower_joint",4),("lower_joint","stem_lower",4),("stem_lower","base_center",8),("base_left","base_center",13),("base_right","base_center",13)]
mask=cv2.imread(str(Path(review["source_mask49"])/"000000.png"),0)>0
v,f,_=regularized_shell(mask,K,A,anchors,segments)
below=v@n+d<0
if below.any():
 uv,_=project(v[below],K,A);v[below]=np.array([plane_intersection(q,K,A,n,d) for q in uv])
rot=rotation_on_plane(n,A[:3,:3].T);center,size,basis=bbox_for_mesh(v,rot);targetpix=[400.,640.];mount=plane_intersection(targetpix,K,A,n,d);mv=v+mount-P
z=dict(source_vertices=v,source_faces=f,source_center=center,source_size=size,source_rotation=basis,source_mount=P,source_unobserved_completion_vertex=np.zeros(len(v),bool),move_vertices=mv,move_faces=f,move_mount=mount,K=cam["K"],w2c_A=cam["w2c_A"],w2c_B=cam["w2c_B"])
np.savez_compressed(out/"geometry.npz",**z)
conf=np.load(N/"inputs/CASE109_NATIVE_POINT_CONFIDENCE49_v1.npz")["world_points_conf"];rast=Rasterizer();records=[]
for i in [0,24,48]:
 rgb=cv2.imread(str(C/"RGB49"/f"{i:06d}.png"));observed=cv2.imread(str(Path(review["source_mask49"])/f"{i:06d}.png"),0)>0
 old_image_mask=cv2.warpPerspective(observed.astype(np.uint8),Hinv,(1280,704),flags=cv2.INTER_NEAREST)
 small=cv2.resize(old_image_mask.astype(np.float32),(raw["depth"].shape[2],raw["depth"].shape[1]),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0
 valid=np.isfinite(conf[i])&(conf[i]>1.)&~small
 sv,sf=depth_surface(raw["depth"][i],raw["K"][i],cam["w2c_A"][i],valid_pixels=valid,max_neighbor_ratio=1.15);scene=rast.depth(sv,sf,cam["K"][i],cam["w2c_A"][i])
 for name,mesh in [("source",v),("move_old_good_left400640",mv)]:
  dep=rast.depth(mesh,f,cam["K"][i],cam["w2c_A"][i]);hit=np.isfinite(dep);layers=visibility(dep,scene);left=rgb.copy();left[hit]=np.rint(.4*left[hit]+.6*np.array([255,0,255])).astype(np.uint8);right=overlay(rgb,layers);contours,_=cv2.findContours(hit.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(right,contours,-1,(0,255,255),1)
  for im,label in [(left,"whole inferred geometry"),(right,"visibility estimate cyan unknown")]:cv2.putText(im,f"109 {name} A{i:02d}: {label}; NOT edited RGB",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
  dst=out/f"{name}_whole_vs_visibility_A{i:02d}.jpg";cv2.imwrite(str(dst),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,96]);records.append(dict(role=name,frame=i,path=str(dst),SHA256=sha(dst),amodal_pixels=int(hit.sum()),visibility={k:int(a.sum()) for k,a in layers.items()}))
assert sha(I/"native_depth_K_E_subset.npz")==rawsha and sha(C/"cameras.npz")==camsha
r=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),case="109",state="true4K_source_and_whole_rigid_move_pilot3_pending_actual_review",source_review=str(reviewpath),source_review_SHA256=sha(reviewpath),support_probe=str(support),source_mount_A00=Ppix,target_mount_A00=targetpix,old_good_clear_floor_location_preserved=True,pure_move_size_orientation_appearance_preserved=True,source_observed_mask_not_amodally_completed=True,source_back_and_thickness_estimated=True,source_front_depth_plane_estimated=True,ground_support_monocular_estimate=True,full_K_homography_derived=True,old_native_depth_K_E_unchanged=True,mask_to_native_source_cell_inverse_H_applied=True,new_depth_inference=False,geometry=str(out/"geometry.npz"),geometry_SHA256=sha(out/"geometry.npz"),geometry_vertices=len(v),geometry_faces=len(f),source_support_plane=plane,anchors={k:v.tolist() for k,v in anchors.items()},actual_pilots=records,GPU_used=False,quality_accepted=False,formal_group_increment=0,seconds=time.time()-t)
(out/"RESULT.json").write_text(json.dumps(r,ensure_ascii=False,indent=2));(out/"JOB.json").write_text(json.dumps(dict(ident,state="terminal0_pending_actual_review",exit_code=0,seconds=time.time()-t),indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ["actual_pilots","anchors","source_support_plane"]}))

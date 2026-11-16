from pathlib import Path
import datetime,fcntl,hashlib,json,os,sys,time
N=Path("/tmp/starreal_uid1007_case19_cpu_v1");Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");I=N/"inputs/case106_native_CPU_v1"
sys.path[:0]=[str(N/"runtime/qh_cpu_py39_v1"),str(I),str(N/"code")]
import cv2,numpy as np
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,plane_intersection,project,visibility
from case19_source_shell_reference_QH_v1 import regularized_shell,rotation_on_plane,overlay
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=(Q/"logs/case106_geometry_component.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cv2.setNumThreads(1);start=time.time()
 out=Q/"component_cache/106/source_whole_rigid_inset_table_pilot3_v1";out.mkdir(exist_ok=False)
 ident=dict(case="106",pid=os.getpid(),uid=os.getuid(),boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),startticks=Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19],GPU_used=False,quality_accepted=False)
 (out/"JOB.json").write_text(json.dumps(dict(ident,state="running_finite_pilot3"),indent=2))
 rawsha=sha(I/"native_depth_K_E_subset.npz");camsha=sha(I/"cameras.npz");source=N/"inputs/case106_source_video_HF_v1/source_A.mkv";sourcesha=sha(source)
 cams=np.load(I/"cameras.npz");raw=np.load(I/"native_depth_K_E_subset.npz");K,A=cams["K"][0],cams["w2c_A"][0];confidence=np.load(N/"inputs/CASE106_NATIVE_POINT_CONFIDENCE49_v1.npz")["world_points_conf"]
 reviewpath=Q/"reviews/CASE106_OBSERVED_SOURCE_SAM49_CURRENT.json";review=json.loads(reviewpath.read_text());assert review["may_compute_private_source_geometry"] and review["reference_frame"]==0
 probe=json.loads((Q/"component_cache/106/actual_tabletop_contact_probe_v1/RESULT.json").read_text());plane=probe["planes"]["clear_right_strip"];n=np.array(plane["normal_world"]);d=plane["d"];Ppix=[967.,547.];Qpix=[1070.,541.];P=plane_intersection(Ppix,K,A,n,d);mount=plane_intersection(Qpix,K,A,n,d)
 forward=A[:3,:3].T[:,2];cn=forward-(forward@n)*n;cn/=np.linalg.norm(cn);cd=-cn@P
 points=dict(finial_top=[961,129],shade_top=[962,252],shade_left=[849,355],shade_right=[1080,355],shade_lower=[963,374],candle_left=[919,389],candle_center=[965,390],candle_right=[1018,389],arm_joint=[964,477],stem_bottom=[966,516],base_left=[923,536],base_right=[1010,536],base_center=Ppix)
 anchors={}
 for key,pix in points.items():
  w=plane_intersection(pix,K,A,cn,cd);anchors[key]=np.r_[pix,float(project(w[None],K,A)[1][0])]
 segments=[("finial_top","shade_top",8),("shade_top","shade_left",110),("shade_top","shade_right",110),("shade_left","shade_lower",110),("shade_right","shade_lower",110),("candle_left","arm_joint",16),("candle_center","arm_joint",16),("candle_right","arm_joint",16),("arm_joint","stem_bottom",18),("stem_bottom","base_center",28),("base_left","base_center",80),("base_right","base_center",80)]
 mask=cv2.imread(str(Path(review["source_mask49"])/"000000.png"),0)>0;v,f,_=regularized_shell(mask,K,A,anchors,segments)
 # Only lower numerical support correction along the original camera ray. No unseen foot/arm painting.
 below=v@n+d<0
 if below.any():
  uv,_=project(v[below],K,A);v[below]=np.array([plane_intersection(x,K,A,n,d) for x in uv])
 rot=rotation_on_plane(n,A[:3,:3].T);center,size,basis=bbox_for_mesh(v,rot);move=v+mount-P
 z=dict(source_vertices=v,source_faces=f,source_center=center,source_size=size,source_rotation=basis,source_mount=P,source_unobserved_completion_vertex=np.zeros(len(v),bool),move_vertices=move,move_faces=f,move_mount=mount,K=cams["K"],w2c_A=cams["w2c_A"],w2c_B=cams["w2c_B"]);np.savez_compressed(out/"geometry.npz",**z)
 cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);inputproof=json.loads((I/"INPUT_MANIFEST.json").read_text());raster=Rasterizer();rows=[]
 for i in range(49):
  ok,rgb=cap.read();assert ok and hashlib.sha256(rgb.tobytes()).hexdigest()==inputproof["RGB49_BGR_array_SHA256"][i]
  if i not in [0,24,48]:continue
  obs=cv2.imread(str(Path(review["source_mask49"])/f"{i:06d}.png"),0)>0;small=cv2.resize(obs.astype(np.float32),(raw["depth"].shape[2],raw["depth"].shape[1]),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0
  valid=np.isfinite(confidence[i])&(confidence[i]>1)&~small;sv,sf=depth_surface(raw["depth"][i],raw["K"][i],cams["w2c_A"][i],valid_pixels=valid,max_neighbor_ratio=1.15);scene=raster.depth(sv,sf,cams["K"][i],cams["w2c_A"][i])
  for role,verts in [("source",v),("rigid_move_inset",move)]:
   dep=raster.depth(verts,f,cams["K"][i],cams["w2c_A"][i]);hit=np.isfinite(dep);layers=visibility(dep,scene);left=rgb.copy();left[hit]=np.rint(.4*left[hit]+.6*np.array([255,0,255])).astype(np.uint8);right=overlay(rgb,layers)
   for im,label in [(left,"whole fixed geometry"),(right,"Native visibility/cyan unknown")]:cv2.putText(im,f"Case106 {role} A{i:02d}: {label}; NOT editedRGB",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
   p=out/f"{role}_whole_vs_visibility_A{i:02d}.jpg";cv2.imwrite(str(p),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,96]);rows.append(dict(role=role,frame=i,path=str(p),SHA256=sha(p),amodal_pixels=int(hit.sum()),visibility={k:int(a.sum()) for k,a in layers.items()}))
 cap.release();assert rawsha==sha(I/"native_depth_K_E_subset.npz") and camsha==sha(I/"cameras.npz") and sourcesha==sha(source)
 result=dict(case="106",state="source_whole_rigid_inset_table_pilot3_pending_actual_review",source_review=str(reviewpath),source_review_SHA256=sha(reviewpath),source_plane=plane,target_plane=plane,source_contact_candidate_A00=Ppix,target_candidate_A00=Qpix,old_target_near_front_lip_A00=[1095,548],geometry=str(out/"geometry.npz"),geometry_SHA256=sha(out/"geometry.npz"),source_anchors={k:v.tolist() for k,v in anchors.items()},source_front_plane_back_thickness_are_estimates=True,source_mask_no_amodal_front_painting=True,source_reference_clean_A00=True,pure_move_size_orientation_appearance_unchanged=True,source_lower_completion_added=False,source_shade_completion_added=False,Native_RGB49_depth_K_E_unchanged=True,actual_pilots=rows,quality_accepted=False,GPU_used=False,elapsed_seconds=time.time()-start)
 (out/"RESULT.json").write_text(json.dumps(result,indent=2));(out/"JOB.json").write_text(json.dumps(dict(ident,state="terminal0_pending_actual_review",exit_code=0,elapsed_seconds=time.time()-start),indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ["actual_pilots","source_anchors"]}),flush=True)
if __name__=="__main__":main()

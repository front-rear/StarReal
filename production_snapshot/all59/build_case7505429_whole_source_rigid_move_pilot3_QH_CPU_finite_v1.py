from pathlib import Path
import os,sys,json,hashlib,fcntl,time,datetime,ast,copy,traceback
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OPENBLAS_NUM_THREADS"]="1"
Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");N=Path("/tmp/starreal_uid1007_case19_cpu_v1")
sys.path[:0]=[str(N/"runtime/qh_cpu_py39_v1"),str(Q/"inputs/case111_native_CPU_v1"),str(N/"code")]
import numpy as np,cv2
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,plane_intersection,project,visibility
from case19_source_shell_reference_QH_v1 import regularized_shell,rotation_on_plane,overlay
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.time()
lock=open(Q/"logs/case7505429_geometry_component.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
parent=Q/"component_cache/7505429";I=parent/"recovered_native_grid_and_clear_support_CPU_v1";D=json.loads((I/"RESULT.json").read_text());raw=np.load(I/"native_depth_K_E_subset.npz");base=np.load(I/"K_and_actual_Native_A_only.npz");K=base["K"];A=base["w2c_A"]
reviewpath=Q/"reviews/CASE7505429_TRUE4K_OBSERVED_SOURCE_SAM49_CURRENT.json";rv=json.loads(reviewpath.read_text());assert rv["all49_actually_viewed"] and rv["may_compute_private_source_geometry"]
S=Q/"source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1/prepared49_from_actual_Original_crop_v1/7505429"
out=parent/"whole_observed_source_rigid_move_pilot3_v1";out.mkdir(exist_ok=False)
job=dict(PID=os.getpid(),uid=os.getuid(),boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),startticks=Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19],state="CPU_geometry_pilot3_running",GPU_used=False)
def put(p,d):
 t=p.with_suffix(".tmp");t.write_text(json.dumps(d,indent=2));t.replace(p)
put(out/"JOB.json",job)
try:
 p=D["planes"]["observed_lantern_bed_contact"];t=D["planes"]["clear_left_bed_away_book_and_people"];n=np.array(p["normal_world"]);d=p["d"];tn=np.array(t["normal_world"]);td=t["d"]
 P=plane_intersection(p["candidate_pixel_A00"],K[0],A[0],n,d);forward=A[0,:3,:3].T[:,2];cn=forward-(forward@n)*n;cn/=np.linalg.norm(cn);cd=float(-cn@P)
 points={"handle_top":[429,298],"handle_left":[395,316],"handle_right":[466,330],"body_top":[428,340],"body_bottom":[426,400],"base_left":[399,406],"base_right":[456,405],"base_center":[431,413]}
 segs=[("handle_top","handle_left",8),("handle_top","handle_right",8),("handle_left","body_top",8),("handle_right","base_right",8),("body_top","body_bottom",65),("body_bottom","base_center",65),("base_left","base_center",65),("base_right","base_center",65)]
 anchors={}
 for name,uv in points.items():
  w=plane_intersection(uv,K[0],A[0],cn,cd);anchors[name]=np.r_[uv,float(project(w[None],K[0],A[0])[1][0])]
 mask=cv2.imread(str(Path(rv["source_mask49"])/"000000.png"),0)>0;v,f,_=regularized_shell(mask,K[0],A[0],anchors,segs)
 below=v@n+d<0
 if below.any():
  uv,_=project(v[below],K[0],A[0]);v[below]=np.array([plane_intersection(u,K[0],A[0],n,d) for u in uv])
 mount=plane_intersection(t["candidate_pixel_A00"],K[0],A[0],tn,td);mv=v+(mount-P);correction=max(0.,-float(np.min(mv@tn+td)))*tn;mv+=correction;mount+=correction
 rot=rotation_on_plane(n,A[0,:3,:3].T);pc,ps,pr=bbox_for_mesh(v,rot);qc,qs,qr=bbox_for_mesh(mv,rot)
 # Execute only the read, reviewed original producer camera function AST on Qinghai CPU.
 prov=Q/"all59_reconstruction_20261115_v1/provenance/CASE7505429_NATIVE_PREPROCESS_AND_CAMERA_PRODUCER_READONLY_20261116_v1.json";s=json.loads(prov.read_text())["rows"][0]["text"];tree=ast.parse(s);nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ["norm","rodrigues"]];run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="run");camfn=next(x for x in run.body if isinstance(x,ast.FunctionDef) and x.name=="camera")
 camera_ast=ast.Module(body=copy.deepcopy(nodes+[camfn]),type_ignores=[]);ast.fix_missing_locations(camera_ast);env={"np":np,"up":n.copy(),"pivot":(pc+qc)/2,"A":A};exec(compile(camera_ast,str(prov),"exec"),env);camera=env["camera"]
 def cost(sign):
  total=0
  for fi in [0,24,48]:
   e=camera(10,fi,sign)
   for verts in [v,mv]:
    uv,z=project(verts[::max(1,len(verts)//2000)],K[fi],e)
    total+=float(np.maximum(-uv[:,0],0).sum()+np.maximum(uv[:,0]-1279,0).sum()+np.maximum(-uv[:,1],0).sum()+np.maximum(uv[:,1]-703,0).sum()+np.maximum(.05-z,0).sum()*1000)
  return total
 sign=-1 if cost(-1)<=cost(1) else 1;B=np.stack([camera(10,i,sign) for i in range(49)]);assert not np.array_equal(A,B);np.testing.assert_allclose(A[0],B[0],atol=1e-12)
 C=parent/"true4K_RGB_camera_component_v2";C.mkdir(exist_ok=False);(C/"RGB49").symlink_to(S/"RGB49",target_is_directory=True);(C/"source_A.mkv").symlink_to(S/"source_A_true4K_Original1280x704_49_FFV1.mkv")
 np.savez_compressed(C/"cameras.npz",K=K,w2c_A=A,w2c_B=B,P=pc,Q=qc,P_size=ps,Q_size=qs,P_rotation=pr,Q_rotation=qr)
 sm=json.loads((S/"RESULT.json").read_text());meta=dict(case="7505429",state="actual_Original4K_RGB49_matching_recovered_Native_grid_with_original_camera_function_new_B",source_preparation=str(S/"RESULT.json"),source_preparation_SHA256=sha(S/"RESULT.json"),source_video=str(C/"source_A.mkv"),source_video_SHA256=sha(C/"source_A.mkv"),camera_file=str(C/"cameras.npz"),camera_file_SHA256=sha(C/"cameras.npz"),original_native_input=str(I),original_native_depth_K_E_SHA256=sha(I/"native_depth_K_E_subset.npz"),H_old_prepared1280_to_new_Original1280=np.eye(3).tolist(),raw_model_input_RGB_was_historical_upscaled640=True,Native_RGB49_exact=False,new_depth_inference_run=False,actual_Native_A_poses_preserved_exact=True,B_newly_derived_original_AST_not_historical_relabel=True,B_producer_readonly_provenance=str(prov),B_producer_SHA256=sha(prov),B_camera_AST_SHA256=hashlib.sha256(ast.dump(camera_ast,include_attributes=False).encode()).hexdigest(),camera_sign=sign,camera_orbit_max_degrees=10.,new_B_ground_truth_RGB_observed=False,RGB49_BGR_array_SHA256=[hashlib.sha256(cv2.imread(str(S/"RGB49"/f"{i:06d}.png")).tobytes()).hexdigest() for i in range(49)],GPU_used=False,quality_accepted=False)
 put(C/"INPUT_MANIFEST.json",meta)
 np.savez_compressed(out/"geometry.npz",source_vertices=v,source_faces=f,source_center=pc,source_size=ps,source_rotation=pr,source_mount=P,source_unobserved_completion_vertex=np.zeros(len(v),bool),move_vertices=mv,move_faces=f,move_mount=mount,K=K,w2c_A=A,w2c_B=B)
 raster=Rasterizer();conf=np.load(I/"world_points_conf.npz")["world_points_conf"];rows=[]
 for fi in [0,24,48]:
  rgb=cv2.imread(str(S/"RGB49"/f"{fi:06d}.png"));observed=cv2.imread(str(Path(rv["source_mask49"])/f"{fi:06d}.png"),0)>0;small=cv2.resize(observed.astype(np.float32),(518,280),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0
  valid=np.isfinite(conf[fi])&(conf[fi]>1)&~small;sv,sf=depth_surface(raw["depth"][fi],raw["K"][fi],A[fi],valid_pixels=valid,max_neighbor_ratio=1.15);scene=raster.depth(sv,sf,K[fi],A[fi])
  for role,verts in [("source",v),("rigid_move",mv)]:
   dep=raster.depth(verts,f,K[fi],A[fi]);hit=np.isfinite(dep);layers=visibility(dep,scene);left=rgb.copy();left[hit]=np.rint(.4*left[hit]+.6*np.array([255,0,255])).astype(np.uint8);right=overlay(rgb,layers)
   for im,label in [(left,"whole inferred3D"),(right,"scene visibility estimate")]:cv2.putText(im,f"7505429 {role} A{fi:02d} {label}; NOT edited RGB or observedB GT",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
   dst=out/f"{role}_whole_vs_visibility_A{fi:02d}.jpg";cv2.imwrite(str(dst),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,96]);rows.append(dict(role=role,frame=fi,path=str(dst),SHA256=sha(dst),amodal_pixels=int(hit.sum()),visibility={k:int(x.sum()) for k,x in layers.items()}))
 result=dict(case="7505429",state="terminal0_whole_observed_source_rigid_move_pilot3_pending_actual_review",source_review=str(reviewpath),source_review_SHA256=sha(reviewpath),source_plane=p,target_plane=t,source_contact_candidate_A00=p["candidate_pixel_A00"],target_candidate_A00=t["candidate_pixel_A00"],target_ground_correction_world=correction.tolist(),source_vertices=len(v),source_faces=len(f),geometry=str(out/"geometry.npz"),geometry_SHA256=sha(out/"geometry.npz"),new_camera_component=str(C),source_front_plane_back_and_thickness_are_estimates=True,pure_move_size_orientation_appearance_preserved=True,source_observed_shape_not_amodally_painted=True,actual_pilots=rows,B_original_camera_function_AST_executed_on_QH_CPU=True,GPU_used=False,quality_accepted=False,seconds=time.time()-start)
 put(out/"RESULT.json",result);job.update(state="terminal0_pending_actual_pilot_review",exit_code=0)
except BaseException as e:
 job.update(state="terminal1_preserve_outputs",exit_code=1,error=repr(e),traceback=traceback.format_exc());raise
finally:put(out/"JOB.json",job)

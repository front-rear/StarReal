from pathlib import Path
import sys,json,hashlib,datetime,fcntl,time,os,concurrent.futures,traceback
Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");N=Path("/tmp/starreal_uid1007_case19_cpu_v1");R=Path("/tmp/starreal_uid1007_case83_cpu_v2")
sys.path.insert(0,str(N/"runtime/qh_cpu_py39_v1"))
import cv2,numpy as np
cv2.setNumThreads(1)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
out=Q/"inputs/case83_actual_all_legacy_HF_v1"
l=open(Q/"logs/case83_historical_input_import.lock","a");fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
state=out/"HISTORY_READER_JOB.json";d=dict(case="83",pid=os.getpid(),uid=os.getuid(),boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),startticks=Path("/proc/self/stat").read_text().split(") ",1)[1].split()[19],GPU_used=False,state="running")
def put():state.write_text(json.dumps(d,indent=2))
put();t=time.time()
try:
 imported=json.loads((out/"HF_IMPORT_MANIFEST.json").read_text());assert len(imported["camera_checks"])==12
 oldfiles={Path(x["local_path"]).resolve():x for x in imported["records"]}
 movies=sorted(p for p in (out/"original_HF_folder").rglob("*") if p.suffix.lower() in [".mp4",".mkv",".mov"])
 def one(p):
  row=oldfiles[p.resolve()];assert p.stat().st_size==row["bytes"] and sha(p)==row["SHA256"]
  package="relative_ops_v1" if "relative_ops_v1" in p.parts else "move_replace_mesh_v1"
  isjoint=p.name=="joint.mkv" or p.name.startswith("joint_B_")
  camera=("orbit_10deg" if "orbit_10deg" in str(p) else "small_4deg" if "small_4deg" in str(p) else None)
  op=p.parts[-3] if package=="relative_ops_v1" and isjoint else "move_replace" if isjoint else None
  cap=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened(),str(p)
  fps=cap.get(5);hashes=[];shapes=set();thumbs=[]
  while True:
   ok,bgr=cap.read()
   if not ok:break
   hashes.append(hashlib.sha256(bgr.tobytes()).hexdigest());shapes.add(tuple(bgr.shape))
   if isjoint:thumbs.append(cv2.resize(bgr,(320,176),interpolation=cv2.INTER_AREA))
  cap.release();assert len(hashes)==49,(str(p),len(hashes));assert fps==10,(str(p),fps)
  v=dict(path=str(p.relative_to(out)),file_SHA256=row["SHA256"],frames_decoded=49,FPS=fps,decoded_shapes=[list(s) for s in sorted(shapes)],historical_generated_relighting_not_observed_edited_GT=not isjoint and p.name!="source_A.mkv")
  if not isjoint:return v,None
  assert op in ["insert","remove","move","replace","move_replace"] and camera
  canvas=np.full((1330,2240,3),24,np.uint8)
  for f,im in enumerate(thumbs):
   x,y=(f%7)*320,(f//7)*190;canvas[y+14:y+190,x:x+320]=im;cv2.putText(canvas,f"83 old {package[:3]} {op} {camera} f{f}",(x+2,y+11),cv2.FONT_HERSHEY_SIMPLEX,.25,(255,255,255),1)
  contact=out/f"HISTORICAL_{package}_{op}_{camera}_all49.jpg";assert not contact.exists()
  assert cv2.imwrite(str(contact),canvas,[cv2.IMWRITE_JPEG_QUALITY,96])
  c=next(x for x in imported["camera_checks"] if x["package"]==package and x["operation"]==op and x["camera_version"]==camera)
  joint=dict(operation=op,camera_version=camera,historical_producer_revision=package,historical_package_path=str(p.relative_to(out)),historical_source_SHA256_unchanged=row["SHA256"],historical_camera_path=str(Path(c["path"]).relative_to(out)),historical_camera_SHA256=c["SHA256"],frames_decoded=49,FPS=fps,RGB_frame_SHA256_BGR=hashes,actual_all49_contact_sheet=str(contact),contact_sheet_SHA256=sha(contact),actual_visual_review_pending=True)
  return v,joint
 decoded=[];joints=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for v,j in pool.map(one,movies):
   decoded.append(v)
   if j:joints.append(j)
 assert len(joints)==12 and {x["operation"] for x in joints}=={"insert","remove","move","replace","move_replace"}
 m=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),case="83",schema="StarReal.actual_legacy_five_operations_and_extra_mesh_MR_two_camera_versions.v1",HF_revision=imported["HF_revision"],rows=joints,additional_historical_camera_versions=[],historical_camera_versions_preserved=12,historical_operation_count=5,all_original_HF_folder_files_preserved=imported["records"],all_legacy_videos_decoded=decoded,source_K_E_mapping=imported["camera_checks"],old_cameras_preserved_exact_with_independent_current_equality_results=True,new_comparison="Current five operations and both PCF49 generated, no old video regenerated",training_GT_available=False,quality_accepted=False,GPU_used=False)
 (out/"MANIFEST.json").write_text(json.dumps(m,indent=2))
 dest=R/"outputs/83/historical_joint_comparison49_QH_v1";assert not dest.exists();dest.symlink_to(out,target_is_directory=True)
 receipt=dict(case="83",historical_actual_joint_clips=12,all_movies_decoded49=len(decoded),all_camera_file_SHA_retained=True,manifest=str(out/"MANIFEST.json"),GPU_used=False,quality_accepted=False)
 (R/"receipts/CASE83_HISTORICAL_CAMERAS_AND_SOURCE49_READBACK_v1.json").write_text(json.dumps(receipt,indent=2))
 d.update(state="terminal0_all_legacy49_decoded_pending_actual12_contact_review",exit_code=0,seconds=time.time()-t,**receipt);put();print(json.dumps(d),flush=True)
except BaseException:
 d.update(state="terminal1_preserve_completed_no_automatic_retry",exit_code=1,seconds=time.time()-t);put();traceback.print_exc();raise

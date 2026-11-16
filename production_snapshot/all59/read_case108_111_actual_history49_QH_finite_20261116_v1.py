from pathlib import Path
import sys,json,hashlib,datetime,fcntl,time
Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");N=Path("/tmp/starreal_uid1007_case19_cpu_v1")
sys.path.insert(0,str(N/"runtime/qh_cpu_py39_v1"))
import cv2,numpy as np
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for case,version in [("108","v4"),("111","v3")]:
 l=open(Q/"logs"/f"case{case}_historical_input_import.lock","a");fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=Q/"inputs"/f"case{case}_historical_mesh_HF_{version}";imported=json.loads((out/"HF_IMPORT_MANIFEST.json").read_text());assert len(imported["camera_checks"])==2
 oldfiles={Path(r["local_path"]).resolve():r for r in imported["records"]};movies=sorted(p for p in (out/"original_HF_folder").rglob("*") if p.suffix.lower() in [".mp4",".mkv",".mov"])
 decoded=[];joints=[]
 for p in movies:
  row=oldfiles[p.resolve()];assert p.stat().st_size==row["bytes"] and sha(p)==row["SHA256"]
  cap=cv2.VideoCapture(str(p));assert cap.isOpened();fps=cap.get(cv2.CAP_PROP_FPS);hashes=[];shapes=set();thumbs=[]
  while True:
   ok,bgr=cap.read()
   if not ok:break
   hashes.append(hashlib.sha256(bgr.tobytes()).hexdigest());shapes.add(tuple(bgr.shape));thumbs.append(cv2.resize(bgr,(320,176),interpolation=cv2.INTER_AREA))
  cap.release();assert len(hashes)==49,p
  v=dict(path=str(p.relative_to(out)),file_SHA256=row["SHA256"],frames_decoded=49,FPS=fps,decoded_shapes=[list(s) for s in sorted(shapes)],historical_generated_relighting_not_observed_edited_GT=("relit" in p.name))
  decoded.append(v)
  if p.name.startswith("joint_B_"):
   camera="orbit_10deg" if "orbit_10deg" in p.name else "small_4deg" if "small_4deg" in p.name else None;assert camera
   canvas=np.full((1330,2240,3),24,np.uint8)
   for f,im in enumerate(thumbs):
    x,y=(f%7)*320,(f//7)*190;canvas[y+14:y+190,x:x+320]=im;cv2.putText(canvas,f"{case} historical {camera} frame{f}",(x+2,y+11),cv2.FONT_HERSHEY_SIMPLEX,.28,(255,255,255),1)
   contact=out/f"HISTORICAL_{camera}_all49.jpg";cv2.imwrite(str(contact),canvas,[cv2.IMWRITE_JPEG_QUALITY,96]);c=next(r for r in imported["camera_checks"] if r["camera_version"]==camera)
   joints.append(dict(operation="move_replace",camera_version=camera,historical_package_path=str(p.relative_to(out)),historical_source_SHA256_unchanged=row["SHA256"],historical_camera_path=str(Path(c["path"]).relative_to(out)),historical_camera_SHA256=c["SHA256"],frames_decoded=49,FPS=fps,RGB_frame_SHA256_BGR=hashes,actual_all49_contact_sheet=str(contact),contact_sheet_SHA256=sha(contact),actual_visual_review_pending=True))
 assert len(joints)==2 and {x["camera_version"] for x in joints}=={"orbit_10deg","small_4deg"}
 m=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),case=case,schema="StarReal.actual_legacy_move_replace_two_camera_versions.v1",HF_revision=imported["HF_revision"],rows=joints,additional_historical_camera_versions=[],historical_camera_versions_preserved=2,historical_operation_count=1,no_relative_ops_v1=True,new_current_has_five_operations=False,new_current_five_operations_pending=True,old_missing_operations_not_fabricated=True,all_original_HF_folder_files_preserved=imported["records"],all_legacy_videos_decoded=decoded,source_K_E_mapping=imported["camera_checks"],new_comparison="Prepared Source/move geometry components only; full current5 variants/twoPCF comparison not yet generated",training_GT_available=False,quality_accepted=False,GPU_used=False)
 (out/"MANIFEST.json").write_text(json.dumps(m,indent=2))
 r=Path("/tmp")/f"starreal_uid1007_case{case}_true4K_cpu_v2";dest=r/"outputs"/case/"historical_joint_comparison49_QH_v1";dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();dest.symlink_to(out,target_is_directory=True)
 (r/"receipts").mkdir(exist_ok=True);(r/"receipts"/f"CASE{case}_HISTORICAL_CAMERAS_AND_SOURCE49_READBACK_v1.json").write_text(json.dumps(dict(case=case,historical_actual_joint_clips=2,all_movies_decoded49=len(decoded),all_camera_file_SHA_retained=True,manifest=str(out/"MANIFEST.json"),GPU_used=False,quality_accepted=False),indent=2))
 print(json.dumps(dict(case=case,all_movies_decoded49=len(decoded),actual_old_joint_clips=2,manifest=str(out/"MANIFEST.json"))),flush=True)
 l.close()

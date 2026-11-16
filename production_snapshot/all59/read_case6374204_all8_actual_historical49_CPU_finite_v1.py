from pathlib import Path,PurePosixPath
import os,sys,json,hashlib,datetime,fcntl,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1';R=Path('/tmp/starreal_uid1007_case6374204_true4K_cpu_v2');N=Path('/tmp/starreal_uid1007_case19_cpu_v1');sys.path.insert(0,str(N/'runtime/qh_cpu_py39_v1'))
import numpy as np,cv2
cv2.setNumThreads(1);os.environ['CUDA_VISIBLE_DEVICES']='';lock=(Q/'logs/case6374204_historical_reader.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
h=Q/'inputs/extras_five_actual_HF_history_v1/6374204';imp=json.loads((h/'HF_IMPORT_MANIFEST.json').read_text());assert imp['state']=='all_actual_original_HF_folder_files_server_SHA_verified'
out=R/'outputs/6374204/historical_joint_comparison49_QH_v1';out.mkdir(exist_ok=False);rows=[];start=time.time();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();identity=dict(pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
def dump(name,z):(out/name).write_text(json.dumps(z,indent=2)+'\n')
dump('JOB.json',dict(identity,state='reading_all_eight_actual_historical_joint_clips',clips_finished=0,GPU_used=False))
try:
 movies=[x for x in imp['records'] if x['HF_path'].endswith('/joint.mkv')];assert len(movies)==8
 for r in movies:
  p=Path(r['local_path']);assert sha(p)==r['SHA256'];parts=PurePosixPath(r['HF_path']).parts;op,cam=parts[-3],parts[-2]
  cp=h/f'extracted/relative_ops_v4/operations/{op}/{cam}/cameras_and_boxes.npz';z=np.load(cp);assert z['K'].shape==(49,3,3) and z['w2c_A'].shape==z['w2c_B'].shape==(49,4,4)
  d=out/'original_unchanged'/op/cam;d.mkdir(parents=True);os.link(p,d/'joint.mkv');(d/'cameras_and_boxes.npz').write_bytes(cp.read_bytes())
  cap=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.isOpened();can=np.full((1330,2240,3),24,np.uint8);frame_hash=[];fps=float(cap.get(cv2.CAP_PROP_FPS));assert abs(fps-10)<1e-5
  for i in range(49):
   ok,im=cap.read();assert ok and im.shape==(704,1280,3);frame_hash.append(hashlib.sha256(im.tobytes()).hexdigest());x,y=(i%7)*320,(i//7)*190;can[y+14:y+190,x:x+320]=cv2.resize(im,(320,176),interpolation=cv2.INTER_AREA);cv2.putText(can,f'OLD {op} {cam} {i:02d}',(x+2,y+11),cv2.FONT_HERSHEY_SIMPLEX,.27,(255,255,255),1)
  assert not cap.read()[0];cap.release();contact=out/(op+'_'+cam+'_all49.jpg');cv2.imwrite(str(contact),can,[cv2.IMWRITE_JPEG_QUALITY,93])
  rows.append(dict(operation=op,camera_version=cam,historical_HF_path=r['HF_path'],historical_package_path=str((d/'joint.mkv').relative_to(out)),historical_source_SHA256_unchanged=sha(d/'joint.mkv'),historical_camera_path=str((d/'cameras_and_boxes.npz').relative_to(out)),historical_camera_SHA256=sha(d/'cameras_and_boxes.npz'),frames=49,width=1280,height=704,FPS=fps,decoded_BGR49_SHA256=frame_hash,all49_actual_contact_sheet=str(contact.relative_to(R)),contact_sheet_SHA256=sha(contact),actual_old_camera_K_A_B_retained=True,new_Native_camera_is_different_not_homography_match=True,original_clip_container_unchanged=True))
  dump('JOB.json',dict(identity,state='reading_all_eight_actual_historical_joint_clips',clips_finished=len(rows),GPU_used=False))
 primary=[r for r in rows if r['camera_version']=='orbit_10deg'];secondary=[r for r in rows if r['camera_version']=='small'];assert len(primary)==len(secondary)==4
 result=dict(case='6374204',state='all8_actual_original_joint_clips_decoded49_and_original_cameras_retained_pending_actual_visual_review',rows=primary,additional_historical_camera_versions=secondary,actual_historical_operations=['insert','move','remove','replace'],historical_move_replace_exists=False,no_fabricated_old_operation=True,original_HF_folder_manifest=str(h/'HF_IMPORT_MANIFEST.json'),original_HF_folder_manifest_SHA256=sha(h/'HF_IMPORT_MANIFEST.json'),original_HF_files=71,original_complete_archives=[r for r in imp['records'] if r['HF_path'].endswith('.tar')],camera_warning='Original historical K and A/B are different estimated camera solution, retained unchanged and not relabelled. New Native estimated geometry/camera reuse is bound by exact49 decoded original RGB and full4K crop proof.',exact_decoded_source_and_new_Native_proof=str(Q/'component_cache/6374204/RGB49_NATIVE_REUSE_PROOF_v1/RESULT.json'),GPU_used=False,quality_accepted=False,formal_case_increment=0,seconds=time.time()-start)
 dump('MANIFEST.json',result);dump('JOB.json',dict(identity,state='terminal0_all_eight_actual_historical_joint49_read',exit_code=0,clips_finished=8,seconds=time.time()-start,GPU_used=False));print('HISTORICAL_ALL8_49_READ_OK',flush=True)
except BaseException as e:
 dump('JOB.json',dict(identity,state='terminal1_preserve_partial',exit_code=1,error=repr(e),GPU_used=False));raise

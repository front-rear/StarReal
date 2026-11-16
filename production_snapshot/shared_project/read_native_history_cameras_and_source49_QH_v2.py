from pathlib import Path
import json,hashlib,sys,os
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');sys.path.insert(0,str(N/'runtime/qh_cpu_py39_v1'))
import cv2,numpy as np
CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal();cv2.setNumThreads(1);I=N/f'inputs/case{CASE}_native_CPU_v1';H=N/f'inputs/case{CASE}_historical_cameras_v1';m=json.loads((I/'INPUT_MANIFEST.json').read_text());cam=np.load(I/'cameras.npz');proof=json.loads((H/'CAMERA_PROVENANCE.json').read_text());rows=[]
for row in proof['records']:
 p=H/row['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==row['SHA256'];z=np.load(p);diff={k:dict(same_shape=z[k].shape==cam[k].shape,exactly_equal=bool(np.array_equal(z[k],cam[k])),max_abs_difference=float(np.max(np.abs(z[k]-cam[k]))) if z[k].shape==cam[k].shape else None) for k in ['K','w2c_A','w2c_B']};rows.append(dict(path=row['path'],SHA256=row['SHA256'],camera_comparison=diff))
videos=[N/f'inputs/case{CASE}_source_video_HF_v1/source_A.mkv',N/f'inputs/case{CASE}_historical_five_ops_HF_v2/media/source_A.mkv'];vids=[]
for p in videos:
 cap=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=cap.get(5)
 for i in range(49):
  ok,bgr=cap.read();assert ok and bgr.shape==(704,1280,3) and hashlib.sha256(bgr.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i]
 assert not cap.read()[0];cap.release();assert fps==10;vids.append(dict(path=str(p),bytes=p.stat().st_size,SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),all49_BGR_arrays_equal_Native=True,fps=fps))
r=dict(case=CASE,state='all10_HF_matched_camera_members_and_both_source_containers49_exact_readback',camera_records=rows,source_videos=vids,container_SHA_difference_not_RGB_change=vids[0]['SHA256']!=vids[1]['SHA256'],GPU_used=False,quality_accepted=False);p=N/f'receipts/CASE{CASE}_HISTORICAL_CAMERAS_AND_SOURCE49_READBACK_v1.json';assert not p.exists();p.write_text(json.dumps(r,indent=2));print(json.dumps(dict(case=CASE,camera_members=10,source_videos_both49_exact=True,camera_equalities=[dict(path=x['path'],K=x['camera_comparison']['K']['exactly_equal'],A=x['camera_comparison']['w2c_A']['exactly_equal'],B=x['camera_comparison']['w2c_B']['exactly_equal']) for x in rows])),flush=True)

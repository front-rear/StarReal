"""Fresh variant streaming Original composition; no frame renderer modifications."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
R=Path(os.environ.get('STARREAL_CPU_ROOT','/tmp/starreal_uid1007_case19_cpu_v1'));CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal()
sys.path[:0]=[str(R/'runtime/qh_cpu_py39_v1'),str(R/'runtime/qh_ffmpeg_py39_v1'),str(R/'code'),str(R)]
import cv2,imageio_ffmpeg,numpy as np
from original_joint_frame_stream_v1 import OriginalJointStream
OUT=R/f'outputs/{CASE}/two_pcf49_fixed_world_yellow_QH_v1'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def video(p,w):
 c=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);fps=c.get(5);n=0
 try:
  while True:
   ok,im=c.read()
   if not ok:break
   assert im.shape==(704,w,3);n+=1
 finally:c.release()
 assert n==49 and abs(fps-10)<1e-6
 return dict(frames_decoded=n,width=w,height=704,fps=fps,SHA256=sha(p),bytes=p.stat().st_size)
def one(name):
 reader=OriginalJointStream(R/'runtime/case18_ordinary_refs_v1/CASE18_bbox_dataset_original_UNMODIFIED.py')
 assert reader.original_file_SHA256=='ae76b9588a94dd5a9db511ff89182a36d48947193b6ad04e0bbf3fae48a68fca'
 assert reader.original_joint_AST_SHA256 in {'50fc516127cb947cbb21f1ff9f0efb5d022fc02eb036246b60339601f322fd30','e8a1c1ef03c4eac9c426a8c0e43b1d9ec5ebac64ffa0f27b6ee40cac06c028d0'},'Python-version-specific AST identity differs'
 folder=OUT/name;stage=folder;assert not any((folder/x).exists() for x in ['regular_1280x704_49f.mp4','new_1280x704_49f.mp4','old_vs_new_2560x704_49f.mp4'])
 ff=imageio_ffmpeg.get_ffmpeg_exe();specs=[('ordinary',1280,'regular_1280x704_49f.mp4'),('depth_aware',1280,'new_1280x704_49f.mp4'),('comparison',2560,'old_vs_new_2560x704_49f.mp4')];procs=[];proof=[];sheets={k:np.full((7*190,7*320,3),24,np.uint8) for k in ['old_joint','new_joint']};new_hashes=[]
 try:
  for mode,w,file in specs:
   p=subprocess.Popen([ff,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x704','-r','10','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','16','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(stage/file)],stdin=subprocess.PIPE)
   procs.append(p)
  for i in range(49):
   rgb=reader.frame(folder/'old_style',i);before=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR);p=folder/'new_joint'/f'{i:06d}.png';h=sha(p);after=cv2.imread(str(p));assert after.shape==before.shape==(704,1280,3)
   new_hashes.append(dict(frame=i,SHA256=h));op=folder/'old_joint'/f'{i:06d}.png'
   if op.exists():
    existing=cv2.imread(str(op));assert existing is not None and np.array_equal(existing,before),'Existing ordinary frame differs; preserve and inspect'
   else:assert cv2.imwrite(str(op),before)
   for proc,arr in zip(procs,[before,after,np.concatenate([before,after],1)]):proc.stdin.write(arr.tobytes())
   proof.extend([dict(variant=name,version='old_joint',frame=i,RGB_SHA256=hashlib.sha256(rgb.tobytes()).hexdigest()),dict(variant=name,version='new_joint',frame=i,RGB_SHA256=hashlib.sha256(after[:,:,::-1].copy().tobytes()).hexdigest())])
   for key,arr in [('old_joint',before),('new_joint',after)]:
    x,y=i%7*320,i//7*190;sheets[key][y+14:y+190,x:x+320]=cv2.resize(arr,(320,176),interpolation=cv2.INTER_AREA);cv2.putText(sheets[key],f'{name} /{i:02d}',(x+2,y+11),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
  for p in procs:p.stdin.close()
  assert all(p.wait()==0 for p in procs)
 except BaseException:
  # Only these owned finite encoders; no signals to scene workers or other jobs.
  for p in procs:
   if p.stdin and not p.stdin.closed:p.stdin.close()
  for p in procs:p.wait(timeout=120)
  raise
 checks={mode:video(stage/file,w) for mode,w,file in specs}
 for row in new_hashes:assert sha(folder/'new_joint'/f'{row["frame"]:06d}.png')==row['SHA256']
 put(stage/'RGB_PIXEL_READBACK.json',dict(records=proof));put(stage/'NEW_RENDERED_PNG_SHA_UNCHANGED.json',dict(records=new_hashes))
 for key,sheet in sheets.items():assert cv2.imwrite(str(OUT/f'{name}_{key}_all49.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,95])
 return name,checks,dict(actual_FFmpeg_PIDs=[p.pid for p in procs],original_joint_AST_SHA256=reader.original_joint_AST_SHA256,new_rendered_frames_recomputed=0)

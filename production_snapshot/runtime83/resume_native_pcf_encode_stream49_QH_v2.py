"""One finite successor for a proven dead encoding pool; never rerender PCF.

Original joint function executes one frame at a time, parity-proven on245 native
frames. Two independent variants, three FFmpeg streams each. All existing partial
files and the predecessor JOB are preserved before installing verified successors.
"""
from pathlib import Path
import concurrent.futures, datetime, fcntl, hashlib, json, os, resource
import shutil, subprocess, sys, time, traceback
R=Path('/tmp/starreal_uid1007_case19_cpu_v1')
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal()
sys.path[:0]=[str(R/'runtime/qh_cpu_py39_v1'),str(R/'runtime/qh_ffmpeg_py39_v1'),str(R/'code')]
import cv2, imageio_ffmpeg, numpy as np
from original_joint_frame_stream_v1 import OriginalJointStream
OUT=R/f'outputs/{CASE}/two_pcf49_fixed_world_yellow_QH_v1'
SOURCE=R/f'outputs/{CASE}/geometry_masks49_cached_QH_v1'
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
def identity_dead(j):
 assert j['uid']==os.getuid();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 p=Path('/proc')/str(j['pid'])
 if boot!=j['boot_id'] or not p.exists():return
 st=(p/'stat').read_text().split(') ',1)[1].split()
 assert st[19]!=str(j['startticks']) or st[0]=='Z','Matching predecessor is still live; no recovery permitted'
def one(name):
 reader=OriginalJointStream(R/'runtime/case18_ordinary_refs_v1/CASE18_bbox_dataset_original_UNMODIFIED.py')
 assert reader.original_file_SHA256=='ae76b9588a94dd5a9db511ff89182a36d48947193b6ad04e0bbf3fae48a68fca'
 assert reader.original_joint_AST_SHA256 in {'50fc516127cb947cbb21f1ff9f0efb5d022fc02eb036246b60339601f322fd30','e8a1c1ef03c4eac9c426a8c0e43b1d9ec5ebac64ffa0f27b6ee40cac06c028d0'},'Python-version-specific AST identity differs'
 folder=OUT/name;stage=OUT/'encoding_stream_successor_v2'/name;stage.mkdir(parents=True,exist_ok=False)
 ff=imageio_ffmpeg.get_ffmpeg_exe();specs=[('ordinary',1280,'regular_1280x704_49f.mp4'),('depth_aware',1280,'new_1280x704_49f.mp4'),('comparison',2560,'old_vs_new_2560x704_49f.mp4')];procs=[];proof=[];sheets={k:np.full((7*190,7*320,3),24,np.uint8) for k in ['old_joint','new_joint']};new_hashes=[]
 # Successful predecessor variants are real completed components: read them back,
 # do not encode them again merely because a sibling worker died.
 reusable=all((folder/f).is_file() for _,_,f in specs) and (folder/'RGB_PIXEL_READBACK.json').is_file() and all((OUT/f'{name}_{k}_all49.jpg').is_file() for k in sheets)
 if reusable:
  checks={mode:video(folder/file,w) for mode,w,file in specs};oldproof=json.loads((folder/'RGB_PIXEL_READBACK.json').read_text())['records'];assert len(oldproof)==98
  by={(x['version'],x['frame']):x['RGB_SHA256'] for x in oldproof}
  for i in range(49):
   rgb=reader.frame(folder/'old_style',i);actual=cv2.imread(str(folder/'old_joint'/f'{i:06d}.png'))
   assert np.array_equal(actual[:,:,::-1],rgb) and hashlib.sha256(rgb.tobytes()).hexdigest()==by['old_joint',i]
   after=cv2.imread(str(folder/'new_joint'/f'{i:06d}.png'));assert after.shape==(704,1280,3) and hashlib.sha256(after[:,:,::-1].copy().tobytes()).hexdigest()==by['new_joint',i]
  for _,_,file in specs:shutil.copy2(folder/file,stage/file)
  shutil.copy2(folder/'RGB_PIXEL_READBACK.json',stage/'RGB_PIXEL_READBACK.json')
  for k in sheets:shutil.copy2(OUT/f'{name}_{k}_all49.jpg',stage/f'{name}_{k}_all49.jpg')
  return name,checks,dict(reused_complete_predecessor_variant=True,all49_original_joint_and_new_RGB_SHA_exact=True,new_rendered_frames_recomputed=0,videos_reencoded=0)
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
 for key,sheet in sheets.items():assert cv2.imwrite(str(stage/f'{name}_{key}_all49.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,95])
 return name,checks,dict(actual_FFmpeg_PIDs=[p.pid for p in procs],original_joint_AST_SHA256=reader.original_joint_AST_SHA256,new_rendered_frames_recomputed=0)
def main():
 cv2.setNumThreads(1);start=time.time();guard=(Q/'logs/reconstructed_joint_cpu.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 failed=json.loads((OUT/'JOB.json').read_text());assert failed['exit_code']==1 and failed['frames_finished']==49 and 'BrokenProcessPool' in failed['error'];identity_dead(failed);assert not (OUT/'RESULT.json').exists()
 for p in Path('/proc').iterdir():
  if not p.name.isdecimal() or p.stat().st_uid!=os.getuid():continue
  try:cmd=(p/'cmdline').read_bytes()
  except OSError:continue
  assert not (b'ffmpeg' in cmd and str(OUT).encode() in cmd),'Predecessor encoder still exists; do not duplicate'
 proof=json.loads((R/'receipts/ORIGINAL_JOINT_STREAM_NATIVE704_ALL245_PIXEL_PARITY_v1.json').read_text());assert proof['state']=='all245_native704_original_joint_frames_pixel_exact' and len(proof['records'])==245
 manifest=json.loads((SOURCE/'MANIFEST.json').read_text());names=list(manifest['operations']);assert len(names)==5
 equality=json.loads((OUT/'NATIVE_FULL_VS_SPARSE_ALL5_EQUALITY.json').read_text());assert len(equality['records'])==5 and all(x['RGB_differing_pixels']==0 for x in equality['records'])
 frames=[]
 for i in range(49):
  row=json.loads((OUT/'frame_receipts'/f'{i:06d}.json').read_text());assert row['frame']==i
  for name in names:
   assert row['variants'][name]['unresolved']==0 and row['variants'][name]['depth_order_all_relevant_rays_passed']
   for rel in ['new_joint','old_style/labels','old_style/geometry/P/wire','old_style/geometry/Q/wire']:assert (OUT/name/rel/f'{i:06d}.png').is_file()
  frames.append(row)
 status=OUT/'ENCODING_STREAM_SUCCESSOR_JOB_v2.json';assert not status.exists();state=dict(actual_worker_pids=failed['actual_worker_pids'],frame_workers=failed['frame_workers'],trace_threads=failed['trace_threads'],case=CASE,pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],state='running_two_variant_streaming_encode_only',workers=2,geometry_rerendered=0,GPU_used=False,quality_accepted=False,automatic_HF=False,producer_SHA256=sha(__file__))
 def save():state.update(elapsed_seconds=time.time()-start,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);put(status,state)
 save();checks={};adoption={}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   for name,v,a in pool.map(one,names):checks[name]=v;adoption[name]=a;state['variants_encoded']=len(checks);save();print(json.dumps(state),flush=True)
  backup=OUT/'predecessor_encoding_pool_failure_v1';backup.mkdir(exist_ok=False);shutil.copy2(OUT/'JOB.json',backup/'JOB.json')
  for name in names:
   folder=OUT/name;stage=OUT/'encoding_stream_successor_v2'/name
   for f in ['regular_1280x704_49f.mp4','new_1280x704_49f.mp4','old_vs_new_2560x704_49f.mp4','RGB_PIXEL_READBACK.json']:
    dest=folder/f
    if dest.exists():old=backup/name/f;old.parent.mkdir(parents=True,exist_ok=True);dest.rename(old)
    shutil.copy2(stage/f,dest);assert sha(stage/f)==sha(dest)
   for key in ['old_joint','new_joint']:
    f=f'{name}_{key}_all49.jpg';dest=OUT/f
    if dest.exists():dest.rename(backup/f)
    shutil.copy2(stage/f,dest)
  policy=json.loads((R/'HUMAN_DEPTH_AWARE_YELLOW_OVERLAP_CURRENT.json').read_text());fixed=[dict(variant=n,geometry_SHA256=sha(SOURCE/n/'fixed_world_geometry.npz')) for n in names]
  result=dict(case=CASE,state='two_PCF49_components_generated_pending_actual_full49_review',frames=49,fps=10,resolution_each_panel=[1280,704],variants=names,semantic_revision=policy['required_mode'],renderer_SHA256=policy['renderer_SHA256'],renderer_modified=False,PCF_parameters=dict(spacing=.5,AA=2,far_cells=64.,max_steps=4096),appearance_parameters=dict(new_box_fill_alpha=1.,new_background_blend=0.,ordinary_comparison_fill_alpha=.58),world_unit='relative monocular units, not calibrated metres',world_geometry_static=True,actual_K_E49_unchanged=True,world_geometry=fixed,frame_depth_checks=frames,video_checks=checks,native_sparse_full_equality='NATIVE_FULL_VS_SPARSE_ALL5_EQUALITY.json',old_style='unchanged Original joint_frames body; one native704 frame at a time with245-frame exact parity',new_style='immutable approved fixed-world opaque shared-depth with visible projected P/Q overlap yellow',GPU_used=False,quality_accepted=False,automatic_HF=False,elapsed_seconds=failed['elapsed_seconds']+time.time()-start,encoding_successor_seconds=time.time()-start,prior_failed_attempt_seconds=failed['elapsed_seconds'],elapsed_scope='sum of finite failed producer and encoding successor active wall times; excludes disconnected gap; not end-to-end throughput',resource_failure_and_successor=dict(failed_JOB='predecessor_encoding_pool_failure_v1/JOB.json',successor_JOB=status.name,all49_rendered_frames_preserved=True,geometry_rerendered=0,actual_stream_adoption=adoption),camera_only_background_cache=dict(used=True,path=str(R/f'outputs/{CASE}/pcf_background49_precomputed_QH_v1'),all49_K_C_RGB_SHA_checked=True,first_frame_background_rerender_exact=True,box_rgb_full_vs_sparse_first_frame_exact=True))
  put(OUT/'RESULT.json',result);state.update(state='terminal0_generated_and_decoded_pending_actual_review',exit_code=0);save();put(OUT/'JOB.json',dict(state,predecessor='predecessor_encoding_pool_failure_v1/JOB.json',frames_finished=49,render_seconds=failed.get('render_seconds'),renderer_SHA256=policy['renderer_SHA256'],renderer_modified=False))
 except BaseException as e:state.update(state='terminal1_no_automatic_retry_preserve_all_outputs',exit_code=1,error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()

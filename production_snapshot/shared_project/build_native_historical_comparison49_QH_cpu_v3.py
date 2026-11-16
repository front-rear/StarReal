"""Preserve every historical camera clip; choose matched primary by actual K/E."""
import json,shutil,subprocess,os,time,datetime,threading,concurrent.futures
from pathlib import Path
import sys,types
R=Path('/tmp/starreal_uid1007_case19_cpu_v1')
sys.path[:0]=[str(R/'runtime/qh_cpu_py39_v1'),str(R/'runtime/qh_pcf_cpu_v1'),str(R/'runtime/qh_ffmpeg_py39_v1'),str(R/'runtime/case19_ordinary_refs_v1')]
import cv2,numpy as np
import imageio_ffmpeg
import hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checkvideo(p,width):
 cap=cv2.VideoCapture(str(p));n=0;fps=cap.get(5)
 while True:
  ok,im=cap.read()
  if not ok:break
  assert im.shape==(704,width,3);n+=1
 cap.release();assert n==49 and abs(fps-10)<1e-6
 return dict(frames_decoded=n,fps=fps,SHA256=sha(p))
_native_cap=cv2.VideoCapture
cv2.VideoCapture=lambda p:_native_cap(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
def check(p,width):
 d=checkvideo(p,width);return dict(decoded_frames=d['frames_decoded'],resolution=[width,704],fps=d['fps'],sha256=d['SHA256'])
R=Path('/tmp/starreal_uid1007_case19_cpu_v1')


CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal();GUARD=threading.Lock();ACTIVE=0;PEAK=0
def one(spec,out,old,new,hashes,ff):
 global ACTIVE,PEAK
 op,variant,cam,choices,primary=spec;pair_start=time.time()
 with GUARD:ACTIVE+=1;PEAK=max(PEAK,ACTIVE)
 try:
  p,diff=choices[cam];rel='media/'+op+'/'+cam+'/joint.mkv';src=old/rel;actual=sha(src)
  recorded=hashes[rel]
  if isinstance(recorded,dict):recorded=recorded.get('sha256',recorded.get('SHA256'))
  assert actual==recorded
  folder=out/op/cam;folder.mkdir(parents=True)
  copy=folder/'old_original_joint.mkv';shutil.copy2(src,copy);assert sha(copy)==actual
  shutil.copy2(p,folder/'historical_cameras_and_boxes.npz')
  oldcheck=check(copy,1280)
  target=folder/'historical_vs_reconstructed_2560x704_49f.mp4'
  cap=cv2.VideoCapture(str(src))
  proc=subprocess.Popen([ff,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s','2560x704','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','16','-threads','2','-pix_fmt','yuv420p',str(target)],stdin=subprocess.PIPE)
  sheet=np.full((7*152,7*512,3),24,np.uint8);pair_hashes=[]
  for i in range(49):
      ok,before=cap.read();assert ok
      after=cv2.imread(str(new/variant/'new_joint'/f'{i:06d}.png'))
      assert before.shape==after.shape==(704,1280,3)
      pair=np.concatenate([before,after],1);assert np.array_equal(pair[:,:1280],before) and np.array_equal(pair[:,1280:],after);pair_hashes.append(hashlib.sha256(pair.tobytes()).hexdigest());proc.stdin.write(pair.tobytes())
      x,y=(i%7)*512,(i//7)*152
      sheet[y+11:y+152,x:x+512]=cv2.resize(pair,(512,141),interpolation=cv2.INTER_AREA)
      cv2.putText(sheet,f'{op} {cam}/{i:02d}: historical | reconstructed',(x+2,y+9),cv2.FONT_HERSHEY_SIMPLEX,.28,(255,255,255),1)
      if i in [0,24,48]:cv2.imwrite(str(folder/f'comparison_A{i:02d}.jpg'),pair,[cv2.IMWRITE_JPEG_QUALITY,96])
  assert not cap.read()[0];cap.release();proc.stdin.close();assert proc.wait()==0
  contact=out/f'{op}_{cam}_historical_vs_reconstructed_all49.jpg'
  cv2.imwrite(str(contact),sheet,[cv2.IMWRITE_JPEG_QUALITY,90])
  row=dict(operation=op,new_variant=variant,historical_camera=cam,historical_source=str(src),historical_package_path=str(copy.relative_to(out)),historical_source_SHA256_unchanged=actual,historical_video_check=oldcheck,comparison_video=str(target.relative_to(out)),comparison_check=check(target,2560),camera_comparison=diff,historical_camera_path=str((folder/'historical_cameras_and_boxes.npz').relative_to(out)),historical_camera_SHA256=sha(p),historical_camera_not_relabelled_as_new=True,actual_full49_contact_sheet=str(contact.relative_to(out)),primary_matched_new_camera=cam==primary[0])
  row['pair_RGB_SHA256']=pair_hashes;row['actual_FFmpeg_pid']=proc.pid;row['actual_worker_thread']=threading.get_ident();row['elapsed_seconds']=time.time()-pair_start

  return row
 finally:
  with GUARD:ACTIVE-=1

def main():
 cv2.setNumThreads(1);start=time.time();old=R/f'inputs/case{CASE}_historical_five_ops_HF_v2';new=R/f'outputs/{CASE}/two_pcf49_fixed_world_yellow_QH_v1';assert json.loads((new/'JOB.json').read_text())['exit_code']==0
 out=R/f'outputs/{CASE}/historical_joint_comparison49_QH_v1';out.mkdir(exist_ok=False);hashes={x['path']:x['SHA256'] for x in json.loads((old/'HF_IMPORT_MANIFEST.json').read_text())['records']};cams=np.load(R/f'outputs/{CASE}/geometry_masks49_cached_QH_v1/cameras49.npz');cm=json.loads((R/f'outputs/{CASE}/geometry_masks49_cached_QH_v1/MANIFEST.json').read_text());variants={v['operation']:k for k,v in cm['operations'].items()};specs=[];ff=imageio_ffmpeg.get_ffmpeg_exe()
 for op in ['insert','remove','move','replace','move_replace']:
  available=sorted({x.split('/')[2] for x in hashes if x.startswith('media/'+op+'/') and x.endswith('/joint.mkv')});assert available;choices={}
  for cam in available:
   p=R/f'inputs/case{CASE}_historical_cameras_v1/operations'/op/cam/'cameras_and_boxes.npz';c=np.load(p);diff={k:dict(same_shape=c[k].shape==cams[k].shape,exactly_equal=bool(np.array_equal(c[k],cams[k])),max_abs_difference=float(np.max(np.abs(c[k]-cams[k]))) if c[k].shape==cams[k].shape else None) for k in ['K','w2c_A','w2c_B']};choices[cam]=(p,diff)
  primary=[cam for cam,(p,diff) in choices.items() if all(x['exactly_equal'] for x in diff.values())];assert len(primary)==1
  for cam in available:specs.append((op,variants[op],cam,choices,primary))
 job=dict(case=CASE,state='running_finite_four_independent_historical_pairs',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],pairs_expected=len(specs),pairs_finished=0,workers=4,GPU_used=False,quality_accepted=False,producer_SHA256=sha(Path(__file__)))
 def putjob():
  p=out/'JOB.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(job,indent=2));t.replace(p)
 putjob();results={}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   futures={pool.submit(one,s,out,old,new,hashes,ff):i for i,s in enumerate(specs)}
   for f in concurrent.futures.as_completed(futures):
    results[futures[f]]=f.result();job.update(pairs_finished=len(results),actual_peak_pairs=PEAK,elapsed_seconds=time.time()-start);putjob();print(json.dumps(dict(case=CASE,pairs_finished=len(results),actual_peak_pairs=PEAK)),flush=True)
  ordered=[results[i] for i in range(len(specs))];rows=[x for x in ordered if x['primary_matched_new_camera']];extras=[x for x in ordered if not x['primary_matched_new_camera']];assert len(rows)==5 and len(ordered)==10 and all(len(x['pair_RGB_SHA256'])==49 for x in ordered)
  result=dict(case=CASE,state='every_historical_camera_video_SHA_and49_comparison_ready_for_actual_review',rows=rows,additional_historical_camera_versions=extras,historical_video_count=len(ordered),comparison_scope='All saved camera versions preserved; primary K/A/B exact, supplemental differences disclosed; conditioning only.',GPU_used=False,quality_accepted=False,parallelism=dict(mode='four_independent_readonly_history_pairs',actual_peak_pairs=PEAK,workers=4,FFmpeg_threads_per_pair=2,producer_SHA256=sha(Path(__file__)),all490_pair_RGB_arrays_checked_before_encoding=True),elapsed_seconds=time.time()-start)
  (out/'MANIFEST.json').write_text(json.dumps(result,indent=2));job.update(state='terminal0_all10_historical_pairs_ready_for_actual_review',exit_code=0,elapsed_seconds=time.time()-start);putjob();print(json.dumps({k:v for k,v in result.items() if k not in ['rows','additional_historical_camera_versions']}),flush=True)
 except BaseException as e:job.update(state='terminal1_preserve_completed_pairs_no_retry',exit_code=1,error=repr(e),elapsed_seconds=time.time()-start);putjob();raise
if __name__=='__main__':main()

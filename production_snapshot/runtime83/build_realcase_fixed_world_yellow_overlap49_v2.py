"""Apply exact already-completed shared-depth renderer to realcase conditions.

No semantic overlay replacement, geometry/camera/control edits or publication.
"""
import argparse,concurrent.futures,fcntl,hashlib,importlib.util,json,multiprocessing,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import cv2,imageio_ffmpeg,numpy as np
from pcf_cpu_accelerator import CPUTrace
from realcase_condition_reader import RealcaseConditionReader
R=Path('/root/autodl-tmp/codex_realcase_reconstruction_20261112_v1');CTX={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def approved():
 policy=json.loads((R/'HUMAN_DEPTH_AWARE_YELLOW_OVERLAP_CURRENT.json').read_text());p=R/policy['renderer_copy'];assert sha(p)==policy['renderer_SHA256'];assert sha(R/policy['base_approved_copy'])==policy['base_approved_SHA256'];spec=importlib.util.spec_from_file_location('approved_depth_fusion_unchanged',p);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module,policy

def init(package,out,threads):
 cv2.setNumThreads(1);mod,policy=approved();mod.trace_lattice=CPUTrace(threads);package=Path(package);m=json.loads((package/'MANIFEST.json').read_text());c=json.loads((package/'conditions/MANIFEST.json').read_text());cams=np.load(package/'conditions/cameras49.npz');A0=cams['w2c_A'][0];variants={}
 for name,meta in c['operations'].items():
  with np.load(package/'conditions'/name/'fixed_world_geometry.npz') as z:g={k:z[k] for k in z.files}
  boxes=[];op=meta['operation']
  for key,label in [('source',1),('target',2)]:
   if key=='source' and op=='insert' or key=='target' and op=='remove':continue
   center=(A0@np.r_[g[key+'_center'],1.])[:3];rot=A0[:3,:3]@g[key+'_rotation'];boxes.append(mod.Box(center,g[key+'_size']/2,rot,label))
  variants[name]=(g,boxes)
 CTX.update(package=package,out=Path(out),mod=mod,policy=policy,cams=cams,A0=A0,variants=variants)

def frame(i):
 c=CTX;mod=c['mod'];cams=c['cams'];K=cams['K'][i];B=cams['w2c_B'][i];C=np.linalg.inv(B@np.linalg.inv(c['A0']));start=time.time();cache=mod.prepare_lattice(C,K,1280,704,2,.5,64.,4096);background,_=mod.render(C,K,[],1280,704,2,.5,64.,4096,prepared=cache);existing=cv2.imread(str(c['package']/'joints/shared_pcf'/f'{i:06d}.png'))[:,:,::-1].copy();diff=np.abs(background.astype(np.int16)-existing.astype(np.int16));rows={}
 for name,(g,boxes) in c['variants'].items():
  for b in boxes:
   key='source' if b.label==1 else 'target';np.testing.assert_allclose((B@np.r_[g[key+'_center'],1.])[:3],(np.linalg.inv(C)@np.r_[b.center,1.])[:3],atol=1e-12)
  rgb,aux=mod.render(C,K,boxes,1280,704,2,.5,64.,4096,prepared=cache);front=np.isfinite(aux['box_depth_m'])&aux['pcf_visible'];back=aux['box_visible']&np.isfinite(aux['pcf_depth_m']);assert not aux['unresolved'].any();assert np.all(aux['pcf_depth_m'][front]<=aux['box_depth_m'][front]);assert np.all(aux['box_depth_m'][back]<aux['pcf_depth_m'][back]);assert rgb.shape==(704,1280,3);yellow=(aux['labels']==3).reshape(704,2,1280,2).all(axis=(1,3));assert np.all(rgb[yellow,0]==rgb[yellow,1]) and np.all(rgb[yellow,2]==0);assert cv2.imwrite(str(c['out']/name/'approved_png49'/f'{i:06d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
  evidence={}
  for kind,mask in [('lattice_in_front',front),('box_in_front',back)]:
   ids=np.flatnonzero(mask.ravel());chosen=ids[np.linspace(0,len(ids)-1,min(8,len(ids))).astype(int)] if len(ids) else []
   evidence[kind]=[dict(AA_pixel_xy=[int(j%(1280*2)),int(j//(1280*2))],PCF_ray_distance=float(aux['pcf_depth_m'].ravel()[j]),box_ray_distance=float(aux['box_depth_m'].ravel()[j])) for j in chosen]
  rows[name]=dict(frame=i,lattice_in_front_AA_samples=int(front.sum()),box_in_front_AA_samples=int(back.sum()),yellow_visible_overlap_AA_samples=int((aux['labels']==3).sum()),full_yellow_pixels=int(yellow.sum()),all_full_yellow_pixels_R_equal_G_and_B0=True,red_only_AA_samples=int((aux['labels']==1).sum()),blue_only_AA_samples=int((aux['labels']==2).sum()),unresolved=0,depth_order_all_relevant_rays_passed=True,fixed_world_projection_equal_actual_B_camera=True,evidence=evidence)
 row=dict(frame=i,pid=os.getpid(),approved_body_SHA256=c['policy']['renderer_SHA256'],body_modified=True,only_HUMAN_yellow_overlap_color_rule_changed=True,unboxed_approved_PCF_vs_existing_RGB=dict(mean_abs_channel_255=float(diff.mean()),p99_abs_channel_255=float(np.quantile(diff,.99)),max_abs_channel_255=int(diff.max()),different_pixels=int(np.any(diff,2).sum())),variants=rows,elapsed_seconds=time.time()-start)
 atomic(c['out']/'frame_receipts'/f'{i:06d}.json',row);return row

def check(p,width):
 cap=cv2.VideoCapture(str(p));n=0;fps=cap.get(5)
 while True:
  ok,f=cap.read()
  if not ok:break
  assert f.shape==(704,width,3);n+=1
 cap.release();assert n==49 and fps==10;return dict(frames=49,resolution=[width,704],fps=10,sha256=sha(p),bytes=p.stat().st_size)
def encode(task):
 package,folder,name=map(Path,task);name=str(name);cv2.setNumThreads(1);reader=RealcaseConditionReader(package/'conditions',package/'media/source_A.mkv',package/'joints');proof=json.loads((package/'PACKAGED_JOINT_PIXEL_READBACK.json').read_text());book={(r['variant'],r['version'],r['frame']):r['RGB_SHA256'] for r in proof['records']};frames=[]
 for i in range(49):
  old=np.asarray(reader.joint_frame(name,'old_joint',i));assert hashlib.sha256(old.tobytes()).hexdigest()==book[(name,'old_joint',i)];old=cv2.cvtColor(old,cv2.COLOR_RGB2BGR);new=cv2.imread(str(folder/name/'approved_png49'/f'{i:06d}.png'));assert new.shape==(704,1280,3);frames.append((old,new))
  if name=='move' and i in [0,24,48]:
   rejected=cv2.cvtColor(np.asarray(reader.joint_frame(name,'new_joint',i)),cv2.COLOR_RGB2BGR);pair2=np.concatenate([rejected,new],1);cv2.putText(pair2,'REJECTED transparent blend | FIXED WORLD + YELLOW OVERLAP',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1);cv2.imwrite(str(folder/f'move_rejected_vs_restored_A{i:02d}.jpg'),pair2,[cv2.IMWRITE_JPEG_QUALITY,96])
 # Stream exact RGB buffers to encoders; no redundant comparison/ordinary PNG49
 # trees. All49 native output PNGs and all3 decoded videos are retained.
 ff=imageio_ffmpeg.get_ffmpeg_exe();checks={}
 for mode,width,dst in [('ordinary',1280,'ordinary_1280x704_49f.mp4'),('approved_depth_aware',1280,'approved_depth_1280x704_49f.mp4'),('comparison',2560,'ordinary_vs_approved_depth_2560x704_49f.mp4')]:
  target=folder/name/dst;cmd=[ff,'-v','error','-n','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x704','-r','10','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','16','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)];proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
  try:
   for old,new in frames:proc.stdin.write((old if mode=='ordinary' else new if mode=='approved_depth_aware' else np.concatenate([old,new],1)).tobytes())
  finally:proc.stdin.close()
  assert proc.wait()==0;checks[mode]=check(target,width)
 for ver,index in [('approved_png49',1),('ordinary_png49',0)]:
  canvas=np.full((7*190,7*320,3),24,np.uint8)
  for i,pair in enumerate(frames):
   x,y=(i%7)*320,(i//7)*190;canvas[y+14:y+190,x:x+320]=cv2.resize(pair[index],(320,176),interpolation=cv2.INTER_AREA);cv2.putText(canvas,f'{name} /{i:02d}',(x+2,y+11),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1)
  cv2.imwrite(str(folder/f'{name}_{ver}_all49.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
 return name,checks

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--package-relative',required=True);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();assert a.case.isdigit() and 1<=a.workers<=8;rel=Path(a.package_relative);assert not rel.is_absolute() and '..' not in rel.parts;package=R/rel;m=json.loads((package/'MANIFEST.json').read_text());assert m['case']==a.case;proof=json.loads((R/'logs/APPROVED_DEPTH_NATIVE15_REFERENCE_EQUALITY_v1.json').read_text());assert proof['approved_body_modified'] is False;assert shutil.disk_usage(R).free>16*1024**3+len(m['variants'])*512*1024**2,'Insufficient measured free space for bounded next-case output; preserve healthy owners and inputs';mod,policy=approved();yellow_proof=json.loads((R/'logs/YELLOW_OVERLAP_FIXED_WORLD_NATIVE_VALIDATION_v2.json').read_text());assert yellow_proof['passed'] and yellow_proof['renderer_SHA256']==policy['renderer_SHA256'];assert proof['approved_renderer_SHA256']==policy['base_approved_SHA256'];quota=Path('/sys/fs/cgroup/cpu.max').read_text().split();assert quota[0]=='max' or a.workers*2<=int(quota[0])/int(quota[1]);guard=(R/'logs/reconstructed_joint_cpu.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX);out=R/'outputs'/a.case/'shared_depth_yellow_overlap49_v2';out.mkdir(exist_ok=False);(out/'frame_receipts').mkdir();names=m['variants'];cams=np.load(package/'conditions/cameras49.npz');A0=cams['w2c_A'][0];fixed=[]
 for name in names:
  for sub in ['approved_png49','ordinary_png49','comparison_png49']:(out/name/sub).mkdir(parents=True)
  z=np.load(package/'conditions'/name/'fixed_world_geometry.npz');fixed.append(dict(variant=name,world_geometry_SHA256=sha(package/'conditions'/name/'fixed_world_geometry.npz')))
 start=time.time();state=dict(case=a.case,pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],state='running_fixed_world_depth_and_HUMAN_yellow_overlap49',producer_code_SHA256=sha(Path(__file__)),encoder='exact_RGB_buffer_stream_no_redundant_PNG_trees',frames_finished=0,variants=len(names),workers=a.workers,trace_threads=2,approved_body_SHA256=policy['renderer_SHA256'],approved_body_modified=True,only_HUMAN_yellow_overlap_color_rule_changed=True,package_source=str(package),fixed_A0_anchor=A0.tolist(),world_geometry=fixed,GPU_used=False,automatic_quality_or_HF=False)
 def save():state['elapsed_seconds']=time.time()-start;atomic(out/'JOB.json',state)
 save();rows={}
 try:
  context=multiprocessing.get_context('spawn')
  with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=context,initializer=init,initargs=(str(package),str(out),2)) as pool:
   for future in concurrent.futures.as_completed([pool.submit(frame,i) for i in range(49)]):
    row=future.result();rows[row['frame']]=row;state.update(frames_finished=len(rows),actual_worker_pids=sorted({x['pid'] for x in rows.values()}));save();print(json.dumps(dict(case=a.case,approved_frames=len(rows),variants=len(names),elapsed=state['elapsed_seconds'])),flush=True)
  state.update(state='encoding_approved_and_ordinary_full49',render_seconds=time.time()-start);save();fcntl.flock(guard,fcntl.LOCK_UN);guard.close();videos={}
  with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=context) as pool:
   for name,checked in pool.map(encode,[(str(package),str(out),n) for n in names]):videos[name]=checked;state['variants_encoded']=len(videos);save()
  result=dict(case=a.case,state='all49_fixed_world_depth_yellow_overlap_outputs_ready_FOR_ACTUAL_REVIEW',approved_renderer_SHA256=policy['renderer_SHA256'],approved_renderer_body_changed=True,only_HUMAN_yellow_overlap_color_rule_changed=True,shader='Opaque nearest-depth kernel preserved from20261112; independently PCF-visible P/Q projected overlap yellow; no PCF background blend',camera_and_source_geometry_unchanged=True,Original_training_gamma_alpha_color_masks_unchanged=True,world_units='relative monocular world, gridspacing0.5 in same units; not calibratedmetres',frames=49,resolution=[1280,704],fps=10,variants=names,fixed_world_anchor=A0.tolist(),world_geometry=fixed,frame_evidence=[rows[i] for i in range(49)],video_checks=videos,render_seconds=state['render_seconds'],elapsed_seconds=time.time()-start,quality_accepted=False,GPU_used=False,automatic_HF_upload=False);atomic(out/'RESULT.json',result);state.update(state='terminal0_approved49_videos_and_depth_proofs_ready_PENDING_ACTUAL_QA',exit_code=0);save();print(json.dumps(dict(case=a.case,state=state['state'],frames=49,variants=len(names),video_count=len(videos)*3,elapsed=state['elapsed_seconds'])),flush=True)
 except BaseException as e:state.update(state='terminal1_no_automatic_retry',exit_code=1,error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()

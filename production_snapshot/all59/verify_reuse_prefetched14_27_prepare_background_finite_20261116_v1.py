from pathlib import Path
import hashlib,json,datetime,sys,os,subprocess,fcntl
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1'
sys.path.insert(0,str(N/'runtime/qh_cpu_py39_v1'))
import numpy as np,cv2
from PIL import Image,ImageDraw
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(x,indent=2))
def link(p,t):
 if p.exists():assert p.resolve()==t.resolve() or (p.is_file() and sha(p)==sha(t));return
 p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(t,target_is_directory=t.is_dir())
def main():
 guard=(U/'VERIFY_PREFETCH14_27_REUSE.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cv2.setNumThreads(1);rows=[]
 for c in ['14','27']:
  d=Q/f'inputs/original_native_readonly_SFTP_14_27_stream_v2/{c}';dur=Q/f'inputs/case{c}_native_CPU_v1';m=json.loads((dur/'INPUT_MANIFEST.json').read_text())
  assert sha(d/'source_A.mkv')==m['source_video_SHA256'] and sha(d/'vggt_raw.npz')==m['raw_original_file_SHA256']
  for row in m['files']:assert (dur/row['path']).stat().st_size==row['bytes'] and sha(dur/row['path'])==row['SHA256']
  with np.load(d/'vggt_raw.npz') as raw,np.load(dur/'native_depth_K_E_subset.npz') as z:
   for k in ['depth','K','extrinsic_3x4']:np.testing.assert_array_equal(raw[k],z[k])
   conf=raw['world_points_conf']
  cams=np.load(dur/'cameras.npz');assert sha(dur/'cameras.npz')==sha(d/'cameras.npz')==m['camera_original_file_SHA256']
  for base in [N,Q]:
   src=base/f'inputs/case{c}_source_video_HF_v1/source_A.mkv';assert src.exists() or base==N
   if src.exists():assert sha(src)==m['source_video_SHA256']
  link(N/f'inputs/case{c}_native_CPU_v1',dur);link(N/f'inputs/case{c}_source_video_HF_v1',Q/f'inputs/case{c}_source_video_HF_v1')
  confp=Q/f'inputs/CASE{c}_NATIVE_POINT_CONFIDENCE49_v1.npz'
  with np.load(confp) as z:np.testing.assert_array_equal(conf,z['world_points_conf'])
  for suf in ['.npz','.json']:link(N/f'inputs/CASE{c}_NATIVE_POINT_CONFIDENCE49_v1{suf}',confp.with_suffix(suf))
  cap=cv2.VideoCapture(str(d/'source_A.mkv'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.get(5)==10;hashes=[]
  review=Q/f'component_cache/{c}/prefetched_observed_RGB_SAM49_review_v1';review.mkdir(parents=True,exist_ok=False);page=Image.new('RGB',(2240,7*196),'black');draw=ImageDraw.Draw(page)
  for i in range(49):
   ok,im=cap.read();assert ok and im.shape==(704,1280,3);np.testing.assert_array_equal(im,cv2.imread(str(d/'original_frames49'/f'{i:06d}.png')));hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
   mask=cv2.imread(str(dur/'original_mask49'/f'{i:06d}.png'),0)>0;np.testing.assert_array_equal(mask,cv2.imread(str(d/'original_mask49'/f'{i:06d}.png'),0)>0)
   rgb=im[:,:,::-1].copy();over=rgb.copy();over[mask]=(over[mask]*.3+np.array([0,255,64])*.7).astype(np.uint8)
   small=Image.fromarray(over);small.thumbnail((320,176));x=i%7*320;y=i//7*196;page.paste(small,(x,y));draw.text((x+3,y+178),f'Case{c} original SAM A{i:02}',fill='white')
   if i in [0,24,48]:
    Image.fromarray(rgb).save(review/f'observed_A{i:02}.jpg',quality=96);Image.fromarray(over).save(review/f'original_SAM_A{i:02}.jpg',quality=96)
  assert not cap.read()[0];cap.release();assert hashes==m['RGB49_BGR_array_SHA256']
  page.save(review/'ALL49_ORIGINAL_OBSERVED_SAM.jpg',quality=94)
  row=dict(case=c,state='all_RGB49_original_frame_SAM_depth_K_E_and_confidence_independently_reverified_QH',original_Source_SHA256=m['source_video_SHA256'],existing_durable_input_reused=True,Mac_payload_files=False,remote_source_writes=False,remote_source_execution=False,quality_accepted=False)
  put(Q/f'receipts/CASE{c}_PREFETCH_REUSE_ALL_ARRAYS49_VERIFIED_v1.json',row);rows.append(row);print(json.dumps(row),flush=True)
 # Exact existing camera caches are reused, no duplicate owner.
 for c in ['14','27']:
  out=N/f'outputs/{c}/pcf_background49_precomputed_QH_v1';dur=Q/f'component_cache/{c}/pcf_background49_precomputed_QH_v1'
  if dur.exists() and not out.exists():link(out,dur)
  if out.exists():
   result=json.loads((out/'RESULT.json').read_text());assert result['frames']==49 and result['camera_source_SHA256']==sha(N/f'inputs/case{c}_native_CPU_v1/cameras.npz');print(json.dumps(dict(case=c,cache='existing_exact49_reused')),flush=True);continue
  env=dict(os.environ,STARREAL_CASE=c,STARREAL_QH_ONLY_COMPONENT_OWNER='1',STARREAL_FRAME_WORKERS='32')
  with (U/f'CASE{c}_CAMERA_BACKGROUND_FINITE_20261116_v2.log').open('xb') as log:
   rc=subprocess.call(['python3',str(N/'code/precompute_native_PCF_background49_QH_only_owner_cpu_v2.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
  assert rc==0
  receipt=Q/f'receipts/CASE{c}_PERSISTENT_CONFIDENCE_AND_BACKGROUND_v1.json'
  if not receipt.exists():
   rc=subprocess.call(['python3',str(U/'persist_completed_background_on_QH_direct_v2.py'),c]);assert rc==0
 put(U/'PREFETCH14_27_REUSE_BACKGROUND_FINITE_RESULT_v1.json',dict(state='terminal0_inputs_and_background49_ready_not_complete_groups',cases=rows,new_complete_groups=0))
if __name__=='__main__':main()

from pathlib import Path
import json,numpy as np,cv2,hashlib,subprocess,sys
B=Path(sys.argv[1]);results=[];records=[]
for p in sorted(B.glob('case*')):
 if not p.is_dir():continue
 sam=json.loads((p/'sampling.json').read_text());row=json.loads((p/'inference.json').read_text());c=np.load(p/'geometry/camera.npz');assert c['w2c'].shape==(49,4,4) and c['intrinsics'].shape==(49,3,3)
 assert np.max(abs(c['w2c'][0]-np.eye(4)))<1e-4;R=c['w2c'][:,:3,:3];assert np.allclose(R@R.transpose(0,2,1),np.eye(3),atol=.02)
 assert len(set(sam['frame_indices']))==49
 nonempty=[];source_nonempty=[]
 for name in ['source','pcf','joint','source_bbox_in_target_view','target_bbox_in_target_view','operation_labels','color','intensity','xyz_preview']:
  fs=sorted((p/name).glob('*.png'));assert len(fs)==49,(p,name,len(fs))
  for f in fs:
   im=cv2.imread(str(f),cv2.IMREAD_UNCHANGED);assert im is not None and im.shape[:2]==(352,640),(p,f)
 for f in range(49):
  read=lambda d:cv2.imread(str(p/d/f'{f:06d}.png'),cv2.IMREAD_UNCHANGED)
  sm=read('source_bbox_in_target_view')>0;tm=read('target_bbox_in_target_view')>0;nonempty.append(int(tm.sum()));source_nonempty.append(int(sm.sum()))
  assert np.array_equal(read('operation_labels'),sm.astype('uint8')+2*tm.astype('uint8'))
  bg=read('pcf');joint=read('joint');assert np.array_equal(bg[~(sm|tm)],joint[~(sm|tm)])
  xyz=np.load(p/'scene_xyz'/f'{f:06d}.npz');assert xyz['xyz'].shape==(352,640,3) and np.isfinite(xyz['xyz']).all()
  for kind in ['color','intensity']:
   z=np.load(p/kind/f'{f:06d}.npz');assert z[kind].dtype==np.float32 and tuple(z['full_shape_hwc'])==(352,640,3)
  records.append({'case':p.name,'frame':f,'source_sha256':hashlib.sha256((p/'source'/f'{f:06d}.png').read_bytes()).hexdigest()})
 def relative(v):
  if isinstance(v,str) and '/real_video_inputs_20261102_v1/' in v:return v.split('/real_video_inputs_20261102_v1/',1)[1]
  if isinstance(v,list):return [relative(x) for x in v]
  if isinstance(v,dict):return {k:relative(x) for k,x in v.items()}
  return v
 portable=relative(row);portable.update(source_video=str((p/'source.mp4').relative_to(B)),pcf_video=str((p/'pcf.mp4').relative_to(B)),joint_video=str((p/'joint.mp4').relative_to(B)),coordinates=str((p/'scene_xyz').relative_to(B)))
 (p/'inference_portable.json').write_text(json.dumps(portable,indent=2))
 results.append({'case':p.name,'complete':True,'frame_count':49,'geometry':'estimated relative camera and depth; non-metric','target_mask_nonempty_frames':sum(x>0 for x in nonempty),'source_mask_nonempty_frames':sum(x>0 for x in source_nonempty),'target_mask_pixel_range':[min(nonempty),max(nonempty)],'photometry':'editable intended controls, not measured','inference':portable})
(B/'INPUT_VALIDATION.json').write_text(json.dumps({'ok':True,'cases':5,'total_frames':245,'validation':'decode all RGB and condition images; camera finite/SO3; per-frame mask labels and background invariance; 3D XYZ; float32 condition shapes','results':results},indent=2))
(B/'model_inputs.json').write_text(json.dumps({'root':'.','purpose':'inference_only_no_target_RGB','samples':[x['inference'] for x in results]},indent=2));print([(x['case'],x['target_mask_nonempty_frames']) for x in results])

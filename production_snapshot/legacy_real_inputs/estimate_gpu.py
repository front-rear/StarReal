from pathlib import Path
import sys,os,time,json,traceback
B=Path('/mnt/atlas/fangrui/LampEdit/real_video_inputs_20261102_v1');sys.path.insert(0,str(B/'deps'));sys.path.insert(0,'/mnt/atlas/fangrui/LUMILENS_QINGHAI_TRAINING_20261010/code/LumiLens/3rdparties/Depth-Anything-3/src')
import torch,cv2,numpy as np
from depth_anything_3.api import DepthAnything3
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.35);assert torch.cuda.is_available()
free,total=torch.cuda.mem_get_info();assert free>10*1024**3,(free,total)
model=DepthAnything3.from_pretrained(str(B/'model')).to('cuda').eval()
for p in sorted(B.glob('case*')):
 if not p.is_dir():continue
 g=p/'geometry';g.mkdir(exist_ok=True);t=time.time()
 if (g/'camera.npz').exists():continue
 (B/'GEOMETRY_STATUS.json').write_text(json.dumps({'stage':'GPU_DA3_running','case':p.name,'time':t,'job':os.environ.get('SLURM_JOB_ID')}))
 try:
  cap=cv2.VideoCapture(str(p/'source.mp4'));frames=[]
  while True:
   ok,im=cap.read()
   if not ok:break
   frames.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
  cap.release();assert len(frames)==49
  pred=model.inference(image=frames,process_res=252,process_res_method='upper_bound_resize',ref_view_strategy='middle')
  E=np.asarray(pred.extrinsics);K=np.asarray(pred.intrinsics,dtype=np.float64);depth=np.asarray(pred.depth,dtype=np.float32)
  if E.shape[-2:]==(3,4):E=np.concatenate([E,np.broadcast_to([0,0,0,1],(49,1,4))],axis=1)
  rawE=E.copy();E=E@np.linalg.inv(E[0]);dh,dw=depth.shape[1:];K[:,0]*=640/dw;K[:,1]*=352/dh
  assert np.isfinite(E).all() and np.isfinite(K).all() and np.isfinite(depth).all()
  sam=json.loads((p/'sampling.json').read_text());np.savez_compressed(g/'camera.npz',intrinsics=K.astype('float32'),w2c=E.astype('float32'),image_size=np.array([352,640]),timestamps=sam['timestamps_seconds'],frame_indices=sam['frame_indices'],scale_known=np.array(False),method=np.array('DA3-SMALL process_res252 full49'))
  np.savez_compressed(g/'predicted_geometry.npz',depth=depth,conf=np.asarray(pred.conf),raw_w2c=rawE)
  (g/'ESTIMATION.json').write_text(json.dumps({'ok':True,'frames':49,'method':'DA3-SMALL','metric_scale_known':False,'process_res':252,'elapsed_s':time.time()-t,'geometry_measured':False,'source':'only source RGB; no target RGB','depth_shape':list(depth.shape),'camera_convention':'OpenCV +X right +Y down +Z forward; frame0 identity','gpu':torch.cuda.get_device_name(),'job':os.environ.get('SLURM_JOB_ID'),'peak_GPU_bytes':torch.cuda.max_memory_allocated()},indent=2));print(p.name,time.time()-t,flush=True)
 except Exception:
  (g/'FAILURE.json').write_text(json.dumps({'error':traceback.format_exc(),'elapsed_s':time.time()-t}));raise
(B/'GEOMETRY_STATUS.json').write_text(json.dumps({'stage':'complete','cases':5,'time':time.time()}))

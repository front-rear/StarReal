from pathlib import Path
import os,time,json,sys,types,traceback
os.environ['OMP_NUM_THREADS']='6';os.environ['MKL_NUM_THREADS']='6'
B=Path('/root/autodl-tmp/real_video_inputs_20261102_v1')
D=Path('/root/autodl-tmp/LampEditBaselineHandoff_20261101_v1/execution_20261101_v1')
sys.path.insert(0,str(D/'src/Depth-Anything-3_3d835ec1a580/src'))
import torch,numpy as np,cv2
from depth_anything_3.api import DepthAnything3
torch.set_num_threads(6);torch.set_num_interop_threads(2)
model=DepthAnything3.from_pretrained(str(D/'models/pisco/da3_small')).to('cpu').eval()
def forward(self,image,extrinsics=None,intrinsics=None,export_feat_layers=None,infer_gs=False,use_ray_pose=False,ref_view_strategy='middle'):
 with torch.inference_mode():return self.model(image.float(),extrinsics,intrinsics,export_feat_layers,infer_gs,use_ray_pose,ref_view_strategy)
model.forward=types.MethodType(forward,model)
for p in sorted(B.glob('case*')):
 if not p.is_dir():continue
 g=p/'geometry';g.mkdir(exist_ok=True)
 if (g/'camera.npz').exists():continue
 t=time.time();(B/'GEOMETRY_STATUS.json').write_text(json.dumps({'case':p.name,'stage':'CPU_DA3_running','start':t}))
 try:
  frames=[str(x) for x in sorted((p/'source').glob('*.png'))];assert len(frames)==49
  pred=model.inference(image=frames,process_res=252,process_res_method='upper_bound_resize',ref_view_strategy='middle')
  E=np.asarray(pred.extrinsics);K=np.asarray(pred.intrinsics,dtype=np.float64);depth=np.asarray(pred.depth,dtype=np.float32)
  if E.shape[-2:]==(3,4):E=np.concatenate([E,np.broadcast_to([0,0,0,1],(49,1,4))],axis=1)
  rawE=E.copy();E=E@np.linalg.inv(E[0]);dh,dw=depth.shape[1:];K[:,0]*=640/dw;K[:,1]*=352/dh
  assert np.isfinite(E).all() and np.isfinite(K).all() and np.isfinite(depth).all()
  sampling=json.loads((p/'sampling.json').read_text())
  np.savez_compressed(g/'camera.npz',intrinsics=K.astype('float32'),w2c=E.astype('float32'),image_size=np.array([352,640]),timestamps=sampling['timestamps_seconds'],frame_indices=sampling['frame_indices'],scale_known=np.array(False),method=np.array('DA3-SMALL CPU float32 process_res252 full49'))
  np.savez_compressed(g/'predicted_geometry.npz',depth=depth,depth_conf=np.asarray(pred.conf,dtype='float32') if hasattr(pred,'conf') and pred.conf is not None else np.ones_like(depth),raw_w2c=rawE)
  (g/'ESTIMATION.json').write_text(json.dumps({'ok':True,'frames':49,'method':'DA3-SMALL','metric_scale_known':False,'process_res':252,'elapsed_s':time.time()-t,'geometry_measured':False,'source':'only source RGB; no target RGB','cpu_float32_forward':'same pretrained network; CPU autocast disabled','depth_shape':list(depth.shape),'camera_convention':'OpenCV +X right +Y down +Z forward; frame0 identity'},indent=2));print(p.name,time.time()-t,flush=True)
 except Exception:
  (g/'FAILURE.json').write_text(json.dumps({'error':traceback.format_exc(),'elapsed_s':time.time()-t}));raise
(B/'GEOMETRY_STATUS.json').write_text(json.dumps({'stage':'complete','cases':5,'time':time.time()}))

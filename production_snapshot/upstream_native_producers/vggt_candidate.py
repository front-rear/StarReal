import os,sys,json,torch,numpy as np,cv2
from pathlib import Path
ROOT=Path('/root/autodl-tmp/kimi_realcase_20261109')
VG=Path('/root/autodl-tmp/real_video_condition_repro_20261105_v1/tools/vggt')
sys.path.insert(0,str(VG))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
case=sys.argv[1];cd=ROOT/'cases'/case;out=ROOT/'samples'/case/'step3_worldpoints';out.mkdir(parents=True,exist_ok=False)
frames=[str(cd/'frames'/f'{i:03}.png') for i in range(49)]
imgs=load_and_preprocess_images(frames).to('cuda');print('input',tuple(imgs.shape),flush=True)
model=VGGT();sd=torch.load('/root/autodl-tmp/real_video_condition_repro_20261105_v1/tools/weights/VGGT-1B/model.pt',map_location='cpu');missing,unexpected=model.load_state_dict(sd,strict=False);model=model.to('cuda').eval()
with torch.no_grad():pred=model(imgs)
H,W=pred['world_points'].shape[2:4];wp=pred['world_points'][0].float().cpu().numpy();wpc=pred['world_points_conf'][0].float().cpu().numpy();dep=pred['depth'][0].float().cpu().numpy()[...,0]
ext,K=pose_encoding_to_extri_intri(pred['pose_enc'],(H,W));ext=ext[0].float().cpu().numpy();K=K[0].float().cpu().numpy()
assert wp.shape==(49,H,W,3) and ext.shape==(49,3,4) and K.shape==(49,3,3)
assert np.isfinite(wp).all() and np.isfinite(ext).all() and np.isfinite(K).all() and (dep>0).any()
# Cross-frame ORB check against the same VGGT world points; this is self-consistency, not calibration truth.
fa,fb=0,48;ga=cv2.imread(frames[fa],0);gb=cv2.imread(frames[fb],0);orb=cv2.ORB_create(5000);ka,da=orb.detectAndCompute(ga,None);kb,db=orb.detectAndCompute(gb,None)
err=[];used=0
if da is not None and db is not None:
    ms=sorted(cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True).match(da,db),key=lambda m:m.distance)[:2000]
    pa=np.array([ka[m.queryIdx].pt for m in ms],np.float64);pb=np.array([kb[m.trainIdx].pt for m in ms],np.float64)
    H0,W0=ga.shape;ua=pa[:,0]*W/W0;va=pa[:,1]*H/H0;ub=pb[:,0]*W/W0;vb=pb[:,1]*H/H0
    xi=np.clip(ua.astype(int),0,W-1);yi=np.clip(va.astype(int),0,H-1);Xw=wp[fa,yi,xi];conf=wpc[fa,yi,xi]
    Xc=Xw@ext[fb][:3,:3].T+ext[fb][:3,3];z=Xc[:,2];good=(z>1e-6)&(conf>1e-3)&np.isfinite(Xw).all(1)
    u=K[fb][0,0]*Xc[:,0]/np.where(good,z,1)+K[fb][0,2];v=K[fb][1,1]*Xc[:,1]/np.where(good,z,1)+K[fb][1,2]
    err=(np.sqrt((u[good]-ub[good])**2+(v[good]-vb[good])**2)*(W0/W)).tolist();used=int(good.sum())
rec={'case':case,'vggt_grid_hw':[int(H),int(W)],'extri':'OpenCV world-to-camera','world_scale':'relative monocular units; metric scale unknown','n_matches_used':used,'median_reproj_px_in_frame_space':float(np.median(err)) if err else None,'p90_reproj_px':float(np.percentile(err,90)) if err else None,'weights_sha256':__import__('hashlib').sha256(Path('/root/autodl-tmp/real_video_condition_repro_20261105_v1/tools/weights/VGGT-1B/model.pt').read_bytes()).hexdigest(),'validation':'VGGT self-consistency only; not calibrated ground truth'}
(out/'WORLDPOINTS.json').write_text(json.dumps(rec,indent=2))
np.savez_compressed(out/'vggt_raw.npz',world_points=wp.astype(np.float32),world_points_conf=wpc.astype(np.float32),depth=dep.astype(np.float32),extrinsic_3x4=ext.astype(np.float32),K=K.astype(np.float32))
print(json.dumps(rec,indent=2),flush=True)

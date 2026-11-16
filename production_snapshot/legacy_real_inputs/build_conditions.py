from pathlib import Path
import sys,json,hashlib,os,time,cv2,numpy as np
B=Path(sys.argv[1]);sys.path.insert(0,str(B/'core_snapshot'))
from bbox import OrientedBoundingBox,render_bbox_conditions
from periodic_pcf import PeriodicPCFRenderer
import torch
assert torch.cuda.is_available()
torch.set_num_threads(2)
r=PeriodicPCFRenderer(640,352,aa=2,environment='legacy_hsv');palette=np.array([[0,0,0],[255,0,0],[0,0,255],[255,255,0]],np.uint8)
rows=[]
for p in sorted(B.glob('case*')):
 if not p.is_dir():continue
 t=time.time();cam=np.load(p/'geometry/camera.npz');pred=np.load(p/'geometry/predicted_geometry.npz');K=cam['intrinsics'];E=cam['w2c'];ds=pred['depth'];ann=json.loads((p/'annotation.json').read_text());xy=ann['source_box_xyxy'];anchor=ann['target_support_pixel'];d0=cv2.resize(ds[0],(640,352))
 sx,sy=(xy[0]+xy[2])/2,(xy[1]+xy[3])/2
 zsrc=float(np.median(d0[max(0,int(sy)-3):int(sy)+4,max(0,int(sx)-3):int(sx)+4]));ztgt=float(np.median(d0[max(0,anchor[1]-3):min(352,anchor[1]+4),max(0,anchor[0]-3):min(640,anchor[0]+4)]))
 unp=lambda x,y,z:np.linalg.inv(K[0])@np.array([x,y,1.])*z
 size=np.array([(xy[2]-xy[0])*zsrc/K[0,0,0],(xy[3]-xy[1])*zsrc/K[0,1,1],(xy[2]-xy[0])*zsrc/K[0,0,0]*.35])
 src=unp(sx,sy,zsrc);predicted_support_depth=ztgt;ztgt=float(size[1]*K[0,1,1]/(ann['target_box_xyxy'][3]-ann['target_box_xyxy'][1]));tgt=unp(anchor[0],anchor[1],ztgt)-[0,size[1]/2,0]
 boxes=[OrientedBoundingBox('source',src,size,np.eye(3),palette[1]),OrientedBoundingBox('target',tgt,size,np.eye(3),palette[2])]
 geometry={'scale':'DA3 relative, not metric','rotation':'proxy aligned to first camera axes; actual lamp pose uncalibrated','source_depth':zsrc,'target_control_depth':ztgt,'DA3_predicted_support_depth':predicted_support_depth,'target_depth_source':'chosen virtual depth from intended projected size, preserving 3D proxy size; not measured support contact','source':{'center_xyz':src.tolist(),'size_xyz':size.tolist()},'target':{'center_xyz':tgt.tolist(),'size_xyz':size.tolist()},'estimated_not_ground_truth':True,'target_support':'visually selected free surface; source depth proxy, contact not measured'}
 (p/'geometry/placement_3d.json').write_text(json.dumps(geometry,indent=2))
 names=['pcf','joint','source_bbox_in_target_view','target_bbox_in_target_view','source_wire','target_wire','operation_labels','color','intensity','scene_xyz','xyz_preview']
 for name in names:(p/name).mkdir(exist_ok=True)
 emit=np.array([1.,.65,.35],np.float32);lumens=500.;iv=np.float32(np.log1p(lumens/100)/np.log1p(100000/100))
 masks=[]
 for kind in ['color','intensity']:(p/kind/kind).mkdir(exist_ok=True)
 for kind in ['color','intensity']:(p/kind/'receipt.json').write_text(json.dumps({'format':'fixture_bbox_lumens_log_float32_roi.v003','encoded_intensity':float(iv),'linear_rgb':emit.tolist(),'target_fixture_total_lumens':lumens,'estimated_photometry':False,'control_preset':True}))
 (p/'joint/RENDER.json').write_text(json.dumps({'schema':'pcf_bbox_factored.v003','background':str(p/'pcf'),'labels':str(p/'operation_labels'),'source_wire':str(p/'source_wire'),'target_wire':str(p/'target_wire'),'palette_rgb':palette.tolist(),'fill_alpha':.58,'background_alpha':.42}))
 yy,xx=np.mgrid[:352,:640]
 for f in range(49):
  bg,_=r.frame(np.linalg.inv(E[f]),K[f],.5)
  results=[render_bbox_conditions([box],K[f],E[f],352,640,edge_width=2,fill_alpha=.58) for box in boxes]
  sm=results[0][2]>0;tm=results[1][2]>0;labels=sm.astype(np.uint8)+2*tm.astype(np.uint8)
  wires=[np.any(x[0]>160,axis=-1)&m for x,m in zip(results,[sm,tm])]
  joint=bg.copy();active=labels>0;joint[active]=np.rint(.42*bg[active]+.58*palette[labels[active]]).astype(np.uint8);wire=wires[0]|wires[1];joint[wire]=palette[labels[wire]]
  write=lambda name,im:cv2.imwrite(str(p/name/f'{f:06d}.png'),im[:,:,::-1] if im.ndim==3 else im,[cv2.IMWRITE_PNG_COMPRESSION,1])
  write('pcf',bg);write('joint',joint);write('source_bbox_in_target_view',sm.astype(np.uint8)*255);write('target_bbox_in_target_view',tm.astype(np.uint8)*255);write('source_wire',wires[0].astype(np.uint8)*255);write('target_wire',wires[1].astype(np.uint8)*255);write('operation_labels',labels)
  col=np.zeros((352,640,3),np.uint8);col[tm]=np.rint(emit*255);inte=np.zeros_like(col);inte[tm]=round(float(iv)*255);write('color',col);write('intensity',inte)
  for kind,scale in [('color',emit),('intensity',np.full(3,iv,np.float32))]:
   ys,xs=np.where(tm);x0,x1=(int(xs.min()),int(xs.max())+1) if len(xs) else (0,1);y0,y1=(int(ys.min()),int(ys.max())+1) if len(ys) else (0,1);roi=np.zeros((y1-y0,x1-x0,3),np.float32);roi[tm[y0:y1,x0:x1]]=scale
   np.savez_compressed(p/kind/f'{f:06d}.npz',**{kind:roi,'origin_xy':np.array([x0,y0]),'full_shape_hwc':np.array([352,640,3])})
   cv2.imwrite(str(p/kind/kind/f'{f:06d}.png'),(col if kind=='color' else inte)[:,:,::-1],[cv2.IMWRITE_PNG_COMPRESSION,1])
  # Dense source scene XYZ is an estimate, distinct from the artificial PCF.
  z=cv2.resize(ds[f],(640,352));rays=np.stack([xx,yy,np.ones_like(xx)],axis=-1)@np.linalg.inv(K[f]).T;xyz_cam=rays*z[...,None];c2w=np.linalg.inv(E[f]);xyz=(xyz_cam@c2w[:3,:3].T+c2w[:3,3]).astype(np.float32)
  np.savez_compressed(p/'scene_xyz'/f'{f:06d}.npz',xyz=xyz,valid=np.isfinite(xyz).all(axis=-1),metric_scale_known=np.array(False),confidence=cv2.resize(pred['conf'][f],(640,352)))
  # Fixed sequence ranges based on first frame, diagnostics only.
  if f==0:lo=np.percentile(xyz.reshape(-1,3),2,axis=0);hi=np.percentile(xyz.reshape(-1,3),98,axis=0)
  vis=np.clip((xyz-lo)/(hi-lo+1e-6)*255,0,255).astype(np.uint8);write('xyz_preview',vis)
  masks.append({'frame':f,'source_pixels':int(sm.sum()),'target_pixels':int(tm.sum()),'target_mask_ratio':float(tm.mean())})
 (p/'frame_diagnostics.json').write_text(json.dumps(masks,indent=2))
 row={'sample_id':p.name+'_move_v1','operation':'move','frame_count':49,'width':640,'height':352,'fps':10,'source':str(p/'source'),'joint':str(p/'joint'),'joint_background':str(p/'pcf'),'color':str(p/'color'),'intensity':str(p/'intensity'),'reference_images':[str(p/'reference.png')],'reference_image_count':1,'reference_sha256':[hashlib.sha256((p/'reference.png').read_bytes()).hexdigest()],'prompt':ann['prompt'],'camera':str(p/'geometry/camera.npz'),'source_bbox_in_target_view':str(p/'source_bbox_in_target_view'),'target_bbox_in_target_view':str(p/'target_bbox_in_target_view'),'target_emission_linear_rgb':emit.tolist(),'target_fixture_total_lumens':lumens,'ambient':1.,'controls_provenance':'color,intensity,ambient are editable requested presets, not inferred photometry','no_ground_truth_available':True,'geometry_is_estimated':True,'metric_scale_known':False,'pcf_renderer':'unchanged PeriodicPCFRenderer legacy_hsv aa2 spacing0.5 in relative DA3 units','joint_encoding':'red source / blue target / yellow overlap; background .42 fill .58','manifest_kind':'inference only; not training manifest','source_mask_kind':'coarse 3D proxy projection; source_lamp_mask_first is separate approximate visible polygon','target_mask_kind':'amodal estimated 3D proxy, not physically calibrated segmentation','gpu_name':torch.cuda.get_device_name(),'GPU_UUID_env':os.environ.get('CUDA_VISIBLE_DEVICES'),'elapsed_s':time.time()-t}
 (p/'inference.json').write_text(json.dumps(row,indent=2));rows.append(row);print(p.name,'done',time.time()-t,flush=True)
(B/'inference_manifest.json').write_text(json.dumps(rows,indent=2));print('all5 complete',flush=True)

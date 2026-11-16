"""Add clearly separated observed-A and projected-A source Remove conditions."""
from pathlib import Path
import hashlib,json,os,sys,io,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';sys.path[:0]=[str(Q/'runtime/qh_cpu_py39_v1'),str(Q/'runtime/case18_ordinary_refs_v1')]
import cv2,numpy as np
from case18_QH_ordinary_reference_v1 import bbox_mask
cv2.setNumThreads(1)
def one(root):
 start=time.time();m=json.loads((root/'MANIFEST.json').read_text());out=root/'A_observed';assert not out.exists();out.mkdir();g=np.load(root/'source_camera_geometry.npz');color=np.asarray(m['source_color_linear_rgb'],np.float32);source=float(m['source_gamma_control']);target=float(m['target_gamma_control']);delta=target-source;rows=[];panels=[]
 for i in range(49):
  lamp=cv2.imread(str(root/'source_observed_A49'/f'{i:06d}.png'),0)>0;box,_=bbox_mask(g['source_center'],g['source_size'],g['source_rotation'],g['K'][i],g['w2c_A'][i]);box=box>0;imgs=[];row=[]
  for family,support in [('lamp_silhouette',lamp),('bbox_silhouette',box)]:
   folder=out/family
   for sub in ['mask49','Color49','Intensity49']:(folder/sub).mkdir(parents=True,exist_ok=True)
   y,x=np.where(support);x0,y0,x1,y1=(int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)) if len(x) else (0,0,1,1);roi=support[y0:y1,x0:x1];c=np.rint(support[...,None]*color*255).astype(np.uint8);mp=folder/'mask49'/f'{i:06d}.png';cp=folder/'Color49'/f'{i:06d}.png';ip=folder/'Intensity49'/f'{i:06d}.npz';assert cv2.imwrite(str(mp),support.astype(np.uint8)*255) and cv2.imwrite(str(cp),c[:,:,::-1]);np.savez_compressed(ip,support=roi,origin_xy=np.asarray([x0,y0],np.int32),full_shape_hw=np.asarray([704,1280],np.int32),source_color_linear_rgb=color,color_linear_rgb=roi[...,None].astype(np.float32)*color,raw_gamma=roi.astype(np.float32)*np.float32(delta),source_gamma_control=np.asarray(source,np.float64),target_gamma_control=np.asarray(target,np.float64))
   with np.load(ip) as z:
    np.testing.assert_array_equal(z['support'],roi);np.testing.assert_array_equal(z['color_linear_rgb'],roi[...,None].astype(np.float32)*color);np.testing.assert_allclose(z['raw_gamma'][roi]/np.float32(1.1),delta/1.1,atol=1e-7)
   np.testing.assert_array_equal(cv2.imread(str(cp))[:,:,::-1],c);row.append(dict(family=family,pixels=int(support.sum()),mask_SHA256=hashlib.sha256(mp.read_bytes()).hexdigest(),Color_SHA256=hashlib.sha256(cp.read_bytes()).hexdigest(),Intensity_SHA256=hashlib.sha256(ip.read_bytes()).hexdigest()));imgs.append(cv2.resize(c[:,:,::-1],(256,141),interpolation=cv2.INTER_NEAREST))
  rows.append(dict(frame=i,families=row));panels.append(np.concatenate(imgs,axis=1))
 canvas=np.full((7*160+35,7*512,3),20,np.uint8)
 for i,image in enumerate(panels):
  xx=(i%7)*512;yy=(i//7)*160+35;canvas[yy:yy+141,xx:xx+512]=image;cv2.putText(canvas,f'{i:02d}',(xx+2,yy+155),cv2.FONT_HERSHEY_SIMPLEX,.35,(220,220,220),1)
 cv2.putText(canvas,f"case{m['case']} A_OBSERVED_lamp_LEFT_3Dbbox_RIGHT_originalRGB_camera",(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);cv2.imwrite(str(root/'review/A_OBSERVED_source_Color_two_families_all49.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,93]);record=dict(state='all49_A_observed_Color_and_signed_Intensity_generated_and_read_back',frames=49,resolution=[1280,704],lamp='actual original package A observed mask, no fabricated hidden completion',bbox='same original source3D bbox with original K/w2c_A',source_gamma_control=source,target_gamma_control=target,raw_gamma=delta,model_relative_gamma=delta/1.1,background_model_gamma=-1,frames_verified=rows,elapsed_seconds=time.time()-start,actual_visual_review_pending=True);(root/'A_OBSERVED_RECORD.json').write_text(json.dumps(record,indent=2));return dict(case=m['case'],key=m['HF_source_archive_LFS_SHA256'],frames=49)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(one(Path(a.root))),flush=True)

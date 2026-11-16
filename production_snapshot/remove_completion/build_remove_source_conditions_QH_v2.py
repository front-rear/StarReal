"""Add source Color/negative Intensity to Remove. Existing packages unchanged."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,io,json,os,sys,tarfile,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';N=Path('/tmp/starreal_uid1007_case19_cpu_v1/reference_completion_20261115_v1')
sys.path[:0]=[str(Q/'runtime/qh_cpu_py39_v1'),str(Q/'runtime/case18_ordinary_refs_v1')]
import cv2,numpy as np
from case18_QH_ordinary_reference_v1 import bbox_mask
cv2.setNumThreads(1)
def sha(b):return hashlib.sha256(b).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def load_npz(b):
 with np.load(io.BytesIO(b),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def build(row):
 start=time.time();p=Path(row['local_path']);assert p.stat().st_size==row['bytes'];key=row['archive_LFS_SHA256'];out=N/'remove_outputs'/key;out.parent.mkdir(exist_ok=True);guard=(out.parent/(key+'.lock')).open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not out.exists();out.mkdir();preview=out/'review';preview.mkdir();records=[]
 with tarfile.open(p) as tf:
  members={m.name:m for m in tf.getmembers() if m.isfile()};root_m=next(m for m in members.values() if m.name.count('/')<=1 and m.name.endswith('/MANIFEST.json'));root=root_m.name.rsplit('/',1)[0];manifest_bytes=tf.extractfile(root_m).read();m=json.loads(manifest_bytes);assert 'remove' in m['variants']
  def get(rel):return tf.extractfile(members[root+'/'+rel]).read()
  cameras_bytes=get('conditions/cameras49.npz');cams=load_npz(cameras_bytes);assert cams['K'].shape==(49,3,3) and cams['w2c_B'].shape==(49,4,4)
  geometry_bytes=get('conditions/remove/fixed_world_geometry.npz');g=load_npz(geometry_bytes);assert np.isfinite(g['source_center']).all() and np.all(g['source_size']>0)
  hint_bytes=get('source_appearance_control_reference.json');hint=json.loads(hint_bytes);color=np.asarray(hint['control_color_linear_rgb'],np.float32);assert color.shape==(3,) and np.isfinite(color).all() and np.all((color>=0)&(color<=1))
  np.savez_compressed(out/'source_camera_geometry.npz',K=cams['K'],w2c_A=cams['w2c_A'],w2c_B=cams['w2c_B'],source_center=g['source_center'],source_size=g['source_size'],source_rotation=g['source_rotation'])
  (out/'SOURCE_APPEARANCE_RECORD_UNCHANGED.json').write_bytes(hint_bytes)
  video_bytes=get(m['source_RGB']['path']);assert sha(video_bytes)==m['source_RGB']['sha256'];video_path=N/('source_'+m['source_RGB']['sha256']+'.mkv')
  if not video_path.exists():video_path.write_bytes(video_bytes)
  cap=cv2.VideoCapture(str(video_path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);rgbframes=[];all49=[];source_masks=[];color_sheets=[];intensity_sheets=[];family_support={'lamp_silhouette':[],'bbox_silhouette':[]};source_gamma_values=[];target_gamma_values=[]
  for i in range(49):
   ok,im=cap.read();assert ok and im.shape==(704,1280,3);rgbframes.append(im)
   observed=get(f'conditions/source_observed_A/{i:06d}.png');obs=cv2.imdecode(np.frombuffer(observed,np.uint8),0);assert obs.shape==(704,1280);source_masks.append(obs>0);dst=out/'source_observed_A49'/f'{i:06d}.png';dst.parent.mkdir(exist_ok=True);dst.write_bytes(observed)
   source_B=get(f'conditions/source_B_estimate/visible_with_unknown_estimate/{i:06d}.png');lamp=cv2.imdecode(np.frombuffer(source_B,np.uint8),0)>0;assert lamp.shape==(704,1280);box,_=bbox_mask(g['source_center'],g['source_size'],g['source_rotation'],cams['K'][i],cams['w2c_B'][i]);box=box>0
   old=load_npz(get(f'conditions/remove/conditions/lamp_silhouette/{i:06d}.npz'));source=float(old['source_gamma_control_removed']);target=float(old['target_gamma_control_removed']);assert source>=0 and target==0 and source>target;source_gamma_values.append(source);target_gamma_values.append(target);delta=target-source;per=[];color_panels=[];intensity_panels=[]
   for family,support in [('lamp_silhouette',lamp),('bbox_silhouette',box)]:
    y,x=np.where(support);x0,y0,x1,y1=(int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)) if len(x) else (0,0,1,1);roi=support[y0:y1,x0:x1];folder=out/family
    for sub in ['mask49','Color49','Intensity49']:(folder/sub).mkdir(parents=True,exist_ok=True)
    mask=support.astype(np.uint8)*255;colors=np.rint(support[...,None]*color*255).astype(np.uint8)
    mp=folder/'mask49'/f'{i:06d}.png';cp=folder/'Color49'/f'{i:06d}.png';ip=folder/'Intensity49'/f'{i:06d}.npz';assert cv2.imwrite(str(mp),mask) and cv2.imwrite(str(cp),colors[:,:,::-1]);np.savez_compressed(ip,support=roi,origin_xy=np.asarray([x0,y0],np.int32),full_shape_hw=np.asarray([704,1280],np.int32),source_color_linear_rgb=color,color_linear_rgb=roi[...,None].astype(np.float32)*color,raw_gamma=roi.astype(np.float32)*np.float32(delta),source_gamma_control=np.asarray(source,np.float64),target_gamma_control=np.asarray(target,np.float64))
    z=load_npz(ip.read_bytes());np.testing.assert_array_equal(z['support'],roi);np.testing.assert_array_equal(z['raw_gamma'],roi.astype(np.float32)*np.float32(delta));np.testing.assert_array_equal(z['color_linear_rgb'],roi[...,None].astype(np.float32)*color);normalized=np.where(roi,z['raw_gamma']/np.float32(1.1),np.float32(-1));assert np.all(normalized[roi]<0) and np.all(normalized[~roi]==-1);np.testing.assert_allclose(normalized[roi],delta/1.1,atol=1e-7)
    color_read=cv2.imread(str(cp))[:,:,::-1];np.testing.assert_array_equal(color_read,colors);np.testing.assert_array_equal(cv2.imread(str(mp),0)>0,support)
    per.append(dict(family=family,support_pixels=int(support.sum()),mask_SHA256=file_sha(mp),Color_SHA256=file_sha(cp),Intensity_SHA256=file_sha(ip),raw_delta=delta,model_gamma=delta/1.1,background_model_gamma=-1))
    family_support[family].append(int(support.sum()));color_panels.append(cv2.resize(color_read[:,:,::-1],(256,141),interpolation=cv2.INTER_NEAREST));vis=np.zeros((704,1280,3),np.uint8);vis[support]=[0,0,round(min(abs(delta/1.1),1)*255)];intensity_panels.append(cv2.resize(vis,(256,141),interpolation=cv2.INTER_NEAREST))
   color_sheets.append(np.concatenate(color_panels,axis=1));intensity_sheets.append(np.concatenate(intensity_panels,axis=1));records.append(dict(frame=i,families=per,source_B_mask_SHA256=sha(source_B),source_A_observed_mask_SHA256=sha(observed)))
   for name in ['unobserved_scene_region','scene_occluded_estimate']:
    rel=f'conditions/source_B_estimate/{name}/{i:06d}.png'
    if root+'/'+rel in members:
     dst=out/'source_B_uncertainty49'/name/f'{i:06d}.png';dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(get(rel))
  assert not cap.read()[0];cap.release();assert len(set(source_gamma_values))==1 and len(set(target_gamma_values))==1
  # Reference is an observed source-video crop, never a fabricated source GLB.
  frame=max(range(49),key=lambda i:int(source_masks[i].sum()));mask=source_masks[frame];assert mask.any();y,x=np.where(mask);x0=max(0,int(x.min())-8);x1=min(1280,int(x.max())+9);y0=max(0,int(y.min())-8);y1=min(704,int(y.max())+9);rgba=np.dstack([rgbframes[frame][:,:,::-1],mask.astype(np.uint8)*255])[y0:y1,x0:x1];cv2.imwrite(str(out/'source_reference.png'),rgba[:,:,[2,1,0,3]])
  observed=rgbframes[frame].copy();observed[mask]=((observed[mask].astype(float)*.55)+np.asarray([0,115,0])).clip(0,255).astype(np.uint8);cv2.imwrite(str(preview/'source_observed_reference.jpg'),observed)
  for label,images in [('source_Color_B_lamp_LEFT_bbox_RIGHT',color_sheets),('source_Intensity_NEGATIVE_red_lamp_LEFT_bbox_RIGHT',intensity_sheets)]:
   canvas=np.full((7*160+35,7*512,3),20,np.uint8)
   for i,image in enumerate(images):
    xx=(i%7)*512;yy=(i//7)*160+35;canvas[yy:yy+141,xx:xx+512]=image;cv2.putText(canvas,f'{i:02d}',(xx+2,yy+155),cv2.FONT_HERSHEY_SIMPLEX,.35,(220,220,220),1)
   cv2.putText(canvas,f"case{m['case']} {key[:12]} {label}; B estimated, NOT RGBGT",(8,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);assert cv2.imwrite(str(preview/(label+'_all49.jpg')),canvas,[cv2.IMWRITE_JPEG_QUALITY,93])
  manifest=dict(schema='StarReal.Remove_source_color_intensity.v1',state='generated_independently_read_back_PENDING_actual_review',case=str(m['case']),case_id=m['case_id'],HF_source_archive=row['HF_path'],HF_source_archive_LFS_SHA256=key,source_archive_verified=row['verified'],frames=49,resolution=[1280,704],view='original package B camera, source lamp before removal',source_A='source_observed_A49',source_B_visibility='reuse exact packaged visible_with_unknown_estimate source mask; unknown and occluded regions retained separately',bbox='Original OrientedBoundingBox projected with source_center/size/rotation and original per-frame K/w2c_B, not a2D enclosing rectangle',source_color=hint,source_color_linear_rgb=color.tolist(),source_gamma_control=source_gamma_values[0],target_gamma_control=0.,raw_gamma=0.-source_gamma_values[0],normalization='/1.1 exactly once at model entry',background_model_gamma=-1,families=['lamp_silhouette','bbox_silhouette'],family_support_pixels49=family_support,source_camera_SHA256=sha(cameras_bytes),source_geometry_SHA256=sha(geometry_bytes),source_manifest_SHA256=sha(manifest_bytes),source_reference=dict(path='source_reference.png',kind='observed_source_RGB_crop_with_actual_A_SAM',frame=frame,crop_xyxy=[x0,y0,x1,y1],SHA256=file_sha(out/'source_reference.png'),source_RGB_container_SHA256=m['source_RGB']['sha256']),target_reference=None,reference_semantics='Source actual lamp is available as reference; there is no post-removal target lamp',actual_source_photometry_measured=False,edited_B_RGB_GT=False,old_archive_changed=False,new_training_RGB=False,frames_verified=records,elapsed_seconds=time.time()-start,GPU_used=False)
  (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(case=m['case'],archive=key,frames=49,output=str(out),seconds=time.time()-start)),flush=True);return manifest

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--row-json',required=True);a=ap.parse_args();row=json.loads(Path(a.row_json).read_text());assert row['verified'] and file_sha(Path(row['local_path']))==row['archive_LFS_SHA256'];build(row)
if __name__=='__main__':main()

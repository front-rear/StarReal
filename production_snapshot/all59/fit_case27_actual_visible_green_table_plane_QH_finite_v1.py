from pathlib import Path
import sys,json,datetime,hashlib
R=Path('/tmp/starreal_uid1007_case27_cpu_v2');sys.path.insert(0,str(R/'runtime/qh_cpu_py39_v1'))
import numpy as np,cv2
raw=np.load(R/'inputs/case27_native_CPU_v1/native_depth_K_E_subset.npz');cam=np.load(R/'inputs/case27_native_CPU_v1/cameras.npz');conf=np.load(R/'inputs/CASE27_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf']
H,W=raw['depth'].shape[1:];dep=raw['depth'][0];K=raw['K'][0];E=cam['w2c_A'][0];y,x=np.indices((H,W));pix=np.stack([x,y,np.ones_like(x)],-1);pc=(pix@np.linalg.inv(K).T)*dep[:,:,None];world=(pc-E[:3,3])@E[:3,:3]
polys=[[[330,376],[560,388],[570,394],[330,382]],[[740,388],[880,395],[920,399],[920,402],[880,400],[740,395]]]
mask=np.zeros((704,1280),np.uint8)
for points in polys:cv2.fillPoly(mask,[np.array(points,np.int32)],1)
small=cv2.resize(mask,(W,H),interpolation=cv2.INTER_NEAREST)>0;valid=small&np.isfinite(dep)&(dep>0)&np.isfinite(conf[0])&(conf[0]>1);points=world[valid];center=points.mean(0);_,s,V=np.linalg.svd(points-center,full_matrices=False);n=V[-1];n*=1 if n@np.array([.394,-.884,-.25])>0 else -1;d=-float(center@n)
out=R/'outputs/27/actual_green_table_support_plane_v1';out.mkdir(exist_ok=False)
np.savez_compressed(out/'actual_points.npz',points_world=points,valid_native_pixels=valid,RGB_support_polygons=np.array(polys,dtype=object),normal_world=n,plane_d=d)
cap=cv2.VideoCapture(str(R/'inputs/case27_source_video_HF_v1/source_A.mkv'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);ok,im=cap.read();assert ok;cap.release();overlay=im.copy();overlay[mask>0]=np.rint(.5*overlay[mask>0]+.5*np.array([0,255,255])).astype(np.uint8)
for poly in polys:cv2.polylines(overlay,[np.array(poly,np.int32)],True,(0,0,255),1)
cv2.putText(overlay,'Case27 ACTUAL TABLE PIXELS ONLY; excludes person/bag/tools; plane remains monocular estimate',(6,24),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1);cv2.imwrite(str(out/'actual_support_pixels_A00.jpg'),overlay,[cv2.IMWRITE_JPEG_QUALITY,97])
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
result=dict(case='27',frame=0,normal_world=n.tolist(),d=d,actual_RGB_polygon_selections=polys,Native_points=len(points),rms_world=float(np.sqrt(np.mean((points@n+d)**2))),singular_values=s.tolist(),camera_normal=(E[:3,:3]@n).tolist(),depth_K_E_unchanged=True,actual_Depth_source_SHA256=sha(R/'inputs/case27_native_CPU_v1/native_depth_K_E_subset.npz'),old_rect_750_300_950_420_mixed_person_not_used=True,plane_is_estimate_not_measured=True,actual_point_source='unchanged Native depth/K/E; original point confidence>1; only visually selected tabletop polygons',new_complete_groups=0,GPU_used=False,quality_accepted=False)
(out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

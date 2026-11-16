from pathlib import Path
import argparse,sys,json,hashlib,fcntl,time,os
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
def main(c):
 assert c in ['74','80'];I=N/f'inputs/case{c}_native_CPU_v1';sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
 import cv2,numpy as np
 from real_geometry import unproject,project,plane_intersection
 cv2.setNumThreads(1);guard=(Q/f'logs/case{c}_geometry_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();review=json.loads((Q/'reviews'/f'CASE{c}_OBSERVED_SOURCE_SAM49_CURRENT.json').read_text());assert review['may_compute_private_source_geometry'];ref=review['reference_frame'];raw=np.load(I/'native_depth_K_E_subset.npz');cam=np.load(I/'cameras.npz');depth=raw['depth'][ref];Kd=raw['K'][ref];A=cam['w2c_A'][ref];K=cam['K'][ref];conf=np.load(N/f'inputs/CASE{c}_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'][ref];m=json.loads((I/'INPUT_MANIFEST.json').read_text());out=Q/f'component_cache/{c}/native_actual_support_probe_v1';out.mkdir(parents=True,exist_ok=False)
 if c=='74':
  polys={'clear_rug_floor':[[962,646],[1096,633],[1116,687],[969,695]],'central_rug_floor':[[975,640],[1085,636],[1094,681],[976,688]]};queries={'source_foot':[931,623],'move_clear_rug':[1030,680],'insert_clear_rug':[1060,645],'MR_clear_rug':[1030,680]};anchors={'shade_top':[943,374],'shade_left':[912,436],'shade_right':[967,436],'shade_bottom':[941,506],'stem_top':[942,523],'stem_mid':[942,565],'stem_lower':[941,603],'foot_left':[924,622],'foot_right':[942,619],'foot_center':[931,623],'cord_left':[907,522],'cord_right':[938,507]};orient=-A[:3,:3].T[:,1]
 else:
  polys={'clear_wall_right_of_plate':[[380,55],[500,50],[500,165],[380,175]],'upper_bare_wall':[[450,30],[800,35],[800,155],[450,155]]};queries={'source_wall_plate_A00':[339,51],'move_upper_wall':[1000,90],'insert_upper_wall':[800,85],'MR_upper_wall':[1000,90]};anchors={'plate_center':[337,96],'plate_top':[335,72],'plate_right':[357,97],'plate_bottom':[338,119],'elbow':[285,90],'stem_mid':[283,59],'upper_bar_right':[281,29],'upper_bar_left':[253,27],'shade_top':[239,51],'shade_left':[191,90],'shade_right':[275,99],'shade_bottom':[233,114]};orient=-A[:3,:3].T[:,2]
 planes={}
 for name,poly in polys.items():
  img=np.zeros((704,1280),np.uint8);cv2.fillPoly(img,[np.array(poly,np.int32)],1);small=cv2.resize(img,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST)>0;yy,xx=np.where(small&np.isfinite(depth)&(depth>0)&np.isfinite(conf)&(conf>1.));pts=unproject(np.c_[xx,yy],depth[yy,xx],Kd,A)
  if len(pts)<3:planes[name]=dict(error='insufficient genuine cached Native points; no fabricated surface',native_points=len(pts));continue
  mu=pts.mean(0);_,sv,vh=np.linalg.svd(pts-mu,full_matrices=False);n=vh[-1];n*=1 if n@orient>=0 else -1;d=-float(n@mu);row=dict(polygon_in_reference_frame=poly,reference_frame=ref,native_points=len(pts),normal_world=n.tolist(),d=d,singular_values=sv.tolist(),rms=float(np.sqrt(np.mean((pts@n+d)**2))),orientation_dot_camera_reference=float(n@orient),relative_monocular_plane_estimate=True,not_metric_measured_contact=True,query_frame=0,query_intersections={})
  for label,pix in queries.items():
   try:w=plane_intersection(pix,cam['K'][0],cam['w2c_A'][0],n,d);row['query_intersections'][label]=dict(pixel_A00=pix,world=w.tolist(),camera_z=float(project(w[None],cam['K'][0],cam['w2c_A'][0])[1][0]))
   except Exception as e:row['query_intersections'][label]=dict(error=repr(e))
  planes[name]=row
 source_anchors={}
 for name,pix in anchors.items():
  x,y=np.round(np.array(pix)*[depth.shape[1]/1280,depth.shape[0]/704]).astype(int);z=float(depth[y,x]);w=unproject(np.array([pix],float),np.array([z]),K,A)[0];source_anchors[name]=dict(pixel_in_reference=pix,Native_camera_z=z,world=w.tolist(),native_cell=[int(x),int(y)],native_confidence=float(conf[y,x]))
 source=N/f'inputs/case{c}_source_video_HF_v1/source_A.mkv';cap=cv2.VideoCapture(str(source));images={}
 for i in range(49):
  ok,im=cap.read();assert ok and hashlib.sha256(im.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i]
  if i in [0,ref]:images[i]=im
 cap.release();ri=images[ref].copy();qi=images[0].copy()
 for name,poly in polys.items():cv2.polylines(ri,[np.array(poly,np.int32)],True,(0,255,255),1);cv2.putText(ri,name,tuple(poly[0]),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,255),1)
 for name,pix in queries.items():cv2.circle(qi,tuple(pix),4,(0,0,255),-1);cv2.putText(qi,name,(max(2,pix[0]-35),min(699,pix[1]+16)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
 cv2.imwrite(str(out/'actual_reference_support_polygons.jpg'),ri,[cv2.IMWRITE_JPEG_QUALITY,97]);cv2.imwrite(str(out/'actual_A00_proposed_mount_queries.jpg'),qi,[cv2.IMWRITE_JPEG_QUALITY,97]);r=dict(case=c,state='ACTUAL_NATIVE_SUPPORT_PROBE_PENDING_GEOMETRY_PILOTS',source_reference_frame=ref,source_mask_review=review,planes=planes,source_anchors=source_anchors,source_RGBA_camera_depth_originals_unchanged=True,query_locations_are_plan_not_placement_approval=True,quality_accepted=False,GPU_used=False,pid=os.getpid(),seconds=time.time()-start);(out/'RESULT.json').write_text(json.dumps(r,indent=2));print(json.dumps(dict(case=c,reference_frame=ref,planes=planes,seconds=time.time()-start)))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--case',required=True);x=a.parse_args();main(x.case)

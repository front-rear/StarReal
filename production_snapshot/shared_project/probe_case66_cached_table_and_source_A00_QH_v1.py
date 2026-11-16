"""One read-only Native table/anchor probe from existing cache, not approval."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,sys,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1')
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
I=N/'inputs/case66_native_CPU_v1'
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
import cv2,numpy as np
from real_geometry import unproject,project,plane_intersection

def main():
 lock=(Q/'logs/case66_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m=json.loads((I/'INPUT_MANIFEST.json').read_text());assert m['case']=='66' and m['case_id']=='rv_fresh_rv_to_siming_v3_7322211-uhd_4096_2160_25'
 assert m['source_video_SHA256']=='2808afaab97c844bd09d29a169cc4f717c1587746b0c28bf5e0dc8f7e9281b1e'
 out=N/'outputs/66/native_table_source_A00_probe_v1';out.mkdir(exist_ok=False);t=time.time()
 ident=dict(case='66',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
 (out/'JOB.json').write_text(json.dumps(dict(ident,state='running_cached_readonly_support_probe',GPU_used=False),indent=2))
 cv2.setNumThreads(1);raw=np.load(I/'native_depth_K_E_subset.npz');cams=np.load(I/'cameras.npz');dep=raw['depth'][0];Kd=raw['K'][0];K=cams['K'][0];A=cams['w2c_A'][0];up=-A[:3,:3].T[:,1]
 confidence=np.load(N/'inputs/CASE66_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'][0]
 polys={'right_clear_surface':[[1030,540],[1157,533],[1158,545],[1030,553]],'right_front_surface':[[1016,544],[1145,539],[1145,548],[1016,556]],'source_right_near_surface':[[1008,533],[1052,531],[1060,539],[1008,542]]}
 planes={}
 for name,poly in polys.items():
  image=np.zeros((704,1280),np.uint8);cv2.fillPoly(image,[np.array(poly,np.int32)],1)
  mask=cv2.resize(image,(dep.shape[1],dep.shape[0]),interpolation=cv2.INTER_NEAREST)>0
  yy,xx=np.where(mask&np.isfinite(dep)&(dep>0)&(confidence>1.))
  points=unproject(np.c_[xx,yy],dep[yy,xx],Kd,A);
  if len(points)<3:planes[name]=dict(polygon_A00=poly,native_points=len(points),error='insufficient actual native points; no invented fit');continue
  mean=points.mean(0);_,sv,vh=np.linalg.svd(points-mean,full_matrices=False);n=vh[-1];n*=1 if n@up>=0 else -1;d=-n@mean
  planes[name]=dict(polygon_A00=poly,native_points=len(points),normal_world=n.tolist(),d=float(d),singular_values=sv.tolist(),rms=float(np.sqrt(np.mean((points@n+d)**2))),normal_dot_camera_up=float(n@up),relative_monocular_plane_estimate=True)
 queries={'source_mount':[970,545],'move_right':[1100,544],'insert_right':[1120,542],'move_replace_right':[1100,550]}
 for row in planes.values():
  if 'error' in row:continue
  n=np.array(row['normal_world']);d=row['d'];row['query_intersections']={}
  for label,pix in queries.items():
   try:
    w=plane_intersection(pix,K,A,n,d);z=float(project(w[None],K,A)[1][0]);row['query_intersections'][label]=dict(pixel=pix,world=w.tolist(),camera_z=z)
   except Exception as ex:row['query_intersections'][label]=dict(error=repr(ex))
 anchors={}
 pixels={'finial_tip':[963,126],'finial_stem':[963,185],'shade_top':[963,254],'shade_left':[850,337],'shade_right':[1079,337],'shade_bottom':[965,373],'candle_left':[921,401],'candle_center':[966,402],'candle_right':[1022,401],'branch_left':[930,449],'branch_right':[1005,449],'stem':[967,479],'base_left':[930,534],'base_center':[970,545],'base_right':[1006,533]}
 for label,pix in pixels.items():
  x,y=np.round([pix[0]*dep.shape[1]/1280,pix[1]*dep.shape[0]/704]).astype(int);z=float(dep[y,x]);world=unproject(np.array([pix],float),np.array([z]),K,A)[0]
  anchors[label]=dict(pixel=pix,native_cell=[int(x),int(y)],Native_camera_z=z,world=world.tolist(),native_confidence=float(confidence[y,x]))
 cap=cv2.VideoCapture(str(N/'inputs/case66_source_video_HF_v1/source_A.mkv'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
 for i in range(1):
  ok,frame=cap.read();assert ok and hashlib.sha256(frame.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i]
 cap.release()
 for name,poly in polys.items():cv2.polylines(frame,[np.array(poly,np.int32)],True,(0,255,255),1);cv2.putText(frame,name,tuple(poly[0]),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
 for name,pix in queries.items():cv2.circle(frame,tuple(pix),4,(0,0,255),-1);cv2.putText(frame,name,(pix[0]-30,pix[1]+18),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)

 image=out/'actual_A00_support_and_queries.jpg';assert cv2.imwrite(str(image),frame,[cv2.IMWRITE_JPEG_QUALITY,98])
 result=dict(case='66',state='readonly_cached_support_probe_NOT_placement_acceptance',planes=planes,source_anchors=anchors,source_SHA256=m['source_video_SHA256'],Native_camera_SHA256=m['camera_original_file_SHA256'],Native_raw_SHA256=m['raw_original_file_SHA256'],source_reference_frame=0,source_base_visible_in_A00=True,hand_occlusion_later_preserved=True,private_observed_mask_repair_required=False,unchanged_Native_K_depth_camera=True,source_original_position_retained=True,quality_accepted=False,GPU_used=False,elapsed_seconds=time.time()-t,actual_image=str(image),actual_image_SHA256=hashlib.sha256(image.read_bytes()).hexdigest())
 (out/'RESULT.json').write_text(json.dumps(result,indent=2));(out/'JOB.json').write_text(json.dumps(dict(ident,state='terminal0_probe_only',exit_code=0,GPU_used=False),indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['planes','source_anchors']}),flush=True)

if __name__=='__main__':main()

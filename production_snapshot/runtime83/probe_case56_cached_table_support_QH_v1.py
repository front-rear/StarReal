"""One read-only Native table/anchor probe from existing cache, not approval."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,sys,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1')
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
I=N/'inputs/case56_native_CPU_v1'
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
import cv2,numpy as np
from real_geometry import unproject,project,plane_intersection

def main():
 assert os.environ.get('ROOT_ORIGINAL_CASE56_COMPONENT_LOCK_HELD')=='1'
 lock=(Q/'logs/case56_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 m=json.loads((I/'INPUT_MANIFEST.json').read_text());assert m['case']=='56' and m['case_id']=='rv_pexels1_6279716-uhd_3840_2160_24'
 assert m['source_video_SHA256']=='3fa23b454bbc97db0006c0072040c2a2a2376266a2e5538d6b162255f76a160e'
 out=N/'outputs/56/native_table_support_probe_v1';out.mkdir(exist_ok=False);t=time.time()
 ident=dict(case='56',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
 (out/'JOB.json').write_text(json.dumps(dict(ident,state='running_cached_readonly_support_probe',GPU_used=False),indent=2))
 cv2.setNumThreads(1);raw=np.load(I/'native_depth_K_E_subset.npz');cams=np.load(I/'cameras.npz');dep=raw['depth'][24];Kd=raw['K'][24];K=cams['K'][24];A=cams['w2c_A'][24];up=-A[:3,:3].T[:,1]
 confidence=np.load(N/'inputs/CASE56_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'][24]
 polys={'left_clear':[[370,523],[480,523],[480,575],[355,575]],'middle_clear':[[550,520],[750,520],[750,585],[550,585]],'source_near':[[820,545],[858,545],[850,585],[820,585]]}
 planes={}
 for name,poly in polys.items():
  image=np.zeros((704,1280),np.uint8);cv2.fillPoly(image,[np.array(poly,np.int32)],1)
  mask=cv2.resize(image,(dep.shape[1],dep.shape[0]),interpolation=cv2.INTER_NEAREST)>0
  yy,xx=np.where(mask&np.isfinite(dep)&(dep>0)&(confidence>1.))
  points=unproject(np.c_[xx,yy],dep[yy,xx],Kd,A);assert len(points)>=3
  mean=points.mean(0);_,sv,vh=np.linalg.svd(points-mean,full_matrices=False);n=vh[-1];n*=1 if n@up>=0 else -1;d=-n@mean
  planes[name]=dict(polygon_A24=poly,native_points=len(points),normal_world=n.tolist(),d=float(d),singular_values=sv.tolist(),rms=float(np.sqrt(np.mean((points@n+d)**2))),normal_dot_camera_up=float(n@up),relative_monocular_plane_estimate=True)
 queries={'source_mount':[914,587],'move_center':[550,590],'insert_left':[390,575],'move_replace_middle':[650,580]}
 for row in planes.values():
  n=np.array(row['normal_world']);d=row['d'];row['query_intersections']={}
  for label,pix in queries.items():
   try:
    w=plane_intersection(pix,K,A,n,d);z=float(project(w[None],K,A)[1][0]);row['query_intersections'][label]=dict(pixel=pix,world=w.tolist(),camera_z=z)
   except Exception as ex:row['query_intersections'][label]=dict(error=repr(ex))
 anchors={}
 pixels={'shade_top':[807,342],'shade_left':[764,386],'shade_right':[855,383],'shade_bottom':[805,406],'elbow':[851,407],'stem_mid':[881,457],'stem_low':[916,509],'base_left':[866,578],'base_center':[914,587],'base_right':[956,583],'cord_exit':[954,596],'cord_middle':[966,650],'cord_frame_end':[945,701]}
 for label,pix in pixels.items():
  x,y=np.round([pix[0]*dep.shape[1]/1280,pix[1]*dep.shape[0]/704]).astype(int);z=float(dep[y,x]);world=unproject(np.array([pix],float),np.array([z]),K,A)[0]
  anchors[label]=dict(pixel=pix,native_cell=[int(x),int(y)],Native_camera_z=z,world=world.tolist(),native_confidence=float(confidence[y,x]))
 cap=cv2.VideoCapture(str(N/'inputs/case56_source_video_HF_v1/source_A.mkv'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
 for i in range(25):
  ok,frame=cap.read();assert ok and hashlib.sha256(frame.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i]
 cap.release()
 for name,poly in polys.items():cv2.polylines(frame,[np.array(poly,np.int32)],True,(0,255,255),1);cv2.putText(frame,name,tuple(poly[0]),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
 for name,pix in queries.items():cv2.circle(frame,tuple(pix),4,(0,0,255),-1);cv2.putText(frame,name,(pix[0]-30,pix[1]+18),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
 for name in ['cord_exit','cord_middle','cord_frame_end']:cv2.circle(frame,tuple(pixels[name]),3,(255,255,0),1)
 image=out/'actual_A24_support_and_queries.jpg';assert cv2.imwrite(str(image),frame,[cv2.IMWRITE_JPEG_QUALITY,98])
 result=dict(case='56',state='readonly_cached_support_probe_NOT_placement_acceptance',planes=planes,source_anchors=anchors,source_SHA256=m['source_video_SHA256'],Native_camera_SHA256=m['camera_original_file_SHA256'],Native_raw_SHA256=m['raw_original_file_SHA256'],source_cord_missing_from_original_SAM_A24_actually_viewed=True,private_observed_cord_repair_pending=True,unchanged_Native_K_depth_camera=True,source_original_position_retained=True,quality_accepted=False,GPU_used=False,elapsed_seconds=time.time()-t,actual_image=str(image),actual_image_SHA256=hashlib.sha256(image.read_bytes()).hexdigest())
 (out/'RESULT.json').write_text(json.dumps(result,indent=2));(out/'JOB.json').write_text(json.dumps(dict(ident,state='terminal0_probe_only',exit_code=0,GPU_used=False),indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['planes','source_anchors']}),flush=True)

if __name__=='__main__':main()

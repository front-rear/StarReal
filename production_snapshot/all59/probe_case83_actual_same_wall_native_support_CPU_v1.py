from pathlib import Path
import sys,json,hashlib,time,fcntl
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');I=N/'inputs/case83_native_CPU_v1'
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I)]
import cv2,numpy as np
from real_geometry import unproject,project,plane_intersection
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lock=(Q/'logs/case83_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);cv2.setNumThreads(1)
 reviewpath=Q/'reviews/CASE83_OBSERVED_SOURCE_SAM49_CURRENT.json';review=json.loads(reviewpath.read_text());assert review['may_compute_private_source_geometry'] and review['reference_frame']==0
 out=Q/'component_cache/83/native_actual_same_wall_support_probe_v1';out.mkdir(exist_ok=False)
 raw=np.load(I/'native_depth_K_E_subset.npz');cam=np.load(I/'cameras.npz');dep=raw['depth'][0];K,A=cam['K'][0],cam['w2c_A'][0]
 confidence=np.load(N/'inputs/CASE83_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'][0]
 polys={'source_wall':[[1040,15],[1210,15],[1210,190],[1040,190]],'clear_wall_above_nightstand':[[500,130],[665,130],[665,285],[500,285]]}
 mask=cv2.imread(str(Path(review['source_mask49'])/'000000.png'),0)>0;exclude=cv2.dilate(mask.astype(np.uint8),np.ones((11,11),np.uint8))
 allpts=[];counts={}
 for name,poly in polys.items():
  roi=np.zeros(mask.shape,np.uint8);cv2.fillPoly(roi,[np.array(poly,np.int32)],1);roi&=(exclude==0)
  valid=cv2.resize(roi,(dep.shape[1],dep.shape[0]),interpolation=cv2.INTER_NEAREST)>0
  valid&=np.isfinite(dep)&(dep>0)&np.isfinite(confidence)&(confidence>1.)
  yy,xx=np.where(valid);pts=unproject(np.c_[xx,yy],dep[yy,xx],raw['K'][0],A);assert len(pts)>=3;allpts.append(pts);counts[name]=len(pts)
 points=np.concatenate(allpts);keep=np.ones(len(points),bool);normal=None
 for _ in range(3):
  mu=points[keep].mean(0);_,sing,vh=np.linalg.svd(points[keep]-mu,full_matrices=False);normal=vh[-1];d=-float(normal@mu);res=np.abs(points@normal+d);mad=np.median(res);keep=res<=max(mad*3,1e-7)
 orient=-A[:3,:3].T[:,2]
 if normal@orient<0:normal=-normal;d=-d
 queries={'source_actual_wallplate':[1084.,55.],'target_clear_wall_above_phone':[625.,170.]}
 hits={k:dict(pixel_A00=p,world=plane_intersection(p,K,A,normal,d).tolist()) for k,p in queries.items()}
 cap=cv2.VideoCapture(str(N/'inputs/case83_source_video_HF_v1/source_A.mkv'));ok,bgr=cap.read();assert ok;cap.release()
 for name,poly in polys.items():cv2.polylines(bgr,[np.array(poly,np.int32)],True,(0,255,255),1)
 for name,p in queries.items():cv2.circle(bgr,tuple(np.array(p,int)),4,(0,0,255),-1);cv2.putText(bgr,name,(int(p[0]-80),int(p[1]+16)),0,.35,(0,0,255),1)
 cv2.imwrite(str(out/'ACTUAL_WALL_AND_TWO_MOUNT_CANDIDATES.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,97])
 result=dict(case='83',state='actual_same_continuous_wall_support_estimate_pending_source_move_pilots',source_review=str(reviewpath),source_review_SHA256=sha(reviewpath),polygons=polys,native_points_by_ROI=counts,robust_fit_points=int(keep.sum()),normal_world=normal.tolist(),d=d,fit_rms=float(np.sqrt(np.mean((points[keep]@normal+d)**2))),wall_normal_facing_camera=True,physical_wall_identity='same actual floral wallpaper wall around original right reading lamp and above center nightstand; target clear of both people, window curtain and objects on table',queries=hits,relative_monocular_plane_estimate=True,not_metric_measured=True,target_is_not_placement_acceptance=True,quality_accepted=False,GPU_used=False)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()

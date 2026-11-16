from pathlib import Path
import sys,json,hashlib,fcntl,datetime,os,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(N/'inputs/case106_native_CPU_v1'),str(N/'code')]
import numpy as np,cv2
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,plane_intersection,project,visibility
from case19_source_shell_reference_QH_v1 import regularized_shell,rotation_on_plane,overlay
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
SPEC={
'14':dict(source=[1025.,531.],target=[1160.,686.],points=dict(clamp_top=[1025,531],clamp_bottom=[1027,545],lower_joint=[988,324],upper_joint=[900,202],shade_top=[802,225],shade_left=[750,276],shade_right=[848,272],shade_bottom=[799,285]),segments=[('clamp_top','lower_joint',18),('lower_joint','upper_joint',18),('upper_joint','shade_top',18),('shade_left','shade_right',100),('shade_top','shade_bottom',100),('clamp_top','clamp_bottom',20)],destination_scope='Private front-right visible desk lip candidate, outside papers and mug; clamp underside can be partly outside image. Must actually review support/orientation and all3 images before adoption.'),
'27':dict(source=[304.,377.],target=[370.,383.],points=dict(clamp_top=[304,377],clamp_bottom=[304,412],elbow=[358,165],head_joint=[520,135],shade_top=[556,110],shade_left=[520,183],shade_right=[593,183],shade_bottom=[557,191]),segments=[('clamp_bottom','clamp_top',22),('clamp_top','elbow',18),('elbow','head_joint',18),('head_joint','shade_top',20),('shade_top','shade_left',80),('shade_top','shade_right',80),('shade_left','shade_right',80)],destination_scope='Private translation along actual front-left workbench lip; avoid red press and hands. Clamp stays at edge rather than placing a clamp in table center. Must review body/arm and bag clearance before adoption.')
}
def one(c):
 start=time.time();cfg=SPEC[c];I=N/f'inputs/case{c}_native_CPU_v1';source=N/f'inputs/case{c}_source_video_HF_v1/source_A.mkv'
 out=Q/f'component_cache/{c}/whole_source_rigid_edge_pilot3_v1';out.mkdir(parents=True,exist_ok=False);camsha=sha(I/'cameras.npz');rawsha=sha(I/'native_depth_K_E_subset.npz');sourcesha=sha(source)
 review=json.loads((Q/f'reviews/CASE{c}_OBSERVED_SOURCE_SAM49_CURRENT.json').read_text());assert review['may_compute_private_source_geometry'] and review['reference_frame']==0
 cams=np.load(I/'cameras.npz');raw=np.load(I/'native_depth_K_E_subset.npz');conf=np.load(N/f'inputs/CASE{c}_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'];plan=json.loads((I/'HISTORICAL_GEOMETRY_PLAN.json').read_text());plane=plan['support'];n=np.array(plane['normal_world']);d=plane['d'];K,A=cams['K'][0],cams['w2c_A'][0];P=plane_intersection(cfg['source'],K,A,n,d);T=plane_intersection(cfg['target'],K,A,n,d)
 forward=A[:3,:3].T[:,2];cn=forward-(forward@n)*n;cn/=np.linalg.norm(cn);cd=-cn@P;anchors={}
 for key,pix in cfg['points'].items():
  w=plane_intersection(pix,K,A,cn,cd);anchors[key]=np.r_[pix,float(project(w[None],K,A)[1][0])]
 mask=cv2.imread(str(Path(review['source_mask49'])/'000000.png'),0)>0;v,f,_=regularized_shell(mask,K,A,anchors,cfg['segments'])
 # A table clamp extends below the table: retain its observed underside, never flatten it to a tabletop/base.
 rotation=rotation_on_plane(n,A[:3,:3].T);center,size,basis=bbox_for_mesh(v,rotation);mv=v+T-P
 np.savez_compressed(out/'geometry.npz',source_vertices=v,source_faces=f,source_center=center,source_size=size,source_rotation=basis,source_mount=P,move_vertices=mv,move_faces=f,move_mount=T,source_unobserved_completion_vertex=np.zeros(len(v),bool),K=cams['K'],w2c_A=cams['w2c_A'],w2c_B=cams['w2c_B'])
 cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);raster=Rasterizer();rows=[];inputproof=json.loads((I/'INPUT_MANIFEST.json').read_text())
 for i in range(49):
  ok,rgb=cap.read();assert ok and hashlib.sha256(rgb.tobytes()).hexdigest()==inputproof['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  obs=cv2.imread(str(Path(review['source_mask49'])/f'{i:06d}.png'),0)>0;small=cv2.resize(obs.astype(np.float32),(raw['depth'].shape[2],raw['depth'].shape[1]),interpolation=cv2.INTER_AREA)>0;small=cv2.dilate(small.astype(np.uint8),np.ones((3,3),np.uint8))>0
  valid=np.isfinite(conf[i])&(conf[i]>1)&~small;sv,sf=depth_surface(raw['depth'][i],raw['K'][i],cams['w2c_A'][i],valid_pixels=valid,max_neighbor_ratio=1.15);scene=raster.depth(sv,sf,cams['K'][i],cams['w2c_A'][i])
  for role,verts in [('source',v),('rigid_move_edge',mv)]:
   dep=raster.depth(verts,f,cams['K'][i],cams['w2c_A'][i]);hit=np.isfinite(dep);layers=visibility(dep,scene);left=rgb.copy();left[hit]=np.rint(.4*left[hit]+.6*np.array([255,0,255])).astype(np.uint8);right=overlay(rgb,layers)
   for im,label in [(left,'whole fixed geometry'),(right,'Native visibility / cyan uncertainty separate')]:cv2.putText(im,f'Case{c} {role} A{i:02}: {label}; NOT editedRGB',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
   p=out/f'{role}_whole_vs_visibility_A{i:02}.jpg';cv2.imwrite(str(p),np.concatenate([left,right],1),[cv2.IMWRITE_JPEG_QUALITY,96]);rows.append(dict(role=role,frame=i,path=str(p),SHA256=sha(p),amodal_pixels=int(hit.sum()),visibility={k:int(z.sum()) for k,z in layers.items()}))
 cap.release();assert camsha==sha(I/'cameras.npz') and rawsha==sha(I/'native_depth_K_E_subset.npz') and sourcesha==sha(source)
 result=dict(case=c,state='terminal0_private_pilot3_pending_actual_support_orientation_review',source_mask_revision=review['source_mask49'],source_contact_A00=cfg['source'],candidate_move_contact_A00=cfg['target'],destination_scope=cfg['destination_scope'],support_plane_estimate_retained=plane,source_front_plane_and_back_thickness_estimated=True,clamp_underside_retained_not_flattened=True,whole_observed_mask_geometry_no_hidden_front_painting=True,geometry=str(out/'geometry.npz'),geometry_SHA256=sha(out/'geometry.npz'),actual_pilots=rows,rigid_translation_only=True,pure_move_size_orientation_appearance_preserved=True,all49_source_RGB_SHA_checked=True,all_Native_originals_SHA_unchanged=True,quality_accepted=False,approved_for_all49=False,new_complete_groups=0,GPU_used=False,seconds=time.time()-start)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['actual_pilots','support_plane_estimate_retained']}),flush=True)
def main():
 cv2.setNumThreads(1);lock=(Q/'logs/whole_clamp14_27_private_pilots.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for c in ['14','27']:one(c)
if __name__=='__main__':main()

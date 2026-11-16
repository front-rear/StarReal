from pathlib import Path
import sys,json,hashlib,fcntl,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');sys.path.insert(0,str(N/'runtime/qh_cpu_py39_v1'))
import cv2,numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cv2.setNumThreads(1);lock=(Q/'logs/case83_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=Q/'component_cache/83/observed_actual_RGB_wire_refine_pilot3_v7';out.mkdir(exist_ok=False);I=N/'inputs/case83_native_CPU_v1';manifest=json.loads((I/'INPUT_MANIFEST.json').read_text());cap=cv2.VideoCapture(str(N/'inputs/case83_source_video_HF_v1/source_A.mkv'));rows=[]
 # These are observed visible cord-center guides, not final masks or hidden completion.
 points=np.array([[1091,75],[1088,86],[1084,97],[1080,105],[1078,115],[1075,124],[1074,131],[1070,147],[1067,164],[1062,182],[1058,198],[1053,215]],np.int32)
 for i in range(49):
  ok,bgr=cap.read();assert ok and hashlib.sha256(bgr.tobytes()).hexdigest()==manifest['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  original=cv2.imread(str(I/'original_mask49'/f'{i:06d}.png'),0)>0;sam=cv2.imread(str(Q/'component_cache/83/observed_visible_parts_SAM_pilot3_v4'/f'{i:06d}_parts.png'),0)>0;body=sam.copy();body[79:]=False
  corridor=np.zeros((704,1280),np.uint8);cv2.polylines(corridor,[points],False,1,9);corridor[99:128,1069:1088]=1
  x0,y0,x1,y1=1046,65,1100,225;crop=bgr[y0:y1,x0:x1].copy();seed=np.full(crop.shape[:2],cv2.GC_BGD,np.uint8);roi=corridor[y0:y1,x0:x1]>0;seed[roi]=cv2.GC_PR_FGD;gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY);used=[]
  for x,y in points:
   patch=gray[y-1:y+2,x-3:x+4];yy,xx=np.unravel_index(int(np.argmin(patch)),patch.shape);px,py=int(x-3+xx),int(y-1+yy)
   if x0<=px<x1 and y0<=py<y1 and corridor[py,px]:seed[py-y0,px-x0]=cv2.GC_FGD;used.append([px,py])
  bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64);cv2.grabCut(crop,seed,None,bg,fg,8,cv2.GC_INIT_WITH_MASK);wire=np.zeros_like(original);wire[y0:y1,x0:x1]=(seed==cv2.GC_FGD)|(seed==cv2.GC_PR_FGD)
  union=original|body|wire;added=union&~original
  for name,a in [('wire',wire),('body',body),('union',union),('additions',added)]:cv2.imwrite(str(out/f'{i:06d}_{name}.png'),a.astype(np.uint8)*255)
  imgs=[]
  for mask,color in [(np.zeros_like(original),[0,0,0]),(union,[0,255,0]),(added,[255,0,255])]:
   im=bgr.copy();im[mask]=np.rint(.5*im[mask]+.5*np.array(color)).astype(np.uint8);imgs.append(im[10:245,1025:1185])
  im=cv2.resize(np.concatenate(imgs,1),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST);cv2.imwrite(str(out/f'A{i:02d}_actual_RGB_union_additions.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,96]);rows.append(dict(frame=i,union_SHA256=sha(out/f'{i:06d}_union.png'),actual_pixel_seed_coordinates=used,wire_pixels=int(wire.sum()),body_pixels=int(body.sum()),original_pixels_retained=True))
 cap.release();r=dict(case='83',state='private_SAM_body_and_actual_RGB_wire_refinement_pending_actual_review',method='SAM observed plate/arm/shade plus local RGB graph-cut wire refinement. Human visible guide only bounds search; final foreground comes from native RGB classification. No final drawn line, no hidden cord/hair completion.',parts=rows,original_RGB_SAM_K_E_depth_unchanged=True,quality_accepted=False,GPU_used=False);(out/'RESULT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
if __name__=='__main__':main()

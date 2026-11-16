from pathlib import Path
import sys,json,hashlib,time,fcntl,os
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
sys.path.insert(0,str(N/'runtime/qh_cpu_py39_v1'))
import cv2,numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def geometry_masks():
 a=np.zeros((704,1280),np.uint8)
 polygons=[
 [[1079,40],[1089,40],[1092,42],[1092,52],[1090,59],[1087,66],[1084,71],[1078,72],[1076,69],[1076,47]],
 [[1126,29],[1132,29],[1137,31],[1137,38],[1134,42],[1125,44],[1120,47],[1118,46],[1121,38],[1123,32]],
 [[1127,23],[1131,23],[1132,25],[1130,29],[1126,30],[1125,27]],
 [[1082,100],[1086,100],[1090,102],[1089,106],[1087,111],[1085,120],[1085,125],[1082,127],[1075,126],[1073,124],[1075,116],[1077,108],[1079,102]]
 ]
 for p in polygons:cv2.fillPoly(a,[np.array(p,np.int32)],1)
 lines=[([[1059,53],[1082,53],[1092,54],[1099,52],[1104,52],[1111,52],[1117,53],[1120,50]],2),([[1093,49],[1101,47],[1108,43],[1116,35],[1123,30],[1127,30]],2),([[1138,32],[1140,44]],2),([[1119,47],[1143,47]],2)]
 for p,width in lines:cv2.polylines(a,[np.array(p,np.int32)],False,1,width)
 return a>0,polygons,lines
def rgb_wire(bgr,guide):
 gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32)
 # A visually annotated guide constrains search. A band is NOT the final mask.
 g=np.array(guide,np.float32);ys=np.arange(int(g[0,1]),int(g[-1,1])+1);gx=np.interp(ys,g[:,1],g[:,0]);xs=np.rint(gx[:,None]+np.arange(-3,4)).astype(int)
 contrast=np.empty(xs.shape,np.float32)
 for row,y in enumerate(ys):
  for k,x in enumerate(xs[row]):
   bg=np.median(np.r_[gray[y,x-9:x-4],gray[y,x+5:x+10]])
   contrast[row,k]=gray[y,x]-bg
 energy=contrast+.55*(xs-gx[:,None])**2
 costs=energy[0].copy();prev=[]
 for row in range(1,len(ys)):
  transition=costs[:,None]+.9*(xs[row-1,:,None]-xs[row,None,:])**2
  back=transition.argmin(0);costs=energy[row]+transition[back,np.arange(7)];prev.append(back)
 k=int(costs.argmin());path=[k]
 for back in prev[::-1]:k=int(back[k]);path.append(k)
 path=path[::-1];out=np.zeros(gray.shape,bool);coords=[];uncertain=[]
 for row,(y,k) in enumerate(zip(ys,path)):
  x=int(xs[row,k]);coords.append([x,int(y)])
  # Native visible dark center and dark edge pixels only; uncertain rows are kept separate.
  if contrast[row,k]>-3.5:uncertain.append(int(y));continue
  out[y,x]=True
  for dx in [-2,-1,1,2]:
   xx=x+dx;bg=np.median(np.r_[gray[y,xx-9:xx-4],gray[y,xx+5:xx+10]])
   if gray[y,xx]-bg < -5.5 and abs(dx)<=1:out[y,xx]=True
 return out,coords,uncertain
def main():
 lock=(Q/'logs/case83_geometry_component.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time();cv2.setNumThreads(1)
 I=N/'inputs/case83_native_CPU_v1';m=json.loads((I/'INPUT_MANIFEST.json').read_text());source=N/'inputs/case83_source_video_HF_v1/source_A.mkv';assert sha(source)==m['source_video_SHA256']
 out=Q/'component_cache/83/observed_native_RGB_annotation_pilot3_v11';out.mkdir(exist_ok=False)
 body,polys,lines=geometry_masks()
 guides=[[[1092,57],[1090,72],[1088,86],[1086,96],[1084,101]],[[1078,127],[1075,135],[1072,145],[1069,160],[1065,175],[1062,190],[1058,205],[1053,216],[1051,220]]]
 cap=cv2.VideoCapture(str(source));rows=[]
 for i in range(49):
  ok,bgr=cap.read();assert ok and hashlib.sha256(bgr.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i]
  if i not in [0,24,48]:continue
  old=cv2.imread(str(I/'original_mask49'/f'{i:06d}.png'),0)>0;wire=np.zeros(old.shape,bool);traces=[]
  for g in guides:
   w,coords,uncertain=rgb_wire(bgr,g);wire|=w;traces.append(dict(actual_RGB_min_cost_path=coords,rows_without_sufficient_observed_dark_contrast=uncertain))
  union=old|body|wire
  for name,arr in [('body',body),('wire',wire),('union',union)]:cv2.imwrite(str(out/f'{i:06d}_{name}.png'),arr.astype(np.uint8)*255)
  imgs=[]
  for mask,color in [(np.zeros_like(old),(0,0,0)),(union,(0,255,0)),(union&~old,(255,0,255))]:
   im=bgr.copy();im[mask]=np.rint(im[mask]*.5+.5*np.array(color)).astype(np.uint8);imgs.append(im[10:245,1025:1185])
  detail=cv2.resize(np.concatenate(imgs,1),None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
  cv2.putText(detail,f'83 A{i:02d}: actualRGB / SAM shade+visible RGB annotations / added visible parts; not accepted',(3,13),0,.36,(0,255,255),1)
  cv2.imwrite(str(out/f'A{i:02d}_actual_RGB_annotation.jpg'),detail,[cv2.IMWRITE_JPEG_QUALITY,97])
  if i==0:cv2.imwrite(str(out/'SOURCE_A00_FULL.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
  rows.append(dict(frame=i,union_SHA256=sha(out/f'{i:06d}_union.png'),original_SAM_pixels_preserved=True,body_pixels=int(body.sum()),wire_pixels=int(wire.sum()),traces=traces))
 cap.release()
 result=dict(case='83',state='private_actual_visible_native_RGB_manual_parts_and_RGB_wire_DP_pilot3_pending_actual_review',method='Original observed SAM shade unchanged. Visible wallplate/arm/head/switch boundaries visually annotated against native RGB coordinate image; cord foreground derived from local RGB contrast along visually observed corridor. Not a SAM inference claim for hand annotations. No hidden cord completion.',polygons_native_xy=polys,lines_native_xy_width=lines,wire_search_guides=guides,source_SHA256=sha(source),original_RGB_SAM_K_E_depth_unchanged=True,GPU_used=False,quality_accepted=False,records=rows,seconds=time.time()-start)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(out=str(out),pilots=3,seconds=result['seconds'],GPU_used=False)))
if __name__=='__main__':main()

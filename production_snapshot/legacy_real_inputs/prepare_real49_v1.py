from pathlib import Path
import cv2,numpy as np,json,subprocess,hashlib,math
B=Path('/Volumes/T9/LampAssetPipeline_handoff/real_video_inputs_20261102_v1');S=Path('/Users/fangrui/Downloads/real')
for i,p in enumerate(sorted(S.glob('*.mp4')),1):
 out=B/f'case{i:02d}_{p.name.split("-")[0]}';(out/'source').mkdir(parents=True,exist_ok=True)
 cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);sw=int(cap.get(3));sh=int(cap.get(4));start=2.0 if i==1 else .5
 ids=[int(round(start*fps+j*fps/10)) for j in range(49)]
 scale=max(640/sw,352/sh);rw=math.ceil(sw*scale);rh=math.ceil(sh*scale);cx=(rw-640)//2;cy=(rh-352)//2
 A=np.array([[rw/sw,0,-cx],[0,rh/sh,-cy],[0,0,1.]])
 cap.set(cv2.CAP_PROP_POS_FRAMES,ids[0]);cur=ids[0];j=0
 while j<49:
  ok,im=cap.read()
  if not ok:raise RuntimeError((p,j))
  if cur==ids[j]:
   if j==0:cv2.imwrite(str(out/'first_frame_original.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,98])
   im=cv2.resize(im,(rw,rh),interpolation=cv2.INTER_AREA)[cy:cy+352,cx:cx+640]
   cv2.imwrite(str(out/'source'/f'{j:06d}.png'),im,[cv2.IMWRITE_PNG_COMPRESSION,1]);j+=1
  cur+=1
 cap.release()
 subprocess.run(['ffmpeg','-y','-loglevel','error','-framerate','10','-i',str(out/'source/%06d.png'),'-frames:v','49','-c:v','libx264','-crf','0','-pix_fmt','yuv444p',str(out/'source.mp4')],check=True)
 meta={'case':i,'source_original':str(p),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_size_wh':[sw,sh],'source_fps':fps,'frame_indices':ids,'timestamps_seconds':[x/fps for x in ids],'output_fps':10,'output_frame_count':49,'output_wh':[640,352],'resize_crop_matrix':A.tolist(),'sampling':'nearest original frames; no interpolation or repeated end frames','crop':'aspect preserving resize then center crop','geometry_status':'pending_estimation','target_selection':'pending_visual'}
 (out/'sampling.json').write_text(json.dumps(meta,indent=2));print(out.name,flush=True)

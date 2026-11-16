import os,sys,json,shutil,tempfile,gc
from pathlib import Path
import numpy as np,cv2,torch
ROOT=Path('/root/autodl-tmp/kimi_realcase_20261109');SAM2=Path('/root/autodl-tmp/real_video_condition_repro_20261105_v1/tools/sam2');sys.path.insert(0,str(SAM2))
from sam2.build_sam import build_sam2_video_predictor
CONFIG={
 'rv_pexels4_14904939_3840_2160_25fps':[
  {'box':[75,130,195,228],'positive':[[125,180],[155,205]],'negative':[[55,180],[210,180]]},
  {'box':[120,225,173,275],'positive':[[145,250]],'negative':[[145,305],[110,295]]}],
 'rv_pexels4_854693-hd_1920_1080_30fps':[
  {'box':[590,110,790,703],'positive':[[680,220],[690,320],[680,500],[680,650]],'negative':[[650,50],[500,300],[850,400]]}]
}
case=sys.argv[1];parts=CONFIG[case];cd=ROOT/'cases'/case;out=ROOT/'samples'/case/'step8_sam2_refined';out.mkdir(parents=True,exist_ok=False)
tmp=Path(tempfile.mkdtemp(prefix='kimi_sam2_refine_',dir='/root/autodl-tmp'))
for i in range(49):
 im=cv2.imread(str(cd/'frames'/f'{i:03}.png'));assert im is not None;cv2.imwrite(str(tmp/f'{i:05}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,98])
pred=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(SAM2/'checkpoints/sam2.1_hiera_large.pt'),device='cuda',vos_optimized=False);state=pred.init_state(video_path=str(tmp))
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for oid,p in enumerate(parts,1):
  pts=np.asarray(p['positive']+p['negative'],np.float32);labels=np.asarray([1]*len(p['positive'])+[0]*len(p['negative']),np.int32)
  pred.add_new_points_or_box(state,frame_idx=24,obj_id=oid,box=np.asarray(p['box'],np.float32),points=pts,labels=labels)
 masks={}
 for rev in (True,False):
  for i,ids,logits in pred.propagate_in_video(state,start_frame_idx=24,reverse=rev):
   masks[i]=cv2.resize(((logits[:,0]>0).any(dim=0)).cpu().numpy().astype(np.uint8),(1280,704),interpolation=cv2.INTER_NEAREST)>0
assert len(masks)==49
md=out/'mask49';md.mkdir();rows=[]
for i in range(49):
 mm=masks[i].astype(np.uint8)*255;cv2.imwrite(str(md/f'{i:06}.png'),mm);n,lab,stats,cents=cv2.connectedComponentsWithStats((mm>0).astype(np.uint8),8);rows.append({'frame':i,'area':int((mm>0).sum()),'components':n-1,'empty':not (mm>0).any()})
 if i in (0,24,48):
  im=cv2.imread(str(cd/'frames'/f'{i:03}.png'));sel=mm>0;im[sel]=np.rint(.55*im[sel]+.45*np.array([0,255,0])).astype(np.uint8);cv2.imwrite(str(out/f'overlay_{i:06}.png'),im)
(out/'SAM2.json').write_text(json.dumps({'case':case,'model':'sam2.1_hiera_large','prompt_frame':24,'parts':parts,'visible_only':True,'review_state':'visual_review_pending','per_frame':rows},indent=2))
del pred,state;torch.cuda.empty_cache();gc.collect();shutil.rmtree(tmp,ignore_errors=True)
print(json.dumps({'case':case,'areas':[r['area'] for r in rows],'components':[r['components'] for r in rows]}),flush=True)

import os,sys,json,shutil,tempfile,gc
from pathlib import Path
import numpy as np,cv2,torch
ROOT=Path('/root/autodl-tmp/kimi_realcase_20261109');SAM2=Path('/root/autodl-tmp/real_video_condition_repro_20261105_v1/tools/sam2');sys.path.insert(0,str(SAM2))
from sam2.build_sam import build_sam2_video_predictor
CONFIG={
 'rv_1-16-v2_c2a91433-10f2-4456-a42e-':[{'box':[30,120,230,530],'positive':[[110,220],[110,340],[110,450]],'negative':[[300,300],[900,300]]}],
 'rv_pexels1_5199857-uhd_3840_2160_25':[{'box':[250,10,520,500],'positive':[[400,100],[340,280],[350,450]],'negative':[[450,520],[550,200]]}],
 'rv_pexels4_13596340_1920_1080_30fps':[{'box':[400,300,520,650],'positive':[[450,430],[450,550],[450,630]],'negative':[[550,500],[300,550]]}],
 'rv_pexels2_14888447_3840_2160_50fps':[{'box':[200,50,550,450],'positive':[[430,130],[300,200],[300,390]],'negative':[[600,350],[450,350]]}],
 'rv_non_limbertian_8479096-hd_1920_1080_24f':[{'box':[0,0,250,703],'positive':[[100,150],[80,400],[80,550]],'negative':[[350,250],[600,300]]}],
 'rv_early_AjJO3_hu7lk_896':[{'box':[600,80,800,550],'positive':[[700,200],[700,350],[700,480]],'negative':[[500,400],[900,400]]}],
 'rv_post_2_12218822_3840_2160_30fps':[{'box':[100,200,350,703],'positive':[[200,350],[200,500],[200,650]],'negative':[[400,300],[600,500]]}],
 'rv_pexels2_14034896-hd_1920_1080_24':[{'box':[100,100,400,703],'positive':[[200,250],[250,450],[250,620]],'negative':[[500,150],[400,400]]}],
 'rv_pexels4_11839713-uhd_3840_2160_2':[{'box':[800,0,1050,703],'positive':[[900,100],[900,400],[900,650]],'negative':[[500,500],[950,350]]}]
}
case=sys.argv[1];parts=CONFIG[case];cd=ROOT/'cases'/case;out=ROOT/'samples'/case/'step8_sam2_refined';out.mkdir(parents=True,exist_ok=False)
tmp=Path(tempfile.mkdtemp(prefix='kimi_sam2_reuse_',dir='/root/autodl-tmp'))
for i in range(49):
 im=cv2.imread(str(cd/'frames'/f'{i:03}.png'));assert im is not None;cv2.imwrite(str(tmp/f'{i:05}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,98])
pred=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(SAM2/'checkpoints/sam2.1_hiera_large.pt'),device='cuda',vos_optimized=False);state=pred.init_state(video_path=str(tmp))
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for oid,p in enumerate(parts,1):
  pts=np.asarray(p['positive']+p['negative'],np.float32);labels=np.asarray([1]*len(p['positive'])+[0]*len(p['negative']),np.int32);pred.add_new_points_or_box(state,frame_idx=24,obj_id=oid,box=np.asarray(p['box'],np.float32),points=pts,labels=labels)
 masks={}
 for rev in (True,False):
  for i,ids,logits in pred.propagate_in_video(state,start_frame_idx=24,reverse=rev):masks[i]=cv2.resize(((logits[:,0]>0).any(dim=0)).cpu().numpy().astype(np.uint8),(1280,704),interpolation=cv2.INTER_NEAREST)>0
assert len(masks)==49
md=out/'mask49';md.mkdir();rows=[]
for i in range(49):
 mm=masks[i].astype(np.uint8)*255;cv2.imwrite(str(md/f'{i:06}.png'),mm);n,lab,stats,cents=cv2.connectedComponentsWithStats((mm>0).astype(np.uint8),8);rows.append({'frame':i,'area':int((mm>0).sum()),'components':n-1,'empty':not (mm>0).any()})
 if i in (0,24,48):
  im=cv2.imread(str(cd/'frames'/f'{i:03}.png'));sel=mm>0;im[sel]=np.rint(.55*im[sel]+.45*np.array([0,255,0])).astype(np.uint8);cv2.imwrite(str(out/f'overlay_{i:06}.png'),im)
(out/'SAM2.json').write_text(json.dumps({'case':case,'model':'sam2.1_hiera_large','prompt_frame':24,'parts':parts,'visible_only':True,'review_state':'visual_review_pending','per_frame':rows},indent=2))
del pred,state;torch.cuda.empty_cache();gc.collect();shutil.rmtree(tmp,ignore_errors=True)
print(json.dumps({'case':case,'areas':[r['area'] for r in rows],'components':[r['components'] for r in rows]}),flush=True)

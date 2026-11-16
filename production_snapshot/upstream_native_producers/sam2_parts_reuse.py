import os,sys,json,shutil,tempfile,gc
from pathlib import Path
import numpy as np,cv2,torch
ROOT=Path('/root/autodl-tmp/kimi_realcase_20261109');SAM2=Path('/root/autodl-tmp/real_video_condition_repro_20261105_v1/tools/sam2');sys.path.insert(0,str(SAM2))
from sam2.build_sam import build_sam2_video_predictor
PARTS={
 'rv_1-16-v2_c2a91433-10f2-4456-a42e-':[{'box':[60,300,170,500],'positive':[[110,330],[110,420]],'negative':[[250,450]]}],
 'rv_pexels1_5199857-uhd_3840_2160_25':[{'box':[250,140,390,430],'positive':[[330,250],[340,380]],'negative':[[450,400]]},{'box':[280,400,420,500],'positive':[[350,450]],'negative':[[450,480]]}],
 'rv_pexels4_13596340_1920_1080_30fps':[{'box':[410,400,500,660],'positive':[[450,520],[450,620]],'negative':[[520,550]]}],
 'rv_pexels2_14888447_3840_2160_50fps':[{'box':[200,100,400,400],'positive':[[300,200],[300,320]],'negative':[[250,100]]},{'box':[220,350,400,450],'positive':[[300,400]],'negative':[[400,400]]}],
 'rv_non_limbertian_8479096-hd_1920_1080_24f':[{'box':[20,250,150,703],'positive':[[80,400],[80,550],[80,650]],'negative':[[300,300]]}],
 'rv_early_AjJO3_hu7lk_896':[{'box':[650,250,750,550],'positive':[[700,350],[700,480]],'negative':[[550,400]]}],
 'rv_post_2_12218822_3840_2160_30fps':[{'box':[120,300,280,703],'positive':[[200,450],[200,600]],'negative':[[300,450]]}],
 'rv_pexels2_14034896-hd_1920_1080_24':[{'box':[150,300,330,703],'positive':[[250,450],[250,600]],'negative':[[350,450]]}],
 'rv_pexels4_11839713-uhd_3840_2160_2':[{'box':[800,200,1000,650],'positive':[[900,400],[900,600]],'negative':[[950,350]]}]
}
case=sys.argv[1];parts=PARTS[case];cd=ROOT/'cases'/case;sd=ROOT/'samples'/case;out=sd/'step8_sam2_parts';out.mkdir(parents=True,exist_ok=False)
old=cv2.imread(str(sd/'step8_sam2_candidate/mask49/000024.png'),0);assert old is not None
ys,xs=np.where(old>0);oldbox=np.array([xs.min(),ys.min(),xs.max(),ys.max()],np.float32);oldpoint=[[float(xs.mean()),float(ys.mean())]]
tmp=Path(tempfile.mkdtemp(prefix='kimi_sam2_parts_',dir='/root/autodl-tmp'))
for i in range(49):
 im=cv2.imread(str(cd/'frames'/f'{i:03}.png'));assert im is not None;cv2.imwrite(str(tmp/f'{i:05}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,98])
pred=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(SAM2/'checkpoints/sam2.1_hiera_large.pt'),device='cuda',vos_optimized=False);state=pred.init_state(video_path=str(tmp))
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 pred.add_new_points_or_box(state,frame_idx=24,obj_id=1,box=oldbox,points=np.asarray(oldpoint,np.float32),labels=np.ones(1,np.int32))
 for oid,p in enumerate(parts,2):
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
(out/'SAM2.json').write_text(json.dumps({'case':case,'model':'sam2.1_hiera_large','prompt_frame':24,'old_object_box':oldbox.tolist(),'old_object_positive':oldpoint,'parts':parts,'visible_only':True,'review_state':'visual_review_pending','per_frame':rows},indent=2))
del pred,state;torch.cuda.empty_cache();gc.collect();shutil.rmtree(tmp,ignore_errors=True)
print(json.dumps({'case':case,'areas':[r['area'] for r in rows],'components':[r['components'] for r in rows]}),flush=True)

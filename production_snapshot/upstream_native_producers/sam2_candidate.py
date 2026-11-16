import os,sys,json,shutil,tempfile,gc
from pathlib import Path
import numpy as np,cv2,torch
ROOT=Path('/root/autodl-tmp/kimi_realcase_20261109')
SAM2=Path('/root/autodl-tmp/real_video_condition_repro_20261105_v1/tools/sam2')
sys.path.insert(0,str(SAM2))
from sam2.build_sam import build_sam2_video_predictor
case=sys.argv[1];cd=ROOT/'cases'/case;out=ROOT/'samples'/case/'step8_sam2_candidate';out.mkdir(parents=True,exist_ok=False)
frames=[cd/'frames'/f'{i:03}.png' for i in range(49)]
tmp=Path(tempfile.mkdtemp(prefix='kimi_sam2_',dir='/root/autodl-tmp'))
for i,f in enumerate(frames):
    im=cv2.imread(str(f));assert im is not None and im.shape[:2]==(704,1280)
    cv2.imwrite(str(tmp/f'{i:05}.jpg'),im,[cv2.IMWRITE_JPEG_QUALITY,98])
pframe=24;m=cv2.imread(str(cd/'segmentation/lamp0'/f'{pframe:03}.png'),0);assert m is not None
num,lab,stats,cents=cv2.connectedComponentsWithStats((m>0).astype(np.uint8),8)
areas=stats[1:,cv2.CC_STAT_AREA];keep=[]
for j,a in enumerate(areas,1):
    if a>=4 and a>=max(4,float(areas.max())*.002):keep.append(j)
if not keep:raise RuntimeError('empty prompt mask')
points=[]
for j in keep:
    ys,xs=np.where(lab==j);idx=np.argmin((xs-cents[j][0])**2+(ys-cents[j][1])**2);points.append([float(xs[idx]),float(ys[idx])])
ys,xs=np.where(m>0);box=np.array([xs.min(),ys.min(),xs.max(),ys.max()],np.float32)
# Limit very noisy prompt sets without discarding the principal fixture components.
if len(points)>24:points=sorted(points,key=lambda p:(p[0]-cents[keep[0]][0])**2+(p[1]-cents[keep[0]][1])**2)[:24]
pred=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_l.yaml',str(SAM2/'checkpoints/sam2.1_hiera_large.pt'),device='cuda',vos_optimized=False)
state=pred.init_state(video_path=str(tmp))
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    pred.add_new_points_or_box(state,frame_idx=pframe,obj_id=1,box=box,points=np.asarray(points,np.float32),labels=np.ones(len(points),np.int32))
    masks={}
    for rev in (True,False):
        for i,ids,logits in pred.propagate_in_video(state,start_frame_idx=pframe,reverse=rev):
            t=(logits[:,0]>0).any(dim=0).cpu().numpy().astype(np.uint8)
            masks[i]=cv2.resize(t,(1280,704),interpolation=cv2.INTER_NEAREST)>0
assert len(masks)==49,sorted(masks)
md=out/'mask49';md.mkdir();rows=[]
for i in range(49):
    mm=masks[i].astype(np.uint8)*255;cv2.imwrite(str(md/f'{i:06}.png'),mm)
    ref=cv2.imread(str(cd/'segmentation/lamp0'/f'{i:03}.png'),0)>0;a=mm>0;inter=np.count_nonzero(a&ref);union=np.count_nonzero(a|ref)
    rows.append({'frame':i,'sam_area':int(a.sum()),'dataset_area':int(ref.sum()),'iou_vs_prompt_mask':float(inter/max(1,union)),'empty':not a.any()})
for i in (0,24,48):
    im=cv2.imread(str(frames[i]));sel=masks[i];im[sel]=np.rint(.55*im[sel]+.45*np.array([0,255,0])).astype(np.uint8);cv2.imwrite(str(out/f'overlay_{i:06}.png'),im)
rec={'case':case,'model':'sam2.1_hiera_large','prompt_frame':pframe,'prompt_box_xyxy':box.tolist(),'positive_points':points,'negative_points':[],'prompt_source':'dataset mask multi-component positive points; candidate, not user confirmation','components_prompted':len(keep),'per_frame':rows,'review_state':'visual_review_pending','visible_only':True,'occluded_geometry':'not included in visible mask'}
(out/'SAM2.json').write_text(json.dumps(rec,indent=2))
del pred,state;torch.cuda.empty_cache();gc.collect();shutil.rmtree(tmp,ignore_errors=True)
print(json.dumps({'case':case,'empty_frames':[r['frame'] for r in rows if r['empty']],'median_iou':float(np.median([r['iou_vs_prompt_mask'] for r in rows])),'output':str(out)}),flush=True)

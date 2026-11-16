"""Independent original archive/reader checks, then review pages. Never autoapprove."""
from pathlib import Path
import concurrent.futures,hashlib,io,json,os,sys,tarfile,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';N=Path('/tmp/starreal_uid1007_case19_cpu_v1/reference_completion_20261115_v1');sys.path[:0]=[str(Q/'runtime/qh_cpu_py39_v1'),str(T/'code')]
import cv2,numpy as np
from remove_condition_reader import load_remove_fields
cv2.setNumThreads(1)
def sha(b):return hashlib.sha256(b).hexdigest()
def arrays(b):
 with np.load(io.BytesIO(b),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def one(row):
 root=Path(row['manifest']).parent;m=json.loads((root/'MANIFEST.json').read_text());a=json.loads((root/'A_OBSERVED_RECORD.json').read_text());assert a['frames']==m['frames']==49;source=INPUTS[m['HF_source_archive']];signature=hashlib.sha256();signature.update(json.dumps([m['source_color_linear_rgb'],m['source_gamma_control'],m['target_gamma_control'],m['source_reference']['SHA256']],sort_keys=True).encode());patches={v:[] for v in ['A','B']};count=0
 with tarfile.open(source['local_path']) as tf:
  members={x.name:x for x in tf.getmembers() if x.isfile()};prefix=next(x for x in members if x.count('/')<=1 and x.endswith('/MANIFEST.json')).rsplit('/',1)[0]
  def get(rel):return tf.extractfile(members[prefix+'/'+rel]).read()
  cameras=arrays(get('conditions/cameras49.npz'));geometry=arrays(get('conditions/remove/fixed_world_geometry.npz'));saved=arrays((root/'source_camera_geometry.npz').read_bytes())
  for k in ['K','w2c_A','w2c_B']:np.testing.assert_array_equal(saved[k],cameras[k])
  for k in ['source_center','source_size','source_rotation']:np.testing.assert_array_equal(saved[k],geometry[k])
  for i in range(49):
   old=arrays(get(f'conditions/remove/conditions/lamp_silhouette/{i:06d}.npz'));s=float(old['source_gamma_control_removed']);t=float(old['target_gamma_control_removed']);assert t==0
   observed=get(f'conditions/source_observed_A/{i:06d}.png');assert sha(observed)==sha((root/'source_observed_A49'/f'{i:06d}.png').read_bytes())
   expectedA=cv2.imdecode(np.frombuffer(observed,np.uint8),0)>0;expectedB=cv2.imdecode(np.frombuffer(get(f'conditions/source_B_estimate/visible_with_unknown_estimate/{i:06d}.png'),np.uint8),0)>0
   for view in ['A','B']:
    pictures=[];masks=[]
    for family in ['lamp_silhouette','bbox_silhouette']:
     base=root/('A_observed' if view=='A' else '')/family;z=load_remove_fields(root,i,family,view);mask=cv2.imread(str(base/'mask49'/f'{i:06d}.png'),0)>0;np.testing.assert_array_equal(z['support'],mask)
     if family=='lamp_silhouette':np.testing.assert_array_equal(mask,expectedA if view=='A' else expectedB)
     color=cv2.imread(str(base/'Color49'/f'{i:06d}.png'))[:,:,::-1];np.testing.assert_array_equal(np.rint(z['color_linear_rgb']*255).astype(np.uint8),color);assert z['source_gamma_control']==s and z['target_gamma_control']==t and z['normalization_count']==1;np.testing.assert_allclose(z['relative_gamma'][mask],(t-s)/1.1,atol=1e-7);assert np.all(z['relative_gamma'][~mask]==-1) and np.all(z['color_linear_rgb'][~mask]==0);np.testing.assert_allclose(z['color_linear_rgb'][mask],np.broadcast_to(np.asarray(m['source_color_linear_rgb'],np.float32),(int(mask.sum()),3)),atol=0,rtol=0)
     signature.update((view+family+str(i)).encode());signature.update(mask.tobytes());signature.update(color.tobytes());pictures.append(color[:,:,::-1]);masks.append(mask);count+=1
    union=masks[0]|masks[1];ys,xs=np.where(union);x0,y0,x1,y1=(max(0,int(xs.min())-5),max(0,int(ys.min())-5),min(1280,int(xs.max())+6),min(704,int(ys.max())+6)) if len(xs) else (0,0,1280,704);cell=np.full((100,154,3),24,np.uint8)
    for j,pic in enumerate(pictures):
     crop=pic[y0:y1,x0:x1];scale=min(74/crop.shape[1],80/crop.shape[0]);im=cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))),interpolation=cv2.INTER_NEAREST);xx=j*77+(74-im.shape[1])//2;yy=(80-im.shape[0])//2;cell[yy:yy+im.shape[0],xx:xx+im.shape[1]]=im
    cv2.putText(cell,f'{i:02d} x{x0} y{y0}',(2,94),cv2.FONT_HERSHEY_SIMPLEX,.28,(200,200,200),1);patches[view].append(cell)
 assert count==196
 return dict(case=m['case'],archive=m['HF_source_archive'],key=m['HF_source_archive_LFS_SHA256'],root=str(root),semantic_fingerprint=signature.hexdigest(),field_reader_checks=count,source_reference=m['source_reference'],source_gamma_control=m['source_gamma_control'],target_gamma_control=m['target_gamma_control'],source_color_linear_rgb=m['source_color_linear_rgb'],patches=patches)

if __name__=='__main__':
 start=time.time();b=json.loads((T/'logs/REMOVE_FINITE48_CURRENT.json').read_text());a=json.loads((T/'logs/REMOVE_A_FINITE48_CURRENT.json').read_text());assert b['completed']==a['completed']==48 and not b['errors'] and not a['errors'];INPUTS={x['HF_path']:x for x in json.loads((T/'inventory/REMOVE_ARCHIVE_INPUTS_v1.json').read_text())['rows']};assert len(INPUTS)==48
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(one,b['jobs']))
 groups={}
 for row in result:groups.setdefault(row['semantic_fingerprint'],[]).append(row)
 pages=T/'review_pages';pages.mkdir(exist_ok=True);records=[]
 for i,(fingerprint,items) in enumerate(sorted(groups.items())):
  row=items[0];canvas=np.full((1610,1078,3),28,np.uint8);cv2.putText(canvas,f"{i:02d} case{row['case']} / {len(items)} exact-equivalent archives / sourceRGB reference",(10,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1);cv2.putText(canvas,f"raw gamma={-row['source_gamma_control']}; /1.1 ONCE; left=lamp right=3D bbox; zoom is REVIEW ONLY",(10,48),cv2.FONT_HERSHEY_SIMPLEX,.45,(220,220,220),1)
  image=cv2.imread(str(Path(row['root'])/'source_reference.png'),cv2.IMREAD_UNCHANGED);image=image[:,:,:3] if image.shape[-1]==3 else np.rint(image[:,:,:3]*(image[:,:,3:]/255.)+128*(1-image[:,:,3:]/255.)).astype(np.uint8);scale=min(300/image.shape[1],110/image.shape[0]);image=cv2.resize(image,(max(1,round(image.shape[1]*scale)),max(1,round(image.shape[0]*scale))),interpolation=cv2.INTER_NEAREST);canvas[60:60+image.shape[0],10:10+image.shape[1]]=image
  for j,view in enumerate(['A','B']):
   top=190+j*710;cv2.putText(canvas,'A actual observed source mask / projected original3D bbox' if view=='A' else 'B source estimated visibility / projected original3D bbox; NOT RGBGT',(10,top-8),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1)
   for f,image in enumerate(row['patches'][view]):
    xx=(f%7)*154;yy=top+(f//7)*100;canvas[yy:yy+100,xx:xx+154]=image
  p=pages/f'group_{i:02d}_case{row["case"]}.jpg';assert cv2.imwrite(str(p),canvas,[cv2.IMWRITE_JPEG_QUALITY,95]);records.append(dict(group=i,case=row['case'],semantic_fingerprint=fingerprint,archives=[{k:v for k,v in x.items() if k!='patches'} for x in items],review_page=str(p),review_page_SHA256=sha(p.read_bytes()),actual_visual_review_pending=True))
 report=dict(state='all48_original_source_camera_geometry_masks_signed_reader_verified_PENDING_actual_visual_review',archives=48,unique_numeric_cases=len({x['case'] for x in result}),independent_field_reader_checks=sum(x['field_reader_checks'] for x in result),source_lamp_A_observed_and_B_masks_exact_to_old_archives=True,all_source_K_A_B_geometry_arrays_exact_to_old_archives=True,all_negative_model_gamma_once_background_minus1=True,exact_equivalent_visual_groups=len(records),groups=records,elapsed_seconds=time.time()-start,overall_goal_complete=False)
 p=T/'inventory/REMOVE_ALL48_INDEPENDENT_READER_AND_REVIEW_INDEX_v1.json';assert not p.exists();p.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='groups'}),flush=True)

from pathlib import Path
import json,hashlib,shutil,concurrent.futures,datetime,math
from PIL import Image,ImageDraw
R=Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal/reference_completion_20261115_v1');BASE=Path('/Volumes/T9/FR_LAMP/renders/clip_all_v1');PUB=R/'publish';PUB.mkdir(exist_ok=True);(R/'reviews/contact_sheets').mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
rows={}
for p in (R/'inventory/tar_manifest_reads').glob('*.json'):
 if p.name.startswith('._') or p.name.endswith('.error.json'):continue
 d=json.loads(p.read_text())
 if 'root_manifest' in d:rows[d['HF_path']]=d
p=R/'inventory/HF_ARCHIVE_MANIFEST_QH_RECOVERY_v1.json'
if p.exists():
 for d in json.loads(p.read_text()):
  if 'root_manifest' in d:rows[d['HF_path']]=d
assets={}
for d in rows.values():
 for a in d['root_manifest']['manifest'].get('assets',[]):
  if not isinstance(a,dict):raise RuntimeError('Unexpected assets schema')
  s=a['sha256'];assert len(s)==64;assets.setdefault(s,dict(records=[]));assets[s]['records'].append(dict(archive_path=d['HF_path'],archive_LFS_SHA256=d['archive_LFS_SHA256'],manifest_member=d['root_manifest']['name'],manifest_SHA256=d['root_manifest']['SHA256'],asset_alias=a['asset'],asset_member=a['package_path']))
cat=json.loads((BASE/'manifest.json').read_text());bysha={x['sha256']:x for x in cat['objects']}
def one(item):
 s,entry=item;row=bysha[s];receipt=json.loads((BASE/'receipts'/f'{row["uid"]}.json').read_text());assert receipt['source_sha256']==s and receipt['uid']==row['uid'];src=Path(row['path']);assert src.exists() and sha(src)==s
 folder=Path(row['output']);checks=[]
 for i in range(4):
  name=f'clip_view_{i:02d}.jpg';p=folder/name;h=sha(p);assert h==receipt['jpg_sha256'][i]
  with Image.open(p) as im:im.load();assert im.size==(512,512) and im.mode=='RGB'
  dst=PUB/'assets'/s/f'view_{i:02d}.jpg';dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert sha(dst)==h
  else:shutil.copyfile(p,dst)
  checks.append(dict(path=str(dst.relative_to(PUB)),SHA256=h,bytes=p.stat().st_size,width=512,height=512,view=i))
 return dict(asset_SHA256=s,uid=row['uid'],name=row['name'],reference=checks[0],views=checks,bound_assets=sorted(entry['records'],key=lambda x:(x['archive_path'],x['asset_alias'])),provenance=dict(kind='existing_whole_original_GLB_material_render',source_raw_GLB_SHA256=s,source_library='FR_LAMP full16037 original raw catalog',render_method=cat['renderer'],render_worker_SHA256=receipt['worker_sha256'],original_render_receipt_status=receipt['status'],original_render_receipt_SHA256=sha(BASE/'receipts'/f'{row["uid"]}.json'),all_four_images_match_recorded_SHA256=True,actual_source_GLB_SHA256_verified=True),actual_visual_review='pending')
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(one,sorted(assets.items())))
for n in range(math.ceil(len(results)/12)):
 group=results[n*12:(n+1)*12];canvas=Image.new('RGB',(1088,3*322+44),(32,32,32));draw=ImageDraw.Draw(canvas);draw.text((12,10),f'StarReal exact-asset references / page {n+1} / all4 views each / NOT yet accepted',fill='white')
 for j,row in enumerate(group):
  x=(j%4)*272;y=(j//4)*322+40;num=n*12+j;draw.text((x+5,y+1),f'{num:03d} {row["uid"][:12]}',fill='white');draw.text((x+5,y+17),row['asset_SHA256'][:18],fill=(220,220,80))
  for k,v in enumerate(row['views']):
   with Image.open(PUB/v['path']) as im:im=im.resize((132,132),Image.Resampling.LANCZOS);canvas.paste(im,(x+(k%2)*134+2,y+34+(k//2)*134))
 p=R/'reviews/contact_sheets'/f'references_{n+1:02d}.jpg';canvas.save(p,quality=95)
manifest=dict(schema='StarReal.asset_reference_bindings.v1',state='prepared_pending_actual_visual_review',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),HF_source_revision=next(iter(rows.values()))['revision'],inventoried_archives=len(rows),unique_assets=len(results),reference_images=len(results)*4,assets=results,training_RGB_or_existing_packages_modified=False,reference_resolution=[512,512],source_photometry_or_UE_illumination_not_claimed=True)
(R/'inventory/EXACT_REFERENCE_BINDINGS_PREPARED_v1.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(unique_assets=len(results),images=4*len(results),exact_source_and_image_SHAs_all_verified=True,contact_pages=math.ceil(len(results)/12),total_JPEG_bytes=sum(v['bytes'] for r in results for v in r['views']))))

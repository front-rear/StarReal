from pathlib import Path
import urllib.request, json, datetime, hashlib
ROOT=Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal/reference_completion_20261115_v1')
def get(url):
 req=urllib.request.Request(url,headers={'User-Agent':'StarReal-reference-completion/1.0'})
 with urllib.request.urlopen(req,timeout=45) as f:return f.read(),f.headers.get('Link','')
info=json.loads(get('https://huggingface.co/api/datasets/FrontRear/VLMdataset')[0]);head=info['sha'];rows=[];url=f'https://huggingface.co/api/datasets/FrontRear/VLMdataset/tree/{head}/StarReal?recursive=true&expand=false&limit=1000';pages=0
while url:
 raw,link=get(url);page=json.loads(raw);assert isinstance(page,list);rows+=page;pages+=1;assert pages<=100
 nxt=[p.split('>')[0].strip().lstrip('<') for p in link.split(',') if 'rel="next"' in p];url=nxt[0] if nxt else None
out=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),repo='FrontRear/VLMdataset',revision=head,public_read_only=True,pages=pages,entries=rows)
p=ROOT/'inventory/HF_STARREAL_PINNED_TREE_v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2));files=[r for r in rows if r['type']=='file'];assets=[r for r in files if r['path'].lower().endswith(('.glb','.obj','.fbx'))];refs=[r for r in files if any(x in r['path'].lower() for x in ('reference','ref_image','appearance','asset_catalog'))]
print(json.dumps(dict(head=head,total_files=len(files),asset_files=len(assets),reference_related_files=len(refs),assets=[x['path'] for x in assets[:15]],reference_paths=[x['path'] for x in refs[:55]])))

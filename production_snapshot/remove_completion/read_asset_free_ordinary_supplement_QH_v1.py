from pathlib import Path
import json,subprocess,shlex
R=Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal/reference_completion_20261115_v1');t=json.loads((R/'inventory/HF_STARREAL_PINNED_TREE_v1.json').read_text());row=next(x for x in t['entries'] if x['type']=='file' and 'ordinary_pcf_bbox_9cases_149videos_v1.tar' in x['path']);cfg=dict(head=t['revision'],row=row)
code=r'''import io,sys,json,hashlib,tarfile,urllib.request
x=json.load(sys.stdin);r=x['row'];end=4194303;url=f"https://huggingface.co/datasets/FrontRear/VLMdataset/resolve/{x['head']}/{r['path']}?download=true";req=urllib.request.Request(url,headers={'Range':f'bytes=0-{end}','User-Agent':'StarReal-reference-completion/1.0'})
with urllib.request.urlopen(req,timeout=90) as f:
 assert f.status==206 and f.headers['Content-Range']==f"bytes 0-{end}/{r['size']}";b=f.read(end+1);assert len(b)==end+1
with tarfile.open(fileobj=io.BytesIO(b)) as tf:
 for m in tf:
  if m.isfile() and m.name.count('/')<=1 and m.name.endswith('/MANIFEST.json'):
   raw=tf.extractfile(m).read();d=json.loads(raw);break
files=d.get('files',[]);glb=[]
def scan(x,path=''):
 if isinstance(x,dict):
  for k,v in x.items():scan(v,path+'/'+str(k))
 elif isinstance(x,list):
  for i,v in enumerate(x):scan(v,path+'/'+str(i))
 elif isinstance(x,str) and x.lower().endswith(('.glb','.obj','.fbx')):glb.append(dict(key=path,value=x))
scan(d)
print(json.dumps(dict(HF_path=r['path'],revision=x['head'],archive_LFS_SHA256=r['lfs']['oid'],manifest_SHA256=hashlib.sha256(raw).hexdigest(),manifest_size=len(raw),keys=list(d),declared_assets=d.get('assets'),GLB_mentions=glb,top_level_shapes={k:len(v) for k,v in d.items() if isinstance(v,(dict,list))})))'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','cscg-qh00','python3 -c '+shlex.quote(code)],input=json.dumps(cfg).encode(),capture_output=True,timeout=150);assert p.returncode==0,p.stderr.decode()[:500];d=json.loads(p.stdout);(R/'inventory/HF_ORDINARY_SUPPLEMENT_ROOT_MANIFEST_CLASSIFICATION_v1.json').write_text(json.dumps(d,indent=2));print(json.dumps(d)[:2500])

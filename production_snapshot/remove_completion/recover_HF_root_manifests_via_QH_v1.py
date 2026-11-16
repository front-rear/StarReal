from pathlib import Path
import json,subprocess,shlex
R=Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal/reference_completion_20261115_v1');d=json.loads((R/'inventory/HF_ARCHIVE_ROOT_MANIFESTS_v1.json').read_text());t=json.loads((R/'inventory/HF_STARREAL_PINNED_TREE_v1.json').read_text());paths={r['HF_path'] for r in d['failures']};cfg=dict(head=t['revision'],rows=[x for x in t['entries'] if x['path'] in paths])
code=r'''from pathlib import Path
import io,json,tarfile,hashlib,urllib.request,sys,concurrent.futures
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');d=json.load(sys.stdin)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def one(row):
 p=Q/'archives'/Path(row['path']).name;names=[]
 try:
  if p.exists():
   assert p.stat().st_size==row['size'] and sha(p)==row['lfs']['oid'];tf=tarfile.open(p);via='persistent_QH_exact_LFS_SHA'
  else:
   url=f"https://huggingface.co/datasets/FrontRear/VLMdataset/resolve/{d['head']}/{row['path']}?download=true";req=urllib.request.Request(url,headers={'Range':'bytes=0-1048575','User-Agent':'StarReal-reference-completion/1.0'})
   with urllib.request.urlopen(req,timeout=60) as f:
    assert f.status==206;cr=f.headers['Content-Range'];assert cr==f"bytes 0-1048575/{row['size']}";b=f.read(1048576);assert len(b)==1048576
   tf=tarfile.open(fileobj=io.BytesIO(b),mode='r:');via='pinned_HF_HTTP_range_QH'
  with tf:
   for m in tf:
    if len(names)<30:names.append(dict(name=m.name,size=m.size))
    if m.isfile() and m.name.count('/')<=1 and Path(m.name).name=='MANIFEST.json':
     raw=tf.extractfile(m).read();return dict(HF_path=row['path'],revision=d['head'],archive_bytes=row['size'],archive_LFS_SHA256=row['lfs']['oid'],via=via,root_manifest=dict(name=m.name,bytes=m.size,offset=m.offset_data,SHA256=hashlib.sha256(raw).hexdigest(),manifest=json.loads(raw)))
  return dict(HF_path=row['path'],names=names,error='root_manifest_not_found')
 except Exception as e:return dict(HF_path=row['path'],error=type(e).__name__,names=names)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,d['rows']))
print(json.dumps(rows))'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','cscg-qh00','python3 -c '+shlex.quote(code)],input=json.dumps(cfg).encode(),capture_output=True,timeout=240);assert p.returncode==0,p.stderr.decode()[:500];rows=json.loads(p.stdout);(R/'inventory/HF_ARCHIVE_MANIFEST_QH_RECOVERY_v1.json').write_text(json.dumps(rows,indent=2));print(json.dumps([dict(path=x['HF_path'],ok='root_manifest' in x,error=x.get('error'),names=x.get('names',[])[:7]) for x in rows]))
for x in rows:
 if 'root_manifest' in x:
  k=hashlib.sha256(x['HF_path'].encode()).hexdigest()[:20] if False else None

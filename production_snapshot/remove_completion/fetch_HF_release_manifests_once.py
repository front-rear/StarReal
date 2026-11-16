from pathlib import Path
import concurrent.futures,urllib.request,json,hashlib
R=Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal/reference_completion_20261115_v1');d=json.loads((R/'inventory/HF_STARREAL_PINNED_TREE_v1.json').read_text());head=d['revision'];files=[x for x in d['entries'] if x['type']=='file' and x['path'].endswith('/RELEASE.json')]
def one(row):
 path=row['path'];req=urllib.request.Request(f'https://huggingface.co/datasets/FrontRear/VLMdataset/resolve/{head}/{path}',headers={'User-Agent':'StarReal-reference-completion/1.0'})
 with urllib.request.urlopen(req,timeout=45) as f:b=f.read()
 assert len(b)==row['size'];assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==row['oid'];return dict(path=path,SHA256=hashlib.sha256(b).hexdigest(),release=json.loads(b))
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(one,files))
p=R/'inventory/HF_RELEASES_VERIFIED_v1.json';assert not p.exists();p.write_text(json.dumps(dict(revision=head,rows=rows),indent=2));print(json.dumps(dict(releases=len(rows),head=head,representative=rows[-1])))

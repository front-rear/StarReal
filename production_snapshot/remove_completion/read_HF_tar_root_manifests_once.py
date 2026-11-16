from pathlib import Path
import concurrent.futures,io,tarfile,json,time,hashlib,requests
R=Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal/reference_completion_20261115_v1');D=R/'inventory';TREE=json.loads((D/'HF_STARREAL_PINNED_TREE_v1.json').read_text());HEAD=TREE['revision'];FILES=[x for x in TREE['entries'] if x['type']=='file' and x['path'].endswith('.tar')];CACHE=D/'tar_manifest_reads';CACHE.mkdir(exist_ok=True)
def one(row):
 path=row['path'];key=hashlib.sha256(path.encode()).hexdigest()[:20];target=CACHE/(key+'.json')
 if target.exists():
  old=json.loads(target.read_text());assert old['HF_path']==path and old['revision']==HEAD;return old
 url=f'https://huggingface.co/datasets/FrontRear/VLMdataset/resolve/{HEAD}/{path}?download=true';errors=[]
 for trial in range(3):
  try:
   with requests.get(url,headers={'Range':'bytes=0-262143','User-Agent':'StarReal-reference-completion/1.0'},timeout=(15,50),stream=True) as resp:
    resp.raise_for_status();assert resp.status_code==206,(resp.status_code,path);cr=resp.headers['Content-Range'];assert cr==f'bytes 0-262143/{row["size"]}',cr;b=resp.raw.read(262144);assert len(b)==262144
   tf=tarfile.open(fileobj=io.BytesIO(b),mode='r:');found=None
   for m in tf:
    if m.isfile() and m.name.count('/')<=1 and Path(m.name).name=='MANIFEST.json':
     assert m.size<240000;raw=tf.extractfile(m).read();found=dict(name=m.name,bytes=m.size,offset=m.offset_data,SHA256=hashlib.sha256(raw).hexdigest(),manifest=json.loads(raw));break
   assert found is not None,'No root manifest in first256KiB'
   result=dict(HF_path=path,revision=HEAD,archive_bytes=row['size'],archive_LFS_SHA256=row['lfs']['oid'],range_response=cr,partial_range_SHA256=hashlib.sha256(b).hexdigest(),root_manifest=found);target.write_text(json.dumps(result,indent=2));print(json.dumps(dict(path=path,assets=len(found['manifest'].get('assets',[])),trial=trial)),flush=True);return result
  except Exception as e:
   errors.append(type(e).__name__+': '+str(e)[:200])
   if trial<2:time.sleep(1)
 result=dict(HF_path=path,revision=HEAD,error=errors);(CACHE/(key+'.error.json')).write_text(json.dumps(result,indent=2));return result
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:rows=list(pool.map(one,FILES))
out=dict(revision=HEAD,archives=len(rows),manifests=sum('root_manifest' in x for x in rows),failures=[x for x in rows if 'error' in x],rows=rows);(D/'HF_ARCHIVE_ROOT_MANIFESTS_v1.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['rows','failures']}));print(json.dumps(out['failures']))

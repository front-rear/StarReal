"""Finite readonly source archive preparation for the explicit Remove supplement."""
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,os,threading,urllib.request,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';N=Path('/tmp/starreal_uid1007_case19_cpu_v1/reference_completion_20261115_v1');N.mkdir(exist_ok=True);(N/'input_archives').mkdir(exist_ok=True);(T/'logs').mkdir(exist_ok=True)
lock=(T/'logs/remove_input_archives.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);scope=json.loads((T/'publish/HF_SOURCE_SCOPE.json').read_text());rows=[];guard=threading.Lock();start=time.time();state=T/'logs/REMOVE_INPUT_ARCHIVES_CURRENT.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def save(row):
 with guard:
  rows.append(row);tmp=state.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state='finite_input_verification',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),elapsed_seconds=time.time()-start,rows=rows,expected=48,GPU_used=False),indent=2));tmp.replace(state)
def one(row):
 name=Path(row['path']).name;local=Q/'archives'/name;cached=N/'input_archives'/name
 try:
  if local.exists() and local.stat().st_size==row['bytes'] and sha(local)==row['LFS_SHA256']:path=local;via='existing_exact_QH_archive'
  elif cached.exists() and cached.stat().st_size==row['bytes'] and sha(cached)==row['LFS_SHA256']:path=cached;via='previously_verified_dedicated_input_copy'
  else:
   path=cached;partial=cached.with_suffix('.tar.partial');url=f"https://huggingface.co/datasets/FrontRear/VLMdataset/resolve/{scope['HF_source_revision']}/{row['path']}?download=true";errors=[]
   for attempt in range(3):
    try:
     offset=partial.stat().st_size if partial.exists() else 0;assert offset<=row['bytes']
     if offset<row['bytes']:
      hdr={'User-Agent':'StarReal-remove-source-read/1.0'}
      if offset:hdr['Range']=f'bytes={offset}-'
      req=urllib.request.Request(url,headers=hdr)
      with urllib.request.urlopen(req,timeout=120) as response:
       if offset:assert response.status==206 and response.headers['Content-Range']==f"bytes {offset}-{row['bytes']-1}/{row['bytes']}"
       else:assert response.status==200
       with partial.open('ab') as f:
        while True:
         b=response.read(8388608)
         if not b:break
         f.write(b)
     assert partial.stat().st_size==row['bytes'];assert sha(partial)==row['LFS_SHA256'];assert not cached.exists();partial.replace(cached);via='pinned_public_HF_archive_new_dedicated_input';break
    except Exception as e:
     errors.append(type(e).__name__)
     if attempt==2:raise RuntimeError(','.join(errors))
   else:raise RuntimeError('finite_download_failed')
  out=dict(HF_path=row['path'],local_path=str(path),bytes=row['bytes'],archive_LFS_SHA256=row['LFS_SHA256'],source_revision=scope['HF_source_revision'],verified=True,via=via)
 except Exception as e:out=dict(HF_path=row['path'],verified=False,error=type(e).__name__+': '+str(e)[:100])
 save(out);print(json.dumps(dict(path=row['path'],verified=out['verified'])),flush=True);return out
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:result=list(pool.map(one,scope['archives']))
manifest=dict(state='all48_input_archives_exact' if all(x['verified'] for x in result) else 'finite_input_read_incomplete_no_render_started',expected=48,verified=sum(x['verified'] for x in result),elapsed_seconds=time.time()-start,rows=result,GPU_used=False,old_archives_modified=False)
p=T/'inventory/REMOVE_ARCHIVE_INPUTS_v1.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(manifest,indent=2));print(json.dumps({k:v for k,v in manifest.items() if k!='rows'}),flush=True)

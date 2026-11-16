from pathlib import Path
import argparse,fcntl,hashlib,json,os,struct,sys,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');N=Path('/tmp/starreal_uid1007_case19_cpu_v1')
def exact(n):
 out=[]
 while n:
  b=sys.stdin.buffer.read(min(n,1024**2))
  if not b:raise EOFError('Finite read-only input stream ended early; preserve partial')
  out.append(b);n-=len(b)
 return b''.join(out)
def message():
 size=struct.unpack('>I',exact(4))[0];assert 0<size<4*1024**2;return json.loads(exact(size))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args();assert all(x.isdecimal() for x in a.name.split('_')) and 1<=len(a.name.split('_'))<=2
 guard=(Q/'logs/READONLY_SFTP_STREAM_PREFETCH.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 spec=message();assert spec['remote_execution'] is False and spec['remote_writes'] is False and spec['Mac_payload_files'] is False
 assert {r['case'] for r in spec['cases']}==set(a.name.split('_'));expected={r['case']+'/'+p['local'] for r in spec['cases'] for p in r['read_only_original_paths']}
 root=Q/('inputs/original_native_readonly_SFTP_'+a.name+'_EXTRA640_CACHE_ONLY_stream_v3');assert not root.exists();root.mkdir();alias=N/('inputs/StarReal_native_readonly_SFTP_'+a.name+'_EXTRA640_CACHE_ONLY_stream_v3');assert not alias.exists() and not alias.is_symlink();alias.symlink_to(root,target_is_directory=True);(root/'SPEC.json').write_text(json.dumps(spec,indent=2));records=[];start=time.time()
 for _ in range(len(expected)):
  row=message();rel=row['path'];assert rel in expected and rel not in {x['path'] for x in records};assert 0<=row['bytes']<=8*1024**3
  dest=root/rel;assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.partial.'+str(os.getpid()));assert not tmp.exists();h=hashlib.sha256();remaining=row['bytes']
  with tmp.open('xb') as f:
   while remaining:
    b=exact(min(remaining,1024**2));f.write(b);h.update(b);remaining-=len(b)
  trailer=message();assert h.hexdigest()==trailer['SHA256'];assert row['source_stat_before']['size']==trailer['source_stat_after']['size']==row['bytes'];assert row['source_stat_before'].get('mtime')==trailer['source_stat_after'].get('mtime');tmp.rename(dest);records.append(dict(path=rel,bytes=dest.stat().st_size,SHA256=h.hexdigest(),source_stat_before=row['source_stat_before'],source_stat_after=trailer['source_stat_after']))
  if len(records)%10==0:(root/'TRANSFER_CURRENT.json').write_text(json.dumps(dict(files=len(records),expected=len(expected),seconds=time.time()-start,Mac_payload_files=False),indent=2))
 result=dict(state='finite_original_input_stream_all_SHA_verified_pending_Native_array_reader',records=records,files=len(records),expected=len(expected),seconds=time.time()-start,source_remote_execution=False,source_remote_writes=False,Mac_payload_files=False);(root/'TRANSFER_RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()

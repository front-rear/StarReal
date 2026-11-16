"""Pack locally before one SCP; safe QH extraction. No 6000D command/compute."""
from pathlib import Path
import argparse,hashlib,json,shlex,subprocess,tarfile,time
L=Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal');N='/tmp/starreal_uid1007_case19_cpu_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);a=ap.parse_args();root=Path(a.root).resolve();assert root.parent==Path('/private/tmp') and root.name.startswith('StarReal_native_readonly_SFTP_');spec=json.loads((root/'SPEC.json').read_text());cases=[x['case'] for x in spec['cases']];assert 1<=len(cases)<=2 and all(x.isdecimal() for x in cases);archive=root.with_suffix('.transfer.tar');assert not archive.exists();start=time.time();records=[]
 paths=[root/'SPEC.json']+[p for c in cases for p in sorted((root/c).rglob('*')) if p.is_file()]
 with tarfile.open(archive,'x') as tf:
  for p in paths:
   assert not p.is_symlink();rel=str(p.relative_to(root));assert '..' not in Path(rel).parts;tf.add(p,arcname=rel,recursive=False);records.append(dict(path=rel,size=p.stat().st_size,SHA256=sha(p)))
 cfg=dict(root_name=root.name,archive_name=archive.name,archive_size=archive.stat().st_size,archive_SHA256=sha(archive),files=records);packing=time.time()-start
 r=subprocess.run(['scp',str(archive),'cscg-qh00:'+N+'/inputs/'+archive.name],capture_output=True,timeout=1200);assert r.returncode==0,r.stderr.decode()
 code=r'''from pathlib import Path
import hashlib,json,shutil,sys,tarfile
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');d=json.load(sys.stdin);p=N/'inputs'/d['archive_name'];out=N/'inputs'/d['root_name'];assert not out.exists();assert p.stat().st_size==d['archive_size']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
assert sha(p)==d['archive_SHA256'];expected={r['path']:r for r in d['files']};out.mkdir()
with tarfile.open(p) as tf:
 members=tf.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
 for m in members:
  assert m.isfile() and not m.issym() and not m.islnk() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.size==expected[m.name]['size'];dst=out/m.name;dst.parent.mkdir(parents=True,exist_ok=True)
  with tf.extractfile(m) as src,dst.open('xb') as f:shutil.copyfileobj(src,f,8*1024**2)
  assert sha(dst)==expected[m.name]['SHA256']
receipt=dict(state='all_original_transferred_files_SHA_verified_pending_Native_review',files=len(expected),destination=str(out),one_network_payload=True,remote_6000D_execution=False,remote_6000D_writes=False,new_complete_groups=0);(out/'BUNDLE_TRANSFER_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))'''
 r=subprocess.run(['ssh','cscg-qh00','python3 -c '+shlex.quote(code)],input=json.dumps(cfg).encode(),capture_output=True,timeout=300);assert r.returncode==0,r.stderr.decode();receipt=dict(state='single_payload_bundle_transfer_verified',cases=cases,packing_seconds=packing,elapsed_seconds=time.time()-start,archive_size=cfg['archive_size'],archive_SHA256=cfg['archive_SHA256'],QH=json.loads(r.stdout),quality_accepted=False,new_complete_groups=0);p=L/'logs'/('ORIGINAL_NATIVE_BUNDLE_'+'_'.join(cases)+'_v1.json');assert not p.exists();p.write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()

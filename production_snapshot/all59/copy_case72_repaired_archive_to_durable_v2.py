"""One bounded verified copy from node-local scratch to persistent StarReal."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, shutil, tarfile, time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1')
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def put(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def main():
    start=time.time();lock=(Q/'logs/case72_durable_archive_copy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    release=json.loads((N/'releases/72/revisions/fixed_world_yellow_depth_v6/RELEASE.json').read_text())
    source=Path(release['archive']);assert source.parent==N/'archives' and release['quality_accepted']
    assert source.stat().st_size==release['archive_size'] and sha(source)==release['archive_SHA256']
    assert shutil.disk_usage(Q).free>source.stat().st_size+16*1024**3
    dest=Q/'archives'/source.name;assert not dest.exists();dest.parent.mkdir(exist_ok=True)
    partials=list(dest.parent.glob(dest.name+'.partial.*'))
    assert len(partials)<=1, 'Multiple partial copies need explicit inspection'
    if partials:
        temp=partials[0]
        assert temp.stat().st_size==release['archive_size'] and sha(temp)==release['archive_SHA256'], 'Preserved previous partial copy mismatches; do not overwrite'
    else:
        temp=dest.with_name(dest.name+'.partial.'+str(os.getpid()));assert not temp.exists()
        with source.open('rb') as a,temp.open('xb') as b:shutil.copyfileobj(a,b,8*1024**2)
    assert temp.stat().st_size==source.stat().st_size and sha(temp)==sha(source)
    with tarfile.open(temp) as tf:
        members=tf.getmembers();assert all(not m.issym() and not m.islnk() and '..' not in Path(m.name).parts and not m.name.startswith('/') for m in members)
        records=json.load(tf.extractfile('case_72/SHA256_FILES.json'))
        expected={'case_72/'+x['path']:x for x in records}
        for m in members:
            if not m.isfile() or m.name=='case_72/SHA256_FILES.json':continue
            row=expected.pop(m.name);assert m.size==row['size'];h=hashlib.sha256()
            with tf.extractfile(m) as f:
                for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
            assert h.hexdigest()==row['sha256'],m.name
        assert not expected
        assert hashlib.sha256(tf.extractfile('case_72/MANIFEST.json').read()).hexdigest()==release['package_manifest_SHA256']
    temp.rename(dest)
    proof=dict(case='72',state='durable_QH_archive_all_members_SHA_verified',archive=str(dest),archive_size=dest.stat().st_size,archive_SHA256=sha(dest),files_in_archive=release['files_in_archive'],package_manifest_SHA256=release['package_manifest_SHA256'],all_members_SHA256_verified=True,persistent_disk=True,source_files_preserved=True,source_node_local_archive=str(source),source_node_local_archive_SHA256=sha(source),archive_creation_backend='QH_node_local_complete_package_then_verified_persistent_copy',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-start)
    receipt=Q/'receipts/CASE72_DURABLE_ARCHIVE_RECEIPT_v2.json';assert not receipt.exists();put(receipt,proof)
    release.update(archive=str(dest),archive_host='cscg-qh00',archive_creation_backend=proof['archive_creation_backend'],durable_archive_receipt='receipts/CASE72_DURABLE_ARCHIVE_RECEIPT_v2.json',durable_archive_receipt_SHA256=sha(receipt))
    target=Q/'releases/72/revisions/fixed_world_yellow_depth_v6/RELEASE.json';assert not target.exists();put(target,release)
    for name in ['MANIFEST.json','PORTABLE_FINAL_AUDIT.json','BUILD_STATE.json']:
        a=N/'packages/72/fixed_world_yellow_depth_v6_staging'/name;b=Q/'remote_package_metadata/72/revisions/fixed_world_yellow_depth_v6'/name;assert not b.exists();b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b);assert sha(a)==sha(b)
    print(json.dumps(proof),flush=True)
if __name__=='__main__':main()

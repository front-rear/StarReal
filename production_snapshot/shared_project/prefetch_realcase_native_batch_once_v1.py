"""Bounded CPU lookahead. Preserve all inputs; never quality-accept or upload.

Usage: python prefetch_realcase_native_batch_once_v1.py --cases 58 64
Each case is identity-pinned from the workboard. No textual per-case adapters.
Original component/global PCF locks remain in the existing finite runner.
"""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, io, json, os, shlex, subprocess, sys, time, zipfile

L = Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal')
N = '/tmp/starreal_uid1007_case19_cpu_v1'
Q = '/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal'
PY = '/root/autodl-tmp/repro_video_20261105/_envs/pisco/bin/python'

def remote(host, command, data=None, timeout=300):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, command],
                       input=data, capture_output=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f'{host}: exit {r.returncode}: '+r.stderr.decode(errors='replace')[-5000:])
    return r.stdout

def qpython(code, cfg, data=None):
    return remote('cscg-qh00', 'env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 python3 -c '+
                  shlex.quote(code)+' '+shlex.quote(json.dumps(cfg)), data)

IMPORT_NATIVE = r'''
from pathlib import Path
import hashlib,io,json,shutil,sys,zipfile
c=json.loads(sys.argv[1]);N=Path(c['N']);Q=Path(c['Q']);data=sys.stdin.buffer.read(96*1024**2+1)
assert 0<len(data)<=96*1024**2
z=zipfile.ZipFile(io.BytesIO(data));names=z.namelist();assert len(names)==len(set(names))
assert all(not Path(x).is_absolute() and '..' not in Path(x).parts for x in names)
m=json.loads(z.read('INPUT_MANIFEST.json'));v=c['version']
assert m['case']==c['case'] and m['case_id']==v['case_id'] and m['source_video_SHA256']==v['sha256']
assert m['source_video_bytes']==v['bytes'] and m['source_video_HF_path']==v['hf_path']
assert m['Native_RGB49_exact'] and m['original_inputs_unchanged']
for row in m['files']:
 b=z.read(row['path']);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['SHA256']
out=N/('inputs/case'+c['case']+'_native_CPU_v1');dest=Q/('inputs/case'+c['case']+'_native_CPU_v1')
assert not out.exists() and not dest.exists(),'Existing inputs: inspect instead of automatic replay'
assert shutil.disk_usage(Q).free>1024**3
out.mkdir();z.extractall(out);dest.mkdir(parents=True)
for name in names:
 a=out/name;b=dest/name;b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
 assert hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest()
print(json.dumps(dict(case=c['case'],state='Native_input_snapshot_verified_and_persistent',files=len(names),ZIP_SHA256=hashlib.sha256(data).hexdigest(),quality_accepted=False,GPU_used=False)))
'''

EXPORT_CONFIDENCE = r'''
from pathlib import Path
import hashlib,io,json,sys,zipfile
import numpy as np
c=json.loads(sys.argv[1]);p=Path('/root/autodl-tmp/kimi_realcase_20261109/samples')/c['version']['case_id']/'step3_worldpoints/vggt_raw.npz'
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
with np.load(p) as raw:
 a=raw['world_points_conf'];assert a.shape==raw['depth'].shape and a.shape[0]==49
 assert a.dtype==np.float32 and np.isfinite(a).all() and (a>=1).all()
 b=io.BytesIO();np.savez_compressed(b,world_points_conf=a);data=b.getvalue()
 np.testing.assert_array_equal(np.load(io.BytesIO(data))['world_points_conf'],a)
 d=dict(case=c['case'],source_path=str(p),source_original_SHA256=h.hexdigest(),array='world_points_conf',shape=list(a.shape),dtype=str(a.dtype),array_SHA256=hashlib.sha256(a.tobytes()).hexdigest(),SHA256=hashlib.sha256(data).hexdigest(),source_array_unchanged=True,exact_activation_floor=1.,floor_pixels49=[int((x==1).sum()) for x in a],scope='Native point-head reliability, not calibrated depth probability',payload_disk_writes_6000D=0)
out=io.BytesIO();stem='CASE'+c['case']+'_NATIVE_POINT_CONFIDENCE49_v1'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 z.writestr(stem+'.npz',data);z.writestr(stem+'.json',json.dumps(d,indent=2))
assert len(out.getvalue())<96*1024**2;sys.stdout.buffer.write(out.getvalue())
'''

IMPORT_CONFIDENCE = r'''
from pathlib import Path
import hashlib,io,json,sys,zipfile
c=json.loads(sys.argv[1]);N=Path(c['N']);Q=Path(c['Q']);data=sys.stdin.buffer.read(96*1024**2+1);assert len(data)<=96*1024**2
z=zipfile.ZipFile(io.BytesIO(data));stem='CASE'+c['case']+'_NATIVE_POINT_CONFIDENCE49_v1';names=[stem+'.npz',stem+'.json'];assert set(z.namelist())==set(names)
d=json.loads(z.read(names[1]));m=json.loads((N/('inputs/case'+c['case']+'_native_CPU_v1/INPUT_MANIFEST.json')).read_text())
assert d['case']==c['case'] and d['source_original_SHA256']==m['raw_original_file_SHA256']
assert hashlib.sha256(z.read(names[0])).hexdigest()==d['SHA256']
for name in names:
 for root in [N,Q]:
  p=root/'inputs'/name;assert not p.exists();p.write_bytes(z.read(name))
print(json.dumps(d))
'''

IMPORT_SOURCE = r'''
from pathlib import Path
import hashlib,json,sys
c=json.loads(sys.argv[1]);N=Path(c['N']);Q=Path(c['Q']);sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps')]
import cv2
from huggingface_hub import HfApi
cv2.setNumThreads(1);v=c['version'];data=sys.stdin.buffer.read(96*1024**2+1)
assert len(data)==v['bytes'] and hashlib.sha256(data).hexdigest()==v['sha256']
m=json.loads((N/('inputs/case'+c['case']+'_native_CPU_v1/INPUT_MANIFEST.json')).read_text())
api=HfApi(token=Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip())
revision=api.repo_info('FrontRear/VLMdataset',repo_type='dataset').sha
rows=api.get_paths_info('FrontRear/VLMdataset',repo_type='dataset',revision=revision,paths=[v['hf_path']])
assert len(rows)==1 and rows[0].size==len(data) and rows[0].lfs.sha256==v['sha256']
paths=[]
for root in [N,Q]:
 d=root/('inputs/case'+c['case']+'_source_video_HF_v1');assert not d.exists();d.mkdir();p=d/'source_A.mkv';p.write_bytes(data);assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];paths.append(str(p))
cap=cv2.VideoCapture(paths[0],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.get(5)==10
for i in range(49):
 ok,im=cap.read();assert ok and im.shape==(704,1280,3) and hashlib.sha256(im.tobytes()).hexdigest()==m['RGB49_BGR_array_SHA256'][i]
assert not cap.read()[0];cap.release()
proof=dict(case=c['case'],state='Native_source49_persistent_and_HF_server_SHA_verified',source_SHA256=v['sha256'],source_HF_path=v['hf_path'],HF_commit=revision,HF_server_size_and_LFS_SHA_matched=True,RGB49_1280x704_10FPS_Native_exact=True,persistent_source=paths[1],quality_accepted=False,new_complete_groups=0,new_HF_cases=0,GPU_used=False)
p=Q/'receipts'/('CASE'+c['case']+'_NATIVE_SOURCE_PERSISTENT_INPUTS_v1.json');assert not p.exists();p.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',nargs='+',required=True);a=ap.parse_args()
    assert 1<=len(a.cases)<=3 and len(set(a.cases))==len(a.cases) and all(x.isdecimal() for x in a.cases)
    lock=(L/'logs/REALCASE_NATIVE_LOOKAHEAD_BATCH.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    board=json.loads((L/'MASTER_WORKBOARD.json').read_text());start=time.time()
    name='NATIVE_LOOKAHEAD_'+'_'.join(a.cases)+'_v1';status=L/'logs'/('REALCASE_'+name+'.json');assert not status.exists()
    result=dict(state='bounded_CPU_lookahead_running',pid=os.getpid(),cases={},GPU_used=False,quality_accepted=False,new_complete_groups=0,new_HF_cases=0,automatic_retry=False)
    def save():status.write_text(json.dumps(result,indent=2))
    save()
    try:
        # Exact input preparation overlaps the human review of the preceding case.
        # Render stages remain sequential under the original shared PCF lock.
        for c in a.cases:
            t=time.time();row=next(x for x in board['cases'] if str(x['case'])==c);v=row['versions'][0]
            assert not row.get('full_case_accepted',False)
            cfg=dict(case=c,version=v,N=N,Q=Q);rec=result['cases'][c]=dict(state='native_inputs_preparing');save()
            payload=remote('6000D','env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 '+PY+' /root/autodl-tmp/StarReal/code/export_native_case_inputs_memory_v1.py --case '+shlex.quote(c))
            rec['native']=json.loads(qpython(IMPORT_NATIVE,cfg,payload));del payload;save()
            payload=remote('6000D','env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 '+PY+' -c '+shlex.quote(EXPORT_CONFIDENCE)+' '+shlex.quote(json.dumps(cfg)))
            conf=json.loads(qpython(IMPORT_CONFIDENCE,cfg,payload));del payload
            p=L/'logs'/('CASE'+c+'_NATIVE_POINT_CONFIDENCE49_v1.json');assert not p.exists();p.write_text(json.dumps(conf,indent=2));rec['confidence_SHA256']=conf['array_SHA256'];save()
            payload=remote('6000D','cat '+shlex.quote(v['version_root']+'/media/source_A.mkv'))
            rec['source']=json.loads(qpython(IMPORT_SOURCE,cfg,payload));del payload
            p=L/'logs'/('CASE'+c+'_NATIVE_SOURCE_PERSISTENT_INPUTS_v1.json');assert not p.exists();p.write_text(json.dumps(rec['source'],indent=2));rec.update(state='Native_inputs_source_confidence_persistent',seconds=time.time()-t);save()
            print(json.dumps(dict(case=c,state=rec['state'],seconds=rec['seconds'])),flush=True)
        for c in a.cases:
            rec=result['cases'][c];tag='BATCH_LOOKAHEAD_BG_V1'
            cmd=[sys.executable,str(L/'code/run_realcase_finite_CPU_component_v1.py'),'--case',c,'--tag',tag,'--kind','pcf','--producer','precompute_native_PCF_background49_QH_cpu_v1.py','--job','outputs/'+c+'/pcf_background49_precomputed_QH_v1/JOB.json','--workers','32']
            log=L/'logs'/('CASE'+c+'_BATCH_LOOKAHEAD_BG_v1.log')
            with log.open('xb') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
            assert r.returncode==0,'Inspect finite background log; no automatic retry'
            r=subprocess.run([sys.executable,str(L/'code/persist_realcase_confidence_background_once_v1.py'),'--case',c],capture_output=True,text=True,timeout=240)
            assert r.returncode==0,r.stderr
            rec['background']=json.loads(r.stdout);rec['state']='exact49_background_and_inputs_verified_persistent';save();print(json.dumps(dict(case=c,state=rec['state'])),flush=True)
        result.update(state='terminal0_bounded_lookahead_complete_NOT_delivery',exit_code=0)
    except BaseException as e:
        result.update(state='terminal1_preserve_successes_inspect_only_failed_component',exit_code=1,error=repr(e));raise
    finally:
        result.update(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-start);save()
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()

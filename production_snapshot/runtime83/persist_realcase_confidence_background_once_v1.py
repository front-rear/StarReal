"""Persist exact completed CPU lookahead inputs; never acceptance or dispatch."""
from pathlib import Path
import argparse,json,subprocess,shlex
L=Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal');ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();assert a.case.isdecimal()
code=r'''
from pathlib import Path
import datetime,hashlib,json,shutil,sys
c=sys.argv[1];assert c.isdecimal();N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');receipt=Q/'receipts'/('CASE'+c+'_PERSISTENT_CONFIDENCE_AND_BACKGROUND_v1.json');assert not receipt.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((N/('inputs/case'+c+'_native_CPU_v1/INPUT_MANIFEST.json')).read_text());conf=N/'inputs'/('CASE'+c+'_NATIVE_POINT_CONFIDENCE49_v1.json');d=json.loads(conf.read_text());npz=conf.with_suffix('.npz');assert d['case']==c and d['source_original_SHA256']==m['raw_original_file_SHA256'] and sha(npz)==d['SHA256']
b=N/'outputs'/c/'pcf_background49_precomputed_QH_v1';job=json.loads((b/'JOB.json').read_text());r=json.loads((b/'RESULT.json').read_text());assert job['exit_code']==0 and job['frames_finished']==r['frames']==49 and not r['renderer_modified'] and r['camera_source_SHA256']==sha(N/('inputs/case'+c+'_native_CPU_v1/cameras.npz'))
for row in r['frame_receipts']:assert sha(b/'shared_pcf'/('%06d.png'%row['frame']))==row['PNG_SHA256']
records=[]
for src,dst in [(conf,Q/'inputs'/conf.name),(npz,Q/'inputs'/npz.name)]+[(x,Q/'component_cache'/c/'pcf_background49_precomputed_QH_v1'/x.relative_to(b)) for x in sorted(b.rglob('*')) if x.is_file() and not any(y.startswith('._') for y in x.relative_to(b).parts)]:
 dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert sha(src)==sha(dst)
 else:shutil.copyfile(src,dst);assert sha(src)==sha(dst)
 records.append(dict(path=str(dst.relative_to(Q)),SHA256=sha(dst),bytes=dst.stat().st_size))
p=dict(case=c,state='Native_confidence_and_exact_camera_background49_persistent_verified',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),confidence_array_SHA256=d['array_SHA256'],camera_SHA256=r['camera_source_SHA256'],background49_seconds=r['elapsed_seconds'],original_inputs_unchanged=True,files=records,quality_accepted=False,new_complete_groups=0,new_HF_cases=0,GPU_used=False);receipt.write_text(json.dumps(p,indent=2));print(json.dumps(p))
'''
r=subprocess.run(['ssh','cscg-qh00','python3 -c '+shlex.quote(code)+' '+shlex.quote(a.case)],capture_output=True,timeout=180);assert r.returncode==0,r.stderr.decode();p=L/'logs'/('CASE'+a.case+'_PERSISTENT_CONFIDENCE_AND_BACKGROUND_v1.json');assert not p.exists();p.write_bytes(r.stdout);d=json.loads(r.stdout);print(json.dumps({k:v for k,v in d.items() if k!='files'}),flush=True)

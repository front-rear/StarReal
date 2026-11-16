"""Record only verified component preparation, with three-root CAS guards."""
from pathlib import Path
import datetime,hashlib,json,shlex,shutil,subprocess
L=Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal')
names=['CONTINUATION_STATE.json','MASTER_WORKBOARD.json','LAMP_DIVERSITY_CURRENT.json','LATEST_FACTS.json']
pn='REALCASE_ACCELERATED_PIPELINE_CURRENT.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
batch=json.loads((L/'logs/REALCASE_NATIVE_LOOKAHEAD_58_64_v1.json').read_text())
assert batch['exit_code']==0 and batch['state']=='terminal0_bounded_lookahead_complete_NOT_delivery'
assert set(batch['cases'])=={'58','64'}
for c,r in batch['cases'].items():
 assert r['state']=='exact49_background_and_inputs_verified_persistent' and r['source']['RGB49_1280x704_10FPS_Native_exact']
 assert r['source']['HF_server_size_and_LFS_SHA_matched'] and not r['source']['quality_accepted']
 support=json.loads((L/'reviews/CASE56_CACHED_TABLE_AND_CORD_ACTUAL_REVIEW_v1.json').read_text())
assert support['case']=='56' and not support['quality_accepted']
old={n:sha(L/n) for n in names+[pn]}
facts=json.loads((L/'LATEST_FACTS.json').read_text());assert facts['current_qualified_conditioning_cases']==30
assert facts['HF_latest_verified_reconstruction_receipt']['case']=='53'
p=json.loads((L/pn).read_text());assert p['validated_delivery']['case']=='53'
# Correct a metadata-only copied-prefix typo against the actual signed HF receipt.
actual_prefix=str(Path(facts['HF_latest_verified_reconstruction_receipt']['add_only_paths'][0]).parent)
prefix_before=p['validated_delivery']['HF_prefix'];p['validated_delivery']['HF_prefix']=actual_prefix
assert actual_prefix=='StarReal/53/revisions/fixed_world_yellow_depth_v5'
p['operational_changes']=list(dict.fromkeys(p['operational_changes']))
p['operational_changes'].append('Generic identity-pinned finite two-case Native/source/confidence preparation replaces per-case script copying. Existing original locks serialize shared PCF backgrounds while current-case placement work proceeds independently.')
p['future_unclaimed_control'].update(native_prefetch_runner='prefetch_realcase_native_batch_once_v1.py',native_prefetch_runner_SHA256=sha(L/'code/prefetch_realcase_native_batch_once_v1.py'),next_case_cache_depth=2,identity_from_actual_workboard=True,no_per_case_text_substitution=True)
look=dict(state='two_next_cases_Native_and_background49_prepared_verified_persistent',primary_work_case='56',prepared_next_cases=['58','64'],cases={c:dict(source_input_receipt='logs/CASE'+c+'_NATIVE_SOURCE_PERSISTENT_INPUTS_v1.json',background_receipt='logs/CASE'+c+'_PERSISTENT_CONFIDENCE_AND_BACKGROUND_v1.json',source_SHA256=r['source']['source_SHA256'],camera_SHA256=r['background']['camera_SHA256'],background49_seconds=r['background']['background49_seconds'],source49_exact=True,background49_exact=True,quality_accepted=False,new_complete_groups=0,new_HF_cases=0) for c,r in batch['cases'].items()},batch_receipt='logs/REALCASE_NATIVE_LOOKAHEAD_58_64_v1.json',batch_elapsed_seconds=batch['elapsed_seconds'],no_per_case_text_substitution=True,GPU_used=False,new_complete_groups=0,new_HF_cases=0)
p.update(state='current56_support_checked_next58_64_two_case_inputs_background49_ready',lookahead_batch=look,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
p['lookahead'].update(source_power_cord_detail_and_native_table_support_pending=False,actual_Native_table_support_probe_complete=True,actual_source_cord_detail_viewed=True,private_observed_cord_mask_repair_pending=True,actual_support_and_target_placement_pilots_pending=True,cached_support_review='reviews/CASE56_CACHED_TABLE_AND_CORD_ACTUAL_REVIEW_v1.json')
rels=['code/prefetch_realcase_native_batch_once_v1.py','code/probe_case56_cached_table_support_QH_v1.py','code/record_realcase_two_case_lookahead_speedup_v1.py','reviews/CASE56_CACHED_TABLE_AND_CORD_ACTUAL_REVIEW_v1.json','logs/REALCASE_NATIVE_LOOKAHEAD_58_64_v1.json','logs/CASE56_NATIVE_SUPPORT_V1_FINITE_CALL_v1.json']
rels += ['logs/CASE'+c+s for c in ['58','64'] for s in ['_NATIVE_SOURCE_PERSISTENT_INPUTS_v1.json','_NATIVE_POINT_CONFIDENCE49_v1.json','_PERSISTENT_CONFIDENCE_AND_BACKGROUND_v1.json','_BATCH_LOOKAHEAD_BG_V1_FINITE_CALL_v1.json']]
files={n:(L/n).read_text() for n in rels}
cfg=dict(old=old,names=names,policy_name=pn,policy=p,files=files,lookahead=look,prefix_before=prefix_before,prefix_after=actual_prefix)
code=r'''
from pathlib import Path
import datetime,fcntl,hashlib,json,shutil,sys
R=Path('/root/autodl-tmp/StarReal');d=json.load(sys.stdin);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=(R/'logs/ROOT_FACTS_UPDATE.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,h in d['old'].items():assert sha(R/n)==h,n
for n,s in d['files'].items():
 p=R/n
 if p.exists():assert p.read_text()==s,n
at=datetime.datetime.now(datetime.timezone.utc);eid='CODEX_REALCASE_TWO_CASE_LOOKAHEAD_SPEEDUP_'+at.strftime('%Y%m%dT%H%M%SZ');rel='events/'+eid+'.json';cp=R/'events/checkpoints'/eid;cp.mkdir(exist_ok=False)
for n in d['old']:shutil.copy2(R/n,cp/n)
def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(x);t.replace(p)
for n,s in d['files'].items():
 p=R/n
 if not p.exists():put(p,s)
for n in d['names']:
 p=R/n;x=json.loads(p.read_text());x.update(at=at.isoformat(),latest_event_path=rel,realcase_CPU_lookahead_batch_current=d['lookahead']);x['realcase_CPU_lookahead_current']=d['policy']['lookahead']
 x['actual_realcase_acceleration']['next_two_cases_prepared_ahead']=d['lookahead']
 x['latest_fixed_world_yellow_v5_release_facts']['next_two_cases_prepared_ahead']=d['lookahead']
 if 'current_qualified_conditioning_cases' in x:assert x['current_qualified_conditioning_cases']==30
 put(p,json.dumps(x,ensure_ascii=False,indent=2))
put(R/d['policy_name'],json.dumps(d['policy'],ensure_ascii=False,indent=2))
put(R/'logs/YELLOW_RELEASE_ROOT_EXPECTED_SHA_v5.json',json.dumps({n:sha(R/n) for n in d['names']},indent=2))
event=dict(event_id=eid,at=at.isoformat(),automation_wake=False,state='finite_two_case_CPU_lookahead_verified',lookahead=d['lookahead'],current56_support_probe_actual_review=d['policy']['lookahead'],metadata_only_prior_prefix_corrected=dict(before=d['prefix_before'],after=d['prefix_after'],actual_HF_receipt_unchanged=True),current_qualified_conditioning_cases=30,new_complete_groups=0,new_HF_cases=0,scope_complete=False)
put(R/rel,json.dumps(event,ensure_ascii=False,indent=2));put(R/'events/LATEST_EVENT_PATH.txt',str(R/rel)+'\n')
ns=list(d['old'])+['logs/YELLOW_RELEASE_ROOT_EXPECTED_SHA_v5.json',rel,'events/LATEST_EVENT_PATH.txt']
print(json.dumps({n:dict(text=(R/n).read_text(),SHA256=sha(R/n)) for n in ns}))
'''
r=subprocess.run(['ssh','6000D','/root/miniconda3/bin/python -c '+shlex.quote(code)],input=json.dumps(cfg).encode(),capture_output=True,timeout=120);assert r.returncode==0,r.stderr.decode();bundle=json.loads(r.stdout)
for n,h in old.items():assert sha(L/n)==h,n
event=next(n for n in bundle if n.startswith('events/CODEX_'))
cp=L/'events/checkpoints'/Path(event).stem;cp.mkdir(exist_ok=False)
for n in old:shutil.copy2(L/n,cp/n)
for n,row in bundle.items():
 p=L/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(row['text']);assert sha(p)==row['SHA256']
qcode=r'''
from pathlib import Path
import fcntl,hashlib,json,shutil,sys
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');d=json.load(sys.stdin);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=(Q/'logs/ROOT_CANONICAL_R_SNAPSHOT_MIRROR.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
for n,h in d['old'].items():assert sha(Q/n)==h,n
cp=Q/'events/checkpoints'/Path(d['event']).stem;cp.mkdir(exist_ok=False)
for n in d['old']:shutil.copy2(Q/n,cp/n)
for n,row in d['bundle'].items():
 p=Q/n if n in d['old'] else Q/'canonical_R_mirror'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(row['text']);assert sha(p)==row['SHA256']
for n,s in d['files'].items():
 if n.startswith(('reviews/','code/')):
  p=Q/n;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert p.read_text()==s,n
  else:p.write_text(s)
print(json.dumps(dict(state='three_roots_CAS_verified_and_SHA_exact',root_SHA256={n:sha(Q/n) for n in d['old']})))
'''
r=subprocess.run(['ssh','cscg-qh00','python3 -c '+shlex.quote(qcode)],input=json.dumps(dict(old=old,bundle=bundle,files=files,event=event)).encode(),capture_output=True,timeout=120);assert r.returncode==0,r.stderr.decode();proof=json.loads(r.stdout)
out=dict(state='speedup_batch_actual_completed_and_three_roots_synced',lookahead=look,QH=proof,current_qualified_conditioning_cases=30,new_complete_groups=0,new_HF_cases=0,source56_cord_mask_repair_pending=True,not_end_to_end_throughput_benchmark=True)
p=L/'logs/TWO_CASE_LOOKAHEAD_SPEEDUP_FINAL_v1.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['QH','lookahead']},ensure_ascii=False),flush=True)

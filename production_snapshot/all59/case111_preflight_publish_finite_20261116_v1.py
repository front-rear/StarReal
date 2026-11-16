from pathlib import Path
import subprocess,json,hashlib,os,time,fcntl
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1'
lock=(U/'CASE111_PREFLIGHT_PUBLISH_FINITE_20261116_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
state=U/'CASE111_PREFLIGHT_PUBLISH_FINITE_20261116_v1.json';d=dict(case='111',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split()[21],state='running_preflight',GPU_used=False,steps=[])
def save():state.write_text(json.dumps(d,indent=2))
save()
try:
 rc=subprocess.call(['python3',str(U/'prepare_new_case_publication_v5.py'),'--case','111','--revision','fixed_world_yellow_depth_v5']);d['steps'].append(dict(step='strict_combined_preflight',exit_code=rc));save();assert rc==0
 p=U/'CASE111_PUBLICATION_PAYLOAD.json';payload=json.loads(p.read_text());pre=payload['preflight'];assert (pre['actual_reader_frames'],pre['actual_joint_frames'],pre['operation_references_checked'])==(588,490,5)
 rel=Q/'releases/111/revisions/fixed_world_yellow_depth_v5/RELEASE.json';release=json.loads(rel.read_text());release.update(latest_contract_ready=True,quality_scope='Combined reviewed core archive plus SHA-bound Remove source and solid color supplements passed actual588 reader,490joints,5references and15videos; pending server verification',required_before_latest_contract_upload=['unique writer server file/size/SHA verification'],latest_combined_visual_review=str(U/'CASE111_LATEST_VISUAL_ACCEPTANCE_v1.json'),latest_combined_visual_review_SHA256=sha(U/'CASE111_LATEST_VISUAL_ACCEPTANCE_v1.json'),actual_latest_reader_condition_frames=588,actual_latest_reader_joint_frames=490)
 rel.write_text(json.dumps(release,ensure_ascii=False,indent=2))
 rows=[x for x in payload['files'] if Path(x['local_path'])==rel];assert len(rows)==1;rows[0].update(bytes=rel.stat().st_size,SHA256=sha(rel));p.write_text(json.dumps(payload,ensure_ascii=False,indent=2))
 d['state']='running_unique_HF_writer';save()
 rc=subprocess.call(['python3',str(U/'publish_preflight_verified_case_v1.py'),'--case','111']);d['steps'].append(dict(step='unique_writer_full_public_GET_SHA',exit_code=rc));save();assert rc==0
 d.update(state='terminal0_HF_final_receipt_verified',exit_code=0);save()
except Exception as e:d.update(state='failed',exit_code=1,error=repr(e));save();raise

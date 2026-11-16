from pathlib import Path
import subprocess,os,json,fcntl
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
lock=(U/'CASE106_STAGE_RELEASE_PREPARE_FINITE_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
d=dict(case='106',pid=os.getpid(),state='running',steps=[],GPU_used=False,HF_upload=False);p=U/'CASE106_STAGE_RELEASE_PREPARE_FINITE_v1.json'
for script,args in [
('stage_case106_reviewed_native_portable_QH_v1.py',[]),
('release_case106_native_node_local_archive_QH_only_v1.py',['--case','106']),
('copy_case106_native_accepted_archive_to_durable_QH_v1.py',[]),
('prepare_case106_native_latest_contract_v1.py',['--case','106','--revision','fixed_world_yellow_depth_v5'])]:
 p.write_text(json.dumps(d));rc=subprocess.call(['python3',str(U/script)]+args);d['steps'].append(dict(script=script,exit_code=rc));p.write_text(json.dumps(d));assert rc==0
d['state']='terminal0_prepared_requires_actual_Remove_final_review';p.write_text(json.dumps(d))

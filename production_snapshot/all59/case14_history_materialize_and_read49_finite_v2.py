from pathlib import Path
import subprocess,json,os,time
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
d=dict(pid=os.getpid(),steps=[],GPU_used=False,HF_upload=False);p=U/'CASE14_HISTORY_MATERIALIZE_READ49_FINITE_v2.json'
for name in ['import_case14_verified_archive_members_history_QH_finite_20261116_v2.py','read_case14_actual_history49_QH_finite_20261116_v1.py']:
 d['step']=name;p.write_text(json.dumps(d));rc=subprocess.call(['python3',str(U/name)]);d['steps'].append(dict(name=name,exit_code=rc));p.write_text(json.dumps(d));assert rc==0
d['state']='terminal0_actual_history2_cameras_13movies49_ready_for_actual_review';p.write_text(json.dumps(d))

from pathlib import Path
import subprocess,os,json
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
p=U/'CASE27_HISTORY_MATERIALIZE_READ49_FINITE_v1.json';d=dict(pid=os.getpid(),steps=[],GPU_used=False,HF_upload=False)
for name in ['import_case27_verified_archive_and_actual_HF_history_QH_finite_20261116_v1.py','read_case27_actual_history49_QH_finite_20261116_v1.py']:
 d['step']=name;p.write_text(json.dumps(d));rc=subprocess.call(['python3','-u',str(U/name)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));d['steps'].append(dict(name=name,exit_code=rc));p.write_text(json.dumps(d));assert rc==0
d.update(state='terminal0_pending_actual_historical_review',exit_code=0);p.write_text(json.dumps(d))

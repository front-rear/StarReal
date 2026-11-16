from pathlib import Path
import os,subprocess,json,datetime
U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
steps=[['python3','-u',str(U/'release_case82_node_local_archive_QH_only_v1.py'),'--case','82'],['python3','-u',str(U/'copy_case82_accepted_archive_to_durable_QH_v2.py')],['python3','-u',str(U/'prepare_new_case_latest_contract_v5.py'),'--case','82','--revision','fixed_world_yellow_depth_v5']]
env=dict(os.environ,STARREAL_QH_ONLY_COMPONENT_OWNER='1',CUDA_VISIBLE_DEVICES='')
for i,argv in enumerate(steps):
 print(json.dumps({'step':i,'argv':argv}),flush=True)
 r=subprocess.run(argv,env=env)
 if r.returncode:raise SystemExit(r.returncode)
(U/'CASE82_RELEASE_COPY_REMOVE_PREP_TERMINAL_20261116_v1.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':0,'automatic_upload':False,'requires_actual_Remove_visual_review':True}))

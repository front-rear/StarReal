import pathlib,subprocess,json,os,datetime
Q=pathlib.Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1';rows=[]
for case in ['108','109']:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',STARREAL_CASE=case,STARREAL_FRAME_WORKERS='32',STARREAL_QH_ONLY_COMPONENT_OWNER='1',STARREAL_TRUE4K_CAMERA_COMPONENT=str(Q/'component_cache'/case/'true4K_RGB_camera_component_v2'))
 log=U/('CASE'+case+'_TRUE4K_PCF_BACKGROUND49_FINITE_20261116_v1.log')
 with log.open('xb') as f:
  proc=subprocess.Popen(['python3',str(U/'precompute_true4K_camera_component_PCF_background49_QH_v3.py')],stdout=f,stderr=subprocess.STDOUT,env=env)
  row=dict(case=case,PID=proc.pid,startticks=pathlib.Path('/proc/'+str(proc.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),log=str(log),state='running_finite49',GPU_used=False)
  (U/'TWO_TRUE4K_BACKGROUND_CURRENT_20261116_v1.json').write_text(json.dumps(dict(current=row,finished=rows),indent=2))
  try:row['returncode']=proc.wait(timeout=900);row['state']='terminal'
  except subprocess.TimeoutExpired:
   row['state']='bounded_controller_wait_ended_child_not_signalled';rows.append(row);break
 rows.append(row)
 if row['returncode']!=0:break
 result=Q/'component_cache'/case/'pcf_background49_true4K_Original_K_v2/RESULT.json';r=json.loads(result.read_text());assert r['frames']==49 and r['renderer_modified']==False;row['verified_frames']=49;row['result']=str(result)
(U/'TWO_TRUE4K_BACKGROUND_RESULT_20261116_v1.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,GPU_used=False,quality_accepted=False,formal_group_increment=0),indent=2))

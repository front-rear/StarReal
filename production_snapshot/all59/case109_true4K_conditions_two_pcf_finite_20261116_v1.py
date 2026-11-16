import pathlib,json,subprocess,os,datetime
R=pathlib.Path('/tmp/starreal_uid1007_case109_true4K_cpu_v2');U=pathlib.Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
env=dict(os.environ,STARREAL_CPU_ROOT=str(R),STARREAL_CASE='109',STARREAL_FRAME_WORKERS='32',STARREAL_GEOMETRY_REVIEW='CASE109_FIVE_ROLE_ALL49_GEOMETRY_ACTUAL_REVIEW_v1.json',CUDA_VISIBLE_DEVICES='')
rows=[]
for name,p in [('conditions',U/'build_case109_true4K_cached_conditions49_QH_v4.py'),('independent_conditions490',R/'verify_native_conditions49_QH_v2.py'),('source_appearance_hint',R/'prepare_native_source_appearance_reference_QH_v2.py'),('two_PCF49',R/'build_native_case_two_pcf49_QH_cpu_v5.py')]:
 log=U/('CASE109_TRUE4K_'+name+'_FINITE_20261116_v1.log')
 with log.open('xb') as f:
  proc=subprocess.Popen(['python3','-u',str(p)],env=env,stdout=f,stderr=subprocess.STDOUT);row=dict(step=name,PID=proc.pid,startticks=pathlib.Path('/proc/'+str(proc.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],log=str(log))
  (U/'CASE109_TRUE4K_CONDITIONS_TWO_PCF_CURRENT_20261116_v1.json').write_text(json.dumps(dict(current=row,finished=rows),indent=2))
  try:row['returncode']=proc.wait(timeout=1200)
  except subprocess.TimeoutExpired:row['state']='finite_controller_wait_expired_child_not_signalled';rows.append(row);break
 rows.append(row)
 if row['returncode']!=0:break
(U/'CASE109_TRUE4K_CONDITIONS_TWO_PCF_RESULT_20261116_v1.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,GPU_used=False,quality_accepted=False,HF_upload_started=False),indent=2))

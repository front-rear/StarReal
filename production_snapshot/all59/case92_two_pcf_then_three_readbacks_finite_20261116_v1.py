from pathlib import Path
import subprocess,concurrent.futures,json
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');U=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1')
r=subprocess.run(['python3','-u',str(N/'code/build_native_case_two_pcf49_QH_cpu_v5.py')])
if r.returncode:raise SystemExit(r.returncode)
assert json.loads((N/'outputs/92/two_pcf49_fixed_world_yellow_QH_v1/JOB.json').read_text())['exit_code']==0
steps=[['python3','-u',str(N/'code/review_native_conditions_and_joint_pixels_QH_v2.py')],['/mnt/atlas/fangrui/repro_video_20261105/_envs/pisco/bin/python','-u',str(N/'code/check_native_actual_Original_entry_QH_v2.py')],['python3','-u',str(N/'code/build_native_historical_comparison49_QH_cpu_v3.py')]]
def run(argv):
 name=Path(argv[-1]).name
 with (U/('CASE92_POST_'+name+'.log')).open('xb') as out:r=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT)
 return dict(script=name,returncode=r.returncode)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:results=list(ex.map(run,steps))
(U/'CASE92_POST_READBACKS_TERMINAL_20261116_v1.json').write_text(json.dumps({'results':results,'quality_or_HF':False},indent=2))
for r in results:print(json.dumps(r),flush=True)
if any(r['returncode'] for r in results):raise SystemExit(1)

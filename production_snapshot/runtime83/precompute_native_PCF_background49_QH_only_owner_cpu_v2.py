"""Finite exact camera-only background cache; no box/condition acceptance."""
from pathlib import Path
import concurrent.futures,fcntl,hashlib,importlib.util,json,multiprocessing,os,shutil,sys,time,traceback
N=Path(os.environ.get('STARREAL_CPU_ROOT','/tmp/starreal_uid1007_case19_cpu_v1'));CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal()
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
I=N/f'inputs/case{CASE}_native_CPU_v1';OUT=N/f'outputs/{CASE}/pcf_background49_precomputed_QH_v1';W=int(os.environ.get('STARREAL_FRAME_WORKERS','32'));assert 1<=W<=32
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(N/'runtime/qh_pcf_cpu_v1')]
import cv2,numpy as np
from pcf_cpu_accelerator import CPUTrace
CTX={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def init():
 cv2.setNumThreads(1);p=json.loads((N/'HUMAN_DEPTH_AWARE_YELLOW_OVERLAP_CURRENT.json').read_text());f=N/p['renderer_copy'];assert sha(f)==p['renderer_SHA256']
 spec=importlib.util.spec_from_file_location('approved_precomputed_lattice',f);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);mod.trace_lattice=CPUTrace(2)
 cams=np.load(I/'cameras.npz');CTX.update(mod=mod,cams=cams,a0=cams['w2c_A'][0],policy=p)
def frame(i):
 c=CTX;K=c['cams']['K'][i];B=c['cams']['w2c_B'][i];C=np.linalg.inv(B@np.linalg.inv(c['a0']))
 prepared=c['mod'].prepare_lattice(C,K,1280,704,2,.5,64.,4096)
 rgb,aux=c['mod'].render(C,K,[],1280,704,2,.5,64.,4096,prepared=prepared);assert not aux['unresolved'].any()
 path=OUT/'shared_pcf'/f'{i:06d}.png';assert cv2.imwrite(str(path),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));read=cv2.imread(str(path))[:,:,::-1].copy();np.testing.assert_array_equal(read,rgb)
 row=dict(frame=i,pid=os.getpid(),PNG_SHA256=sha(path),RGB_array_SHA256=hashlib.sha256(rgb.tobytes()).hexdigest(),K_array_SHA256=hashlib.sha256(K.tobytes()).hexdigest(),normalized_c2w_array_SHA256=hashlib.sha256(C.tobytes()).hexdigest(),unresolved=0)
 put(OUT/'frame_receipts'/f'{i:06d}.json',row);return row
def main():
 assert os.environ.get('STARREAL_QH_ONLY_COMPONENT_OWNER')=='1'
 lock=(Q/'logs/reconstructed_joint_cpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert len(os.sched_getaffinity(0))>=W*2 and shutil.disk_usage(N).free>16*1024**3
 mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024;assert mem>W*2*1024**3
 native=json.loads((I/'INPUT_MANIFEST.json').read_text());assert native['case']==CASE and native['Native_RGB49_exact']
 parity=json.loads((N/'receipts/QH_PRIVATE_PCF_KERNEL_REFERENCE_EXACT_v1.json').read_text());assert parity['parity']['full_RGB_and_all_depth_aux_arrays_exact']
 p=json.loads((N/'HUMAN_DEPTH_AWARE_YELLOW_OVERLAP_CURRENT.json').read_text());assert sha(N/p['renderer_copy'])==p['renderer_SHA256']
 cams=np.load(I/'cameras.npz');assert cams['K'].shape==(49,3,3) and cams['w2c_A'].shape==cams['w2c_B'].shape==(49,4,4)
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'shared_pcf').mkdir();(OUT/'frame_receipts').mkdir();camera_sha=sha(I/'cameras.npz');start=time.time();rows={}
 state=dict(execution_owner_scope='QH_original_CPU_PCF_lock_only_no6000D_owner_claim',case=CASE,state='running_camera_only_background49',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],frames_finished=0,frames_expected=49,frame_workers=W,GPU_used=False,quality_accepted=False,new_complete_groups=0)
 put(OUT/'JOB.json',state)
 try:
  with concurrent.futures.ProcessPoolExecutor(max_workers=W,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
   for f in concurrent.futures.as_completed([pool.submit(frame,i) for i in range(49)]):
    row=f.result();rows[row['frame']]=row;state.update(frames_finished=len(rows),actual_worker_pids=sorted({x['pid'] for x in rows.values()}),elapsed_seconds=time.time()-start);put(OUT/'JOB.json',state)
    if len(rows)%7==0:print(json.dumps(dict(frames_finished=len(rows),elapsed_seconds=state['elapsed_seconds'])),flush=True)
  assert sha(I/'cameras.npz')==camera_sha
  result=dict(case=CASE,state='exact_camera_only_background49_complete_not_conditioning_delivery',frames=49,resolution=[1280,704],AA=2,spacing=.5,far_cells=64.,max_steps=4096,renderer_SHA256=p['renderer_SHA256'],renderer_modified=False,camera_source=str(I/'cameras.npz'),camera_source_SHA256=camera_sha,frame_receipts=[rows[i] for i in range(49)],world_normalization='inverse(B_i @ inverse(A_0))',Native_K_E_unchanged=True,consumer_requires_exact_all_K_C_arrays_and_renderer_and_parameters=True,consumer_first_frame_full_vs_sparse_validation_required=True,GPU_used=False,quality_accepted=False,new_complete_groups=0,new_HF_cases=0,elapsed_seconds=time.time()-start)
  put(OUT/'RESULT.json',result);state.update(state='terminal0_exact_background49_not_delivery',exit_code=0,elapsed_seconds=time.time()-start);put(OUT/'JOB.json',state);print(json.dumps({k:v for k,v in result.items() if k!='frame_receipts'}),flush=True)
 except BaseException as e:state.update(state='terminal1_preserve_partial_no_retry',exit_code=1,error=repr(e),traceback=traceback.format_exc());put(OUT/'JOB.json',state);raise
if __name__=='__main__':main()

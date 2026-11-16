"""Measure the restored exact driver against full rendering on actual case72 cameras/boxes."""
from pathlib import Path
import fcntl, hashlib, importlib.util, json, os, sys, time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1')
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
U=Q/'all59_reconstruction_20261115_v1'
os.environ.update(STARREAL_CASE='72',STARREAL_CPU_ROOT=str(N),STARREAL_QH_ONLY_COMPONENT_OWNER='1',STARREAL_FRAME_WORKERS='32',CUDA_VISIBLE_DEVICES='')
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(N/'runtime/qh_pcf_cpu_v1')]
import cv2,numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.time();cv2.setNumThreads(1);guard=(Q/'logs/reconstructed_joint_cpu.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 consumer=N/'build_native_case_two_pcf49_QH_cpu_v5.py';spec=importlib.util.spec_from_file_location('case72_restored_driver_under_test',consumer);driver=importlib.util.module_from_spec(spec);sys.modules[spec.name]=driver;spec.loader.exec_module(driver);mod,policy=driver.approved()
 cache=driver.BG_CACHE;meta=json.loads((cache/'RESULT.json').read_text());job=json.loads((cache/'JOB.json').read_text());assert job['exit_code']==0 and job['frames_finished']==49
 cams=np.load(driver.SOURCE/'cameras49.npz');a0=cams['w2c_A'][0];cm=json.loads((driver.SOURCE/'MANIFEST.json').read_text());jobs={}
 for name,info in cm['operations'].items():
  with np.load(driver.SOURCE/name/'fixed_world_geometry.npz') as z:g={k:z[k] for k in z.files}
  boxes=[]
  for key,label in [('source',1),('target',2)]:
   if key=='source' and info['operation']=='insert' or key=='target' and info['operation']=='remove':continue
   boxes.append(mod.Box((a0@np.r_[g[key+'_center'],1.])[:3],g[key+'_size']/2,a0[:3,:3]@g[key+'_rotation'],label))
  jobs[name]=boxes
 records=[]
 for i in range(49):
  K=cams['K'][i];C=np.linalg.inv(cams['w2c_B'][i]@np.linalg.inv(a0));row=meta['frame_receipts'][i];p=cache/'shared_pcf'/f'{i:06d}.png'
  assert row['frame']==i and sha(p)==row['PNG_SHA256'];assert hashlib.sha256(K.tobytes()).hexdigest()==row['K_array_SHA256'] and hashlib.sha256(C.tobytes()).hexdigest()==row['normalized_c2w_array_SHA256']
  rgb=cv2.imread(str(p))[:,:,::-1].copy();assert rgb.shape==(704,1280,3) and hashlib.sha256(rgb.tobytes()).hexdigest()==row['RGB_array_SHA256']
  if i not in [0,24,48]:continue
  prepared=mod.prepare_lattice(C,K,1280,704,2,.5,64.,4096);reference,_=mod.render(C,K,[],1280,704,2,.5,64.,4096,prepared=prepared);np.testing.assert_array_equal(reference,rgb)
  for name,boxes in jobs.items():
   sparse,saux=mod.render(C,K,boxes,1280,704,2,.5,64.,4096,base_rgb=rgb);full,faux=mod.render(C,K,boxes,1280,704,2,.5,64.,4096,prepared=prepared);np.testing.assert_array_equal(sparse,full);selected=saux['depth_computed']
   for key in ['pcf_depth_m','box_depth_m','box_visible','pcf_visible','labels','unresolved']:np.testing.assert_array_equal(saux[key][selected],faux[key][selected])
   assert driver.stats(saux)==driver.stats(faux);records.append(dict(frame=i,variant=name,RGB_differing_pixels=0,all_selected_depth_aux_exact=True))
  print(json.dumps(dict(frame=i,all5_variants_exact=True,seconds=time.time()-start)),flush=True)
 result=dict(state='restored_exact_driver_cache_full_render_equivalence_verified',case='72',all49_cached_PNG_K_C_array_SHA_readback=49,full_render_sample_frames=[0,24,48],all_native_RGB_and_selected_depth_aux_exact=True,records=records,consumer_SHA256=sha(consumer),renderer_SHA256=policy['renderer_SHA256'],renderer_modified=False,scientific_PCFlattice_parameters_unchanged=True,not_visual_acceptance=True,GPU_used=False,elapsed_seconds=time.time()-start)
 out=U/'CASE72_RESTORED_DRIVER_ACTUAL_PARITY_v1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
 dst=N/'receipts/EXACT_BACKGROUND_CACHE_CONSUMER_STREAM_V4_INTEGRATION_v1.json';backup=dst.with_name('EXACT_BACKGROUND_CACHE_CONSUMER_STREAM_V4_INTEGRATION_PREVIOUS_RESTORED_SOURCE.json');assert not backup.exists();dst.rename(backup);dst.write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()

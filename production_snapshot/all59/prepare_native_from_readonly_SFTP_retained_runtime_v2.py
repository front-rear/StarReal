"""Verify original files moved by SFTP, perform all decoding/array work on QH CPU."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,io,json,os,shutil,sys,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');sys.path.insert(0,str(N/'runtime/qh_cpu_py39_v1'))
import cv2,numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists();p.write_text(json.dumps(x,indent=2))
def main():
 a=argparse.ArgumentParser();a.add_argument('--transfer',required=True);a.add_argument('--spec',required=True);a.add_argument('--cases',nargs='+',required=True);args=a.parse_args();root=Path(args.transfer);assert root.parent==N/'inputs';s=json.loads(Path(args.spec).read_text());s['cases']=[x for x in s['cases'] if x['case'] in args.cases];assert 1<=len(s['cases'])<=2 and {x['case'] for x in s['cases']}==set(args.cases);cv2.setNumThreads(1);start=time.time();rows=[]
 guard=(Q/'logs/READONLY_SFTP_NATIVE_PREPARATION.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 sys.path.insert(0,str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps'))
 from huggingface_hub import HfApi
 api=HfApi(token=Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip());head=api.repo_info('FrontRear/VLMdataset',repo_type='dataset').sha
 for cfg in s['cases']:
  c=cfg['case'];v=cfg['version'];server=api.get_paths_info('FrontRear/VLMdataset',repo_type='dataset',revision=head,paths=[v['hf_path']]);assert len(server)==1 and server[0].size==v['bytes'] and server[0].lfs.sha256==v['sha256'];d=root/c;out=N/f'inputs/case{c}_native_CPU_v1';dur=Q/f'inputs/case{c}_native_CPU_v1';assert not out.exists() and not dur.exists();source=d/'source_A.mkv';assert sha(source)==v['sha256'] and source.stat().st_size==v['bytes'];rawpath=d/'vggt_raw.npz';original_raw=next(r['remote'] for r in cfg['read_only_original_paths'] if r['local']=='vggt_raw.npz');rawhash=sha(rawpath)
  with np.load(rawpath) as raw:
   arrays={k:raw[k] for k in ['depth','K','extrinsic_3x4']};conf=raw['world_points_conf'];assert arrays['depth'].shape[0]==49 and conf.shape==arrays['depth'].shape and conf.dtype==np.float32 and np.isfinite(conf).all() and (conf>=1).all()
  b=io.BytesIO();np.savez_compressed(b,**arrays);subset=b.getvalue();proof={}
  with np.load(io.BytesIO(subset)) as z:
   for k,x in arrays.items():np.testing.assert_array_equal(x,z[k]);proof[k]=dict(shape=list(x.shape),dtype=str(x.dtype),array_SHA256=ah(x))
  cap=cv2.VideoCapture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);assert cap.get(5)==10;pixels=[]
  for i in range(49):
   ok,im=cap.read();assert ok and im.shape==(704,1280,3);original=cv2.imread(str(d/'original_frames49'/f'{i:06d}.png'));np.testing.assert_array_equal(im,original);pixels.append(ah(im));mask=cv2.imread(str(d/'original_mask49'/f'{i:06d}.png'),0);assert mask is not None and mask.shape==(704,1280)
  assert not cap.read()[0];cap.release();cam=np.load(d/'cameras.npz');K=arrays['K'].astype(float).copy();K[:,0]*=1280/arrays['depth'].shape[2];K[:,1]*=704/arrays['depth'].shape[1];np.testing.assert_array_equal(K,cam['K']);E=np.broadcast_to(np.eye(4),(49,4,4)).copy();E[:,:3]=arrays['extrinsic_3x4'];np.testing.assert_array_equal(E,cam['w2c_A']);assert cam['w2c_B'].shape==(49,4,4)
  payload={'native_depth_K_E_subset.npz':subset}
  for i in range(49):payload[f'original_mask49/{i:06d}.png']=(d/'original_mask49'/f'{i:06d}.png').read_bytes()
  for name in ['cameras.npz','HISTORICAL_GEOMETRY_PLAN.json','HISTORICAL_MANIFEST.json','real_geometry.py','mesh_zbuffer.so']:payload[name]=(d/name).read_bytes()
  for name in ['real_geometry.py','mesh_zbuffer.so']:assert hashlib.sha256(payload[name]).hexdigest()==sha(N/'inputs/case72_native_CPU_v1'/name),'Unexpected original runtime revision: inspect, do not import code'
  m=dict(case=c,case_id=v['case_id'],state='Native_CPU_snapshot_exact_original_arrays_and_observed_RGB49_verified',source_video_HF_path=v['hf_path'],source_video_SHA256=sha(source),source_video_bytes=source.stat().st_size,RGB49_BGR_array_SHA256=pixels,Native_RGB49_exact=True,raw_original_path=original_raw,raw_original_file_SHA256=rawhash,subset_arrays_exact=proof,camera_original_path=v['version_root']+'/orbit_10deg/cameras_and_boxes.npz',camera_original_file_SHA256=sha(d/'cameras.npz'),observed_SAM_original_paths=next(r['remote'].rsplit('/',1)[0] for r in cfg['read_only_original_paths'] if r['local']=='original_mask49/000000.png'),original_inputs_unchanged=True,GPU_used=False,quality_accepted=False,payload_disk_writes_6000D=0,remote_CPU_execution=False,transfer='Read-only SFTP exact files; all decoding/array verification on Qinghai CPU',files=[dict(path=k,bytes=len(x),SHA256=hashlib.sha256(x).hexdigest()) for k,x in payload.items()]);payload['INPUT_MANIFEST.json']=json.dumps(m,indent=2).encode()
  for target in [out,dur]:
   target.mkdir(parents=True,exist_ok=False)
   for rel,data in payload.items():p=target/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(data).digest()
  for base in [N,Q]:
   dest=base/f'inputs/case{c}_source_video_HF_v1';dest.mkdir(exist_ok=False);shutil.copyfile(source,dest/'source_A.mkv');assert sha(dest/'source_A.mkv')==v['sha256']
  b=io.BytesIO();np.savez_compressed(b,world_points_conf=conf);cb=b.getvalue();ci=dict(case=c,source_path=original_raw,source_original_SHA256=rawhash,array='world_points_conf',shape=list(conf.shape),dtype=str(conf.dtype),array_SHA256=ah(conf),SHA256=hashlib.sha256(cb).hexdigest(),source_array_unchanged=True,exact_activation_floor=1.,floor_pixels49=[int((x==1).sum()) for x in conf],scope='Native point-head reliability, not calibrated depth probability',payload_disk_writes_6000D=0,remote_CPU_execution=False)
  for base in [N,Q]:
   stem=base/f'inputs/CASE{c}_NATIVE_POINT_CONFIDENCE49_v1';assert not stem.with_suffix('.npz').exists();stem.with_suffix('.npz').write_bytes(cb);put(stem.with_suffix('.json'),ci)
  row=dict(case=c,state='Native_arrays_RGB49_original_frames_SAM_K_E_verified_persistent',source_SHA256=v['sha256'],HF_commit=head,HF_server_size_and_LFS_SHA_matched=True,raw_original_SHA256=rawhash,camera_SHA256=sha(d/'cameras.npz'),remote_execution=False,remote_writes=False,GPU_used=False,quality_accepted=False,new_complete_groups=0)
  put(Q/f'receipts/CASE{c}_READONLY_SFTP_NATIVE_INPUTS_v1.json',row);rows.append(row);print(json.dumps(row),flush=True)
 put(Q/('receipts/READONLY_SFTP_NATIVE_'+'_'.join(args.cases)+'_v1.json'),dict(cases=rows,elapsed_seconds=time.time()-start,remote_CPU_execution=False,all_verification_QH_CPU=True,new_complete_groups=0))
if __name__=='__main__':main()

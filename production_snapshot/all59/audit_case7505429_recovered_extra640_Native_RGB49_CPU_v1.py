from pathlib import Path
import os,sys,json,hashlib,datetime,time
os.environ['CUDA_VISIBLE_DEVICES']=''
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1'
sys.path.insert(0,'/tmp/starreal_uid1007_case19_cpu_v1/runtime/qh_cpu_py39_v1')
import numpy as np,cv2
cv2.setNumThreads(1)
root=Q/'inputs/original_native_readonly_SFTP_7505429_EXTRA640_CACHE_ONLY_stream_v3';I=root/'7505429';out=Q/'component_cache/7505429/recovered_extra640_Native_INPUT_AUDIT_CPU_v1';out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
job=dict(case='7505429',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],GPU_used=False,quality_accepted=False,state='reading_recovered_Native_and_actual49_RGB');(out/'JOB.json').write_text(json.dumps(job,indent=2))
start=time.time()
try:
 tr=json.loads((root/'TRANSFER_RESULT.json').read_text());assert tr['files']==tr['expected']==103 and not tr['source_remote_execution'] and not tr['source_remote_writes'] and not tr['Mac_payload_files']
 for row in tr['records']:assert sha(root/row['path'])==row['SHA256']
 meta=json.loads((I/'case.json').read_text());assert sha(I/'source_input_640x352_UNCHANGED.mp4')==meta['source_sha256']=='d0404500542e445b099fb4e2b8de9dd42494495a15eb5afd67435dc01464fa58'
 raw=np.load(I/'vggt_raw.npz',allow_pickle=False)
 specs={'world_points':(49,280,518,3),'world_points_conf':(49,280,518),'depth':(49,280,518),'extrinsic_3x4':(49,3,4),'K':(49,3,3)}
 arrays={}
 for k,shape in specs.items():
  a=raw[k];assert a.shape==shape and a.dtype==np.float32
  arrays[k]=dict(shape=list(a.shape),dtype=str(a.dtype),array_SHA256=hashlib.sha256(a.tobytes()).hexdigest(),finite=int(np.isfinite(a).sum()),total=int(a.size),finite_min=float(np.min(a[np.isfinite(a)])),finite_max=float(np.max(a[np.isfinite(a)])))
 P=Q/'source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1/prepared49_from_actual_Original_crop_v1/7505429';pmeta=json.loads((P/'RESULT.json').read_text());native=[];true=[];frames=[]
 for i in range(49):
  f=I/'frames'/('%03d.png'%i);g=P/'RGB49'/('%06d.png'%i)
  if not g.exists():g=P/'RGB49'/('%03d.png'%i)
  im=cv2.imread(str(f));new=cv2.imread(str(g));mask=cv2.imread(str(sorted((I/'original_SAM_candidate49').glob('*.png'))[i]),0)
  assert im.shape==new.shape==(704,1280,3) and mask.shape==(704,1280)
  a=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),(256,140),interpolation=cv2.INTER_AREA).astype(float);b=cv2.resize(cv2.cvtColor(new,cv2.COLOR_BGR2GRAY),(256,140),interpolation=cv2.INTER_AREA).astype(float)
  native.append(a-a.mean());true.append(b-b.mean());frames.append(dict(frame=i,Native_input_PNG_SHA256=sha(f),Native_input_RGB_array_SHA256=hashlib.sha256(im[:,:,::-1].copy().tobytes()).hexdigest(),true4K_RGB_array_SHA256=hashlib.sha256(new[:,:,::-1].copy().tobytes()).hexdigest(),same_decoded_pixels=bool(np.array_equal(im,new)),mean_absolute_RGB_difference=float(np.abs(im.astype(float)-new.astype(float)).mean()),original_SAM_candidate_nonzero=int((mask>0).sum())))
  if i in [0,24,48]:
   ov=new.copy();ov[mask>0]=np.rint(.5*ov[mask>0]+.5*np.array([0,255,0])).astype(np.uint8);page=np.hstack([im,new,ov]);cv2.putText(page,'640-derived Native cache input ONLY | actual Original4K training RGB | OLD SAM candidate NOT accepted',(15,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),1);cv2.imwrite(str(out/('SOURCE_NATIVE_TRUE4K_OLD_SAM_A%02d.jpg'%i)),page,[cv2.IMWRITE_JPEG_QUALITY,93])
 native=np.array(native);true=np.array(true);mse=np.mean((true[:,None]-native[None,:])**2,axis=(2,3));best=np.argmin(mse,axis=1);np.savez_compressed(out/'ACTUAL_RGB49_TEMPORAL_DISTANCE_DIAGNOSTIC.npz',MSE=mse,best_Native_input_index=best)
 result=dict(case='7505429',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='recovered_actual_Native103files_full_SHA_arrays49_and_RGB_diagnostic_read_pending_geometry_camera_SAM_review',source_access='Native Mac OpenSSH SFTP read-only RAM relay; no source computation/writes and no Mac payload files',Native_original_archive_SHA256=sha(I/'vggt_raw.npz'),Native_array_readback=arrays,actual_Native_input_frames=49,Native_input_unique_RGB=len({r['Native_input_RGB_array_SHA256'] for r in frames}),true4K_input_unique_RGB=len({r['true4K_RGB_array_SHA256'] for r in frames}),actual_best_temporal_Native_indices=best.tolist(),all_best_temporal_indices_identity=bool(np.array_equal(best,np.arange(49))),new_Original4K_preparation=str(P/'RESULT.json'),new_Original4K_preparation_SHA256=sha(P/'RESULT.json'),input_frame_metadata_source=meta,frames=frames,Native_cache_source_resolution=[640,352],Native_cache_input_was_legacy_upscaled_not_new_native704=True,training_RGB_is_true_Original4K_crop_not_upscaled640=True,Native_RGB_and_Original4K_RGB_byte_identity_not_claimed=True,metric_geometry_not_claimed=True,old_SAM_candidate_not_accepted=True,camera_derivation_pending=True,quality_accepted=False,HF_upload=False,GPU_used=False,seconds=time.time()-start)
 (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));job.update(state='terminal0_input_audit_pending_actual_review',exit_code=0,seconds=time.time()-start);(out/'JOB.json').write_text(json.dumps(job,indent=2));print(json.dumps({k:result[k] for k in ['case','state','Native_original_archive_SHA256','Native_input_unique_RGB','true4K_input_unique_RGB','actual_best_temporal_Native_indices','all_best_temporal_indices_identity','GPU_used','seconds']}),flush=True)
except BaseException as e:
 job.update(state='terminal1_preserve_recovered_files',exit_code=1,error=repr(e));(out/'JOB.json').write_text(json.dumps(job,indent=2));raise

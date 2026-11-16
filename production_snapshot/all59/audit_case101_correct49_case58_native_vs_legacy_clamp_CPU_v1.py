from pathlib import Path
import os,sys,json,hashlib,time,datetime,fcntl
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ["OPENBLAS_NUM_THREADS"]="1"
Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");N=Path("/tmp/starreal_uid1007_case19_cpu_v1");sys.path.insert(0,str(N/"runtime/qh_cpu_py39_v1"))
import cv2,numpy as np
cv2.setNumThreads(1);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=Q/"component_cache/101/correct49_case58_native_vs_legacy_clamp_CPU_audit_v1";out.mkdir(parents=True,exist_ok=False);start=time.time()
I=Q/"inputs/case58_native_CPU_v1";m=json.loads((I/"INPUT_MANIFEST.json").read_text());new=Q/"inputs/case58_source_video_HF_v1/source_A.mkv"
old=Q/"inputs/case101_source_video_HF_v1/source_A.mkv"
if not old.exists():old=N/"inputs/case101_source_video_HF_v1/source_A.mkv"
om=json.loads((N/"inputs/case101_native_CPU_v1/INPUT_MANIFEST.json").read_text())
assert sha(new)==m["source_video_SHA256"] and sha(old)==om["source_video_SHA256"]
def read(p):
 cap=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);rows=[];small=[];raw=[];keys={}
 for i in range(49):
  ok,a=cap.read();assert ok and a.shape==(704,1280,3);raw.append(hashlib.sha256(a.tobytes()).hexdigest());small.append(cv2.resize(a,(160,88),interpolation=cv2.INTER_AREA).astype(np.float32))
  if i in [0,24,48]:keys[i]=a
 assert not cap.read()[0];cap.release();return raw,np.array(small),keys
nr,nf,nk=read(new);orr,of,ok=read(old);assert nr==m["RGB49_BGR_array_SHA256"] and orr==om["RGB49_BGR_array_SHA256"]
dist=np.mean((of[:,None]-nf[None,:])**2,axis=(2,3,4));nearest=np.argmin(dist,axis=1);np.savez_compressed(out/"ACTUAL_SOURCE49_RGB_DISTANCE_MATRIX.npz",mean_squared_error=dist,legacy_to_correct_Native_indices=nearest)
for f in [0,24,48]:
 a=nk[f];mask=cv2.imread(str(I/"original_mask49"/f"{f:06d}.png"),0)>0;ov=a.copy();ov[mask]=np.rint(.5*ov[mask]+.5*np.array([0,255,0])).astype(np.uint8)
 cv2.putText(a.copy(),"Actual49 same underlying video; verified Native input58, NOT yet target101 admitted",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,255,255),1)
 can=np.concatenate([a,ov,ok[f]],1);cv2.putText(can,f"Native58 correct49 A{f:02d} / originalSAM candidate / legacy101 resampled-clamped A{f:02d}",(8,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),1)
 cv2.imwrite(str(out/f"CORRECT_NATIVE58_OLD_SAM_LEGACY101_A{f:02d}.jpg"),can,[cv2.IMWRITE_JPEG_QUALITY,95])
result=dict(case="101",at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state="correct_alias58_real49_RGB_Native_and_legacy101_clamp_compared_pending_actual_images_and_source_route_review",actual_candidate_native_case_id=m["case_id"],candidate_input_manifest=str(I/"INPUT_MANIFEST.json"),candidate_input_manifest_SHA256=sha(I/"INPUT_MANIFEST.json"),candidate_source_video=str(new),candidate_source_SHA256=sha(new),candidate_unique_RGB=len(set(nr)),legacy_input_manifest=str(N/"inputs/case101_native_CPU_v1/INPUT_MANIFEST.json"),legacy_source_video=str(old),legacy_source_SHA256=sha(old),legacy_unique_RGB=len(set(orr)),legacy_to_correct_Native_nearest_indices=nearest.tolist(),nearest_MSE=[float(dist[i,nearest[i]]) for i in range(49)],legacy_rgb_byte_exact_correct_input_indices=[nr.index(h) if h in nr else None for h in orr],current_candidate_RGB49_all_exact_to_actual_Native_manifest=True,wrong_legacy_Native_depth_K_E_NOT_reused=True,new_Native_inference_run=False,GPU_used=False,quality_accepted=False,new_complete_groups=0,seconds=time.time()-start)
(out/"RESULT.json").write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ["nearest_MSE","legacy_rgb_byte_exact_correct_input_indices"]}),flush=True)

from pathlib import Path
import os,sys,json,hashlib,datetime
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1';N=Path('/tmp/starreal_uid1007_case19_cpu_v1');sys.path.insert(0,str(N/'runtime/qh_cpu_py39_v1'))
import cv2,numpy as np
cv2.setNumThreads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for c,a in [('6374204','109'),('10302763','108'),('7505473','111')]:
 hist=Q/'inputs/extras_five_actual_HF_history_v1'/c;imp=json.loads((hist/'HF_IMPORT_MANIFEST.json').read_text());assert imp['state']=='all_actual_original_HF_folder_files_server_SHA_verified'
 old=N/f'inputs/case{a}_native_CPU_v1';oldmeta=json.loads((old/'INPUT_MANIFEST.json').read_text());Rold=Path(f'/tmp/starreal_uid1007_case{a}_true4K_cpu_v2');Iold=Rold/f'inputs/case{a}_native_CPU_v1';newmeta=json.loads((Iold/'INPUT_MANIFEST.json').read_text())
 rr=next(x for x in imp['records'] if x['HF_path'].endswith('media/source_A.mkv'));assert sha(rr['local_path'])==rr['SHA256']
 cap=cv2.VideoCapture(rr['local_path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);oldrgb=[]
 for i in range(49):
  ok,b=cap.read();assert ok;h=hashlib.sha256(b.tobytes()).hexdigest();assert h==oldmeta['RGB49_BGR_array_SHA256'][i];oldrgb.append(h)
 assert not cap.read()[0];cap.release()
 assert oldmeta['Native_RGB49_exact'] and oldmeta['subset_arrays_exact']
 prep=Q/f'source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1/prepared49_from_actual_Original_crop_v1/{c}/RESULT.json';p=json.loads(prep.read_text());video=p['source_video_output'];assert sha(video)==newmeta['source_video_SHA256']==p['source_video_output_SHA256']
 cap=cv2.VideoCapture(video,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);newrgb=[]
 for i in range(49):
  ok,b=cap.read();assert ok;h=hashlib.sha256(b.tobytes()).hexdigest();assert h==newmeta['RGB49_BGR_array_SHA256'][i];newrgb.append(h)
 assert not cap.read()[0];cap.release()
 nc=np.load(Iold/'cameras.npz');oc=np.load(old/'cameras.npz');H=np.array(newmeta['H_old_prepared1280_to_new_Original1280']);np.testing.assert_array_equal(nc['K'],H@oc['K'])
 for k in ['w2c_A','w2c_B']:np.testing.assert_array_equal(nc[k],oc[k])
 assert sha(Iold/'native_depth_K_E_subset.npz')==sha(old/'native_depth_K_E_subset.npz')
 rev='relative_ops_v1' if c=='7505473' else 'relative_ops_v4';oldcam=hist/f'extracted/{rev}/operations/move/orbit_10deg/cameras_and_boxes.npz';z=np.load(oldcam);diff={k:float(np.max(np.abs(z[k]-oc[k]))) for k in ['K','w2c_A','w2c_B']};assert diff['K']>1
 proof=dict(case=c,actual_same_RGB_input_case=a,state='EXACT_OLD49_RGB_AND_NEW_ORIGINAL4K49_RGB_PROVED_NATIVE_REUSE_ALLOWED_HISTORICAL_CAMERAS_NOT_REUSED',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_source_HF_path=rr['HF_path'],old_source_container_SHA256=rr['SHA256'],old_alias_container_SHA256=oldmeta['source_video_SHA256'],container_SHA_equal=rr['SHA256']==oldmeta['source_video_SHA256'],old_decoded_RGB49_BGR_SHA256=oldrgb,all49_decoded_RGB_exact=True,old_alias_original_native_manifest=str(old/'INPUT_MANIFEST.json'),old_alias_original_native_manifest_SHA256=sha(old/'INPUT_MANIFEST.json'),Native_RGB49_exact=True,original_native_raw_path=oldmeta['raw_original_path'],original_native_raw_file_SHA256=oldmeta['raw_original_file_SHA256'],Native_subset_arrays_exact=True,Native_subset_SHA256=sha(old/'native_depth_K_E_subset.npz'),new_original4K_preparation=str(prep),new_original4K_preparation_SHA256=sha(prep),new_source_video=video,new_source_video_SHA256=sha(video),new_all49_RGB_BGR_SHA256=newrgb,new_intrinsics_derived_by_known_image_homography=H.tolist(),new_intrinsics_exact=True,new_world_A_B_poses_exact=True,historical_camera_path=str(oldcam),historical_camera_SHA256=sha(oldcam),historical_vs_Native_max_abs_difference=diff,historical_camera_A_B_K_NOT_reused=True,new_native_estimation_run=False,same_RGB_estimated_Native_geometry_reused_not_relabelled_old_pose=True,raw_depth_not_metric_ground_truth=True,historical_conditions_preserved=True,new_semantic_scene_increment=0,new_case_delivery_increment=0,quality_accepted=False,GPU_used=False)
 out=Q/f'component_cache/{c}/RGB49_NATIVE_REUSE_PROOF_v1';out.mkdir(parents=True,exist_ok=False);(out/'RESULT.json').write_text(json.dumps(proof,indent=2)+'\n')
 R=Path(f'/tmp/starreal_uid1007_case{c}_true4K_cpu_v2');R.mkdir(exist_ok=False)
 for name in ['inputs','outputs','reviews','receipts','packages','releases']: (R/name).mkdir()
 (R/'code').symlink_to(Rold/'code',target_is_directory=True);(R/'runtime').symlink_to(N/'runtime',target_is_directory=True);(R/f'outputs/{c}').mkdir()
 I=R/f'inputs/case{c}_native_CPU_v1';I.mkdir()
 for f in Iold.iterdir():
  if f.name=='INPUT_MANIFEST.json':continue
  (I/f.name).symlink_to(f,target_is_directory=f.is_dir())
 m=dict(newmeta);m.update(case=c,case_id='extra_case02_6374204' if c=='6374204' else 'extra_case01_10302763' if c=='10302763' else 'extra_case04_7505473',source_video_HF_historical_path=rr['HF_path'],source_reused_input_case=a,source_reused_input_proof=str(out/'RESULT.json'),source_reused_input_proof_SHA256=sha(out/'RESULT.json'),historical_camera_not_reused=True)
 (I/'INPUT_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');(I/'REUSED_NATIVE_INPUT_MANIFEST_UNCHANGED.json').write_bytes((Iold/'INPUT_MANIFEST.json').read_bytes())
 (R/f'inputs/case{c}_source_video_true4K_v2').symlink_to(Rold/f'inputs/case{a}_source_video_true4K_v2',target_is_directory=True)
 conf=Rold/f'inputs/CASE{a}_NATIVE_POINT_CONFIDENCE49_v1.npz';(R/f'inputs/CASE{c}_NATIVE_POINT_CONFIDENCE49_v1.npz').symlink_to(conf)
 bg=Rold/f'outputs/{a}/pcf_background49_precomputed_QH_v1';assert bg.exists();(R/f'outputs/{c}/pcf_background49_precomputed_QH_v1').symlink_to(bg,target_is_directory=True)
 obs=json.loads((Q/f'reviews/CASE{a}_TRUE4K_OBSERVED_SOURCE_SAM49_CURRENT.json').read_text());obs.update(case=c,exact_reused_RGB_native_proof=str(out/'RESULT.json'),exact_reused_input_case=a,reused_component=True,original_review_path=str(Q/f'reviews/CASE{a}_TRUE4K_OBSERVED_SOURCE_SAM49_CURRENT.json'),quality_accepted=False)
 (Q/f'reviews/CASE{c}_TRUE4K_OBSERVED_SOURCE_SAM49_CURRENT.json').write_text(json.dumps(obs,indent=2)+'\n')
 rows.append(dict(case=c,same_RGB_native_case=a,proof=str(out/'RESULT.json'),root=str(R),ready_for_fresh_three_supported_whole_targets=True,not_a_complete_group=True))
(U/'EXTRA3_EXACT49_RGB_NATIVE_BINDINGS_CURRENT.json').write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=rows,quality_accepted=False,new_complete_groups=0,GPU_used=False),indent=2)+'\n')
print(json.dumps(rows))

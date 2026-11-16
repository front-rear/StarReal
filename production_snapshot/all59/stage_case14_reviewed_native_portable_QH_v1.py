"""Stage reviewed case14 on node-local QH CPU scratch; never auto-release."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,sys,time
R=Path('/tmp/starreal_uid1007_case14_cpu_v2')
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
sys.path[:0]=[str(R/'runtime/qh_cpu_py39_v1'),str(R/'code')]
import cv2,numpy as np
from realcase_condition_reader import RealcaseConditionReader
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def files(p):return [x for x in sorted(p.rglob('*')) if x.is_file() and not any(z.startswith('._') for z in x.relative_to(p).parts)]
def cp(a,b):
    if not a.exists() and a.parent==R and (R/'code'/a.name).exists():a=R/'code'/a.name
    if a.is_dir():
        b.mkdir(parents=True,exist_ok=True)
        for p in files(a):cp(p,b/p.relative_to(a))
    else:
        b.parent.mkdir(parents=True,exist_ok=True)
        if b.exists():assert sha(a)==sha(b);return
        shutil.copyfile(a,b);assert sha(a)==sha(b)
def put(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
def video(p,width):
    cap=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);n=0;fps=cap.get(5)
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(704,width,3);n+=1
    cap.release();assert n==49 and fps==10
    return dict(frames=n,resolution=[width,704],fps=fps,sha256=sha(p),bytes=p.stat().st_size)
def main():
    cv2.setNumThreads(1);start=time.time();guard=(R/'logs/case14_portable_QH.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cond=R/'outputs/14/geometry_masks49_cached_QH_v1';joints=R/'outputs/14/two_pcf49_fixed_world_yellow_QH_v1';geom=R/'outputs/14/five_role_geometry49_QH_v1';hist=R/'outputs/14/historical_joint_comparison49_QH_v1';snap=R/'inputs/case14_native_CPU_v1';refs=R/'inputs/case53_portable_reference_v1'
    review=R/'reviews/CASE14_FIXED_WORLD_YELLOW_FINAL_ALL49_ACTUAL_REVIEW_v1.json';visual=json.loads(review.read_text());policy=json.loads((R/'HUMAN_DEPTH_AWARE_YELLOW_OVERLAP_CURRENT.json').read_text());c=json.loads((cond/'MANIFEST.json').read_text());v=json.loads((joints/'RESULT.json').read_text())
    assert visual['verdict']=='PASS_RENDERED_DEPTH_COLOR_VIDEO_COMPONENT_ONLY' and len(visual['actual_all49_new_joint_sheets'])==len(visual['actual_all49_ordinary_joint_sheets'])==5
    assert visual['renderer_SHA256']==v['renderer_SHA256']==policy['renderer_SHA256']==sha(R/policy['renderer_copy'])
    assert v['semantic_revision']==policy['required_mode'] and json.loads((cond/'INDEPENDENT_CONDITION_READBACK.json').read_text())['files_read']==490
    assert json.loads((R/'reviews/CASE14_HISTORICAL2_ALL49_ACTUAL_REVIEW_v1.json').read_text())['verdict']=='ALL2_ACTUAL_HISTORICAL_JOINT49_CLIPS_VIEWED_PRESERVED_K_CROP_H_MATCHED'
    signed=json.loads((R/'logs/CASE14_NATIVE_FAST5_SIGNED_ORIGINAL_ENTRY_READBACK_v1.json').read_text());hint=json.loads((R/'logs/CASE14_NATIVE_FAST5_SOURCE_HINT_ORIGINAL_ENTRY_READBACK_v1.json').read_text())
    assert signed['tests']==26 and signed['normalization_count']==1 and not signed['body_changed'];assert len(hint['records'])==2 and all(x['frames']==49 and x['Original_entry_passed'] for x in hint['records'])
    for row in visual['actual_all49_new_joint_sheets']+visual['actual_all49_ordinary_joint_sheets']:assert row['frames_viewed']==list(range(49)) and sha(R/row['all49_actual_contact_sheet'])==row['sha256']
    assert shutil.disk_usage(R).free>16*1024**3
    out=R/'packages/14/fixed_world_yellow_depth_v5_staging';out.parent.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=False)
    def state(s,**kw):put(out/'BUILD_STATE.json',dict(case='14',state=s,pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],GPU_used=False,quality_accepted=False,elapsed_seconds=time.time()-start,**kw))
    state('copying_actual_reviewed_conditions_and_native_inputs')
    try:
        cp(cond,out/'conditions');cp(geom/'RESULT.json',out/'geometry_review_provenance/RESULT.json');cp(R/'outputs/14/source_appearance_control_reference.json',out/'source_appearance_control_reference.json')
        cp(R/'inputs/case14_source_video_HF_v1/source_A.mkv',out/'media/source_A.mkv');cp(Path('/tmp/starreal_uid1007_case19_cpu_v1/inputs/case14_native_CPU_v1/cameras.npz'),out/'provenance/ORIGINAL_CAMERA_ARCHIVE_UNCHANGED.npz');cp(snap/'cameras.npz',out/'provenance/NATIVE_CAMERA_ARCHIVE_UNCHANGED.npz')
        cp(R/'inputs/CASE14_NATIVE_POINT_CONFIDENCE49_v1.npz',out/'source_depth/ORIGINAL_POINT_CONFIDENCE49.npz');cp(R/'inputs/CASE14_NATIVE_POINT_CONFIDENCE49_v1.json',out/'source_depth/ORIGINAL_POINT_CONFIDENCE_PROVENANCE.json');cp(snap/'native_depth_K_E_subset.npz',out/'source_depth/native_depth_K_E_LOSSLESS_SUBSET.npz');cp(snap/'INPUT_MANIFEST.json',out/'source_depth/ORIGINAL_FULL_SOURCE_AND_EXACT_SUBSET_PROVENANCE.json');cp(R/'inputs/CASE14_INPUT_BACKEND_QH_v1.json',out/'source_depth/INPUT_BACKEND_PROVENANCE.json');cp(geom/'source/fixed_world_geometry.npz',out/'source_geometry_estimate/source_P_geometry.npz');cp(geom/'move/fixed_world_geometry.npz',out/'source_geometry_estimate/moved_rigid_geometry.npz');cp(geom/'visibility49',out/'source_geometry_estimate/visibility49');cp(R/'inputs/CASE14_GEOMETRY49_CONFIG_v1.json',out/'source_geometry_estimate/GEOMETRY49_CONFIG.json');cp(Q/'component_cache/14/whole_source_rigid_edge_pilot3_v2/geometry.npz',out/'source_geometry_estimate/source_and_rigid_move_pilot_geometry.npz')
        cp(R/'inputs/case14_three_fresh_whole_models_v2/ORIGINAL_LIBRARY_RAW_CANDIDATES_PROOF.json',out/'provenance/ORIGINAL_LIBRARY_RAW_CANDIDATES_PROOF.json')
        cp(R/'inputs/case14_three_fresh_whole_models_v2/MESH_PARSE_PROOF.json',out/'provenance/WHOLE_RAW_MESH_PARSE_PROOF.json')
        geometry_config=json.loads((R/'inputs/CASE14_GEOMETRY49_CONFIG_v1.json').read_text());aux_rows=[]
        for role,spec in geometry_config['geometry_sources'].items():
            if role in ['source','move']:continue
            rel=spec['path'];assert '..' not in Path(rel).parts;actual_target=(R/rel).resolve();actual_target.relative_to((R/'outputs/14').resolve())
            src=R/rel;dst=out/'whole_target_geometry'/role/'geometry.npz';cp(src,dst);assert sha(src)==sha(dst);aux_rows.append(dict(role=role,source=rel,package_path=str(dst.relative_to(out)),SHA256=sha(dst)))
        assert len(aux_rows)==3;put(out/'AUXILIARY_GEOMETRY_SAME_CASE_SHA_CHECK.json',dict(case='14',all_same_case_and_SHA_verified=True,geometries=aux_rows))
        raw=np.load(snap/'native_depth_K_E_subset.npz');cam=np.load(cond/'cameras49.npz');K=raw['K'].astype(float).copy();K[:,0]*=1280/raw['depth'].shape[2];K[:,1]*=704/raw['depth'].shape[1];oldcam=np.load(out/'provenance/ORIGINAL_CAMERA_ARCHIVE_UNCHANGED.npz');np.testing.assert_array_equal(K,oldcam['K']);np.testing.assert_array_equal(K,cam['K']);E=np.broadcast_to(np.eye(4),(49,4,4)).copy();E[:,:3]=raw['extrinsic_3x4'];np.testing.assert_array_equal(E,cam['w2c_A'])
        proof=c['assets'];candidates=proof['actual_exact_raw_SHA_matched'];assert proof['qualified500_claimed'] is False and proof['schema']=='original_user_lamp_library_exact_raw_candidates_v1' and len(candidates)==3 and len({x['actual_raw_SHA256'] for x in candidates})==3
        diversity=json.loads((refs/'LAMP_DIVERSITY_CURRENT.json').read_text());used={x['actual_mesh_SHA256'] for x in diversity['actual_usage_ledger']};assets=[]
        for done in (Q/'remote_package_metadata').glob('*/MANIFEST.json'):
            previous=json.loads(done.read_text())
            if previous.get('quality_accepted'):used.update(x['sha256'] for x in previous['assets'])
        for row in candidates:
            idx=row['source_index'];p=R/'inputs/case14_three_fresh_whole_models_v2/raw_assets'/('lamp_'+idx+'.glb');assert sha(p)==row['actual_raw_SHA256'] and sha(p) not in used
            dst=out/'assets'/('lamp_'+idx+'.glb');cp(p,dst);assets.append(dict(asset='original_library_'+idx,package_path=str(dst.relative_to(out)),sha256=sha(dst),source_path=row['source_original_path'],qualified500_indices_by_exact_source_hash=row['qualified500_indices_by_exact_source_hash'],catalog_metadata=row['catalog_metadata'],whole_raw_mesh_preserved=True,rendered_mesh_authority='conditions/<variant>/fixed_world_geometry.npz',selection=row))
        cp(joints/'shared_pcf',out/'joints/shared_pcf');put(out/'joints/FACTORED_JOINTS.json',dict(schema='shared_PCF_plus_uint8_RGB_patch_v1',frames=49,resolution=[1280,704],pixel_dtype='uint8_RGB',no_lossy_reconstruction=True,variants=list(c['operations']),versions=['old_joint','new_joint']))
        records=[];checks={};state('factoring_exact_native_RGB_without_rerender')
        for name in c['operations']:
            for version in ['old_joint','new_joint']:
                dst=out/'joints'/name/version;dst.mkdir(parents=True)
                for i in range(49):
                    bg=cv2.imread(str(joints/'shared_pcf'/f'{i:06d}.png'))[:,:,::-1].copy();rgb=cv2.imread(str(joints/name/version/f'{i:06d}.png'))[:,:,::-1].copy();assert bg.shape==rgb.shape==(704,1280,3)
                    yy,xx=np.where(np.any(rgb!=bg,2));x,y,x1,y1=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)) if len(xx) else (0,0,1,1)
                    np.savez_compressed(dst/f'{i:06d}.npz',origin_xy=np.array([x,y],np.int32),patch_rgb_uint8=rgb[y:y1,x:x1]);records.append(dict(variant=name,version=version,frame=i,RGB_SHA256=hashlib.sha256(rgb.tobytes()).hexdigest()))
            checks[name]={}
            for key,src,directory,width in [('ordinary','regular_1280x704_49f.mp4','regular_pcf_bbox',1280),('approved_depth_aware','new_1280x704_49f.mp4','depth_aware_pcf_bbox',1280),('comparison','old_vs_new_2560x704_49f.mp4','comparison',2560)]:
                dst=out/directory/(name+('_regular_vs_depth_aware' if key=='comparison' else '')+'.mp4');cp(joints/name/src,dst);checks[name][key]=video(dst,width)
            print(json.dumps(dict(case='14',variant=name,factored_both49=True,videos_decoded=3)),flush=True)
        put(out/'VIDEO_INDEX.json',dict(variants=list(c['operations']),checks=checks,all15_videos_decoded49=True));cp(hist,out/'historical_comparison');history=json.loads((hist/'MANIFEST.json').read_text());
        if history.get('parallelism'):
            cp(R/'build_native_historical_comparison49_QH_cpu_v3.py',out/'pipeline_reference/build_native_historical_comparison49_QH_cpu_v3.py')
        rows=history['rows']+history['additional_historical_camera_versions'];assert len(rows)==2
        for row in rows:assert sha(out/'historical_comparison'/row['historical_package_path'])==row['historical_source_SHA256_unchanged'] and sha(out/'historical_comparison'/row['historical_camera_path'])==row['historical_camera_SHA256']
        old=R/'inputs/case14_historical_mesh_HF_v4'
        cp(old/'PINNED_SERVER_TREE.json',out/'provenance/HISTORICAL_ACTUAL_MOVE_REPLACE_SERVER_TREE.json');cp(old/'HF_IMPORT_MANIFEST.json',out/'provenance/HISTORICAL_HF_IMPORT_MANIFEST.json')
        cp(snap/'INPUT_MANIFEST.json',out/'provenance/NATIVE_UNCHANGED_RGB_CAMERA_COMPONENT.json')

        for p in sorted((R/'reviews').glob('CASE14*.json')):
            if not p.name.startswith('._'):cp(p,out/'audits'/p.name)
        for row in visual['actual_all49_new_joint_sheets']+visual['actual_all49_ordinary_joint_sheets']:cp(R/row['all49_actual_contact_sheet'],out/'audits/visual_contact_sheets'/Path(row['all49_actual_contact_sheet']).name)
        cp(R/'outputs/14/actual_condition_palette_review_QH_v1',out/'audits/actual_color_condition_sheets')
        for name in ['CASE14_JOINT_RGB_AND_PALETTE_INDEPENDENT_READBACK_v1.json','QH_PRIVATE_PCF_KERNEL_REFERENCE_EXACT_v1.json','CASE14_QH_ACTUAL_ORIGINAL_ENTRY_v2.json','CASE14_HISTORICAL_CAMERAS_AND_SOURCE49_READBACK_v1.json']:cp(R/'receipts'/name,out/'provenance'/name)
        for name in ['CASE14_NATIVE_FAST5_SIGNED_ORIGINAL_ENTRY_READBACK_v1.json','CASE14_NATIVE_FAST5_SOURCE_HINT_ORIGINAL_ENTRY_READBACK_v1.json']:cp(R/'logs'/name,out/'provenance'/name)
        cp(joints/'JOB.json',out/'provenance/PCF_STREAM_V4_ACTUAL_JOB.json')
        cp(joints/'RESULT.json',out/'provenance/PCF_FIXED_WORLD_ALL49_RENDER_READBACK.json');cp(joints/'NATIVE_FULL_VS_SPARSE_ALL5_EQUALITY.json',out/'provenance/NATIVE_FULL_VS_SPARSE_ALL5_EQUALITY.json')
        for name in ['Original_user_render_pcf.py','Original_wan_video_new_UNMODIFIED.py','Original_base_pipeline_UNMODIFIED.py']:cp(refs/'provenance'/name,out/'pipeline_reference'/name)
        assert sha(out/'pipeline_reference/Original_wan_video_new_UNMODIFIED.py')==signed['Original_SHA256'] and sha(out/'pipeline_reference/Original_base_pipeline_UNMODIFIED.py')==signed['BasePipeline_SHA256']
        cp(refs/'code/mesh_zbuffer.cpp',out/'pipeline_reference/mesh_zbuffer.cpp');cp(snap/'real_geometry.py',out/'pipeline_reference/real_geometry.py');cp(R/'code/case19_source_shell_reference_QH_v1.py',out/'pipeline_reference/case19_source_shell_reference_QH_v1.py');cp(Q/'component_cache/14/whole_source_rigid_edge_pilot3_v2/RESULT.json',out/'source_geometry_estimate/SOURCE_PROXY_AND_RIGID_MOVE_RECIPE.json');cp(R/'outputs/14/three_whole_supported_targets_pilot3_QH_v1/RESULT.json',out/'source_geometry_estimate/THREE_WHOLE_TARGETS_RECIPE.json')
        for name in ['depth_fusion_fixed_world_yellow_overlap_v2.py','realcase_condition_reader.py']:cp(R/'code'/name,out/'pipeline_reference'/name)
        for name in ['pcf_cpu_accelerator.py','pcf_trace_cpu.cpp']:cp(R/'runtime/qh_pcf_cpu_v1'/name,out/'pipeline_reference'/name)
        for p in sorted((R/'runtime/case18_ordinary_refs_v1').glob('*.py')):
            if not p.name.startswith('._'):cp(p,out/'pipeline_reference/ordinary_reference'/p.name)
        for name in ['original_joint_frame_stream_v1.py','encode_native_pcf_variant_stream49_v1.py','review_native_conditions_and_joint_pixels_QH_v2.py']:cp(R/name,out/'pipeline_reference'/name)
        cp(R/'receipts/ORIGINAL_JOINT_STREAM_NATIVE704_ALL245_PIXEL_PARITY_v1.json',out/'qa/ORIGINAL_JOINT_STREAM_NATIVE704_ALL245_PIXEL_PARITY_v1.json')
        for name in ['build_native_five_role_geometry49_solidmask_QH_cpu_v3.py','build_native_cached_conditions49_QH_v3.py','build_native_case_two_pcf49_QH_cpu_v5.py','build_native_historical_comparison49_QH_cpu_v3.py']:cp(R/'code'/name,out/'pipeline_reference'/name)
        cp(R/'code/realcase_condition_reader.py',out/'reader/realcase_condition_reader.py');cp(R/'HUMAN_DEPTH_AWARE_YELLOW_OVERLAP_CURRENT.json',out/'HUMAN_DEPTH_AWARE_YELLOW_OVERLAP_CURRENT.json');cp(refs/'HUMAN_FAST_REALCASE_THREE_FRESH_FIVE_OPS_TWO_PCF_CURRENT.json',out/'HUMAN_POLICY.json')
        adaptations=['build_case14_rigid_right_edge_pilot3_QH_finite_20261116_v2.py','build_case14_three_whole_base_and_true_clamp_targets_pilot3_QH_v1.py','build_case14_native_cached_conditions49_QH_v4.py']
        copied=[]
        for name in adaptations:
            src=Q/'all59_reconstruction_20261115_v1'/name;cp(src,out/'pipeline_reference/actual_native_adaptations'/name);copied.append(dict(path='pipeline_reference/actual_native_adaptations/'+name,SHA256=sha(src)))
        put(out/'pipeline_reference/ACTUAL_NATIVE_ADAPTATION_PROVENANCE.json',dict(new_crop_not_applied=True,raw_RGB_model_depth_K_E_preserved=True,actual_code=copied))
        state('independently_reading_all490_packaged_RGB_and_source49');reader=RealcaseConditionReader(out/'conditions',out/'media/source_A.mkv',out/'joints');reference=RealcaseConditionReader(cond,R/'inputs/case14_source_video_HF_v1/source_A.mkv',joints)
        for row in records:assert hashlib.sha256(np.asarray(reader.joint_frame(row['variant'],row['version'],row['frame'])).tobytes()).hexdigest()==row['RGB_SHA256']
        for i in range(49):np.testing.assert_array_equal(np.asarray(reader.observed_frame(i)),np.asarray(reference.observed_frame(i)))
        put(out/'PACKAGED_JOINT_PIXEL_READBACK.json',dict(state='all_old_new_joint_frames_pixel_exact_after_factoring',RGB_frames_checked=490,records=records,observed_source_RGB_all49_pixel_exact=True))
        inventory={str(p.relative_to(cond)):sha(p) for p in files(cond)};assert inventory=={str(p.relative_to(out/'conditions')):sha(p) for p in files(out/'conditions')}
        put(out/'UNCHANGED_INPUT_SHA_READBACK.json',dict(all_condition_file_SHA256=inventory,all_condition_files_unchanged=True,source_RGB_SHA256=sha(out/'media/source_A.mkv'),reader_SHA256=sha(out/'reader/realcase_condition_reader.py'),raw_Native_K_E_all49_exact=True,Native_full_K_exact_scaled_raw49=True,Native_depth_subset_exact_to_original_export=True))
        manifest=dict(schema='realcase_fresh_fixed_world_yellow_portable_v5',case='14',case_id=c['case_id'],state='portable_fast5_audited_PENDING_SPECIFIC_REVISION_QUALITY_RELEASE',frames=49,resolution=[1280,704],fps=10,variants=list(c['operations']),assets=assets,source_RGB=dict(path='media/source_A.mkv',sha256=sha(out/'media/source_A.mkv'),role='observed inputA, not edited targetB'),conditions_path='conditions',joint_path='joints',reader_path='reader/realcase_condition_reader.py',source_depth_path='source_depth/native_depth_K_E_LOSSLESS_SUBSET.npz',source_depth_archive_scope='Exact depth/K/extrinsic arrays from unchanged full Native archive; full archive source path and SHA preserved in provenance and original archive provenance, not mislabeled as full archive in this package',source_B_estimate_path='conditions/source_B_estimate',original_observed_SAM_path='conditions/source_original_SAM_candidate_A',source_mask_revision=c['source_mask_revision'],source_original_lamp_3D=json.loads((R/'inputs/CASE14_GEOMETRY49_CONFIG_v1.json').read_text())['source_hidden_geometry_policy'],mask_families=['lamp_silhouette','bbox_silhouette'],relative_controls='raw signed target-source; gamma/1.1 alpha/.9 once at reader entry',pure_move_reference='source_appearance_control_reference.json',world_units='relative monocular, not calibrated metres',source_original_support_contact_observed=c['source_original_support_contact_observed'],hidden_source_lower_completion_explicit=c['source_lower_completion_explicitly_estimated'],source_base_contact_visual_reviewed=c['source_base_contact_visual_reviewed'],metric_support_not_claimed=True,historical_camera_versions_preserved=2,historical_old_versions='historical_comparison/MANIFEST.json',audited_visual_revision=review.name,joint_revision=policy['required_mode'],renderer_SHA256=policy['renderer_SHA256'],edited_B_RGB=None,training_GT_available=False,quality_accepted=False,uploaded=False,required_remaining=['specific Quality release','HF server verification'])
        manifest['scene_visibility_uncertainty_revision']=c['scene_visibility_uncertainty_revision'];manifest['source_hidden_back_not_observed_GT']=True;manifest['source_lower_completion_added']=c['source_lower_completion_explicitly_estimated'];manifest['source_shade_completion_added']=c['source_shade_completion_explicitly_estimated'];manifest['hidden_completion_mask_semantics']=c['hidden_completion_mask_semantics'];put(out/'MANIFEST.json',manifest)
        (out/'README.txt').write_text('StarReal case14: unchanged Native RGB/depth/K/E 1280x704/49/10FPS real-video CONDITIONING only; no observed editedB RGB GT. Five operations and exactly three distinct whole lamps: two full bases on clear tabletop and one complete table-clamp fixed by its actual upper-jaw underside to the original table edge; original full articulated arms/clamp observed,3D support plane and source back/thickness are monocular estimates from original user library, raw hashes and qualification limits preserved. Both ordinary and approved opaque shared fixed-world depth PCF49, visible projected redP/blueQ overlap yellow. Both silhouette families, eight explicit colors, signed target-source gamma/1.1 alpha/.9 once at actual Original entry. Puremove unchanged appearance/size/orientation. '+manifest['source_original_lamp_3D']+' Two real HF historical MR videos/cameras preserved SHA-exact. Primary historical orbit10 K and A/B poses unchanged; no new crop or K transform applied; small4 supplemental B views differ and remain identified; no camera relabeling. Full59/98+167 source-history scope remains incomplete.\n')
        audit=dict(case='14',state='portable_audit_passed_pending_release_record',joint_revision=policy['required_mode'],renderer_SHA256=policy['renderer_SHA256'],Native_depth_K_E_checked_all49=True,Original_reader_code_and_controls_unchanged=True,all_explicit_videos_decoded49=15,all_condition_files_SHA_unchanged=len(inventory),variants=5,actual_fresh_whole_GLBs=3,full_K_exact_scaled_raw49=True,condition_files_fully_read=490,packaged_joints_exact_readback_frames=490,observed_source_RGB_exact_readback_frames=49,all_historical_clips_and_cameras_SHA_preserved=2,full_native_archive_not_claimed_in_package=True,quality_accepted=False,GPU_used=False,elapsed_seconds=time.time()-start)
        put(out/'PORTABLE_FINAL_AUDIT.json',audit);state('terminal0_fast5_portable_audit_passed_pending_release',exit_code=0);print(json.dumps(audit),flush=True)
    except BaseException as e:state('terminal1_preserve_partial_no_automatic_retry',exit_code=1,error=repr(e));raise
if __name__=='__main__':main()

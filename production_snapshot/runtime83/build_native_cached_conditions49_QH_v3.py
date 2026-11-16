"""Five reviewed operations; reuse exact all49 visibility, preserve raw controls.

Local CPU/storage adaptation only. No rerasterization, RGB resynthesis, QA or HF.
"""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,sys,time,traceback
R=Path('/tmp/starreal_uid1007_case83_cpu_v2')
CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal()
sys.path[:0]=[str(R/'runtime/qh_cpu_py39_v1'),str(R/f'inputs/case{CASE}_native_CPU_v1'),str(R/'code')]
import cv2,numpy as np
from build_case1_masks49 import COLORS,fields,gamma_pairs,png

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)

def main():
    cv2.setNumThreads(1);start=time.time()
    lock=(R/f'logs/case{CASE}_QH_conditions49.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reviewpath=R/f'reviews/CASE{CASE}_FIVE_ROLE_ALL49_GEOMETRY_ACTUAL_REVIEW_v1.json'
    review=json.loads(reviewpath.read_text());assert review['may_generate_full49_conditions'] and not review['quality_accepted']
    geo=R/review['component_root'];result=json.loads((geo/'RESULT.json').read_text())
    assert result['frames']==49 and result['Native_RGB49_SAM_depth_K_E_unchanged']
    for row in review['actual_viewed_all49_sheets']:assert sha(R/row['path'])==row['SHA256']
    backend=json.loads((R/f'inputs/CASE{CASE}_INPUT_BACKEND_QH_v1.json').read_text());snap=R/f'inputs/case{CASE}_native_CPU_v1'
    for row in backend['inputs'].values():assert sha(row['local_path'])==row['SHA256']
    for i in range(49):assert sha(Path(backend['observed_mask_directory'])/f'{i:06d}.png')==backend['all49_observed_mask_SHA256'][i]['SHA256']
    out=R/f'outputs/{CASE}/geometry_masks49_cached_QH_v1';out.mkdir(exist_ok=False)
    state=dict(case=f'{CASE}',state='running_finite_cached_conditions49',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],frames_finished=0,GPU_used=False,quality_accepted=False)
    def save():state['elapsed_seconds']=time.time()-start;put(out/'JOB.json',state)
    save()
    try:
        camera=np.load(snap/'cameras.npz');np.savez_compressed(out/'cameras49.npz',**{k:camera[k] for k in ['K','w2c_A','w2c_B']})
        source=np.load(geo/'source/fixed_world_geometry.npz');jobs={}
        cfg=json.loads((R/f'inputs/CASE{CASE}_GEOMETRY49_CONFIG_v1.json').read_text());asset_proof=json.loads((R/cfg['assets_proof']).read_text());candidates=asset_proof['actual_exact_raw_SHA_matched'];byop={x['role']:x['source_index'] for x in candidates};assert set(byop)=={'insert','replace','move_replace'}
        specs=[('remove','remove',None,None),('move','move',None,'move')]+[(op+'_'+byop[op],op,byop[op],op+'_'+byop[op]) for op in ['insert','replace','move_replace']]
        for name,op,asset,role in specs:
            folder=out/name;folder.mkdir();target=np.load(geo/role/'fixed_world_geometry.npz') if role else None
            data={'source_'+k:source[k] for k in ['vertices','faces','center','size','rotation']}
            for k in ['vertices','faces','center','size','rotation']:
                data['target_'+k]=target[k] if target is not None else (np.empty((0,3),np.int32 if k=='faces' else float) if k in ['vertices','faces'] else np.eye(3) if k=='rotation' else np.zeros(3))
            data['source_unobserved_completion_vertex']=source['unobserved_completion_vertex']
            if target is not None and 'unobserved_completion_vertex' in target:data['target_unobserved_completion_vertex']=target['unobserved_completion_vertex']
            np.savez_compressed(folder/'fixed_world_geometry.npz',**data)
            jobs[name]=dict(operation=op,asset_id=asset,cache_role=role,frames=[])
        for i in range(49):
            assert shutil.disk_usage(R).free>4*1024**3
            cachepath=geo/'visibility49'/f'{i:06d}.npz';z=np.load(cachepath)
            for inputdir,outputdir in [(Path(backend['observed_mask_directory']),'source_observed_A'),(snap/'original_mask49','source_original_SAM_candidate_A')]:
                (out/outputdir).mkdir(exist_ok=True);shutil.copy2(inputdir/f'{i:06d}.png',out/outputdir/f'{i:06d}.png')
            assert sha(out/'source_observed_A'/f'{i:06d}.png')==backend['all49_observed_mask_SHA256'][i]['SHA256']
            keys=['amodal','scene_occluded_estimate','visible_known_estimate','unobserved_scene_region','visible_with_unknown_estimate']
            for key in keys:png(out/'source_B_estimate'/key/f'{i:06d}.png',z['source_B_'+key])
            for role in ['source','move']:
                for view in ['A','B']:png(out/'hidden_completion_estimates'/role/view/f'{i:06d}.png',z[role+'_'+view+'_unobserved_lower_completion_estimate'])
            for name,d in jobs.items():
                role=d['cache_role'];folder=out/name
                layers={key:z[role+'_B_'+key] if role else np.zeros((704,1280),bool) for key in keys}
                box=z[role+'_B_bbox_projected'] if role else np.zeros((704,1280),bool)
                boxvisible=z[role+'_B_bbox_scene_visible_estimate'] if role else box
                for key,value in layers.items():png(folder/'masks'/key/f'{i:06d}.png',value)
                png(folder/'masks/bbox_projected_support'/f'{i:06d}.png',box);png(folder/'masks/bbox_scene_visible_estimate'/f'{i:06d}.png',boxvisible)
                lf=fields(folder/'conditions/lamp_silhouette'/f'{i:06d}.npz',layers['visible_with_unknown_estimate'],gamma_pairs(d['operation']))
                bf=fields(folder/'conditions/bbox_silhouette'/f'{i:06d}.npz',box,gamma_pairs(d['operation']))
                d['frames'].append(dict(frame=i,lamp=lf,bbox=bf,scene_occluded_pixels=int(layers['scene_occluded_estimate'].sum()),unknown_scene_pixels=int(layers['unobserved_scene_region'].sum()),visibility_cache_SHA256=sha(cachepath)))
            z.close();state['frames_finished']=i+1;save()
            if i%7==0:print(json.dumps(dict(frames_finished=i+1,elapsed_seconds=time.time()-start)),flush=True)
        native=json.loads((snap/'INPUT_MANIFEST.json').read_text());assert native['case']==CASE
        original_mask=snap/'original_mask49';mask_changed=Path(backend['observed_mask_directory'])!=original_mask
        lower=bool(cfg['source_lower_completion_estimated']);shade=bool(cfg['source_shade_completion_estimated'])
        manifest=dict(case=CASE,case_id=native['case_id'],state='all49_conditions_generated_pending_full_reader_and_final_review',frames=49,resolution=[1280,704],actual_geometry_review=str(reviewpath),actual_geometry_review_SHA256=sha(reviewpath),source_depth=backend['inputs']['source_depth'],camera_path=backend['inputs']['camera'],source_original_observed_mask=str(original_mask),source_mask_revision=dict(original_candidate=str(original_mask),current=backend['observed_mask_directory'],correction_applied=mask_changed,correction=review['source_hidden_geometry_policy'],review=reviewpath.name,upstream_record=str(snap/'INPUT_MANIFEST.json'),hidden_geometry_prior_not_observed_mask=True),source_hidden_geometry_estimated=True,metric_scale_known=False,source_original_support_contact_observed=cfg.get('source_original_support_contact_observed',True),source_lower_completion_explicitly_estimated=lower,source_shade_completion_explicitly_estimated=shade,source_base_contact_visual_reviewed=cfg.get('source_base_contact_visual_reviewed',True),reviewed_operation_references_A24=cfg['reviewed_operation_references_A24'],colors_linear_rgb=COLORS,gamma_control_pairs_by_operation={op:gamma_pairs(op) for op in ['insert','remove','move','replace','move_replace']},control_reference_not_measured_photometry=True,gamma_semantics='signed target-source stored raw; /1.1 exactly once at model entry',alpha_control_reference_pair=[.5,.5],alpha_raw_difference=0.,alpha_reader='/.9 exactly once; no ambient edit',color_palette_policy='explicit requested8 colors; not Original random8 sample claim',source_original_fixture_retained_for_insert=True,pure_move_default='source identity/geometry/dimensions/orientation and appearance retained; optional recolor/intensity variants separately requested',bbox_family='full projected3D-box silhouette; scene-visible estimate separate',unobserved_scene_policy='explicitly separate unknown; estimated B, never observed GT',edited_B_RGB=None,operations=jobs,assets=asset_proof,quality_accepted=False,automatic_HF=False)
        manifest['scene_visibility_uncertainty_revision']={k:result[k] for k in ['source_hidden_geometry_policy','source_lower_completion_separately_flagged','Native_RGB49_SAM_depth_K_E_unchanged']}
        manifest['scene_visibility_uncertainty_revision'].update(case19_wall_annotation_applied=False,Native_point_confidence_proof=result['Native_point_confidence_proof'],occlusion_eligibility=result['occlusion_eligibility'],fixed_world_geometry_static=True,precomputed_PCF_background_camera_only=True)
        manifest['hidden_completion_mask_semantics']=('Vertex-flagged estimated hidden geometry: '+cfg['source_hidden_geometry_policy']) if lower or shade else 'No source lower or shade completion added; legacy hidden-completion masks are zero, source back/thickness still estimated. Never observed SAM or GT.'
        manifest['original_library_geometry_qualification']='Three actual original user lamp-library raw SHA identities checked; whole scene nodes retained; no UE illumination qualification claimed; source photometry and hidden geometry are estimates'
        put(out/'MANIFEST.json',manifest);state.update(state='terminal0_conditions49_pending_independent_full_readback',exit_code=0);save()
    except BaseException as e:state.update(state='terminal1_no_automatic_retry',exit_code=1,error=repr(e),traceback=traceback.format_exc());save();raise

if __name__=='__main__':main()

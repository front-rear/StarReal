"""Independent full-file readback of both condition families and signed controls.

This checks data-interface semantics, not edited RGB/model inference or quality.
"""
import json,os
from pathlib import Path
import time

import sys
sys.path.insert(0,'/tmp/starreal_uid1007_case19_cpu_v1'+'/runtime/qh_cpu_py39_v1')
import cv2
import numpy as np

CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal()
ROOT=Path('/tmp/starreal_uid1007_case19_cpu_v1')/f'outputs/{CASE}/geometry_masks49_cached_QH_v1'


def main():
    manifest=json.load(open(ROOT/'MANIFEST.json'))
    variants=manifest['operations'];report=[];total=0;start=time.time()
    q=Path('/tmp/starreal_uid1007_case19_cpu_v1');geo=q/f'outputs/{CASE}/five_role_geometry49_QH_v1';backend=json.loads((q/f'inputs/CASE{CASE}_INPUT_BACKEND_QH_v1.json').read_text());assert backend['case']==manifest['case']==CASE
    assert set(x['operation'] for x in variants.values())==set(['insert','remove','move','replace','move_replace']) and len(variants)==5
    source=np.load(geo/'source/fixed_world_geometry.npz')
    import hashlib
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for i in range(49):
        assert sha(ROOT/'source_observed_A'/f'{i:06d}.png')==backend['all49_observed_mask_SHA256'][i]['SHA256']
        assert sha(ROOT/'source_original_SAM_candidate_A'/f'{i:06d}.png')==sha(Path(backend['original_observed_mask_directory'])/f'{i:06d}.png')
    for name,row in variants.items():
        geom=np.load(ROOT/name/'fixed_world_geometry.npz')
        for key in ['vertices','faces','center','size','rotation']:np.testing.assert_array_equal(geom['source_'+key],source[key])
        target=np.load(geo/row['cache_role']/'fixed_world_geometry.npz') if row['cache_role'] else None
        if target is None:
            assert geom['target_vertices'].shape==(0,3) and geom['target_faces'].shape==(0,3) and not geom['target_size'].any()
        else:
            for key in ['vertices','faces','center','size','rotation']:np.testing.assert_array_equal(geom['target_'+key],target[key])
    assert manifest['alpha_control_reference_pair']==[.5,.5] and manifest['alpha_raw_difference']==0.

    for name,operation in variants.items():
        op=operation['operation']
        expected_pairs=manifest['gamma_control_pairs_by_operation'][op]
        rows=[]
        for family,mask_name in [('lamp_silhouette','visible_with_unknown_estimate'),('bbox_silhouette','bbox_projected_support')]:
            paths=sorted((ROOT/name/'conditions'/family).glob('*.npz'))
            assert len(paths)==49
            for i,path in enumerate(paths):
                assert path.stem==f'{i:06d}'
                mask=cv2.imread(str(ROOT/name/'masks'/mask_name/f'{i:06d}.png'),0)
                assert mask is not None and mask.shape==(704,1280)
                with np.load(path) as z:
                    support=z['support'];x,y=map(int,z['origin_xy']);shape=tuple(z['full_shape_hw'])
                    assert shape==(704,1280) and support.dtype==np.bool_
                    h,w=support.shape;assert x>=0 and y>=0 and x+w<=1280 and y+h<=704
                    full=np.zeros(shape,bool);full[y:y+h,x:x+w]=support
                    np.testing.assert_array_equal(full,mask>0)
                    if op=='remove':assert not full.any()
                    for color,rgb in manifest['colors_linear_rgb'].items():
                        field=z['color_'+color]
                        assert field.dtype==np.float32 and field.shape==(h,w,3)
                        assert np.isfinite(field).all() and (field>=0).all() and (field<=1).all()
                        assert not field[~support].any()
                        if support.any():np.testing.assert_allclose(field[support],np.broadcast_to(np.asarray(rgb),field[support].shape),atol=1e-7)
                    for level,(source,target) in expected_pairs.items():
                        stored_source=float(z['source_gamma_control_'+level]);stored_target=float(z['target_gamma_control_'+level])
                        assert 0<=stored_source<=1.1 and 0<=stored_target<=1.1
                        assert stored_source==source and stored_target==target
                        if op=='insert':assert stored_source==0 and stored_target>0
                        if op=='remove':assert stored_target==0
                        raw=stored_target-stored_source
                        spatial=z['raw_gamma_'+level]
                        assert spatial.dtype==np.float32 and np.isfinite(spatial).all()
                        assert not spatial[~support].any()
                        np.testing.assert_allclose(spatial[support],raw,atol=1e-7)
                        np.testing.assert_allclose(spatial[support]/1.1,raw/1.1,atol=1e-7)
                    total+=1
                rows.append({'family':family,'frame':i,'visible_support_pixels':int(full.sum())})
        report.append({'variant':name,'operation':op,'files_fully_read':len(rows),'states_correct_for_absence':True})
    result={'case':CASE,'status':'all_condition_files_and_control_semantics_readback_passed_not_final_visual_acceptance',
        'variants':len(variants),'frames_per_family':49,'families':2,'files_read':total,'details':report,
        'resolution':[1280,704],'gamma_contract':'signed unnormalized target-source; /1.1 exactly once at model entry',
        'input_RGB_is_observed_not_B_ground_truth':True,'full_model_inference_run':False,'quality_accepted':False,
        'elapsed_seconds':time.time()-start}
    (ROOT/'INDEPENDENT_CONDITION_READBACK.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='details'}),flush=True)


if __name__=='__main__':main()

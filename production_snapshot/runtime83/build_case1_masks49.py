"""Finite all-frame geometry and two condition families, never auto-accepts."""
import datetime
import fcntl
import itertools
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,visibility

ROOT=Path('/root/autodl-tmp/codex_realcase_reconstruction_20261112_v1')
OLD=Path('/root/autodl-tmp/codex_user8_move_replace_20261109')
CASE='rv_pexels3_12237997_1920_1080_24fps'
COLORS={'purple':[.72,.02,1.],'green':[.02,1.,.03],'blue':[.02,.08,1.],'warm':[1.,.75,.42],
        'red':[1.,0,0],'cyan':[0,1.,1.],'yellow':[1.,1.,0],'magenta':[1.,0,1.]}
def gamma_pairs(op):
    # Interface control references, not measured source-video photometry.
    if op=='insert':return {'low':(0.,.15),'medium':(0.,.5),'high':(0.,1.05)}
    if op=='remove':return {'removed':(.5,0.)}
    return {'decrease':(.5,.15),'unchanged':(.5,.5),'increase':(.5,1.05)}
SIGNS=np.array(list(itertools.product((-1,1),repeat=3)),float)
BOX_FACES=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                    [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int32)


def png(path,mask):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert cv2.imwrite(str(path),mask.astype(np.uint8)*255)


def fields(path,support,pairs):
    path.parent.mkdir(parents=True,exist_ok=True)
    yy,xx=np.where(support)
    x0,y0,x1,y1=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)) if len(xx) else (0,0,1,1)
    m=support[y0:y1,x0:x1]
    arrays={'support':m,'origin_xy':np.array([x0,y0]),'full_shape_hw':np.array([704,1280])}
    for name,rgb in COLORS.items():arrays['color_'+name]=m[...,None]*np.array(rgb,np.float32)
    for name,(source,target) in pairs.items():
        assert 0<=source<=1.1 and 0<=target<=1.1
        arrays['raw_gamma_'+name]=m.astype(np.float32)*(target-source)
        arrays['source_gamma_control_'+name]=np.array(source,np.float64)
        arrays['target_gamma_control_'+name]=np.array(target,np.float64)
    np.savez_compressed(path,**arrays)
    with np.load(path) as z:
        np.testing.assert_array_equal(z['support'],m)
        for name,rgb in COLORS.items():
            expected=m[...,None]*np.array(rgb,np.float32)
            np.testing.assert_array_equal(z['color_'+name],expected)
        for name,(source,target) in pairs.items():
            gamma=target-source
            raw=z['raw_gamma_'+name]
            np.testing.assert_array_equal(raw,m.astype(np.float32)*gamma)
            np.testing.assert_allclose(raw[m]/1.1,gamma/1.1,atol=1e-7)
    return {'support_pixels':int(support.sum()),'roi_xyxy':[x0,y0,x1,y1]}


def main():
    lock=(ROOT/'logs/case1_masks49.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=ROOT/'outputs/1/geometry_masks49_v1';out.mkdir(parents=True,exist_ok=False)
    pilot=ROOT/'outputs/1/geometry_review_pilot_v5'
    report=json.load(open(pilot/'REVIEW_REPORT.json'))
    jobs={}
    for row in report['operations']:
        op=row['operation'];number=row['asset_cache_number'];name=op+(f'_asset{number}' if number else '')
        geom=np.load(pilot/name/'geometry.npz')
        d={k:geom[k] for k in geom.files}
        d.update(op=op,number=number,stats=[])
        rotation=(d['source_rotation'] if op=='move' else np.asarray(report['assets'][str(number)]['world_rotation'])) if op!='remove' else np.eye(3)
        if len(d['target_vertices']):
            c,size,R=bbox_for_mesh(d['target_vertices'],rotation)
            d['bbox_vertices']=(SIGNS*size/2)@R.T+c
            d['target_center'],d['target_size'],d['target_rotation']=c,size,R
        else:d['bbox_vertices']=np.empty((0,3))
        folder=out/name;folder.mkdir()
        np.savez_compressed(folder/'fixed_world_geometry.npz',**{k:v for k,v in d.items() if isinstance(v,np.ndarray)})
        jobs[name]=d
    cameras=np.load(OLD/'support_anchor_v2/1/cameras_and_boxes.npz')
    np.savez_compressed(out/'cameras49.npz',K=cameras['K'],w2c_A=cameras['w2c_A'],w2c_B=cameras['w2c_B'])
    depth_source=Path('/root/autodl-tmp/codex_aligned_v2/samples')/CASE/'step3_worldpoints/vggt_raw.npz'
    depths=np.load(depth_source)['depth']
    raster=Rasterizer();started=time.time()
    job={'state':'running_finite_49_frames','pid':os.getpid(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
         'startticks':Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],'frames_finished':0,'variants':len(jobs),'GPU_used':False}
    (out/'JOB.json').write_text(json.dumps(job,indent=2))
    for i in range(49):
        frame_start=time.time();K=cameras['K'][i];A=cameras['w2c_A'][i];B=cameras['w2c_B'][i]
        source_mask=cv2.imread(str(OLD/'sam_visible_refined_v3/1/mask'/f'{i:06d}.png'),0)>0
        dep=depths[i];h,w=dep.shape;nativeK=K.copy();nativeK[0]*=w/1280;nativeK[1]*=h/704
        native_source=cv2.resize(source_mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
        sv,sf=depth_surface(dep,nativeK,A,max_neighbor_ratio=1.15)
        rv,rf=depth_surface(dep,nativeK,A,valid_pixels=~native_source,max_neighbor_ratio=1.15)
        scene_original=raster.depth(sv,sf,K,B);scene_removed=raster.depth(rv,rf,K,B)
        for name,d in jobs.items():
            folder=out/name;source_scene=scene_original if d['op']=='insert' else scene_removed
            target_depth=raster.depth(d['target_vertices'],d['target_faces'],K,B)
            layers=visibility(target_depth,source_scene)
            for key,mask in layers.items():png(folder/'masks'/key/f'{i:06d}.png',mask)
            if len(d['bbox_vertices']):
                box_depth=raster.depth(d['bbox_vertices'],BOX_FACES,K,B)
                box_support=np.isfinite(box_depth)
                box_visible=visibility(box_depth,source_scene)['visible_with_unknown_estimate']
            else:box_support=np.zeros((704,1280),bool);box_visible=box_support
            png(folder/'masks/bbox_projected_support'/f'{i:06d}.png',box_support)
            png(folder/'masks/bbox_scene_visible_estimate'/f'{i:06d}.png',box_visible)
            lamp_fields=fields(folder/'conditions/lamp_silhouette'/f'{i:06d}.npz',layers['visible_with_unknown_estimate'],gamma_pairs(d['op']))
            bbox_fields=fields(folder/'conditions/bbox_silhouette'/f'{i:06d}.npz',box_support,gamma_pairs(d['op']))
            d['stats'].append({'frame':i,'lamp':lamp_fields,'bbox':bbox_fields,
                'scene_occluded_pixels':int(layers['scene_occluded_estimate'].sum()),
                'unknown_scene_pixels':int(layers['unobserved_scene_region'].sum())})
        job.update(frames_finished=i+1,last_frame_seconds=time.time()-frame_start,elapsed_seconds=time.time()-started)
        temp=out/'JOB.json.tmp';temp.write_text(json.dumps(job,indent=2));temp.replace(out/'JOB.json')
        print(json.dumps({k:job[k] for k in ['frames_finished','variants','last_frame_seconds','elapsed_seconds']}),flush=True)
    manifest={'state':'all_frame_geometry_and_control_fields_generated_not_final_case_accepted',
        'case':'1','case_id':CASE,'source_geometry_review':str(pilot/'REVIEW_REPORT.json'),
        'frames':49,'resolution':[1280,704],'camera_version':'existing 10deg B; actual matrices preserved',
        'colors_linear_rgb':COLORS,'gamma_control_pairs_by_operation':{op:gamma_pairs(op) for op in ['insert','remove','move','replace','move_replace']},
        'control_reference_not_measured_photometry':True,
        'color_palette_policy':'explicit requested control colors, not a claim of samples from Original random8 synthesis; no real RGB resynthesis performed',
        'gamma_semantics':'target-source, stored signed float; reader /1.1 once, not applied upstream',
        'alpha_control_reference_pair':[.5,.5],'alpha_raw_difference':0.,'alpha_reader':'/.9 once at entry; no ambient edit here',
        'lamp_family':'actual new asset meshes, or depth-derived explicitly estimated original fixture; scene occlusion estimated from per-frame depth',
        'bbox_family':'separate full projected 3D-box silhouette; scene-visible estimate also supplied separately',
        'unobserved_scene_policy':'saved separately, included as an estimate in complete lamp support, never labeled observed GT',
        'pure_move_default':'preserve source fixture identity, color and intensity; sampled color/intensity fields are additional requested-edit variants, not claimed pure-move ground truth',
        'source_original_fixture_retained_for_insert':True,'edited_B_RGB':None,
        'source_observed_mask':'existing SAM v3 retained; original observed data unchanged',
        'metric_scale_known':False,'source_hidden_geometry_estimated':True,
        'operations':{name:{'operation':d['op'],'asset_cache_number':d['number'],'frames':d['stats']} for name,d in jobs.items()},
        'remaining':['old-style and depth-fused PCF+bbox for every variant','all-frame visual review and support validation',
                     'full dataset loader readback','versioned packaging and release after final review'],
        'quality_accepted':False,'elapsed_seconds':time.time()-started}
    (out/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    job['state']='fields_generated_needs_visual_and_full_package_validation';job['exit_code']=0
    (out/'JOB.json').write_text(json.dumps(job,indent=2))


if __name__=='__main__':main()

"""Finite native49 ordinary + immutable approved shared-world PCF, QH CPU.

Up to49 independent frame workers, two reference-exact trace threads each. No automatic
Quality acceptance or upload; all source geometry and K/E remain unchanged.
"""
from pathlib import Path
import concurrent.futures, fcntl, gc, hashlib, importlib.util, json
import multiprocessing, os, shutil, subprocess, sys, time, traceback
DURABLE=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
R=Path(os.environ.get('STARREAL_CPU_ROOT',str(DURABLE)));CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal()
WORKERS=int(os.environ.get('STARREAL_FRAME_WORKERS','49'));assert 1<=WORKERS<=49
sys.path[:0]=[str(R/'runtime/qh_cpu_py39_v1'),str(R/'runtime/qh_pcf_cpu_v1'),str(R/'runtime/qh_ffmpeg_py39_v1'),str(R/'runtime/case18_ordinary_refs_v1'),str(R/'code')]
import cv2, imageio_ffmpeg, numpy as np
from pcf_cpu_accelerator import CPUTrace
from case18_QH_ordinary_reference_v1 import bbox_mask, load_joint
SOURCE=R/f'outputs/{CASE}/geometry_masks49_cached_QH_v1'
OUT=R/f'outputs/{CASE}/two_pcf49_fixed_world_yellow_QH_v1'
BG_CACHE=R/f'outputs/{CASE}/pcf_background49_precomputed_QH_v1'
CTX={}
_capture=cv2.VideoCapture
def capture(source):return _capture(str(source),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def put(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def approved():
    policy=json.loads((R/'HUMAN_DEPTH_AWARE_YELLOW_OVERLAP_CURRENT.json').read_text())
    p=R/policy['renderer_copy'];assert sha(p)==policy['renderer_SHA256']
    assert sha(R/policy['base_approved_copy'])==policy['base_approved_SHA256']
    spec=importlib.util.spec_from_file_location('approved_case19_fusion',p)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.trace_lattice=CPUTrace(2)
    return module,policy
def init():
    cv2.setNumThreads(1);mod,policy=approved()
    m=json.loads((SOURCE/'MANIFEST.json').read_text());cams=np.load(SOURCE/'cameras49.npz');a0=cams['w2c_A'][0]
    variants={}
    for name,meta in m['operations'].items():
        with np.load(SOURCE/name/'fixed_world_geometry.npz') as z:g={k:z[k] for k in z.files}
        boxes=[];op=meta['operation']
        for key,label in [('source',1),('target',2)]:
            if key=='source' and op=='insert' or key=='target' and op=='remove':continue
            boxes.append(mod.Box((a0@np.r_[g[key+'_center'],1.])[:3],g[key+'_size']/2,a0[:3,:3]@g[key+'_rotation'],label))
        variants[name]=(meta,g,boxes)
    cache=json.loads((BG_CACHE/'RESULT.json').read_text()) if (BG_CACHE/'RESULT.json').exists() else None
    CTX.update(mod=mod,policy=policy,cams=cams,a0=a0,variants=variants,background_cache=cache)
def stats(aux):
    front=np.isfinite(aux['box_depth_m'])&aux['pcf_visible'];back=aux['box_visible']&np.isfinite(aux['pcf_depth_m'])
    assert np.all(aux['pcf_depth_m'][front]<=aux['box_depth_m'][front])
    assert np.all(aux['box_depth_m'][back]<aux['pcf_depth_m'][back])
    assert not aux['unresolved'].any()
    evidence={}
    for name,mask in [('lattice_in_front',front),('box_in_front',back)]:
        ids=np.flatnonzero(mask.ravel())
        ids=ids[np.linspace(0,len(ids)-1,min(8,len(ids))).astype(int)] if len(ids) else []
        evidence[name]=[dict(AA_xy=[int(j%2560),int(j//2560)],PCF_distance=float(aux['pcf_depth_m'].ravel()[j]),box_distance=float(aux['box_depth_m'].ravel()[j])) for j in ids]
    return dict(pcf_in_front_of_box_AA_samples=int(front.sum()),box_in_front_AA_samples=int(back.sum()),yellow_overlap_AA_samples=int((aux['labels']==3).sum()),red_only_AA_samples=int((aux['labels']==1).sum()),blue_only_AA_samples=int((aux['labels']==2).sum()),unresolved=0,depth_order_all_relevant_rays_passed=True,evidence=evidence)
def frame(i):
    c=CTX;mod=c['mod'];cams=c['cams'];start=time.time();K=cams['K'][i];B=cams['w2c_B'][i]
    C=np.linalg.inv(B@np.linalg.inv(c['a0']));prepared=None
    if c['background_cache'] is not None:
        row=c['background_cache']['frame_receipts'][i];path=BG_CACHE/'shared_pcf'/f'{i:06d}.png'
        assert row['frame']==i and sha(path)==row['PNG_SHA256']
        assert hashlib.sha256(K.tobytes()).hexdigest()==row['K_array_SHA256']
        assert hashlib.sha256(C.tobytes()).hexdigest()==row['normalized_c2w_array_SHA256']
        bg=cv2.imread(str(path))[:,:,::-1].copy();assert hashlib.sha256(bg.tobytes()).hexdigest()==row['RGB_array_SHA256']
        if i==0:
            prepared=mod.prepare_lattice(C,K,1280,704,2,.5,64.,4096)
            checked,aux=mod.render(C,K,[],1280,704,2,.5,64.,4096,prepared=prepared);np.testing.assert_array_equal(bg,checked);del checked,aux
    else:
        prepared=mod.prepare_lattice(C,K,1280,704,2,.5,64.,4096)
        bg,aux=mod.render(C,K,[],1280,704,2,.5,64.,4096,prepared=prepared);del aux
    assert cv2.imwrite(str(OUT/'shared_pcf'/f'{i:06d}.png'),cv2.cvtColor(bg,cv2.COLOR_RGB2BGR))
    blank=np.zeros((704,1280),np.uint8);rows={};proof=[]
    for name,(meta,g,boxes) in c['variants'].items():
        folder=OUT/name;op=meta['operation']
        pm,pw=(blank,blank) if op=='insert' else bbox_mask(g['source_center'],g['source_size'],g['source_rotation'],K,B)
        qm,qw=(blank,blank) if op=='remove' else bbox_mask(g['target_center'],g['target_size'],g['target_rotation'],K,B)
        labels=(pm>0).astype(np.uint8)+2*(qm>0).astype(np.uint8)
        for rel,value in [('labels',labels),('geometry/P/wire',pw),('geometry/Q/wire',qw)]:
            assert cv2.imwrite(str(folder/'old_style'/rel/f'{i:06d}.png'),value)
        for box in boxes:
            key='source' if box.label==1 else 'target'
            np.testing.assert_allclose((B@np.r_[g[key+'_center'],1.])[:3],(np.linalg.inv(C)@np.r_[box.center,1.])[:3],atol=1e-12)
        rgb,aux=mod.render(C,K,boxes,1280,704,2,.5,64.,4096,base_rgb=bg)
        row=stats(aux);yellow=(aux['labels']==3).reshape(704,2,1280,2).all(axis=(1,3))
        assert np.all(rgb[yellow,0]==rgb[yellow,1]) and np.all(rgb[yellow,2]==0)
        row.update(full_yellow_pixels=int(yellow.sum()),full_yellow_R_equals_G_B0=True,fixed_world_projection_equal_actual_B_camera=True)
        if i==0:
            full,fullaux=mod.render(C,K,boxes,1280,704,2,.5,64.,4096,prepared=prepared)
            np.testing.assert_array_equal(rgb,full);assert stats(aux)==stats(fullaux)
            selected=aux['depth_computed']
            for key in ['pcf_depth_m','box_depth_m','box_visible','pcf_visible','labels','unresolved']:
                np.testing.assert_array_equal(aux[key][selected],fullaux[key][selected])
            proof.append(dict(operation=name,RGB_differing_pixels=0,all_selected_depth_labels_exact=True,all_intersection_statistics_exact=True))
            del full,fullaux
        assert cv2.imwrite(str(folder/'new_joint'/f'{i:06d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        rows[name]=row;del rgb,aux
    if i==0:put(OUT/'NATIVE_FULL_VS_SPARSE_ALL5_EQUALITY.json',dict(state='all5_native_frame0_RGB_depth_labels_exact',records=proof,renderer_SHA256=c['policy']['renderer_SHA256'],shader_modified=False))
    row=dict(frame=i,pid=os.getpid(),variants=rows,elapsed_seconds=time.time()-start)
    put(OUT/'frame_receipts'/f'{i:06d}.json',row);gc.collect();return row
def checkvideo(path,width):
    cap=capture(path);n=0;fps=cap.get(5)
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(704,width,3);n+=1
    cap.release();assert n==49 and abs(fps-10)<1e-6
    return dict(frames_decoded=n,width=width,height=704,fps=fps,SHA256=sha(path),bytes=path.stat().st_size)
def encode(name):
    from encode_native_pcf_variant_stream49_v1 import one
    return one(name)

def main():
    cv2.setNumThreads(1);ff=imageio_ffmpeg.get_ffmpeg_exe();subprocess.run([ff,'-version'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);mod,policy=approved();rb=json.loads((SOURCE/'INDEPENDENT_CONDITION_READBACK.json').read_text());assert rb['files_read']==490
    review=json.loads((R/'reviews'/os.environ['STARREAL_GEOMETRY_REVIEW']).read_text());assert str(review['case'])==CASE and review['may_generate_full49_conditions']
    parity=json.loads((R/'receipts/QH_PRIVATE_PCF_KERNEL_REFERENCE_EXACT_v1.json').read_text());assert parity['parity']['full_RGB_and_all_depth_aux_arrays_exact']
    assert len(os.sched_getaffinity(0))>=WORKERS*2 and shutil.disk_usage(R).free>16*1024**3
    available_kb=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')));assert available_kb*1024>WORKERS*2*1024**3
    guard=(DURABLE/'logs/reconstructed_joint_cpu.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if BG_CACHE.exists():
        integration=json.loads((R/'receipts/EXACT_BACKGROUND_CACHE_CONSUMER_STREAM_V4_INTEGRATION_v1.json').read_text())
        assert integration['all_native_RGB_and_selected_depth_aux_exact'] and integration['consumer_SHA256']==sha(__file__) and integration['renderer_SHA256']==policy['renderer_SHA256']
        cache=json.loads((BG_CACHE/'RESULT.json').read_text());cj=json.loads((BG_CACHE/'JOB.json').read_text())
        assert cj['exit_code']==0 and cj['frames_finished']==cache['frames']==49
        assert cache['renderer_SHA256']==policy['renderer_SHA256'] and not cache['renderer_modified']
        assert cache['resolution']==[1280,704] and cache['AA']==2 and cache['spacing']==.5 and cache['far_cells']==64. and cache['max_steps']==4096
        assert len(cache['frame_receipts'])==49 and all(x['unresolved']==0 for x in cache['frame_receipts'])
    OUT.mkdir(exist_ok=False);(OUT/'shared_pcf').mkdir();(OUT/'frame_receipts').mkdir()
    m=json.loads((SOURCE/'MANIFEST.json').read_text());names=list(m['operations'])
    recipe=dict(schema='pcf_bbox_factored.v003',background='bbox_v3/../../shared_pcf',labels='bbox_v3/labels',source_wire='bbox_v3/geometry/P/wire',target_wire='bbox_v3/geometry/Q/wire',palette_rgb=[[0,0,0],[255,0,0],[0,0,255],[255,255,0]],fill_alpha=.58,background_alpha=.42)
    fixed=[]
    for name in names:
        for sub in ['old_style/labels','old_style/geometry/P/wire','old_style/geometry/Q/wire','old_joint','new_joint']:(OUT/name/sub).mkdir(parents=True)
        put(OUT/name/'old_style/RENDER.json',recipe);fixed.append(dict(variant=name,geometry_SHA256=sha(SOURCE/name/'fixed_world_geometry.npz')))
    start=time.time();state=dict(case=CASE,state='running_finite_QH_CPU_two_PCF49',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],frames_finished=0,frames_expected=49,frame_workers=WORKERS,trace_threads=2,GPU_used=False,quality_accepted=False,automatic_HF=False,renderer_SHA256=policy['renderer_SHA256'],renderer_modified=False,producer_SHA256=sha(__file__))
    def save():state['elapsed_seconds']=time.time()-start;put(OUT/'JOB.json',state)
    save();rows={};context=multiprocessing.get_context('spawn')
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=context,initializer=init) as pool:
            for future in concurrent.futures.as_completed([pool.submit(frame,i) for i in range(49)]):
                row=future.result();rows[row['frame']]=row;state.update(frames_finished=len(rows),actual_worker_pids=sorted({x['pid'] for x in rows.values()}));save();print(json.dumps(state),flush=True)
        state.update(state='encoding_and_independently_decoding15_videos',render_seconds=time.time()-start);save()
        videos={};encoding_adoption={}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for name,checks,adoption in pool.map(encode,names):videos[name]=checks;encoding_adoption[name]=adoption;state['variants_encoded']=len(videos);state['encoding_profile']='two_variants_three_streams_original_body_one_frame_at_a_time';save()
        result=dict(case=CASE,state='two_PCF49_components_generated_pending_actual_full49_review',frames=49,fps=10,resolution_each_panel=[1280,704],variants=names,semantic_revision=policy['required_mode'],renderer_SHA256=policy['renderer_SHA256'],renderer_modified=False,PCF_parameters=dict(spacing=.5,AA=2,far_cells=64,max_steps=4096),appearance_parameters=dict(new_box_fill_alpha=1.,new_background_blend=0.,ordinary_comparison_fill_alpha=.58),world_unit='relative monocular units, not calibrated metres',world_geometry_static=True,actual_K_E49_unchanged=True,world_geometry=fixed,frame_depth_checks=[rows[i] for i in range(49)],video_checks=videos,native_sparse_full_equality='NATIVE_FULL_VS_SPARSE_ALL5_EQUALITY.json',old_style='unmodified ordinary bbox math and original joint_frames; only native704 dimensions and logical dataset Path enumeration adapted',new_style='immutable approved fixed-world opaque shared-depth with visible projected P/Q overlap yellow',GPU_used=False,quality_accepted=False,automatic_HF=False,elapsed_seconds=time.time()-start)
        result['encoding_profile']=dict(mode='two_independent_variants_three_FFmpeg_streams_each',variant_workers=2,FFmpeg_threads_each=2,original_composition='unchanged Original joint_frames body, native704 one-frame IO,245-frame parity proven',actual_adoption=encoding_adoption)
        result['camera_only_background_cache']=dict(used=BG_CACHE.exists(),path=str(BG_CACHE) if BG_CACHE.exists() else None,cache_result_SHA256=sha(BG_CACHE/'RESULT.json') if BG_CACHE.exists() else None,all49_K_C_RGB_SHA_checked=True,first_frame_background_rerender_exact=True,box_rgb_full_vs_sparse_first_frame_exact=True);put(OUT/'RESULT.json',result);state.update(state='terminal0_generated_and_decoded_pending_actual_review',exit_code=0);save()
    except BaseException as e:
        state.update(state='terminal1_no_automatic_retry',exit_code=1,error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()

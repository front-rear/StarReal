"""Finite sixteen-worker geometry/visibility cache with unchanged Native inputs."""
from pathlib import Path
import concurrent.futures,fcntl,hashlib,json,multiprocessing,os,shutil,sys,time,traceback
N=Path(os.environ.get('STARREAL_CPU_ROOT','/tmp/starreal_uid1007_case19_cpu_v1'));CASE=os.environ['STARREAL_CASE'];assert CASE in ['108','111']
D=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');I=N/f'inputs/case{CASE}_native_CPU_v1';OUT=N/f'outputs/{CASE}/source_move_geometry49_COMPONENT_ONLY_QH_v1'
sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(I),str(N/'code')]
import cv2,numpy as np
from real_geometry import Rasterizer,bbox_for_mesh,depth_surface,visibility
from case19_source_shell_reference_QH_v1 import overlay
SIGNS=np.array([[a,b,c] for a in [-1,1] for b in [-1,1] for c in [-1,1]],float)
BF=np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int32);CTX={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def init():
    cv2.setNumThreads(1);cfg=json.loads((N/f'inputs/CASE{CASE}_GEOMETRY49_CONFIG_v1.json').read_text());raw=np.load(I/'native_depth_K_E_subset.npz');cam=np.load(I/'cameras.npz');roles={}
    for name in cfg['geometry_sources']:
        z=dict(np.load(OUT/name/'fixed_world_geometry.npz'));z['box_vertices']=(SIGNS*z['size']/2)@z['rotation'].T+z['center'];roles[name]=z
    backend=json.loads((N/f'inputs/CASE{CASE}_INPUT_BACKEND_QH_v1.json').read_text());confidence=np.load(N/f'inputs/CASE{CASE}_NATIVE_POINT_CONFIDENCE49_v1.npz')['world_points_conf'];assert confidence.shape==raw['depth'].shape
    CTX.update(cfg=cfg,raw=raw,cam=cam,roles=roles,raster=Rasterizer(),rgb=np.load(OUT/'observed_RGB49_BGR.npy',mmap_mode='r'),maskdir=Path(backend['observed_mask_directory']),confidence=confidence)
def frame(i):
    c=CTX;raw=c['raw'];cam=c['cam'];r=c['raster'];K,A=cam['K'][i],cam['w2c_A'][i];maskpath=c['maskdir']/f'{i:06d}.png';observed=cv2.imread(str(maskpath),0)>0
    Hinv=np.linalg.inv(np.asarray(c['cfg']['image_homography_old_prepared_to_new']));old_image_observed=cv2.warpPerspective(observed.astype(np.uint8),Hinv,(1280,704),flags=cv2.INTER_NEAREST)>0;source_cell=cv2.resize(old_image_observed.astype(np.float32),(raw['depth'].shape[2],raw['depth'].shape[1]),interpolation=cv2.INTER_AREA)>0;source_cell=cv2.dilate(source_cell.astype(np.uint8),np.ones((3,3),np.uint8))>0
    eligible=np.isfinite(c['confidence'][i])&(c['confidence'][i]>np.float32(1.))
    fv,ff=depth_surface(raw['depth'][i],raw['K'][i],A,valid_pixels=eligible,max_neighbor_ratio=1.15);rv,rf=depth_surface(raw['depth'][i],raw['K'][i],A,valid_pixels=eligible&~source_cell,max_neighbor_ratio=1.15)
    arrays={'source_cell_coverage_native':source_cell,'occluder_eligible_native':eligible,'exact_confidence_floor_unknown_native':~eligible};row=dict(frame=i,worker_pid=os.getpid(),observed_A_mask_SHA256=sha(maskpath),original_A_SAM_SHA256=sha(I/'original_mask49'/f'{i:06d}.png'),views={})
    for view,E in [('A',A),('B',cam['w2c_B'][i])]:
        full=r.depth(fv,ff,K,E);removed=r.depth(rv,rf,K,E);row['views'][view]={}
        for name,z in c['roles'].items():
            dep=r.depth(z['vertices'],z['faces'],K,E);layers=visibility(dep,full if name.startswith('insert_') else removed)
            for key,value in layers.items():arrays[name+'_'+view+'_'+key]=value
            bd=r.depth(z['box_vertices'],BF,K,E);arrays[name+'_'+view+'_bbox_projected']=np.isfinite(bd);arrays[name+'_'+view+'_bbox_scene_visible_estimate']=visibility(bd,full if name.startswith('insert_') else removed)['visible_with_unknown_estimate']
            if 'unobserved_completion_vertex' in z:
                faces=z['faces'][np.any(z['unobserved_completion_vertex'][z['faces']],axis=1)];hd=r.depth(z['vertices'],faces,K,E);arrays[name+'_'+view+'_unobserved_lower_completion_estimate']=np.isfinite(hd)&(hd<=dep+1e-6)
            row['views'][view][name]={k:int(a.sum()) for k,a in layers.items()}
            if view=='A':
                if name=='source':
                    image=c['rgb'][i].copy();image[layers['amodal']]=np.rint(.5*image[layers['amodal']]+.5*np.array([0,255,0])).astype(np.uint8)
                else:image=overlay(c['rgb'][i],layers)
                sub='A_overlay49';label='actualRGB geometry diagnostic; cyan UNKNOWN; no editedRGB'
            else:
                # Public mask preview has exactly one foreground RGB. Keep
                # uncertainty in the existing numeric/boolean cache, never
                # encode it as another Color on the same lamp silhouette.
                image=np.zeros((704,1280,3),np.uint8);image[layers['visible_with_unknown_estimate']]=(0,255,0) if name=='source' else (255,0,255);sub='B_estimated49';label='single-color estimated silhouette; uncertainty separate; not observedGT'
            if view=='B':
                for leaf,value in [('B_solid_silhouette49',image),('B_uncertainty49',layers['unobserved_scene_region'].astype(np.uint8)*255)]:
                    (OUT/name/leaf).mkdir(exist_ok=True);assert cv2.imwrite(str(OUT/name/leaf/f'{i:06d}.png'),value)
            cv2.putText(image,f'Case{CASE} {name} {view}{i:02d}: '+label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,255,255),1);cv2.imwrite(str(OUT/name/sub/f'{i:06d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,95])
    np.savez_compressed(OUT/'visibility49'/f'{i:06d}.npz',**arrays);put(OUT/'frame_receipts'/f'{i:06d}.json',row);return row
def main():
    assert os.environ.get('ROOT_ORIGINAL_CASE'+CASE+'_COMPONENT_LOCK_HELD')=='1';cv2.setNumThreads(1);guard=(D/'logs'/f'case{CASE}_geometry_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((N/f'inputs/CASE{CASE}_GEOMETRY49_CONFIG_v1.json').read_text());review=N/cfg['actual_pilot_review'];assert sha(review)==cfg['actual_pilot_review_SHA256'];verdict=json.loads(review.read_text());assert verdict['may_compute_finite_source_move49_component'] and not verdict['quality_accepted']
    assert sha(Path(verdict['geometry']))==verdict['geometry_SHA256']
    manifest=json.loads((I/'INPUT_MANIFEST.json').read_text());assert manifest['RGB49_camera_homography_verified'] and not manifest['new_Native_depth_inference']
    comp=Path(manifest['true4K_component_manifest']);assert sha(comp)==manifest['true4K_component_manifest_SHA256'];proof=json.loads(comp.read_text());oldcams=np.load(Path(proof['original_native_input'])/'cameras.npz');newcams=np.load(I/'cameras.npz');H=np.asarray(proof['H_old_prepared1280_to_new_Original1280']);np.testing.assert_array_equal(newcams['K'],H@oldcams['K'])
    for key in oldcams.files:
        if key!='K':np.testing.assert_array_equal(newcams[key],oldcams[key])
    assert sha(I/'native_depth_K_E_subset.npz')==proof['original_native_depth_K_E_SHA256']
    before={f:sha(I/f) for f in ['cameras.npz','native_depth_K_E_subset.npz']};assert len(os.sched_getaffinity(0))>=16 and shutil.disk_usage(N).free>16*1024**3
    OUT.mkdir(exist_ok=False);(OUT/'visibility49').mkdir();(OUT/'frame_receipts').mkdir();geometry_proof={}
    for name,row in cfg['geometry_sources'].items():
        src=N/row['path'];z=np.load(src);v,f,rot=[z[row[k]] for k in ['vertices_key','faces_key','rotation_key']];c,size,basis=bbox_for_mesh(v,rot);a=dict(vertices=v,faces=f,center=c,size=size,rotation=basis)
        if name in ['source','move']:a['unobserved_completion_vertex']=z['source_unobserved_completion_vertex']
        folder=OUT/name;folder.mkdir();(folder/'A_overlay49').mkdir();(folder/'B_estimated49').mkdir();np.savez_compressed(folder/'fixed_world_geometry.npz',**a);geometry_proof[name]=dict(source_path=str(src),source_SHA256=sha(src),fixed_world_geometry_SHA256=sha(folder/'fixed_world_geometry.npz'))
    manifest=json.loads((I/'INPUT_MANIFEST.json').read_text());video=N/f'inputs/case{CASE}_source_video_true4K_v2/source_A.mkv';assert sha(video)==manifest['source_video_SHA256'];rgb=np.lib.format.open_memmap(OUT/'observed_RGB49_BGR.npy',mode='w+',dtype=np.uint8,shape=(49,704,1280,3));cap=cv2.VideoCapture(str(video),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
    for i in range(49):
        ok,im=cap.read();assert ok and hashlib.sha256(im.tobytes()).hexdigest()==manifest['RGB49_BGR_array_SHA256'][i];rgb[i]=im
    assert not cap.read()[0];cap.release();rgb.flush();del rgb;start=time.time();state=dict(case=CASE,state='running_finite_full49_geometry',pid=os.getpid(),uid=os.getuid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),startticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],frames_finished=0,frame_workers=16,GPU_used=False,quality_accepted=False);rows={}
    def save():state['elapsed_seconds']=time.time()-start;put(OUT/'JOB.json',state)
    save()
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=16,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
            for f in concurrent.futures.as_completed([pool.submit(frame,i) for i in range(49)]):
                row=f.result();rows[row['frame']]=row;state.update(frames_finished=len(rows),actual_worker_pids=sorted({v['worker_pid'] for v in rows.values()}));save()
        sheets=[]
        for name in cfg['geometry_sources']:
            for view in ['A','B']:
                canvas=np.full((1330,2240,3),24,np.uint8)
                for i in range(49):
                    p=OUT/name/('A_overlay49' if view=='A' else 'B_estimated49')/f'{i:06d}.jpg';im=cv2.imread(str(p));assert im.shape==(704,1280,3);x,y=(i%7)*320,(i//7)*190;canvas[y+14:y+190,x:x+320]=cv2.resize(im,(320,176),interpolation=cv2.INTER_AREA);cv2.putText(canvas,f'{name} {view}{i:02d}',(x+2,y+11),cv2.FONT_HERSHEY_SIMPLEX,.28,(255,255,255),1)
                p=OUT/f'{name}_{view}_all49.jpg';cv2.imwrite(str(p),canvas,[cv2.IMWRITE_JPEG_QUALITY,96]);sheets.append(dict(role=name,view=view,path=str(p.relative_to(N)),SHA256=sha(p),frames=49))
        assert before=={f:sha(I/f) for f in before};result=dict(case=CASE,state='all49_two_role_SOURCE_MOVE_COMPONENT_ONLY_READY_FOR_ACTUAL_REVIEW',frames=49,resolution=[1280,704],roles=list(cfg['geometry_sources']),sheets=sheets,statistics=[rows[i] for i in range(49)],geometry_proof=geometry_proof,world_geometry_static_across_all49=True,Original4K_RGB49_SAM_unchanged_during_run=True,original_raw_Native_depth_K_E_preserved=True,full_K_homography_derived=True,source_cell_inverse_H_applied=True,source_hidden_geometry_policy=cfg['source_hidden_geometry_policy'],source_lower_completion_separately_flagged=True,visibility='Native monocular scene surface; missing/removed source regions explicitly UNKNOWN; novel-B silhouette is estimate not observedGT',quality_accepted=False,automatic_HF=False,GPU_used=False,elapsed_seconds=time.time()-start);result['Native_point_confidence_proof']=json.loads((N/f'inputs/CASE{CASE}_NATIVE_POINT_CONFIDENCE49_v1.json').read_text());result['occlusion_eligibility']=dict(rule='finite Native point-head confidence strictly greater than exact expp1 activation floor1; no calibrated depth probability claim',no_new_numeric_threshold=True,unknown_not_asserted_observed_visible=True);result['observed_mask_backend']=str(N/f'inputs/CASE{CASE}_INPUT_BACKEND_QH_v1.json');put(OUT/'RESULT.json',result);state.update(state='terminal0_SOURCE_MOVE_COMPONENT_ONLY49_READY_FOR_ACTUAL_REVIEW',exit_code=0);save();print(json.dumps({k:v for k,v in result.items() if k not in ['statistics','sheets','geometry_proof']}),flush=True)
    except BaseException as e:state.update(state='terminal1_preserve_partial_no_automatic_retry',exit_code=1,error=repr(e),traceback=traceback.format_exc());save();raise
if __name__=='__main__':main()

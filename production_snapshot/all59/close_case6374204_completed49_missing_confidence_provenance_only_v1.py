from pathlib import Path
import json,hashlib,sys,os,datetime,time,contextlib,io
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1';N=Path('/tmp/starreal_uid1007_case6374204_true4K_cpu_v2');A=Path('/tmp/starreal_uid1007_case109_true4K_cpu_v2');OUT=N/'outputs/6374204/five_role_geometry49_QH_v1';CASE='6374204';I=N/'inputs/case6374204_native_CPU_v1';sys.path.insert(0,str(N/'runtime/qh_cpu_py39_v1'));import numpy as np
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((OUT/'JOB.json').read_text());assert old['exit_code']==1 and old['frames_finished']==49 and 'NATIVE_POINT_CONFIDENCE49_v1.json' in old['traceback'];assert not Path('/proc/'+str(old['pid'])).exists()
cfg=json.loads((N/'inputs/CASE6374204_GEOMETRY49_CONFIG_v1.json').read_text());backend=json.loads((N/'inputs/CASE6374204_INPUT_BACKEND_QH_v1.json').read_text());rows={};sheets=[];geometry_proof={};raw=np.load(I/'native_depth_K_E_subset.npz');native_shape=tuple(raw['depth'].shape[1:])
p=N/'inputs/CASE6374204_NATIVE_POINT_CONFIDENCE49_v1.json';assert p.resolve()==(A/'inputs/CASE109_NATIVE_POINT_CONFIDENCE49_v1.json').resolve()
for i in range(49):
 row=json.loads((OUT/'frame_receipts'/f'{i:06d}.json').read_text());assert row['frame']==i;rows[i]=row
 with np.load(OUT/'visibility49'/f'{i:06d}.npz') as z:
  for k in z.files:
   ar=z[k];assert ar.shape in [(704,1280),native_shape],(k,ar.shape)
  for role in cfg['geometry_sources']:
   for view in ['A','B']:
    for name in ['amodal','scene_occluded_estimate','visible_known_estimate','unobserved_scene_region','visible_with_unknown_estimate']:
     assert z[role+'_'+view+'_'+name].shape==(704,1280)
for name,b in cfg['geometry_sources'].items():
 for view in ['A','B']:
  p=OUT/f'{name}_{view}_all49.jpg';assert p.exists();sheets.append(dict(role=name,view=view,path=str(p.relative_to(N)),SHA256=sha(p)))
 geometry_proof[name]=dict(source_path=b['path'],source_SHA256=sha(b['path']),fixed_world_geometry_SHA256=sha(OUT/name/'fixed_world_geometry.npz'))
cams=np.load(I/'cameras.npz');c=np.load(A/'inputs/case109_native_CPU_v1/cameras.npz')
for k in ['K','w2c_A','w2c_B']:np.testing.assert_array_equal(cams[k],c[k])
rgb=np.load(OUT/'observed_RGB49_BGR.npy',mmap_mode='r');manifest=json.loads((I/'INPUT_MANIFEST.json').read_text());assert [hashlib.sha256(rgb[i].tobytes()).hexdigest() for i in range(49)]==manifest['RGB49_BGR_array_SHA256']
before={'cameras.npz':backend['inputs']['camera']['SHA256'],'native_depth_K_E_subset.npz':backend['inputs']['source_depth']['SHA256']};assert before=={f:sha(I/f) for f in before}
(OUT/'JOB_BEFORE_MISSING_CONFIDENCE_METADATA_REPAIR.json').write_text(json.dumps(old,indent=2)+'\n')
def put(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
state=dict(old);state.pop('error',None);state.pop('traceback',None);state['metadata_only_repair']='Exact unchanged confidence provenance JSON alias restored after all49 done; existing cache arrays/RGB/cameras independently read, no render repeated'
start=time.time()-old['elapsed_seconds']
def save():put(OUT/'JOB.json',state)
lines=(U/'build_case6374204_true4K_five_role_geometry49_QH_v1.py').read_text().splitlines();line=next(s.strip() for s in lines if "assert before==" in s and "result=dict" in s)
with contextlib.redirect_stdout(io.StringIO()):exec(compile(line,'exact_existing_producer_metadata_finalization_line','exec'))
assert state['exit_code']==0
put(U/'CASE6374204_GEOMETRY_METADATA_ONLY_CLOSURE_v1.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='all49_existing_geometry_reader_validated_and_missing_provenance_pointer_restored',original_terminal_receipt=str(OUT/'JOB_BEFORE_MISSING_CONFIDENCE_METADATA_REPAIR.json'),source_RGB_camera_depth_unchanged=True,all49_cache_arrays_read=True,new_render_jobs=0,missing_file=str(N/'inputs/CASE6374204_NATIVE_POINT_CONFIDENCE49_v1.json'),metadata_only=True,exact_existing_producer_metadata_line_executed=True,quality_accepted=False))
print('EXISTING_ALL49_METADATA_CLOSED_NO_RERENDER')

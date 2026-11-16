from pathlib import Path
import argparse,hashlib,json,sys,shutil,datetime,fcntl
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');sys.path.insert(0,'/tmp/starreal_uid1007_case19_cpu_v1/runtime/qh_cpu_py39_v1')
import cv2,numpy as np
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--case',required=True,choices=['14','27']);c=a.parse_args().case;cv2.setNumThreads(1)
 R=Path(f'/tmp/starreal_uid1007_case{c}_cpu_v2');D=R/f'outputs/{c}/source_move_geometry49_COMPONENT_ONLY_QH_v1';review=Q/f'reviews/CASE{c}_SOURCE_MOVE_COMPONENTS_ALL49_ACTUAL_REVIEW_v1.json';v=json.loads(review.read_text());assert not v['quality_accepted'] and len(v['sheets'])==4
 for row in v['sheets']:assert sha(Path(row['path']))==row['SHA256'] and row['frames']==list(range(49))
 result=json.loads((D/'RESULT.json').read_text());assert result['frames']==49 and set(result['roles'])=={'source','move'} and result['world_geometry_static_across_all49'] and not result['quality_accepted']
 native=json.loads((R/f'inputs/case{c}_native_CPU_v1/INPUT_MANIFEST.json').read_text());rgb=np.load(D/'observed_RGB49_BGR.npy',mmap_mode='r')
 for i in range(49):
  assert hashlib.sha256(rgb[i].tobytes()).hexdigest()==native['RGB49_BGR_array_SHA256'][i]
  with np.load(D/'visibility49'/f'{i:06d}.npz') as z:
   for key in z.files:assert z[key].dtype==bool
   for role,color in [('source',[0,255,0]),('move',[255,0,255])]:
    support=z[role+'_B_visible_with_unknown_estimate'];im=cv2.imread(str(D/role/'B_solid_silhouette49'/f'{i:06d}.png'));assert im.shape==(704,1280,3);expected=np.zeros_like(im);expected[support]=color;np.testing.assert_array_equal(expected,im);unc=cv2.imread(str(D/role/'B_uncertainty49'/f'{i:06d}.png'),0)>0;np.testing.assert_array_equal(unc,z[role+'_B_unobserved_scene_region'])
 z1=np.load(D/'source/fixed_world_geometry.npz');z2=np.load(D/'move/fixed_world_geometry.npz');cfg=json.loads((R/f'inputs/CASE{c}_GEOMETRY49_CONFIG_v1.json').read_text());original=np.load(Path(cfg['geometry_sources']['source']['path']));translation=original['move_mount']-original['source_mount'];np.testing.assert_allclose(z2['vertices']-z1['vertices'],np.broadcast_to(translation,z1['vertices'].shape),atol=1e-12);np.testing.assert_array_equal(z1['faces'],z2['faces']);np.testing.assert_allclose(z1['size'],z2['size'],atol=1e-12);np.testing.assert_allclose(z1['rotation'],z2['rotation'],atol=1e-12)
 files=[p for p in sorted(D.rglob('*')) if p.is_file()];dest=Q/f'component_cache/{c}/source_move_geometry49_COMPONENT_ONLY_QH_v1';assert not dest.exists();dest.mkdir();rows=[]
 for src in files:
  rel=src.relative_to(D);dst=dest/rel;dst.parent.mkdir(parents=True,exist_ok=True);digest=sha(src);shutil.copyfile(src,dst);assert dst.stat().st_size==src.stat().st_size and sha(dst)==digest;rows.append(dict(path=str(rel),bytes=src.stat().st_size,SHA256=digest))
 proof=dict(case=c,state='reviewed_source_move_all49_independent_reader_and_durable_file_SHA_verified_COMPONENT_ONLY',files=len(rows),total_bytes=sum(x['bytes'] for x in rows),records=rows,all4_sheets49_actually_viewed=True,RGB49_native_array_exact=True,all98_formal_B_solid_silhouettes_pixel_exact=True,all98_uncertainty_maps_separate=True,all49_visibility_bool_arrays_read=True,same_faces_size_rotation_rigid_translation_verified=True,geometry_back_thickness_and_Native_depth_are_estimates=True,review=str(review),review_SHA256=sha(review),quality_accepted=False,whole_case_complete=False,new_complete_groups=0,GPU_used=False,at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 p=dest/'DURABLE_FILE_SHA_PROOF.json';p.write_text(json.dumps(proof,indent=2));receipt=Q/f'receipts/CASE{c}_SOURCE_MOVE_ALL49_DURABLE_COMPONENT_v1.json';assert not receipt.exists();receipt.write_text(json.dumps({k:v for k,v in proof.items() if k!='records'},indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='records'}),flush=True)
if __name__=='__main__':main()

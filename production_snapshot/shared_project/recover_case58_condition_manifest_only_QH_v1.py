"""Recover one proven missing metadata key without regenerating any arrays."""
from pathlib import Path
import fcntl,hashlib,json,os,sys,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');CASE='58';sys.path[:0]=[str(N/'runtime/qh_cpu_py39_v1'),str(N/'code'),str(N/'inputs/case58_native_CPU_v1')]
import numpy as np
from build_case1_masks49 import COLORS,gamma_pairs
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
 start=time.time();out=N/'outputs/58/geometry_masks49_cached_QH_v1';guard=(N/'logs/case58_QH_conditions49.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);old=json.loads((out/'JOB.json').read_text());assert old['exit_code']==1 and old['frames_finished']==49 and old['error']=="KeyError('source_hidden_geometry_policy')";assert not(out/'MANIFEST.json').exists()
 before={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='JOB.json'};put(out/'FAILED_BUILD_STATE_METADATA_KEYERROR_v1.json',old)
 cfg=json.loads((N/'inputs/CASE58_GEOMETRY49_CONFIG_v1.json').read_text());backend=json.loads((N/'inputs/CASE58_INPUT_BACKEND_QH_v1.json').read_text());reviewpath=N/'reviews/CASE58_FIVE_ROLE_ALL49_GEOMETRY_ACTUAL_REVIEW_v1.json';review=json.loads(reviewpath.read_text());assert 'source_hidden_geometry_policy' not in review
 review['source_hidden_geometry_policy']=cfg['source_hidden_geometry_policy'];snap=N/'inputs/case58_native_CPU_v1';geo=N/review['component_root'];result=json.loads((geo/'RESULT.json').read_text());asset_proof=json.loads((N/cfg['assets_proof']).read_text());byop={x['role']:x['source_index'] for x in asset_proof['actual_exact_raw_SHA_matched']};jobs={}
 for name,op,asset,role in [('remove','remove',None,None),('move','move',None,'move')]+[(op+'_'+byop[op],op,byop[op],op+'_'+byop[op]) for op in ['insert','replace','move_replace']]:
  rows=[]
  for i in range(49):
   summaries={}
   for family,short in [('lamp_silhouette','lamp'),('bbox_silhouette','bbox')]:
    with np.load(out/name/'conditions'/family/f'{i:06d}.npz') as z:
     mask=z['support'];x,y=map(int,z['origin_xy']);hh,ww=mask.shape;summaries[short]=dict(support_pixels=int(mask.sum()),roi_xyxy=[x,y,x+ww,y+hh]);np.testing.assert_array_equal(z['full_shape_hw'],[704,1280])
   cache=geo/'visibility49'/f'{i:06d}.npz'
   with np.load(cache) as z:
    occ=int(z[role+'_B_scene_occluded_estimate'].sum()) if role else 0;unknown=int(z[role+'_B_unobserved_scene_region'].sum()) if role else 0
   rows.append(dict(frame=i,**summaries,scene_occluded_pixels=occ,unknown_scene_pixels=unknown,visibility_cache_SHA256=sha(cache)))
  jobs[name]=dict(operation=op,asset_id=asset,cache_role=role,frames=rows)
 # Execute only the unchanged original manifest-construction statements.
 source=(N/'build_native_cached_conditions49_QH_v3.py').read_text();block=source[source.index("        native=json.loads"):source.index("        put(out/'MANIFEST.json'")];block='\n'.join(line[8:] for line in block.splitlines());scope=dict(CASE=CASE,json=json,snap=snap,Path=Path,backend=backend,cfg=cfg,reviewpath=reviewpath,review=review,sha=sha,COLORS=COLORS,gamma_pairs=gamma_pairs,jobs=jobs,asset_proof=asset_proof,result=result);exec(compile(block,'unchanged_manifest_constructor','exec'),scope);manifest=scope['manifest'];assert manifest['source_original_support_contact_observed'] is False
 for rel,h in before.items():assert sha(out/rel)==h
 put(out/'MANIFEST.json',manifest);proof=dict(case='58',state='manifest_only_recovered_all490_field_arrays_unchanged',prior_failure_preserved='FAILED_BUILD_STATE_METADATA_KEYERROR_v1.json',missing_policy_key_read_from_same_reviewed_config=True,original_review_file_unchanged=True,unchanged_output_file_count=len(before),SHA256_unchanged=before,fields_regenerated=0,geometry_rerasterized=0,quality_accepted=False,elapsed_seconds=time.time()-start);put(out/'MANIFEST_METADATA_RECOVERY_v1.json',proof);state=dict(old);state.update(state='terminal0_conditions49_metadata_recovered_pending_independent_readback',exit_code=0,original_error_preserved=True,recovery_receipt='MANIFEST_METADATA_RECOVERY_v1.json');put(out/'JOB.json',state);print(json.dumps({k:v for k,v in proof.items() if k!='SHA256_unchanged'}),flush=True)
if __name__=='__main__':main()

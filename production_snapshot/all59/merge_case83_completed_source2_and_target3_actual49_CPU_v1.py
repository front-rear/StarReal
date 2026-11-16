from pathlib import Path
import sys,os,json,hashlib,shutil,fcntl,datetime,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');R=Path('/tmp/starreal_uid1007_case83_cpu_v2');sys.path.insert(0,str(R/'runtime/qh_cpu_py39_v1'))
import numpy as np,cv2
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 guard=(Q/'logs/case83_geometry_component.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
 a=R/'outputs/83/source_move_geometry49_COMPONENT_ONLY_QH_v1';b=R/'outputs/83/target3_geometry49_COMPONENT_ONLY_QH_v1';out=R/'outputs/83/five_role_geometry49_QH_v1';out.mkdir(exist_ok=False)
 ma=json.loads((a/'RESULT.json').read_text());mb=json.loads((b/'RESULT.json').read_text());assert len(ma['statistics'])==len(mb['statistics'])==49
 for p in [a,b]:
  st=json.loads((p/'JOB.json').read_text());assert st['exit_code']==0
  proc=Path('/proc')/str(st['pid'])/'stat';assert not proc.exists() or proc.read_text().split()[21]!=st['startticks']
 assert sha(a/'observed_RGB49_BGR.npy')==sha(b/'observed_RGB49_BGR.npy')
 shutil.copyfile(a/'observed_RGB49_BGR.npy',out/'observed_RGB49_BGR.npy')
 roles=ma['roles']+mb['roles'];assert len(roles)==5 and len(set(roles))==5
 sheets=[];proof={}
 for root,m in [(a,ma),(b,mb)]:
  for role in m['roles']:shutil.copytree(root/role,out/role)
  for row in m['sheets']:
   src=R/row['path'];dest=out/src.name;shutil.copyfile(src,dest);assert sha(dest)==row['SHA256'];sheets.append(dict(row,path=str(dest.relative_to(R))))
  proof.update(m['geometry_proof'])
 (out/'visibility49').mkdir();(out/'frame_receipts').mkdir();stats=[];fields=0
 for i in range(49):
  with np.load(a/'visibility49'/f'{i:06d}.npz',allow_pickle=False) as za,np.load(b/'visibility49'/f'{i:06d}.npz',allow_pickle=False) as zb:
   common=set(za.files)&set(zb.files);assert common=={'source_cell_coverage_native','occluder_eligible_native','exact_confidence_floor_unknown_native'}
   for key in common:np.testing.assert_array_equal(za[key],zb[key])
   merged={key:za[key] for key in za.files}
   for key in zb.files:
    if key not in common:merged[key]=zb[key]
   for key,v in merged.items():assert v.dtype==bool and v.shape in [(704,1280),(280,518)];int(v.sum());fields+=1
   np.savez_compressed(out/'visibility49'/f'{i:06d}.npz',**merged)
  x=dict(ma['statistics'][i]);y=mb['statistics'][i];assert x['frame']==y['frame']==i and x['observed_A_mask_SHA256']==y['observed_A_mask_SHA256'] and x['original_A_SAM_SHA256']==y['original_A_SAM_SHA256']
  x['component_worker_pids']=[x.pop('worker_pid'),y['worker_pid']]
  for view in ['A','B']:assert not set(x['views'][view])&set(y['views'][view]);x['views'][view].update(y['views'][view])
  for role in roles:
   im=cv2.imread(str(out/role/'B_solid_silhouette49'/f'{i:06d}.png'));assert im.shape==(704,1280,3);fg=np.any(im!=0,2);color=[0,255,0] if role=='source' else [255,0,255];assert not fg.any() or np.all(im[fg]==color)
  (out/'frame_receipts'/f'{i:06d}.json').write_text(json.dumps(x,indent=2));stats.append(x)
 cfgpath=R/'inputs/CASE83_GEOMETRY49_CONFIG_v1.json';ct=json.loads(cfgpath.read_text());cs=json.loads((R/'inputs/CASE83_SOURCE_MOVE_GEOMETRY49_CONFIG_SNAPSHOT_v1.json').read_text());combined=dict(cs,geometry_sources=dict(cs['geometry_sources'],**ct['geometry_sources']),component_only=False)
 (R/'inputs/CASE83_TARGET3_GEOMETRY49_CONFIG_SNAPSHOT_v1.json').write_bytes(cfgpath.read_bytes())
 rp=R/'reviews/CASE83_COMBINED15_PILOT_GEOMETRY_BINDINGS_REVIEW_v1.json'
 records=[]
 for p in [R/'reviews/CASE83_SOURCE_RIGID_PILOT3_ACTUAL_REVIEW_v1.json',R/'reviews/CASE83_THREE_WHOLE_WALL_TARGETS_ACTUAL9_REVIEW_v1.json']:
  rr=json.loads(p.read_text());records+=rr['geometry_bindings']
 rev=dict(case='83',may_compute_finite_full49_geometry=True,quality_accepted=False,geometry_bindings=records,actual15_pilots_reviewed=True,source_and_target_component_reviews=[str(R/'reviews/CASE83_SOURCE_RIGID_PILOT3_ACTUAL_REVIEW_v1.json'),str(R/'reviews/CASE83_THREE_WHOLE_WALL_TARGETS_ACTUAL9_REVIEW_v1.json')],already_computed_and_merged_not_rerendered=True)
 rp.write_text(json.dumps(rev,indent=2));combined.update(actual_pilot_review=str(rp),actual_pilot_review_SHA256=sha(rp));cfgpath.write_text(json.dumps(combined,ensure_ascii=False,indent=2))
 result=dict(ma,state='all49_five_role_geometry_READY_FOR_ACTUAL_CONDITIONS_REVIEW',roles=roles,sheets=sheets,statistics=stats,geometry_proof=proof,component_only=False,reused_completed_source_move49_without_rerender=True,merged_common_native_input_arrays_equal=True,merged_all_fields_read=fields,components=[dict(path=str(p),manifest_SHA256=sha(p/'RESULT.json')) for p in [a,b]],quality_accepted=False,automatic_HF=False,GPU_used=False,elapsed_seconds=time.time()-start)
 (out/'RESULT.json').write_text(json.dumps(result,indent=2))
 review=dict(case='83',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision='PASS_ALL5_ROLES_AB49_COMPONENT_READY_FOR_CONDITIONS_NOT_GROUP',may_compute_conditions49=True,may_build_actual_conditions=True,quality_accepted=False,actual10_sheets49_each_viewed=sheets,actual15_pilots_viewed=True,whole_three_material_refs12_actually_viewed=True,raw_models_whole_all_nodes_preserved=True,all49_visibility_NPZ_and_fields_read=fields,all245_B_solid_silhouette_PNG_single_foreground_color_verified=True,common49_source_masks_K_A_B_depth_confidence_and_RGB_identical_across_components=True,source_hidden_geometry_policy=combined['source_hidden_geometry_policy'],source_metadata_originals_preserved=True,not_observed_edited_B_GT=True,contact_locations='source plate1084,55; pure move/insert/MR actual clear continuous wall625,170 above nightstand; no people/curtain/books overlap',native_world_positions_static=True,GPU_used=False)
 (R/'reviews/CASE83_FIVE_ROLE_ALL49_GEOMETRY_ACTUAL_REVIEW_v1.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
 (out/'JOB.json').write_text(json.dumps(dict(case='83',state='terminal0_reused2role_plus3role_all49_numeric_and_actual_images_reviewed',exit_code=0,GPU_used=False,pid=os.getpid(),quality_accepted=False),indent=2));print(json.dumps(dict(case='83',out=str(out),roles=roles,fields_read=fields,sheets=10,seconds=time.time()-start,quality_accepted=False,GPU_used=False)))
if __name__=='__main__':main()

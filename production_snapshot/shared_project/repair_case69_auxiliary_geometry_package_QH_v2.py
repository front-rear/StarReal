"""Correct only three copied auxiliary pilot geometries, preserve invalid local v1 in full."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,time
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 t=time.time();lock=(Q/'logs/case69_portable_aux_correction.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 old=N/'packages/69/fixed_world_yellow_depth_v5_staging';out=N/'packages/69/fixed_world_yellow_depth_v5_staging_v2';chain=json.loads((N/'logs/CASE69_FINITE_REVIEWED_PACKAGE_RELEASE_v1.json').read_text());assert all(x['exit_code']==0 for x in chain['steps']);assert not out.exists()
 assert not (Q/'releases/69/revisions/fixed_world_yellow_depth_v5/HF_RECEIPT.json').exists()
 plan=json.loads((N/'outputs/69/three_whole_table_targets_pilot3_QH_v1/RESULT.json').read_text());assert len(plan['models'])==3
 excluded={'whole_target_geometry','MANIFEST.json','QUALITY_RELEASE.json','SHA256_FILES.json','BUILD_STATE.json','PORTABLE_FINAL_AUDIT.json'}
 preserved=[];out.mkdir()
 for p in sorted(old.rglob('*')):
  rel=p.relative_to(old)
  if not p.is_file() or rel.parts[0] in excluded:continue
  assert not p.is_symlink();dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);assert sha(p)==sha(dst);preserved.append(dict(path=str(rel),SHA256=sha(dst)))
 corrected=[]
 for role,row in plan['models'].items():
  src=N/'outputs/69/three_whole_table_targets_pilot3_QH_v1'/role/'geometry.npz';assert sha(src)==row['geometry_SHA256'];rel=Path('whole_target_geometry')/role/'geometry.npz';dst=out/rel;dst.parent.mkdir(parents=True);shutil.copyfile(src,dst);assert sha(dst)==sha(src)
  fixed=N/'outputs/69/five_role_geometry49_QH_v1'/role/'fixed_world_geometry.npz';assert fixed.is_file()
  corrected.append(dict(role=role,source=str(src),package_path=str(rel),correct_SHA256=sha(src),actual_full49_geometry_SHA256=sha(fixed),raw_SHA256=row['raw_SHA256']))
 assert {p.name for p in (out/'whole_target_geometry').iterdir()}==set(plan['models'])
 proof=dict(case='69',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='specific_auxiliary_package_geometry_corrected_before_any_HF_upload',error='v1 adapter copied three auxiliary pilot geometries from case66; rendered conditions, raw assets and final PCF were correct69. v1 remains local-only invalid candidate, not publishable.',correct_geometries=corrected,all_other_package_files_SHA_unchanged=True,unchanged_files=preserved,rendering_replayed=False,prior_local_candidate_preserved=True,old_local_revision_not_for_upload='fixed_world_yellow_depth_v5',publishable_revision='fixed_world_yellow_depth_v5_packaging_v2',elapsed_seconds=time.time()-t)
 (out/'AUXILIARY_GEOMETRY_CORRECTION_v2.json').write_text(json.dumps(proof,indent=2))
 m=json.loads((old/'MANIFEST.json').read_text());m.update(state='portable_fast5_audited_PENDING_SPECIFIC_REVISION_QUALITY_RELEASE',quality_accepted=False,required_remaining=['specific Quality release','HF server verification'],auxiliary_geometry_correction='AUXILIARY_GEOMETRY_CORRECTION_v2.json');m.pop('quality_release',None);(out/'MANIFEST.json').write_text(json.dumps(m,indent=2))
 audit=json.loads((old/'PORTABLE_FINAL_AUDIT.json').read_text());audit.update(auxiliary_geometries_correct_case69_SHA_verified=3,all_original_pixel_and_control_audits_reused_with_exact_file_SHA=True,auxiliary_correction_seconds=time.time()-t,quality_accepted=False);(out/'PORTABLE_FINAL_AUDIT.json').write_text(json.dumps(audit,indent=2));(out/'BUILD_STATE.json').write_text(json.dumps(dict(case='69',state='terminal0_corrected_auxiliary_geometry_audit_pending_specific_release',exit_code=0,quality_accepted=False,elapsed_seconds=time.time()-t),indent=2))
 for base in [N,Q]:
  p=base/'receipts/CASE69_AUXILIARY_PACKAGE_GEOMETRY_CORRECTION_v2.json';assert not p.exists();p.write_text(json.dumps(proof,indent=2))
 print(json.dumps({k:v for k,v in proof.items() if k!='unchanged_files'}),flush=True)
if __name__=='__main__':main()

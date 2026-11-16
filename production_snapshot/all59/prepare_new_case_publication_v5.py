"""Append one reviewed case to all current reading manifests, then exercise the actual reader."""
from pathlib import Path
import argparse, collections, datetime, hashlib, importlib.util, io, json, os, shutil, sys, tarfile, time, urllib.request
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
U=Q/'all59_reconstruction_20261115_v1'
T=Q/'reference_completion_20261115_v1'
OLD='StarReal/supplements/reference_and_remove_completion_v1'
SOLID='StarReal/supplements/solid_color_masks_20261115_v1'
NEW='StarReal/supplements/all59_completion_20261115_v1'
sys.path.insert(0,str(Q/'runtime/qh_cpu_py39_v1'))
import cv2, numpy as np
from PIL import Image
cv2.setNumThreads(1)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--revision',required=True);a=ap.parse_args();c=a.case;start=time.time()
 prep=json.loads((U/'prepared'/c/a.revision/'PREPARED.json').read_text());visual_path=U/f'CASE{c}_LATEST_VISUAL_ACCEPTANCE_v1.json';visual=json.loads(visual_path.read_text())
 assert visual['case']==c and visual['may_prepare_publication_after_full_numeric_and_reader_checks']
 for r in visual['actually_viewed']:assert sha(U/r['path'])==r['SHA256']
 assert a.revision in ['fixed_world_yellow_depth_v5','fixed_world_yellow_depth_v6']
 assert prep['source']['HF_path'].split('/')[-2]==a.revision
 pub=U/('publication_'+c);pub.mkdir(exist_ok=False);payload={}
 def put(rel,obj):
  p=pub/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');payload[rel]=p;return p
 def add(rel,p):
  assert rel not in payload;payload[rel]=Path(p)
 def fetch(path):
  with urllib.request.urlopen('https://huggingface.co/datasets/FrontRear/VLMdataset/resolve/'+head+'/'+path,timeout=90) as r:return json.load(r)
 with urllib.request.urlopen('https://huggingface.co/api/datasets/FrontRear/VLMdataset/revision/main',timeout=60) as r:head=json.load(r)['sha']
 pointer=fetch('StarReal/DATASET_CURRENT.json');cat=fetch(pointer['catalog']);assert not any(r['case']==c for r in cat['current'])
 remove_book=fetch(OLD+'/REMOVE_BINDINGS.json');ops=fetch(OLD+'/OPERATION_REFERENCE_BINDINGS.json');refs=fetch(OLD+'/REFERENCE_BINDINGS.json');solid_book=fetch(SOLID+'/SOLID_COLOR_BINDINGS.json');solid_policy=fetch('StarReal/SOLID_COLOR_MASK_CURRENT.json')
 original_binding_rows=json.dumps(remove_book['rows'],sort_keys=True)
 rroot=Path(prep['remove_root']);binding=prep['remove_binding'];archive=Path(prep['source']['local_path']);base=prep['source'];m=json.loads((U/'prepared'/c/a.revision/'SOURCE_MANIFEST.json').read_text());assert sha(archive)==base['archive_LFS_SHA256']
 release_path=Q/'releases'/c/'revisions'/a.revision/'RELEASE.json';release=json.loads(release_path.read_text());assert release['quality_accepted']
 # No original archive edits. Finish and package the additional Remove fields only.
 rm=json.loads((rroot/'MANIFEST.json').read_text());rm['state']='ACCEPTED_REVIEWED_SOURCE_REMOVE_COLOR_AND_SIGNED_INTENSITY';rm['actual_visual_review_pending']=False
 (rroot/'MANIFEST.json').write_text(json.dumps(rm,indent=2))
 quality=dict(case=c,decision='ACCEPTED_REMOVE_SOURCE_COLOR_INTENSITY_SUPPLEMENT',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_archive=base['HF_path'],source_archive_LFS_SHA256=base['archive_LFS_SHA256'],frames=49,resolution=[1280,704],views=['observed_A','estimated_B'],families=['lamp_silhouette','bbox_silhouette'],source_Color_present=True,source_signed_Intensity_present=True,raw_gamma=rm['raw_gamma'],model_gamma=rm['raw_gamma']/1.1,normalization_count=1,background_model_gamma=-1,source_Color_photometry_is_measured=False,source_RGB_unchanged=True,old_archive_changed=False,edited_B_RGB_GT=False,latest_actual_review_SHA256=sha(visual_path))
 (rroot/'QUALITY_RELEASE.json').write_text(json.dumps(quality,indent=2));shutil.copyfile(T/'code/remove_condition_reader.py',rroot/'remove_condition_reader.py')
 fields=pngs=0
 for p in rroot.rglob('*.npz'):
  if '/Intensity49/' not in str(p):continue
  with np.load(p,allow_pickle=False) as z:
   support=z['support'];assert support.dtype==bool and tuple(z['full_shape_hw'])==(704,1280);src=float(z['source_gamma_control']);tgt=float(z['target_gamma_control']);assert src>0 and tgt==0
   np.testing.assert_allclose(z['raw_gamma'][support],tgt-src,atol=1e-7);np.testing.assert_array_equal(z['color_linear_rgb'],support[...,None]*np.asarray(rm['source_color_linear_rgb'],np.float32));fields+=1
 for p in rroot.rglob('*.png'):
  if any('/'+k+'/' in str(p) for k in ['Color49','mask49']):
   with Image.open(p) as im:im.load();assert im.size==(1280,704);pngs+=1
 assert fields==196 and pngs==392
 files=[p for p in sorted(rroot.rglob('*')) if p.is_file()];inventory=[dict(path=str(p.relative_to(rroot)),bytes=p.stat().st_size,SHA256=sha(p)) for p in files]
 (rroot/'FILE_MANIFEST.json').write_text(json.dumps(dict(files=inventory,self_excluded=True),indent=2));files.append(rroot/'FILE_MANIFEST.json')
 dest=pub/OLD/binding['new_condition_archive']['path'];dest.parent.mkdir(parents=True,exist_ok=True);root='remove_'+base['archive_LFS_SHA256']
 with tarfile.open(dest,'w:gz',compresslevel=6) as tf:
  for p in files:tf.add(p,arcname=root+'/'+str(p.relative_to(rroot)),recursive=False)
 expected={x['path']:(x['bytes'],x['SHA256']) for x in inventory};read_members=0
 with tarfile.open(dest,'r|gz') as tf:
  for member in tf:
   assert member.isfile() and member.name.startswith(root+'/');b=tf.extractfile(member).read();rel=member.name[len(root)+1:]
   if rel!='FILE_MANIFEST.json':assert expected[rel]==(len(b),hashlib.sha256(b).hexdigest())
   read_members+=1
 assert read_members==len(inventory)+1
 add(OLD+'/'+binding['new_condition_archive']['path'],dest)
 binding['new_condition_archive'].update(SHA256=sha(dest),bytes=dest.stat().st_size,files=read_members,all_member_SHA256_verified=True,all196_signed_Intensity_NPZ_decoded=True,all392_Color_and_mask_PNG_1280x704_decoded=True)
 binding['quality_release']=quality;remove_book['rows'].append(binding);assert json.dumps(remove_book['rows'][:-1],sort_keys=True)==original_binding_rows
 remove_book.update(state='all_published_Remove_source_fields_reviewed_and_portable',archive_versions=len(remove_book['rows']),numeric_cases=len({x['case'] for x in remove_book['rows']}))
 add(OLD+'/'+binding['source_reference']['path'],U/'remove_publish'/binding['source_reference']['path'])
 # Actual material reference images are SHA-bound to the whole original GLBs.
 additions=json.loads((U/'reference_additions'/('REFERENCE_ADDITIONS_'+c+'_REPAIRED.json')).read_text());newassets=[r for r in additions['assets'] if r['case']==c];assert len(newassets)==3
 byrole={}
 for row in newassets:
  byrole[row['role']]=row
  for v in row['views']:
   p=U/'reference_additions'/v['path'];assert sha(p)==v['SHA256'];add(OLD+'/'+v['path'],p)
  bound=dict(archive_path=base['HF_path'],archive_LFS_SHA256=base['archive_LFS_SHA256'],manifest_member='case_'+c+'/MANIFEST.json',manifest_SHA256=release['package_manifest_SHA256'],asset_alias=row['asset_alias'],asset_member=next(x['package_path'] for x in m['assets'] if x['sha256']==row['asset_SHA256']))
  hits=[x for x in refs['assets'] if x['asset_SHA256']==row['asset_SHA256']]
  if hits:hits[0]['bound_assets'].append(bound)
  else:refs['assets'].append(dict(asset_SHA256=row['asset_SHA256'],uid=row['uid'],name=row['name'],reference=row['reference'],views=row['views'],bound_assets=[bound],provenance=row['provenance'],actual_visual_review=dict(case=c,review_SHA256=sha(visual_path))))
 refs.update(unique_assets=len(refs['assets']),reference_images=sum(len(x['views']) for x in refs['assets']),state='all_published_assets_exact_references_reviewed')
 with tarfile.open(archive) as tf:
  members={x.name:x for x in tf.getmembers() if x.isfile()};root='case_'+c+'/'
  def get(rel):return tf.extractfile(members[root+rel]).read()
  inventory=json.loads(get('SHA256_FILES.json'));assert len(inventory)>1000
  for r in inventory:
   b=get(r['path']);assert len(b)==r['size'] and hashlib.sha256(b).hexdigest()==r['sha256']
  controls=json.loads(get('conditions/MANIFEST.json'));variants=[]
  # Full videos decoded again from the exact archive, without re-rendering anything.
  video_checks=[];scratch=U/'video_readback'/c;scratch.mkdir(parents=True,exist_ok=False)
  for var in m['variants']:
   op=controls['operations'][var]['operation'];vpaths={}
   for kind,rel,width in [('ordinary',f'regular_pcf_bbox/{var}.mp4',1280),('depth_aware',f'depth_aware_pcf_bbox/{var}.mp4',1280),('comparison',f'comparison/{var}_regular_vs_depth_aware.mp4',2560)]:
    b=get(rel);tmp=scratch/(var+'_'+kind+'.mp4');tmp.write_bytes(b);cap=cv2.VideoCapture(str(tmp));assert cap.isOpened() and abs(cap.get(cv2.CAP_PROP_FPS)-10)<.001;num=0
    while True:
     ok,image=cap.read()
     if not ok:break
     assert image.shape==(704,width,3);num+=1
    cap.release();assert num==49;video_checks.append(dict(variant=var,kind=kind,frames=49,resolution=[width,704],fps=10,SHA256=hashlib.sha256(b).hexdigest()));vpaths[kind]=rel
   reference=binding['source_reference'] if op in ['move','remove'] else byrole[op]['reference']
   op_row=dict(HF_source_archive=base['HF_path'],archive_LFS_SHA256=base['archive_LFS_SHA256'],case=c,variant=var,operation=op,reference_role='source_before_removal' if op=='remove' else 'source_identity_preserved' if op=='move' else 'exact_target_whole_GLB',target_reference_required=op not in ['move','remove'],reference=reference,target_reference=None if op=='remove' else reference)
   if op=='remove':op_row['new_source_Color_Intensity_package']=binding['new_condition_archive']
   ops['bindings'].append(op_row)
   variants.append(dict(variant=var,operation=op,conditions={f:dict(frames=49,Remove_source_override_required=op=='remove') for f in ['lamp_silhouette','bbox_silhouette']},joints={v:dict(frames=49) for v in ['old_joint','new_joint']},video_paths=vpaths))
  item=dict(case=c,case_id=m['case_id'],HF_archive=base['HF_path'],archive_LFS_SHA256=base['archive_LFS_SHA256'],bytes=base['bytes'],extracted_root='case_'+c,status='current_fixed_world_opaque_yellow_reviewed_conditioning',frames=49,resolution=[1280,704],fps=10,joint_revision=m['joint_revision'],renderer_SHA256=m['renderer_SHA256'],source_RGB=m['source_RGB'],source_camera_A_B_SHA256=prep['source_camera_SHA256'],mask_families=['lamp_silhouette','bbox_silhouette'],colors_linear_rgb=controls['colors_linear_rgb'],gamma_control_pairs_by_operation=controls['gamma_control_pairs_by_operation'],relative_gamma='target-source divided by1.1 exactly once',relative_alpha='target-source divided by0.9 exactly once',variants=variants,assets=m['assets'],Remove_completion_archive=OLD+'/'+binding['new_condition_archive']['path'],Remove_completion_SHA256=binding['new_condition_archive']['SHA256'],Remove_extracted_root=binding['new_condition_archive']['extracted_root'],Remove_A_observed_and_B_estimated_available=True,operation_reference_bindings=OLD+'/OPERATION_REFERENCE_BINDINGS.json',edited_B_RGB=None,training_GT_available=False,all_original_archive_bytes_unchanged=True)
 ops.update(state='all_published_operation_references_exactly_bound',expected_archives=len(remove_book['rows']),operations=len(ops['bindings']),operation_counts=dict(collections.Counter(r['operation'] for r in ops['bindings'])),unbound=[])
 sr={k:v for k,v in prep['solid'].items() if k!='local_archive'};solid_book['rows'].append(sr);solid_book.update(cases=len(solid_book['rows']),variants=sum(r['variants'] for r in solid_book['rows']),state='all_current_solid_Color_archives_verified')
 solid_book['counts']=dict(sum((collections.Counter(r['counters']) for r in solid_book['rows']),collections.Counter()))
 add(sr['HF_archive'],Path(prep['solid']['local_archive']))
 item['solid_Color_archive']=sr['HF_archive'];item['solid_Color_SHA256']=sr['SHA256'];cat['current'].append(item);cat['current'].sort(key=lambda r:int(r['case']));cat['all_versions'].append(item);cat['all_versions'].sort(key=lambda r:(int(r['case']),r['HF_archive']))
 # Reconcile the old case70 flag against its existing canonical server object; no archive rewrite or repeat upload.
 existing70=next(r for r in cat['current'] if r['case']=='70')
 sys.path.insert(0,str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps'))
 from huggingface_hub import HfApi
 remote70=HfApi().get_paths_info('FrontRear/VLMdataset',[existing70['HF_archive']],repo_type='dataset',revision=head)[0]
 assert remote70.size==existing70['bytes'] and remote70.lfs.sha256==existing70['archive_LFS_SHA256']
 assert existing70['archive_LFS_SHA256']=='266f553b2b3a6a3203808bc35087296d54283b9b2e749cabb00d6f6ae1f6d41b'
 pointer['case70_uploaded']=True;cat['case70_uploaded']=True
 counts=collections.Counter(v['operation'] for r in cat['current'] for v in r['variants'])
 cat.update(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_HF_revision=head,numeric_cases_current=len(cat['current']),case_archive_versions=len(cat['all_versions']),total_original_archives=len(cat['all_versions'])+cat['asset_free_ordinary_supplement_archives'],current_operation_variants=sum(counts.values()),current_operation_counts=dict(counts),all_published_operation_variants=sum(len(r['variants']) for r in cat['all_versions']),unique_raw_GLB_assets=len(refs['assets']),library_reference_views=refs['reference_images'],observed_source_reference_crops=len(remove_book['rows']),old_full_reconstruction_goal_resumed=True,all59_required=59,all59_completed=len(cat['current']),all59_complete=False)
 pointer.update(state='CURRENT_AUDITED_CASES_FULL59_RECONSTRUCTION_IN_PROGRESS',source_HF_revision=head,current_cases=len(cat['current']),original_case_archive_versions=len(cat['all_versions']),total_original_archives=cat['total_original_archives'],all59_required=59,all59_completed=len(cat['current']),all59_complete=False,latest_verification=NEW+'/CASE'+c+'_FINAL_VERIFIED.json',all_current_solid_Color_export_complete=True)
 pointer.pop('all35_current_solid_Color_export_complete',None)
 solid_policy.update(complete_cases=[r['case'] for r in cat['current']],target_current_cases=len(cat['current']),cases_complete=len(cat['current']),variants_complete=cat['current_operation_variants'],all59_required=59,all59_goal_active=True,server_verification=NEW+'/CASE'+c+'_FINAL_VERIFIED.json')
 put(OLD+'/REMOVE_BINDINGS.json',remove_book);put(OLD+'/OPERATION_REFERENCE_BINDINGS.json',ops);put(OLD+'/REFERENCE_BINDINGS.json',refs);put(SOLID+'/SOLID_COLOR_BINDINGS.json',solid_book);put(SOLID+'/DATASET_INDEX.json',cat);put('StarReal/DATASET_CURRENT.json',pointer);put('StarReal/SOLID_COLOR_MASK_CURRENT.json',solid_policy)
 put('StarReal/'+c+'/CURRENT.json',dict(case=c,state='REVIEWED_LATEST_CONTRACT_CURRENT',archive=item,reader=pointer['reader'],reference_manifest=cat['reference_manifest'],Remove_manifest=cat['Remove_manifest'],solid_color_bindings=SOLID+'/SOLID_COLOR_BINDINGS.json'))
 add(base['HF_path'],archive);add(release['HF_prefix']+'/RELEASE.json',release_path);add(NEW+'/CASE'+c+'_ACTUAL_VISUAL_REVIEW.json',visual_path)
 readme=pub/'StarReal/README.md';readme.write_text('# StarReal\n\nCurrent reviewed conditioning cases: '+str(len(cat['current']))+'/59. Full59 reconstruction in progress.\n\nRead DATASET_CURRENT.json, then its catalog and strict reader. Each current case has five operations, native1280x704/49/10FPS, ordinary and fixed-world depth-aware PCF+bbox with yellow overlap, both mask families, signed controls normalized once, Remove source fields, exact material references and separate uncertainty. These real-video packages are conditioning data; no observed edited-B RGB ground truth is claimed. Historical revisions are retained.\n');add('StarReal/README.md',readme)
 for row in visual['actually_viewed']:
  add(NEW+'/visual_review/'+c+'/'+Path(row['path']).name,U/row['path'])
 # A repository-shaped temporary view proves the unchanged actual public reader works.
 view=U/('reader_view_'+c);view.mkdir(exist_ok=False)
 for rel,p in payload.items():
  dst=view/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.symlink_to(p)
 reader_dir=view/'runtime';reader_dir.mkdir()
 for name,src in [('starreal_complete_reader.py',Q/'solid_mask_audit_20261115_v1/code/starreal_complete_reader.py'),('strict_base_reader.py',Q/'solid_mask_audit_20261115_v1/strict_base_reader.py'),('ARCHIVE_BINDING_ALIASES.json',Q/'solid_mask_audit_20261115_v1/ARCHIVE_BINDING_ALIASES.json')]:shutil.copyfile(src,reader_dir/name)
 sys.path.insert(0,str(reader_dir));from starreal_complete_reader import StarRealCompleteReader
 rows=joint_frames=references=0
 with StarRealCompleteReader(view,base['HF_path']) as reader:
  for var in m['variants']:
   op=reader.controls['operations'][var]['operation'];im,ref=reader.reference_image(var);im.load();references+=1
   for family in ['lamp_silhouette','bbox_silhouette']:
    for frame in range(49):
     for v in (['A','B'] if op=='remove' else ['B']):
      x=reader.condition_frame(var,family,frame,view=v);assert x['single_color_foreground_verified'] and x['normalization_count']==1
      assert abs(x['normalized_gamma_for_model']-(x['target_gamma_control']-x['source_gamma_control'])/1.1)<1e-8;rows+=1
   for version in ['old_joint','new_joint']:
    for frame in range(49):im=reader.joint_frame(var,frame,version);assert im.size==(1280,704);joint_frames+=1
 assert (rows,joint_frames,references)==(588,490,5)
 report=dict(case=c,state='full_current_contract_actual_reader_and_visual_checks_passed',source_archive=base['HF_path'],source_archive_SHA256=base['archive_LFS_SHA256'],original_internal_file_SHA256_checked=len(inventory),Remove_NPZ_checked=fields,Remove_condition_PNG_decoded=pngs,actual_reader_frames=rows,actual_joint_frames=joint_frames,operation_references_checked=references,videos_decoded=video_checks,solid_export=sr,actual_visual_review_SHA256=sha(visual_path),old_archives_and_controls_unchanged=True,all59_required=59,current_cases_after_publication=len(cat['current']),all59_complete=False,elapsed_seconds=time.time()-start)
 put(NEW+'/CASE'+c+'_PREFLIGHT_VERIFIED.json',report)
 # This is a declaration of the checks just executed. Server verification follows upload.
 put(NEW+'/CASE'+c+'_FINAL_VERIFIED.json',dict(report,state='local_preflight_passed_pending_server_commit_verification'))
 state=dict(case=c,state='reviewed_publication_payload_ready',parent_HF_head=head,files=[dict(HF_path=rel,local_path=str(p),bytes=p.stat().st_size,SHA256=sha(p)) for rel,p in payload.items()],current_cases=len(cat['current']),all59_complete=False,preflight=report)
 (U/('CASE'+c+'_PUBLICATION_PAYLOAD.json')).write_text(json.dumps(state,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in state.items() if k not in ['files','preflight']}),flush=True)
if __name__=='__main__':main()

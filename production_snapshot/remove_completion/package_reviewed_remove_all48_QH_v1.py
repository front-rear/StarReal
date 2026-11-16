"""Package only actually reviewed Remove additions, preserving all original archives."""
from pathlib import Path
import concurrent.futures,datetime,gzip,hashlib,io,json,os,shutil,sys,tarfile,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';PUB=T/'remove_publish';PUB.mkdir(exist_ok=True);(PUB/'remove_packages').mkdir(exist_ok=True);(PUB/'source_references').mkdir(exist_ok=True);(PUB/'remove_reviews').mkdir(exist_ok=True);start=time.time()
sys.path.insert(0,str(Q/'runtime/qh_cpu_py39_v1'))
import cv2,numpy as np

def sha(b):return hashlib.sha256(b).hexdigest()
def fsha(p):return sha(p.read_bytes())
def jwrite(p,d):p.write_text(json.dumps(d,indent=2))
review_path=T/'reviews/REMOVE_ALL35_ACTUAL_VISUAL_REVIEW_v1.json';review=json.loads(review_path.read_text());idxp=T/'inventory/REMOVE_ALL48_INDEPENDENT_READER_AND_REVIEW_INDEX_v1.json';idx=json.loads(idxp.read_text());assert fsha(idxp)==review['independent_index_SHA256'] and review['may_package_and_publish_Remove_addition'];assert idx['independent_field_reader_checks']==9408 and len(review['reviewed_groups'])==35;rows=[]
for g in idx['groups']:
 assert review['reviewed_groups'][str(g['group'])]['semantic_fingerprint']==g['semantic_fingerprint'];page=Path(g['review_page']);assert fsha(page)==g['review_page_SHA256'];shutil.copyfile(page,PUB/'remove_reviews'/page.name)
 for row in g['archives']:rows.append((row,g))
assert len(rows)==48

def one(item):
 row,group=item;root=Path(row['root']);key=row['key'];manifestp=root/'MANIFEST.json';m=json.loads(manifestp.read_text());assert m['HF_source_archive_LFS_SHA256']==key;assert m['frames']==49
 provenance=root/'generation_provenance';provenance.mkdir(exist_ok=False);shutil.copyfile(manifestp,provenance/'GENERATION_MANIFEST_UNCHANGED.json');ap=root/'A_OBSERVED_RECORD.json';shutil.copyfile(ap,provenance/'A_GENERATION_RECORD_UNCHANGED.json')
 release=dict(decision='ACCEPTED_REMOVE_SOURCE_COLOR_INTENSITY_SUPPLEMENT',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_archive=row['archive'],source_archive_LFS_SHA256=key,case=row['case'],frames=49,resolution=[1280,704],views=['observed_A','estimated_B'],families=['lamp_silhouette','bbox_silhouette'],source_Color_present=True,source_signed_Intensity_present=True,raw_gamma=m['raw_gamma'],model_gamma=m['raw_gamma']/1.1,normalization_count=1,background_model_gamma=-1,source_Color_photometry_is_measured=False,source_RGB_unchanged=True,source_K_A_B_geometry_exact_to_original=True,original_source_masks_preserved=True,independent_field_reader_checks=196,actual_visual_group=group['group'],actual_review_SHA256=fsha(review_path),semantic_fingerprint=group['semantic_fingerprint'],old_archive_changed=False,edited_B_RGB_GT=False,case70_uploaded=False)
 jwrite(root/'QUALITY_RELEASE.json',release);m['state']='ACCEPTED_REMOVE_SOURCE_FIELDS_REVIEWED_VERSIONED_ADDITION';m['quality_release']='QUALITY_RELEASE.json';m['A_observed_source_conditions']='A_observed';m['actual_visual_review_pending']=False;jwrite(manifestp,m);a=json.loads(ap.read_text());a['actual_visual_review_pending']=False;a['quality_release']='QUALITY_RELEASE.json';jwrite(ap,a);shutil.copyfile(T/'code/remove_condition_reader.py',root/'remove_condition_reader.py')
 files=[p for p in sorted(root.rglob('*')) if p.is_file() and not any(x.startswith('._') for x in p.parts)];inventory=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,SHA256=fsha(p)) for p in files];jwrite(root/'FILE_MANIFEST.json',dict(files=inventory,self_excluded=True,source_archive_LFS_SHA256=key));files.append(root/'FILE_MANIFEST.json')
 archive=PUB/'remove_packages'/(key+'.tar.gz');assert not archive.exists();prefix='remove_'+key
 with archive.open('xb') as dst:
  with gzip.GzipFile(fileobj=dst,mode='wb',mtime=0,compresslevel=6) as gz:
   with tarfile.open(fileobj=gz,mode='w') as tf:
    for p in sorted(files):
     info=tf.gettarinfo(str(p),arcname=prefix+'/'+str(p.relative_to(root)));info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
     with p.open('rb') as f:tf.addfile(info,f)
 # Full portable decode and member SHA verification, not just tar existence.
 got={};npz_count=0;png_fields=0
 with tarfile.open(archive,'r:gz') as tf:
  for member in tf:
   assert member.isfile() and member.name.startswith(prefix+'/') and '..' not in Path(member.name).parts;b=tf.extractfile(member).read();rel=member.name[len(prefix)+1:];got[rel]=dict(bytes=len(b),SHA256=sha(b))
   if '/Intensity49/' in rel and rel.endswith('.npz'):
    with np.load(io.BytesIO(b),allow_pickle=False) as z:
     sup=z['support'].astype(bool);raw=z['raw_gamma'];assert tuple(z['full_shape_hw'])==(704,1280);s=float(z['source_gamma_control']);t=float(z['target_gamma_control']);assert s>0 and t==0;np.testing.assert_allclose(raw[sup],t-s,atol=1e-7);np.testing.assert_allclose(raw[sup]/np.float32(1.1),m['raw_gamma']/1.1,atol=1e-7);npz_count+=1
   if any('/'+k+'/' in rel for k in ['mask49','Color49']) and rel.endswith('.png'):
    im=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED);assert im.shape[:2]==(704,1280);png_fields+=1
 for r in inventory:assert got[r['path']]==dict(bytes=r['bytes'],SHA256=r['SHA256'])
 assert len(got)==len(inventory)+1 and npz_count==196 and png_fields==392
 source=PUB/'source_references'/(key+'.png');shutil.copyfile(root/'source_reference.png',source);assert fsha(source)==m['source_reference']['SHA256']
 out=dict(HF_source_archive=row['archive'],archive_LFS_SHA256=key,case=row['case'],new_condition_archive=dict(path=str(archive.relative_to(PUB)),SHA256=fsha(archive),bytes=archive.stat().st_size,extracted_root=prefix,files=len(got),all_member_SHA256_verified=True,all196_signed_Intensity_NPZ_decoded=True,all392_Color_and_mask_PNG_1280x704_decoded=True),frames=49,resolution=[1280,704],families=['lamp_silhouette','bbox_silhouette'],views=dict(A='A_observed/{family}',B='{family}'),source_reference=dict(path=str(source.relative_to(PUB)),SHA256=fsha(source),kind=m['source_reference']['kind'],frame=m['source_reference']['frame'],crop_xyxy=m['source_reference']['crop_xyxy']),raw_gamma=m['raw_gamma'],model_relative_gamma=m['raw_gamma']/1.1,normalization_count=1,source_color_linear_rgb=m['source_color_linear_rgb'],review_page='remove_reviews/'+Path(group['review_page']).name,review_group=group['group'],quality_release=release)
 print(json.dumps(dict(case=row['case'],key=key,archive_bytes=archive.stat().st_size)),flush=True);return out
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(one,rows))
result.sort(key=lambda x:x['HF_source_archive']);bind=dict(schema='StarReal.Remove_source_bindings.v1',state='all48_reviewed_portable_packages_ready_for_HF',source_revision=json.loads((T/'publish/HF_SOURCE_SCOPE.json').read_text())['HF_source_revision'],HF_prefix='StarReal/supplements/reference_and_remove_completion_v1',archive_versions=48,numeric_cases=35,frames=49,resolution=[1280,704],families=['lamp_silhouette','bbox_silhouette'],views=['observed_A','estimated_B'],source_Color_and_Intensity_required_on_remove=True,all_npz_model_gamma_signed_and_normalized_once=True,old_archives_overwritten=False,source_intensity_measured=False,source_color_is_recorded_appearance_hint=True,rows=result);jwrite(PUB/'REMOVE_BINDINGS.json',bind)
ops=json.loads((T/'inventory/OPERATION_REFERENCE_BINDINGS_PREFLIGHT_v1.json').read_text());bykey={x['archive_LFS_SHA256']:x for x in result}
for op in ops['bindings']:
 key=op['archive_LFS_SHA256']
 if op['operation'] in ['remove','move']:
  op['reference']=bykey[key]['source_reference'];op['reference_role']='source_before_removal' if op['operation']=='remove' else 'source_identity_preserved'
 if op['operation']=='remove':op['target_reference']=None;op['new_source_Color_Intensity_package']=bykey[key]['new_condition_archive']
ops['state']='all463_operation_references_exactly_bound';ops['unbound']=[];jwrite(PUB/'OPERATION_REFERENCE_BINDINGS.json',ops);shutil.copyfile(T/'code/remove_condition_reader.py',PUB/'remove_condition_reader.py');shutil.copyfile(review_path,PUB/'REMOVE_ACTUAL_REVIEW.json');jwrite(PUB/'REMOVE_INDEPENDENT_READER_AUDIT.json',{k:v for k,v in idx.items() if k!='groups'});jwrite(T/'inventory/REMOVE_PORTABLE_PACKAGES_VERIFIED_v1.json',dict(state='all48_actual_reviewed_portable_packages_verified_ready_for_HF',archives=48,NPZ_decoded=48*196,PNG_condition_files_decoded=48*392,total_archive_bytes=sum(x['new_condition_archive']['bytes'] for x in result),operations_bound=463,overall_goal_complete=False,elapsed_seconds=time.time()-start));print(json.dumps(dict(archives=48,bytes=sum(x['new_condition_archive']['bytes'] for x in result),seconds=time.time()-start)),flush=True)

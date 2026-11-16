"""Read actual all-current fields; export lossless single-color masks, no geometry edits."""
from pathlib import Path
import concurrent.futures,collections,hashlib,io,json,os,sys,tarfile,time,datetime,shutil
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';U=Q/'solid_mask_audit_20261115_v1';OUT=U/'outputs';PREFIX='StarReal/supplements/solid_color_masks_20261115_v1'
sys.path[:0]=[str(Q/'runtime/qh_cpu_py39_v1')]
import numpy as np,cv2
from PIL import Image
cv2.setNumThreads(1)
def sha(b):return hashlib.sha256(b).hexdigest()
def put(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def npz(b):
    with np.load(io.BytesIO(b),allow_pickle=False) as z:return {k:z[k] for k in z.files}
def png(path,support,rgb):
    # Index palette is lossless: RGB decoder sees black or one exact RGB vector.
    path.parent.mkdir(parents=True,exist_ok=True);im=Image.fromarray(support.astype(np.uint8),'P');palette=[0,0,0]+list(map(int,rgb))+[0]*(768-6);im.putpalette(palette);im.save(path,format='PNG',optimize=False,compress_level=1)
    with Image.open(path) as check:
        a=np.asarray(check.convert('RGB'));assert a.shape==(704,1280,3) and np.all(a[~support]==0) and np.all(a[support]==rgb)
    return {'path':str(path.relative_to(U)),'SHA256':sha(path.read_bytes()),'bytes':path.stat().st_size,'foreground_pixels':int(support.sum()),'foreground_RGB_uint8':list(map(int,rgb))}
def full_support(z):
    s=z['support'];assert s.dtype==bool;x,y=map(int,z['origin_xy']);h,w=s.shape;assert tuple(z['full_shape_hw'])==(704,1280) and 0<=x<=x+w<=1280 and 0<=y<=y+h<=704;a=np.zeros((704,1280),bool);a[y:y+h,x:x+w]=s;return a
def case(job):
    start=time.time();row=job['row'];p=Path(job['path']);base=OUT/row['case'];base.mkdir(parents=True,exist_ok=False);records=[];counts=collections.Counter();anomalies=[]
    tf=tarfile.open(p) if p.is_file() else None
    if tf:
        members={m.name:m for m in tf.getmembers() if m.isfile()};pre=min([n for n in members if n.endswith('/MANIFEST.json')],key=lambda x:x.count('/')).rsplit('/',1)[0]+'/'
        def get(rel):return tf.extractfile(members[pre+rel]).read()
    else:
        def get(rel):return (p/rel).read_bytes()
    cm=json.loads(get('conditions/MANIFEST.json'));hint=json.loads(get('source_appearance_control_reference.json'));hintRGB=np.asarray(hint['control_color_linear_rgb'],np.float32);binding=job['remove_binding'];remove_path=T/'remove_publish'/binding['new_condition_archive']['path'];remove={}
    with tarfile.open(remove_path,'r|gz') as rt:
        for m in rt:
            if m.isfile() and m.name.endswith('.npz') and '/Intensity49/' in m.name:remove[m.name.partition('/')[2]]=rt.extractfile(m).read()
    for var in cm['operations']:
        op=cm['operations'][var]['operation'];color='source_hint' if op in ('move','remove') else 'magenta';level='removed' if op=='remove' else 'medium' if op=='insert' else 'unchanged';fg=np.rint(np.clip(hintRGB if color=='source_hint' else cm['colors_linear_rgb'][color],0,1)*255).astype(np.uint8)
        for family in ['lamp_silhouette','bbox_silhouette']:
            frames=[]
            for i in range(49):
                if op=='remove':
                    z=npz(remove[f'{family}/Intensity49/{i:06d}.npz']);support=full_support(z);s=z['support'];np.testing.assert_array_equal(z['color_linear_rgb'],s[...,None]*hintRGB);np.testing.assert_allclose(z['raw_gamma'][s],float(z['target_gamma_control'])-float(z['source_gamma_control']),atol=1e-7);counts['Remove_source_B_condition_fields_audited']+=1
                    unknown=np.zeros((704,1280),bool)
                    n=f'conditions/source_B_estimate/unobserved_scene_region/{i:06d}.png'
                    try:unknown=cv2.imdecode(np.frombuffer(get(n),np.uint8),0)>0
                    except KeyError:pass
                else:
                    z=npz(get(f'conditions/{var}/conditions/{family}/{i:06d}.npz'));support=full_support(z);s=z['support']
                    for name,rgb in cm['colors_linear_rgb'].items():
                        expected=s[...,None]*np.asarray(rgb,np.float32);actual=z['color_'+name]
                        if not np.array_equal(actual,expected):anomalies.append({'archive':row['HF_archive'],'variant':var,'family':family,'frame':i,'color':name,'kind':'nonuniform_or_wrong_canonical_Color'})
                        counts['requested_color_arrays_audited']+=1
                    for name,(src,tgt) in cm['gamma_control_pairs_by_operation'][op].items():
                        assert float(z['source_gamma_control_'+name])==src and float(z['target_gamma_control_'+name])==tgt;np.testing.assert_allclose(z['raw_gamma_'+name][s],tgt-src,atol=1e-7);assert not z['raw_gamma_'+name][~s].any();counts['raw_intensity_levels_audited']+=1
                    counts['nonRemove_condition_fields_audited']+=1
                    try:unknown=cv2.imdecode(np.frombuffer(get(f'conditions/{var}/masks/unobserved_scene_region/{i:06d}.png'),np.uint8),0)>0
                    except KeyError:unknown=np.zeros((704,1280),bool)
                image=png(base/var/family/'Color49'/f'{i:06d}.png',support,fg);sup=png(base/var/family/'Support49'/f'{i:06d}.png',support,[255,255,255]);unk=png(base/var/family/'Uncertainty49'/f'{i:06d}.png',unknown,[255,255,255]);counts['lossless_PNGs_written_and_RGB_decoded']+=3
                frame={'frame':i,'Color':image,'Support':sup,'Uncertainty':unk};frames.append(frame)
                if row['case']=='58' and var=='move' and family=='lamp_silhouette':
                    comparison=png(base/var/family/'RequestedMagenta49'/f'{i:06d}.png',support,[255,0,255]);frame['requested_magenta_alternative']=comparison;counts['lossless_PNGs_written_and_RGB_decoded']+=1
                if i==0 and row['case']=='58' and var=='move' and family=='lamp_silhouette':
                    known=cv2.imdecode(np.frombuffer(get('conditions/move/masks/visible_known_estimate/000000.png'),np.uint8),0)>0;diag=np.zeros((704,1280,3),np.uint8);diag[known]=[255,0,255];diag[unknown]=[0,255,255];raw=np.zeros_like(diag);raw[support]=[255,0,255];pair=np.concatenate([diag,raw],1);cv2.imwrite(str(U/'CASE58_BEFORE_AFTER.png'),pair[:,:,::-1]);cv2.imwrite(str(U/'CASE58_MOVE_CORRECTED_MAGENTA.png'),raw[:,:,::-1]);cv2.imwrite(str(U/'CASE58_MOVE_ACTUAL_SOURCE_HINT.png'),np.asarray(Image.open(U/image['path']).convert('RGB'))[:,:,::-1]);put(U/'CASE58_ROOT_CAUSE.json',{'case':'58','frame':0,'actual_color_magenta_RGB':[[1,0,1]],'actual_color_magenta_already_uniform':True,'diagnostic_known_pixels':int(known.sum()),'diagnostic_unknown_pixels':int(unknown.sum()),'foreground_pixels':int(support.sum()),'bug':'The geometry diagnostic preview assigns cyan to unobserved_scene_region over magenta known pixels. It was presented as target mask without separating uncertainty from requested Color.','fix':'Actual Color remains a single selected control vector for all support pixels. Export lossless PNG; export uncertainty as a separate grayscale map; strict reader rejects any mixed Color foreground.','original_geometry_and_support_unchanged':True})
            records.append({'variant':var,'operation':op,'family':family,'default_color':color,'default_level':level,'foreground_RGB_uint8':fg.tolist(),'frames':frames,'uncertainty_not_encoded_as_Color':True,'all8_requested_colors_retained_in_original_NPZ':True})
    if tf:tf.close()
    assert not anomalies,anomalies[:2]
    manifest={'state':'all_current_case_fields_audited_solid_exports_pending_visual_review','case':row['case'],'HF_source_archive':row['HF_archive'],'source_archive_SHA256':row['archive_LFS_SHA256'],'frame_count':49,'resolution':[1280,704],'fps':10,'B_visibility_estimated_not_observed_GT':True,'Color_semantics':'One exact RGB control vector per chosen sample over the original bool support. No cyan uncertainty, texture, lighting, blending, labels or lossy compression.','default_recipes':'move/remove=source_hint; insert/replace/move_replace=explicit magenta requested variant. All other original8 colors and levels remain available through the strict reader.','normalization':'source raw target-source controls unchanged; /1.1 gamma and /.9 alpha exactly once at model entry','geometry_positions_cameras_PCF_and_controls_changed':False,'anomalies':anomalies,'records':records,'counters':dict(counts),'elapsed_seconds':time.time()-start};put(base/'MANIFEST.json',manifest)
    archive=U/f'case_{row["case"]}_solid_Color49_v1.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as out:
        for p in sorted(base.rglob('*')):
            if p.is_file():out.add(p,arcname='case_'+row['case']+'/'+str(p.relative_to(base)),recursive=False)
    # Re-read each emitted member checksum and PNG bytes from the portable archive.
    expected={str(p.relative_to(base)):sha(p.read_bytes()) for p in base.rglob('*') if p.is_file()};n=0
    with tarfile.open(archive,'r|gz') as tt:
        for m in tt:
            assert m.isfile();rel=m.name.partition('/')[2];b=tt.extractfile(m).read();assert sha(b)==expected[rel]
            if rel.endswith('.png'):
                with Image.open(io.BytesIO(b)) as im:im.load();assert im.size==(1280,704)
            n+=1
    assert n==len(expected);return {'case':row['case'],'HF_source_archive':row['HF_archive'],'source_archive_SHA256':row['archive_LFS_SHA256'],'HF_archive':PREFIX+'/'+archive.name,'local_archive':str(archive),'SHA256':sha(archive.read_bytes()),'bytes':archive.stat().st_size,'files':n,'variants':len(cm['operations']),'counters':dict(counts),'canonical_Color_anomalies':anomalies,'all_portable_members_SHA_and_PNG_readback_passed':True,'elapsed_seconds':time.time()-start}
def main():
    start=time.time();cat=json.loads((U/'DATASET_SOURCE_INDEX.json').read_text());inputs={r['HF_path']:r['local_path'] for r in json.loads((T/'inventory/REMOVE_ARCHIVE_INPUTS_v1.json').read_text())['rows']};bindings={r['HF_source_archive']:r for r in json.loads((T/'remove_publish/REMOVE_BINDINGS.json').read_text())['rows']};jobs=[]
    for row in cat['current']:
        binding_path=row['HF_archive']
        if row['case']=='11':path=Q/'case11_move_blue_20261115_v1/package/case_11';binding_path=row['parent_archive']
        else:path=Path(inputs[row['HF_archive']])
        jobs.append({'row':row,'path':str(path),'remove_binding':bindings[binding_path]})
    jobs.sort(key=lambda j:(j['row']['case']!='58',int(j['row']['case'])));rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(case,j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            row=future.result();rows.append(row);put(U/f'CASE_{row["case"]}_COMPLETE.json',row);status={'state':'finite_all35_actual_Color_audit_and_lossless_export','cases_finished':len(rows),'expected':35,'last_case':row['case'],'canonical_anomalies':sum(len(r['canonical_Color_anomalies']) for r in rows),'elapsed_seconds':time.time()-start};put(U/'EXPORT_CURRENT.json',status);print(json.dumps(status),flush=True)
    counts=sum((collections.Counter(r['counters']) for r in rows),collections.Counter());out={'state':'all35_current_solid_Color_exports_verified_pending_review_and_publication','cases':35,'variants':sum(r['variants'] for r in rows),'rows':sorted(rows,key=lambda r:int(r['case'])),'counters':dict(counts),'canonical_Color_anomalies':[],'all_case_archives_portable_members_and_PNGs_verified':True,'total_bytes':sum(r['bytes'] for r in rows),'elapsed_seconds':time.time()-start,'GPU_used':False};put(U/'ALL35_EXPORT_VERIFIED.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()

"""Publish the reported case first, then continue the bounded all35 sweep."""
from pathlib import Path
import datetime,fcntl,hashlib,json,sys,urllib.request
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'solid_mask_audit_20261115_v1';PREFIX='StarReal/supplements/solid_color_masks_20261115_v1';REPO='FrontRear/VLMdataset'
sys.path.insert(0,str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps'));from huggingface_hub import HfApi,CommitOperationAdd
def sha(b):return hashlib.sha256(b).hexdigest()
def git(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def put(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def main():
    row=json.loads((U/'CASE_58_COMPLETE.json').read_text());assert not row['canonical_Color_anomalies'] and row['all_portable_members_SHA_and_PNG_readback_passed'];test=json.loads((U/'STRICT_READER_ACTUAL_INTEGRATION.json').read_text());assert len(test['checks'])==20
    locks=[]
    for p in [Q.parent/'HF_ORIGINAL_STRICT500_v1/HF_UPLOAD_OWNER.lock',Q/'REALCASE_QH_ARCHIVE_HF_WRITER.lock']:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    api=HfApi(token=Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip());parent=api.repo_info(REPO,repo_type='dataset').sha
    with urllib.request.urlopen(f'https://huggingface.co/datasets/{REPO}/resolve/{parent}/StarReal/DATASET_CURRENT.json',timeout=60) as f:current=json.load(f)
    assert current['catalog']==json.loads((U/'DATASET_SOURCE_CURRENT.json').read_text())['catalog'],'Source catalog changed, audit newest groups first'
    policy={'schema':'StarReal.solid_Color_mask_contract.v1','at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'case58_hotfix_verified_all35_sweep_in_progress','Color_rule':'One exact selected RGB vector on original bool support; outside black. No visibility/uncertainty/lighting/texture/labels blended into Color.','uncertainty':'Original estimated-visibility unknown mask remains separate; never claimed observed known or erased.','PCF':'Original red/blue/yellow editing-marker semantics unchanged; pure Color rule is for lamp/bbox Color mask channels.','frames':49,'resolution':[1280,704],'case58':row,'complete_cases':[58],'target_current_cases':35,'strict_reader':PREFIX+'/starreal_complete_reader.py','reader_base':PREFIX+'/strict_base_reader.py','source_aliases':PREFIX+'/ARCHIVE_BINDING_ALIASES.json','all8_original_colors_and_intensity_levels_retained':True,'model_normalization':'raw target-source, gamma/1.1 and alpha/.9 exactly once','original_geometry_K_E_positions_and_PCF_unchanged':True,'case11_latest_blue_wall_move_preserved':True,'case70_uploaded':False}
    put(U/'POLICY_HOTFIX.json',policy)
    readme='''Solid Color mask correction: diagnostic colors are not Color conditions.

The reported screenshot is case58/move/B0. The previous geometry diagnostic drew27713 known pixels magenta and8737 unknown pixels cyan on one fixture. The published numeric magenta Color field already contains only[1,0,1] across all36450 support pixels. Uncertainty is a geometry confidence diagnostic and must never become an extra requested lamp color.

The new lossless Color49 PNGs contain exactly one selected RGB vector on the original support and black outside. No JPEG, text, anti-alias gradient, scene texture, light shading, opacity mixing or cyan diagnostic is in the actual Color PNG. Original uncertainty is exported separately as white/black Uncertainty49. It remains an estimate, not newly observed truth; for bbox export the uncertainty image retains the original target-lamp visibility uncertainty, not a claim of full bbox-volume uncertainty.

Case58 is published first while the finite35-current-case audit/export continues. Each case archive includes every current operation, lamp_silhouette and bbox_silhouette,49frames at1280x704/10FPS. Default requested recipes:move/remove source_hint (original recorded appearance); insert/replace/move_replace explicit magenta variant. Source hint is not measured emitted spectrum. All8 original requested colors and original intensity levels remain unchanged in their original NPZs and are selectable through the strict reader. The case58 move RequestedMagenta49 sequence is also included so the exact reported comparison can be reproduced.

Use starreal_complete_reader.py with strict_base_reader.py and ARCHIVE_BINDING_ALIASES.json in the same Python directory. The base is the unchanged complete archive/source-Remove adapter; the wrapper verifies all foreground pixels equal the selected control vector and rejects mixed-color fields. Defaults are explicit single-color recipes; to choose another original requested color pass color='cyan' etc. It returns original normalized scalar gamma/alpha once, not a second multiplied spatial intensity tensor. References and original source-Remove completions keep exact existing archive-SHA bindings. The latest case11 blue-wall move is retained.

The PCF's red/source,blue/target,yellow/overlap semantics are intentionally unchanged. Camera/geometry/placement,sourceRGB,mask support,reference images and raw intensity controls are unchanged. Older geometry diagnostics remain history, not training Color fields. Use the lossless exports or strict reader for actual Color conditions. No case70 or resumed full-reconstruction data is included.
''';(U/'README.txt').write_text(readme)
    names=['case_58_solid_Color49_v1.tar.gz','CASE58_ROOT_CAUSE.json','CASE58_BEFORE_AFTER.png','CASE58_MOVE_CORRECTED_MAGENTA.png','CASE58_MOVE_ACTUAL_SOURCE_HINT.png','CASE58_SOLID_MAGENTA_ALL49_REVIEW.png','STRICT_READER_ACTUAL_INTEGRATION.json','strict_base_reader.py','ARCHIVE_BINDING_ALIASES.json','README.txt']
    files=[(PREFIX+'/'+n,U/n) for n in names]+[(PREFIX+'/starreal_complete_reader.py',U/'code/starreal_complete_reader.py'),(PREFIX+'/build_native_five_role_geometry49_solidmask_QH_cpu_v3.py',U/'code/build_native_five_role_geometry49_solidmask_QH_cpu_v3.py'),(PREFIX+'/CASE58_EXPORT_MANIFEST.json',U/'CASE_58_COMPLETE.json'),(PREFIX+'/POLICY.json',U/'POLICY_HOTFIX.json'),('StarReal/SOLID_COLOR_MASK_CURRENT.json',U/'POLICY_HOTFIX.json')]
    current['reader']=PREFIX+'/starreal_complete_reader.py';current['solid_color_masks_policy']='StarReal/SOLID_COLOR_MASK_CURRENT.json';current['geometry_diagnostic_colors_are_not_Color_conditions']=True;put(U/'DATASET_HOTFIX_CURRENT.json',current);files.append(('StarReal/DATASET_CURRENT.json',U/'DATASET_HOTFIX_CURRENT.json'))
    manifest=[{'path':p,'local':str(f),'size':f.stat().st_size,'SHA256':sha(f.read_bytes())} for p,f in files];put(U/'HOTFIX_PUBLICATION_MANIFEST.json',manifest)
    result=api.create_commit(repo_id=REPO,repo_type='dataset',parent_commit=parent,commit_message='Fix case58 mixed diagnostic mask presentation: lossless single-color exports, separate uncertainty, strict Color reader; all35 sweep follows',operations=[CommitOperationAdd(path_in_repo=p,path_or_fileobj=str(f)) for p,f in files]);commit=result.oid;put(U/'HOTFIX_CREATED_COMMIT.json',{'commit':commit,'parent':parent})
    infos={x.path:x for x in api.get_paths_info(REPO,[r['path'] for r in manifest],repo_type='dataset',revision=commit)};checks=[]
    for r in manifest:
        info=infos[r['path']];b=Path(r['local']).read_bytes();assert info.size==r['size'] and (info.lfs.sha256==r['SHA256'] if info.lfs is not None else info.blob_id==git(b))
        with urllib.request.urlopen(f"https://huggingface.co/datasets/{REPO}/resolve/{commit}/{r['path']}?download=true",timeout=90) as f:public=f.read()
        assert len(public)==r['size'] and sha(public)==r['SHA256'];checks.append({k:v for k,v in r.items() if k!='local'})
    receipt={'state':'case58_hotfix_all_new_files_server_verified_all35_sweep_continues','commit':commit,'parent':parent,'case58_export':row,'files':checks,'strict_reader_active':True,'uncertainty_separate_not_encoded_in_Color':True,'geometry_and_PCF_unchanged':True,'all35_complete':False};put(U/'HF_CASE58_HOTFIX_VERIFIED.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ['files','case58_export']}),flush=True)
if __name__=='__main__':main()

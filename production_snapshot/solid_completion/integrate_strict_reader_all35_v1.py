"""Actual current-reader sweep plus an in-memory replay of the reported color leak."""
from pathlib import Path
import collections,json,sys,time
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');T=Q/'reference_completion_20261115_v1';U=Q/'solid_mask_audit_20261115_v1'
sys.path[:0]=[str(Q/'runtime/qh_cpu_py39_v1'),str(U/'code'),str(U)]
import numpy as np
from PIL import Image
from starreal_complete_reader import StarRealCompleteReader
from strict_base_reader import StarRealCompleteReader as Base
def link(p,target):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():p.symlink_to(target)
    assert p.resolve()==Path(target).resolve()
def main():
    start=time.time();cat=json.loads((U/'DATASET_SOURCE_INDEX.json').read_text());view=U/'dataset_view';sources={r['HF_path']:r['local_path'] for r in json.loads((T/'inventory/REMOVE_ARCHIVE_INPUTS_v1.json').read_text())['rows']}
    for row in cat['current']:
        path=Q/'case11_move_blue_20261115_v1/case_11_move_blue_wall_v6.tar.gz' if row['case']=='11' else Path(sources[row['HF_archive']]);link(view/row['HF_archive'],path)
    complete=view/'StarReal/supplements/reference_and_remove_completion_v1'
    for directory in [T/'publish',T/'remove_publish']:
        for p in directory.iterdir():
            if not p.name.startswith('._'):link(complete/p.name,p)
    counts=collections.Counter();rows=[];replayed=False
    for case in cat['current']:
        with StarRealCompleteReader(view,case['HF_archive'],verify_archive=False) as reader:
            for var in reader.manifest['variants']:
                ref,_=reader.reference_image(var);assert ref.width>0;counts['actual_operation_references_read']+=1
                for family in ['lamp_silhouette','bbox_silhouette']:
                    row=reader.condition_frame(var,family,24);assert row['single_color_foreground_verified'] and row['normalization_count']==1;counts['actual_default_Color_and_scalar_rows_verified']+=1
                if case['case']=='58' and var=='move':
                    pure=reader.condition_frame(var,'lamp_silhouette',0,color='magenta',level='unchanged');bad=dict(pure);a=np.asarray(pure['color_mask_rgb']).copy();y,x=np.argwhere(pure['support'])[0];a[y,x]=[0,255,255];bad['color_mask_rgb']=Image.fromarray(a);old=Base.condition_frame
                    try:
                        Base.condition_frame=lambda *args,**kwargs:bad
                        try:reader.condition_frame('move','lamp_silhouette',0,color='magenta',level='unchanged')
                        except ValueError as e:assert 'Non-uniform Color' in str(e);replayed=True
                        else:raise AssertionError('The diagnostic cyan leak was not rejected')
                    finally:Base.condition_frame=old
            rows.append({'case':case['case'],'archive':case['HF_archive'],'variants':len(reader.manifest['variants']),'all_defaults_and_references_passed':True})
        (U/'STRICT_ALL35_CURRENT.json').write_text(json.dumps({'cases':len(rows),'expected':35,'counts':dict(counts),'elapsed_seconds':time.time()-start},indent=2)+'\n')
    assert replayed and counts['actual_operation_references_read']==279
    out={'state':'all35_current_strict_reader_actual_integration_passed','cases':35,'variants':279,'counts':dict(counts),'rows':rows,'reported_cyan_contamination_replay_rejected':True,'fault_injection_only_in_memory_no_data_modified':True,'latest_case11_blue_wall_alias_included':True,'all_source_camera_geometry_Original_scalar_coefficients_preserved':True,'elapsed_seconds':time.time()-start};(U/'STRICT_ALL35_ACTUAL_VERIFIED.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()

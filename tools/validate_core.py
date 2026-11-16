"""Meaningful CPU checks against frozen math and optional actual production pixels."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import cv2
from starreal.core import render_pair,depth_renderer,_load,export_control_fields
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def full_support(z):
    h,w=map(int,z['full_shape_hw']);x,y=map(int,z['origin_xy']);m=z['support'].astype(bool)
    a=np.zeros((h,w),bool);a[y:y+m.shape[0],x:x+m.shape[1]]=m;return a
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--production-root',type=Path)
    ap.add_argument('--report',type=Path,required=True);args=ap.parse_args()
    start=time.time();cv2.setNumThreads(1)
    ref=depth_renderer(False);sys.modules['depth_fusion']=ref
    accel=_load('accel','render/pcf_cpu_accelerator.py')
    parity=accel.verify()
    report=dict(state='PASS',GPU_used=False,CPU_reference_parity=parity,
                production_case_component_test=None,no_new_quality_acceptance=True)
    if args.production_root:
        R=args.production_root;geo=R/'outputs/83/geometry_masks49_cached_QH_v1';frame=24;variant='replace_orig_7095c535'
        with np.load(geo/'cameras49.npz',allow_pickle=False) as z:cams={k:z[k] for k in z.files}
        with np.load(geo/variant/'fixed_world_geometry.npz',allow_pickle=False) as z:g={k:z[k] for k in z.files}
        boxes=[dict(center=g[key+'_center'],size=g[key+'_size'],rotation=g[key+'_rotation'],label=label)
               for key,label in [('source',1),('target',2)]]
        old,new,aux=render_pair(cams['K'][frame],cams['w2c_B'][frame],boxes,
                               first_w2c_A=cams['w2c_A'][0],width=1280,height=704,aa=2)
        proof=[]
        for leaf,actual in [('old_joint',old),('new_joint',new)]:
            p=R/'outputs/83/two_pcf49_fixed_world_yellow_QH_v1'/variant/leaf/f'{frame:06d}.png'
            expected=cv2.imread(str(p))[:,:,::-1]
            assert expected.shape==(704,1280,3)
            np.testing.assert_array_equal(actual,expected)
            proof.append(dict(role=leaf,production_PNG_SHA256=digest(p),differing_pixels=0))
        front=np.isfinite(aux['box_depth_m'])&aux['pcf_visible']
        back=aux['box_visible']&np.isfinite(aux['pcf_depth_m'])
        assert front.any() and back.any() and not aux['unresolved'].any()
        overlap=(aux['labels']==3).reshape(704,2,1280,2).all(axis=(1,3))
        assert overlap.any()
        assert np.all(new[overlap,0]==new[overlap,1]) and np.all(new[overlap,2]==0)
        supports={};original={}
        for family in ['lamp_silhouette','bbox_silhouette']:
            p=geo/'move'/'conditions'/family/f'{frame:06d}.npz'
            with np.load(p,allow_pickle=False) as z:original[family]={k:z[k] for k in z.files}
            supports[family]=full_support(original[family])
        hint=json.loads((R/'outputs/83/source_appearance_control_reference.json').read_text())['control_color_linear_rgb']
        compared=[]
        with tempfile.TemporaryDirectory(prefix='starreal_code_fields_') as tmp:
            dest=Path(tmp)
            export_control_fields(supports,dest/'move','move',frame,hint)
            for family in supports:
                with np.load(dest/'move'/family/f'{frame:06d}.npz',allow_pickle=False) as z:
                    count=0
                    for key in z.files:
                        if key in original[family] and key!='source_color_linear_rgb':
                            np.testing.assert_array_equal(z[key],original[family][key]);count+=1
                    compared.append(dict(family=family,production_matching_arrays=count,source_support_pixels=int(supports[family].sum())))
            export_control_fields(supports,dest/'remove','remove',frame,hint)
            for family in supports:
                with np.load(dest/'remove'/family/f'{frame:06d}.npz',allow_pickle=False) as z:
                    m=z['support'];assert m.any()
                    np.testing.assert_array_equal(z['raw_gamma_removed'],m.astype(np.float32)*np.float32(-.5))
                    np.testing.assert_array_equal(z['color_source_hint'],m[...,None].astype(np.float32)*np.asarray(hint,np.float32))
                    np.testing.assert_allclose(z['raw_gamma_removed'][m]/np.float32(1.1),-.5/1.1,atol=1e-7)
                    assert np.unique(z['color_source_hint'][m],axis=0).shape==(1,3)
            for family in supports:supports[family][:]=False
            try:export_control_fields(supports,dest/'invalid_remove','remove',frame,hint)
            except ValueError:pass
            else:raise AssertionError('Empty source Remove incorrectly allowed')
        report['production_case_component_test']=dict(case='83',frame=frame,width=1280,height=704,AA=2,
             variant=variant,PCF_pixels=proof,fields_variant="move",fields=compared,
             lattice_in_front_subrays=int(front.sum()),box_in_front_subrays=int(back.sum()),
             full_yellow_pixels=int(overlap.sum()),yellow_R_equals_G_B0=True,
             Remove_negative_raw_gamma_and_source_solid_Color_checked=True,
             world_geometry_and_camera_inputs_changed=False)
    report.update(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.time()-start)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

"""Explicit approximate source-color references for pure-move intent.

These are appearance controls, never measured lamp emission or absolute power.
Observed RGB and the eight requested recolor vectors remain unchanged.
"""
import hashlib,os
import json
from pathlib import Path

import sys
sys.path.insert(0,'/tmp/starreal_uid1007_case19_cpu_v1/runtime/qh_cpu_py39_v1')
import cv2
import numpy as np

import sys
sys.path.insert(0,'/tmp/starreal_uid1007_case19_cpu_v1/runtime/qh_cpu_py39_v1')
ROOT=Path('/tmp/starreal_uid1007_case19_cpu_v1')


def main():
    records=[]
    CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal() and CASE!='1'
    for case in [CASE]:
        base=ROOT/'outputs'/case
        manifest=json.loads((base/'geometry_masks49_cached_QH_v1/MANIFEST.json').read_text())
        frames=ROOT/f'outputs/{case}/five_role_geometry49_QH_v1/observed_RGB49_BGR.npy';RGB=np.load(frames,mmap_mode='r')
        maskdir=(Path('/root/autodl-tmp/codex_user8_move_replace_20261109/sam_visible_refined_v3/1/mask')
                 if case=='1' else base/'geometry_masks49_cached_QH_v1/source_observed_A')
        values=[]
        # Same declared brightest20% appearance-hint convention as historical
        # step47, now using the actually retained observed SAM for this revision.
        for i in range(49):
            rgb=RGB[i][:,:,::-1].astype(np.float64)/255
            mask=cv2.imread(str(maskdir/f'{i:06d}.png'),0)>0
            pix=rgb[mask];assert len(pix)
            linear=np.where(pix<=.04045,pix/12.92,((pix+.055)/1.055)**2.4)
            luminance=linear@np.array([.2126,.7152,.0722])
            selected=linear[luminance>=np.quantile(luminance,.8)]
            values.append(selected.mean(0))
        color=np.median(values,axis=0);color/=max(float(color.max()),1e-12)
        assert np.isfinite(color).all() and color.max()>0
        result={'case':case,'case_id':manifest['case_id'],'control_color_linear_rgb':color.tolist(),
            'source':'all49 actual observed RGB and retained SAM, brightest20% per frame then temporal median and max normalization',
            'source_RGB':str(frames),'source_mask':str(maskdir),'source_emission_is_measured':False,
            'limitations':'surface reflectance, clipping, auto exposure and indirect light may affect the appearance hint; unknown spectral emission and absolute intensity',
            'pure_move_recipe':{'variant':'move','color':'source_hint','level':'unchanged',
                'source_gamma_control':.5,'target_gamma_control':.5,'raw_gamma':0.,'model_relative_gamma':0.,
                'appearance_intent':'preserve source appearance; no requested recolor or brightness change'},
            'recolor_recipes':'eight explicit requested colors remain available separately',
            'unchanged_observed_RGB':True,'GPU_used':False,'quality_accepted':False}
        target=base/'source_appearance_control_reference.json';assert not target.exists()
        target.write_text(json.dumps(result,indent=2));records.append(result)
    (ROOT/f'logs/CASE{CASE}_FAST5_SOURCE_APPEARANCE_REFERENCES.json').write_text(json.dumps({'records':records},indent=2))
    print(json.dumps({'records':records}),flush=True)


if __name__=='__main__':main()

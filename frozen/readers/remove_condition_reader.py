"""Read versioned source-lamp Remove Color/Intensity without changing old packages."""
from pathlib import Path
import json,numpy as np

def load_remove_fields(root,frame,family='lamp_silhouette',view='B'):
    root=Path(root)
    if family not in ('lamp_silhouette','bbox_silhouette') or view not in ('A','B') or not 0<=frame<49:raise ValueError('Explicit family/view/frame required')
    base=root/('A_observed' if view=='A' else '')/family
    with np.load(base/'Intensity49'/f'{frame:06d}.npz',allow_pickle=False) as z:
        support=z['support'].astype(bool);x,y=map(int,z['origin_xy']);H,W=map(int,z['full_shape_hw']);raw=z['raw_gamma'].astype(np.float32);color=z['color_linear_rgb'].astype(np.float32);source=float(z['source_gamma_control']);target=float(z['target_gamma_control'])
    if (H,W)!=(704,1280) or source<=target or target!=0:raise ValueError('Expected unchanged704 source-lamp removal controls')
    if raw.shape!=support.shape or color.shape!=(*support.shape,3):raise ValueError('Malformed field shape')
    np.testing.assert_allclose(raw[support],target-source,atol=1e-7)
    mask=np.zeros((H,W),bool);mask[y:y+support.shape[0],x:x+support.shape[1]]=support
    colors=np.zeros((H,W,3),np.float32);colors[y:y+support.shape[0],x:x+support.shape[1]]=color
    gamma=np.full((H,W),-1.,np.float32);patch=gamma[y:y+support.shape[0],x:x+support.shape[1]];patch[support]=raw[support]/np.float32(1.1)
    return dict(color_linear_rgb=colors,relative_gamma=gamma,support=mask,source_gamma_control=source,target_gamma_control=target,view='observed_A' if view=='A' else 'estimated_B',normalization_count=1,operation='remove',target_present=False,source_present=True)

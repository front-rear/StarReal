"""Single-reference, bbox-aligned conditions. Independent of legacy six-view loaders."""
from pathlib import Path
from functools import lru_cache
import json
import numpy as np
from PIL import Image

SCHEMA = 'LampEdit.bbox_conditions.v003'

def resolve(value, legacy_root, upgrade_root):
    if value.startswith('bbox_v3/'):
        return Path(upgrade_root) / value[len('bbox_v3/'):]
    if value.startswith('data_v2/'):
        return Path(legacy_root) / value[len('data_v2/'):]
    raise ValueError('Only versioned portable paths are runtime inputs: ' + value)

def frames(path, gray=False):
    paths=sorted(Path(path).glob('*.png'), key=lambda p:int(p.stem))
    assert len(paths)==49 and [int(p.stem) for p in paths]==list(range(49))
    result=[]
    for p in paths:
        with Image.open(p) as im:
            im=im.convert('L' if gray else 'RGB')
            if im.size!=(640,352):
                assert not gray, 'Never resize bbox labels at load time'
                im=im.resize((640,352),Image.Resampling.LANCZOS)
            result.append(np.asarray(im).copy())
    out=np.stack(result)
    return out[...,None].astype(np.float32)/255 if gray else out

def field_frames(path, kind):
    paths=sorted(Path(path).glob('*.npz'),key=lambda p:int(p.stem));assert len(paths)==49
    result=[]
    for p in paths:
        with np.load(p,allow_pickle=False) as z:
            roi=z[kind];origin=z['origin_xy'];shape=z['full_shape_hwc']
            assert roi.dtype==np.float32 and tuple(shape)==(352,640,3)
            assert np.isfinite(roi).all() and (roi>=0).all() and (roi<=1).all()
            x,y=map(int,origin);h,w=roi.shape[:2]
            assert roi.shape==(h,w,3) and 0<=x<=640-w and 0<=y<=352-h
            full=np.zeros((352,640,3),np.float32);full[y:y+h,x:x+w]=roi
            result.append(full)
    return np.stack(result)

@lru_cache(maxsize=2)
def background_frames(path):
    value=frames(path);value.setflags(write=False)
    return value

def joint_frames(folder,legacy_root,upgrade_root):
    recipe=json.loads((Path(folder)/'RENDER.json').read_text())
    background=background_frames(str(resolve(recipe['background'],legacy_root,upgrade_root)))
    labels=np.stack([np.asarray(Image.open(p)) for p in sorted(resolve(recipe['labels'],legacy_root,upgrade_root).glob('*.png'))])
    sw=frames(resolve(recipe['source_wire'],legacy_root,upgrade_root),True)[...,0]>0
    tw=frames(resolve(recipe['target_wire'],legacy_root,upgrade_root),True)[...,0]>0
    palette=np.asarray(recipe['palette_rgb'],np.uint8);alpha=recipe['fill_alpha']
    assert labels.shape==background.shape[:3]
    result=background.copy();occupied=labels>0
    result[occupied]=np.rint(recipe['background_alpha']*background[occupied]+alpha*palette[labels[occupied]]).astype(np.uint8)
    wire=sw|tw;result[wire]=palette[labels[wire]]
    return result

def load_row(row, legacy_root, upgrade_root):
    assert row['condition_schema']==SCHEMA and row['frame_count']==49
    path=lambda name:resolve(row[name],legacy_root,upgrade_root)
    source=frames(path('source'));target=frames(path('target'))
    joint=joint_frames(path('joint'),legacy_root,upgrade_root)
    color=field_frames(path('color'),'color');intensity=field_frames(path('intensity'),'intensity')
    sm=frames(path('source_bbox_in_target_view'),True)
    tm=frames(path('target_bbox_in_target_view'),True)
    label_paths=sorted(path('operation_labels').glob('*.png'))
    assert len(label_paths)==49
    labels=np.stack([np.asarray(Image.open(p)) for p in label_paths])
    assert np.array_equal(((labels==1)|(labels==3)),sm[...,0]>0)
    assert np.array_equal(((labels==2)|(labels==3)),tm[...,0]>0)
    if row['target_light_active']:
        assert np.array_equal(np.any(color>0,axis=-1),tm[...,0]>0)
        assert np.array_equal(np.any(intensity>0,axis=-1),tm[...,0]>0)
    else:
        assert not color.any() and not intensity.any()
    expected=1 if row['target_fixture_present'] else 0
    assert len(row['reference_images'])==expected
    refs=[]
    for p in row['reference_images']:
        with Image.open(resolve(p,legacy_root,upgrade_root)) as im:
            refs.append(np.asarray(im.convert('RGB').resize((512,512),Image.Resampling.LANCZOS)))
    reference=(np.stack(refs).astype(np.float32)/255. if refs else np.empty((0,512,512,3),np.float32))
    source_mask=frames(path('source_mask_coarse'),True)
    source_in_frame=np.any(source_mask>0,axis=(1,2,3))
    target_in_frame=np.any(tm>0,axis=(1,2,3))
    ambient=float(row['ambient'])
    assert np.isfinite(ambient)
    metadata={k:row[k] for k in ('sample_id','operation','joint_edit_operation','edit_operations',
              'camera_relation','source_view','target_view','source_entity','target_entity',
              'source_fixture_present','source_light_active','source_ambient','target_ambient',
              'reference_asset_id','reference_v2_identity','reference_condition_required',
              'reference_image_count','reference_sha256') if k in row}
    return {**metadata,'source':source,'target':target,'joint':joint,'color':color,'intensity':intensity,
            'source_mask_video':source_mask,'ambient':ambient,
            'target_mask_video':tm,'source_bbox_in_target_view':sm,'operation_labels':labels,
            'fixture_reference_images':reference,'prompt':row['prompt'],
            'target_fixture_present':row['target_fixture_present'],
            'target_light_active':row['target_light_active'],
            'target_mask_in_frame':target_in_frame,'source_mask_in_frame':source_in_frame,
            'target_mask_black_frames':np.flatnonzero(~target_in_frame).tolist(),
            'source_mask_black_frames':np.flatnonzero(~source_in_frame).tolist(),
            'source_physical_light_metadata_json':json.dumps(row.get('source_physical_light_metadata',{})),
            'reference_value_range':'float32_RGB_0_1',
            'target_physical_light_metadata_json':json.dumps(row['target_physical_light_metadata']),
            'condition_schema':SCHEMA,'cache_key':row['conditioning_cache_key']}

class BBoxConditionDataset:
    def __init__(self,manifest,legacy_root,upgrade_root):
        self.rows=json.loads(Path(manifest).read_text());self.legacy_root=legacy_root;self.upgrade_root=upgrade_root
        assert self.rows and all(r['condition_schema']==SCHEMA for r in self.rows)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return load_row(self.rows[index],self.legacy_root,self.upgrade_root)

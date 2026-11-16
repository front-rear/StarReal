import json,cv2,numpy as np
from pathlib import Path
ROOT=Path('/root/autodl-tmp/kimi_realcase_20261109');LEG=Path('/root/autodl-tmp/real_video_condition_repro_20261105_v1/cases');S=Path('/root/autodl-tmp/codex_aligned_v2/samples')
rows=[]
for cfg in json.loads((ROOT/'master_prepare.json').read_text()):
 case=cfg['case_id'];old=case.removeprefix('rv_fresh_');dest=ROOT/'samples'/case/'step3_worldpoints';source=S/old/'step3_worldpoints/vggt_raw.npz';record={'case':case,'legacy_case':old,'cache':'not_reused'}
 if dest.exists():record.update(cache='already_present');rows.append(record);continue
 if not source.exists() or not (LEG/old/'frames').is_dir():rows.append(record);continue
 same=True
 for i in range(49):
  a=cv2.imread(str(ROOT/'cases'/case/'frames'/f'{i:03}.png'));b=cv2.imread(str(LEG/old/'frames'/f'{i:03}.png'))
  if a is None or b is None or a.shape!=b.shape or not np.array_equal(a,b):same=False;record['first_pixel_difference_frame']=i;break
 if not same:rows.append(record);continue
 with np.load(source) as d:
  assert d['world_points'].shape[0]==49 and d['extrinsic_3x4'].shape==(49,3,4) and d['K'].shape==(49,3,3)
  assert np.isfinite(d['world_points']).all() and np.isfinite(d['extrinsic_3x4']).all() and np.isfinite(d['K']).all()
 dest.mkdir(parents=True);(dest/'vggt_raw.npz').symlink_to(source)
 record.update(cache='reused_all_49_frames_pixel_identical',source=str(source));rows.append(record);print(record,flush=True)
(ROOT/'master_vggt_cache_reuse.json').write_text(json.dumps(rows,indent=2))
print({'reused':sum(r['cache'].startswith('reused') for r in rows),'already':sum(r['cache']=='already_present' for r in rows),'total':len(rows)})

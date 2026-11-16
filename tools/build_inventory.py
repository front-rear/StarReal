"""Locate AI code references without executing any production script."""
from pathlib import Path
import datetime,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
patterns={
 'VGGT':r'(?i)\bVGGT\b|world_points_conf',
 'SAM2':r'build_sam2|SAM2ImagePredictor|SAM2VideoPredictor|propagate_in_video|sam2\.build_sam',
 'DA3':r'DepthAnything3|depth_anything_3|DA3-SMALL',
 'model_runtime':r'torch\.inference_mode|torch\.no_grad|model\(imgs|predictor\.predict',
}
rows=[]
for origin in json.loads((ROOT/'provenance/SOURCE_ORIGINS.json').read_text())['files']:
 p=ROOT/origin['path']
 if p.suffix not in ('.py','.cpp','.h'):continue
 lines=p.read_text().splitlines();matches=[]
 for model,pattern in patterns.items():
  hits=[i+1 for i,l in enumerate(lines) if re.search(pattern,l)]
  if hits:matches.append(dict(reference=model,lines=hits,note='Static reference only; may be cached-data IO, probe or historical path, not proof of executed inference'))
 name=p.name.lower()
 if 'original' in origin['path'].lower() or 'unmodified' in name:role='Original or legacy reference; provenance SHA authoritative'
 elif 'upstream_native_producers' in origin['path']:role='Historical upstream actual model-entry source; not universal current case recipe'
 elif any(w in name for w in ('probe','pilot','annotation')):role='Case-specific candidate/probe/visible annotation; individual review determines use'
 elif any(w in name for w in ('publish','upload','release','verify','review','stage','reader','audit')):role='Deterministic packaging/readback/review support; final visual judgment remains outside automation'
 else:role='Geometry/render/IO orchestration; inspect imports for runtime model use'
 rows.append(dict(path=origin['path'],SHA256=origin['SHA256'],role=role,AI_references=matches,
                  authorship='AI-assisted project work where evidenced; original author/exact generated lines not inferable from static source alone'))
out=dict(schema='StarReal.AI_code_inventory.v1',generated=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         interpretation='Separate model inference, AI-assisted code authorship and external AI visual judgment. File references are not execution evidence.',
         files=rows,external_AI_judgment=['case RGB semantic placement/support selection','asset diversity/type/size/pose selection','SAM prompt/visible-part annotation review','all49 geometry/mask/PCF visual review before release'],
         no_automatic_quality_acceptance=True)
(ROOT/'provenance/AI_CODE_INVENTORY.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(dict(inventoried_code_files=len(rows),files_with_model_references=sum(bool(x['AI_references']) for x in rows))))

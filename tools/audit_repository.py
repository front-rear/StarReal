"""Read-only source audit. No model inference, data download, approval or upload."""
from pathlib import Path
import ast,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    origins=json.loads((ROOT/'provenance/SOURCE_ORIGINS.json').read_text())
    checked=0
    for row in origins['files']:
        p=ROOT/row['path']
        assert p.is_file(),row['path']
        assert p.stat().st_size==row['bytes'] and sha(p)==row['SHA256'],row['path']
        checked+=1
    canonical=ROOT/'frozen/render/depth_fusion_fixed_world_yellow_overlap_v2.py'
    assert sha(canonical)=='a6c6c71bb8ce48f07c0e04e1fbfedd1af3d51ddf5a707f01de6c08de6c231f3f'
    mirror=0
    for p in (ROOT/'src/starreal/_frozen').rglob('*'):
        if p.is_file() and p.suffix in ('.py','.cpp','.h'):
            assert sha(p)==sha(ROOT/'frozen'/p.relative_to(ROOT/'src/starreal/_frozen')),str(p)
            mirror+=1
    forbidden_suffixes={'.mp4','.mkv','.glb','.gltf','.pt','.pth','.safetensors','.tar','.zip','.npz','.npy'}
    credential=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{32,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY)')
    scanned=0;syntax=0;historical=[]
    exclusions={x["path"]:x for x in json.loads((ROOT/"provenance/HISTORICAL_NONRUNNABLE.json").read_text())["files"]}
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT)
        if not p.is_file() or '.git' in rel.parts or '__pycache__' in rel.parts or 'build' in rel.parts or 'dist' in rel.parts or any(x.endswith('.egg-info') for x in rel.parts) or p.suffix in ('.so','.pyc','.dylib'):
            continue
        assert p.suffix not in forbidden_suffixes,str(rel)
        assert not p.name.startswith('._') and p.name not in {'hosts.yml','.env'},str(rel)
        b=p.read_bytes()
        assert not credential.search(b),'Credential-like content in '+str(rel)
        if p.suffix=='.py':
            try: ast.parse(b,filename=str(rel));syntax+=1
            except SyntaxError as e:
                row=exclusions.get(str(rel)); assert row and row["SHA256"]==sha(p) and row["line"]==e.lineno and str(rel).startswith("production_snapshot/"),str(rel)
                historical.append(str(rel))
        scanned+=1
    required=['README.md','docs/AI_USAGE.md','docs/PCF_AND_BBOX.md','docs/MASKS_AND_CONTROLS.md',
              'docs/PIPELINE_AND_PARAMETERS.md','docs/REPRODUCE.md','docs/FAILURES_AND_LIMITS.md',
              'provenance/AI_CODE_INVENTORY.json','provenance/MODELS_AND_EXTERNAL_DEPENDENCIES.json']
    assert all((ROOT/x).is_file() for x in required)
    for target in re.findall(r'\]\(([^)]+)\)',(ROOT/'README.md').read_text()):
        if '://' not in target:assert (ROOT/target.split('#')[0]).exists(),target
    result=dict(state='PASS',original_sources_SHA_verified=checked,package_mirrors_SHA_verified=mirror,
                scanned_text_files=scanned,Python_syntax_files=syntax,canonical_renderer_unchanged=True,historical_nonrunnable_candidates=historical,
                model_weights_media_tokens_included=False,GPU_used=False)
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':main()

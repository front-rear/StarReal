from pathlib import Path
import sys,io,json,hashlib,zipfile,fcntl
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');M=Q/'inputs/case83_three_fresh_whole_models_v1'
sys.path.insert(0,'/tmp/starreal_uid1007_case19_cpu_v1/runtime/qh_cpu_py39_v1')
from PIL import Image,ImageDraw
raw=sys.stdin.buffer.read(12*1024**2+1);assert len(raw)<12*1024**2;z=zipfile.ZipFile(io.BytesIO(raw));refs=json.loads(z.read('MATERIAL_REFERENCE_SOURCE_BINDINGS.json'));assert refs['case']=='83' and len(refs['records'])==3
used={x['asset_SHA256'] for x in json.loads((Q/'supplements/reference_and_remove_completion_v1/REFERENCE_BINDINGS.json').read_text())['assets']}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for r in refs['records']:
 assert r['asset_SHA256']==sha(M/'raw_assets'/('lamp_'+r['index']+'.glb')) and r['asset_SHA256'] not in used
 assert len(r['views'])==4
 for v in r['views']:
  b=z.read(v['path']);assert hashlib.sha256(b).hexdigest()==v['SHA256'] and len(b)==v['bytes']
  im=Image.open(io.BytesIO(b));im.load();assert im.size==(512,512)
for name in z.namelist():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 dest=M/p;assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
canvas=Image.new('RGB',(2048,1620),(235,235,235));draw=ImageDraw.Draw(canvas)
for row,r in enumerate(refs['records']):
 draw.text((5,row*540+5),r['role']+' '+r['index']+' exact source material references',fill=(0,0,0))
 for j,v in enumerate(r['views']):canvas.paste(Image.open(M/v['path']).convert('RGB'),(j*512,row*540+28))
canvas.save(M/'ALL12_EXACT_MATERIAL_REFERENCE_REVIEW.jpg',quality=95)
print(json.dumps(dict(case='83',reference_images=12,all_source_PNG_and_raw_SHA_verified=True,out=str(M),payload_bytes=len(raw),GPU_used=False,quality_accepted=False)))

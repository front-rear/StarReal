from pathlib import Path
import sys,json,concurrent.futures,time,os
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');U=Q/'all59_reconstruction_20261115_v1';sys.path.insert(0,str(Q.parent/'HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps'))
from huggingface_hub import HfApi
api=HfApi(token=Path('/home/fangrui/.cache/lampedit-vlmdataset-upload/token').read_text().strip());head='eea2594ed5653d79017e39bf862a104ed3183b78';cases=['6374204','7505473','10302763','100','101','103','7505429','7578553'];start=time.time()
def one(case):
 prefix='realcase/'+case;rows=[]
 try:
  for x in api.list_repo_tree('FrontRear/VLMdataset',path_in_repo=prefix,repo_type='dataset',revision=head,recursive=True):
   if not hasattr(x,'size'):continue
   rows.append(dict(path=x.path,size=x.size,LFS_SHA256=(x.lfs.sha256 if x.lfs else None),blob_id=x.blob_id))
  p=U/('CASE'+case+'_ACTUAL_HF_LAYOUT_20261116_v1.json');p.write_text(json.dumps(dict(case=case,HF_revision=head,files=rows),indent=2))
  key=[r for r in rows if r['path'].endswith(('.json','.py','.npz','.tar','.tar.gz','source_A.mkv')) and '/frames/' not in r['path']]
  return dict(case=case,files=len(rows),key_files=key,metadata_path=str(p))
 except Exception as e:return dict(case=case,error=repr(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(one,cases))
p=U/'REMAINING_ACTUAL_HF_LAYOUT_LOOKAHEAD_20261116_v1.json';p.write_text(json.dumps(dict(rows=rows,HF_revision=head,read_only=True,no_6000D_execution=True,GPU_used=False,seconds=time.time()-start),indent=2));print(json.dumps(dict(result=str(p),rows=[dict(case=x['case'],files=x.get('files'),error=x.get('error'),key_file_count=len(x.get('key_files',[]))) for x in rows],seconds=time.time()-start)))

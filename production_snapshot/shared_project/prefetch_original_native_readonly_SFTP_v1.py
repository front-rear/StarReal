"""Finite exact-file SFTP reads only; no 6000D command/code/compute or writes."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time
L=Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal')
a=argparse.ArgumentParser();a.add_argument('--cases',nargs='+',required=True);a.add_argument('--to-qh-bundle',action='store_true');args=a.parse_args();c=args.cases;assert 1<=len(c)<=2 and all(x.isdecimal() for x in c)
board=json.loads((L/'MASTER_WORKBOARD.json').read_text());root=Path('/private/tmp/StarReal_native_readonly_SFTP_'+'_'.join(c)+'_v1');root.mkdir(exist_ok=False);batch=[];spec=[]
for case in c:
 row=next(x for x in board['cases'] if str(x['case'])==case);assert not row.get('full_case_accepted',False);v=row['versions'][0];d=root/case;d.mkdir();(d/'original_frames49').mkdir();(d/'original_mask49').mkdir();base='/root/autodl-tmp/kimi_realcase_20261109';cid=v['case_id'];pairs=[(v['selected_local_source'],'source_A.mkv'),(v['version_root']+'/orbit_10deg/cameras_and_boxes.npz','cameras.npz'),(v['version_root']+'/geometry_plan.json','HISTORICAL_GEOMETRY_PLAN.json'),(v['version_root']+'/manifest.json','HISTORICAL_MANIFEST.json'),(base+'/samples/'+cid+'/step3_worldpoints/vggt_raw.npz','vggt_raw.npz'),('/root/autodl-tmp/StarReal/code/real_geometry.py','real_geometry.py'),('/root/autodl-tmp/StarReal/code/mesh_zbuffer.so','mesh_zbuffer.so')]
 for i in range(49):pairs += [(base+'/cases/'+cid+'/frames/'+f'{i:03d}.png','original_frames49/'+f'{i:06d}.png'),(base+'/samples/'+cid+'/step8_sam2_candidate/mask49/'+f'{i:06d}.png','original_mask49/'+f'{i:06d}.png')]
 for remote,local in pairs:
  assert '"' not in remote and '\n' not in remote;batch.append('get "'+remote+'" "'+str(d/local)+'"')
 spec.append(dict(case=case,version=v,read_only_original_paths=[dict(remote=r,local=z) for r,z in pairs]))
(root/'SPEC.json').write_text(json.dumps(dict(cases=spec,remote_execution=False,remote_writes=False,goal='Exact existing inputs to Qinghai CPU'),indent=2));p=root/'READ_ONLY_GET_BATCH.sftp';p.write_text('\n'.join(batch)+'\n');t=time.time()
with (root/'SFTP_READ_ONLY.log').open('x') as f:r=subprocess.run(['sftp','-oBatchMode=yes','-oConnectTimeout=15','-b',str(p),'6000D'],stdout=f,stderr=subprocess.STDOUT,timeout=1200)
receipt=dict(cases=c,root=str(root),sftp_exit_code=r.returncode,elapsed_seconds=time.time()-t,remote_execution=False,remote_writes=False,stage='files_copied_pending_QH_Native_array_and_RGB_verification' if r.returncode==0 else 'read_only_transfer_failed_preserve_partial_no_automatic_retry',quality_accepted=False)
(root/'SFTP_READ_ONLY_RESULT.json').write_text(json.dumps(receipt,indent=2));(L/'logs'/('ORIGINAL_NATIVE_SFTP_'+'_'.join(c)+'_v1.json')).write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True);assert r.returncode==0

if args.to_qh_bundle:
 subprocess.run([sys.executable,str(L/'code/transfer_readonly_native_bundle_to_QH_v1.py'),'--root',str(root)],check=True)

from pathlib import Path,PurePosixPath
import sys,os,json,datetime,hashlib,time,concurrent.futures,tarfile,fcntl,traceback
Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");N=Path("/tmp/starreal_uid1007_case19_cpu_v1");R=Path("/tmp/starreal_uid1007_case111_true4K_cpu_v2")
sys.path[:0]=[str(Q.parent/"HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps"),str(N/"runtime/qh_cpu_py39_v1")]
from huggingface_hub import HfApi,hf_hub_url
import requests,numpy as np,cv2
lock=open(Q/"logs/case111_historical_input_import.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=Q/"inputs/case111_historical_mesh_HF_v3";out.mkdir(exist_ok=False);start=time.time()
token=Path("/home/fangrui/.cache/lampedit-vlmdataset-upload/token").read_text().strip();api=HfApi(token=token);head=api.repo_info("FrontRear/VLMdataset",repo_type="dataset").sha;prefix="realcase/111/move_replace_mesh_v1";tree=[]
for row in api.list_repo_tree("FrontRear/VLMdataset",path_in_repo=prefix,repo_type="dataset",revision=head,recursive=True):
 if hasattr(row,"size"):
  lfs=getattr(row,"lfs",None);tree.append(dict(path=row.path,size=row.size,blob_id=row.blob_id,LFS_SHA256=lfs.sha256 if lfs else None))
assert tree and sum(x["path"].endswith("conditions.tar") for x in tree)==1
(out/"PINNED_SERVER_TREE.json").write_text(json.dumps(dict(HF_revision=head,files=tree,case="111",historical_operation="move_replace_only",no_fake_relative_ops=True),indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def fetch(row):
 rel=PurePosixPath(row["path"]).relative_to(prefix);assert ".." not in rel.parts
 dest=out/"original_HF_folder"/str(rel);dest.parent.mkdir(parents=True,exist_ok=True);part=dest.with_name(dest.name+".part");h=hashlib.sha256();g=hashlib.sha1(b"blob "+str(row["size"]).encode()+b"\0");n=0
 with requests.get(hf_hub_url("FrontRear/VLMdataset",row["path"],repo_type="dataset",revision=head),headers={"Authorization":"Bearer "+token},stream=True,timeout=(30,120)) as resp:
  resp.raise_for_status()
  with part.open("xb") as f:
   for b in resp.iter_content(4*1024**2):
    if b:f.write(b);h.update(b);g.update(b);n+=len(b)
 assert n==row["size"]
 if row["LFS_SHA256"]:assert h.hexdigest()==row["LFS_SHA256"]
 else:assert g.hexdigest()==row["blob_id"]
 part.replace(dest);return dict(HF_path=row["path"],local_path=str(dest),bytes=n,SHA256=h.hexdigest(),server_LFS_SHA256=row["LFS_SHA256"],server_blob_id=row["blob_id"])
records=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for row in pool.map(fetch,tree):
  records.append(row);(out/"DOWNLOAD_CURRENT.json").write_text(json.dumps(dict(files_complete=len(records),files_expected=len(tree),bytes_complete=sum(x["bytes"] for x in records),GPU_used=False),indent=2))
archive=out/"original_HF_folder/conditions.tar";camout=out/"extracted_cameras";camout.mkdir();cams=[]
with tarfile.open(archive) as tf:
 for mem in tf.getmembers():
  if not mem.isfile() or not mem.name.endswith("/cameras_and_boxes.npz"):continue
  assert mem.size<32*1024**2;data=tf.extractfile(mem).read();name="orbit_10deg" if "orbit_10deg" in mem.name else "small_4deg" if "small_4deg" in mem.name else None;assert name,"Unknown actual camera name: "+mem.name
  p=camout/(name+".npz");assert not p.exists();p.write_bytes(data);cams.append(dict(camera_version=name,archive_member=mem.name,path=str(p),bytes=len(data),SHA256=sha(p)))
assert len(cams)==2 and {x["camera_version"] for x in cams}=={"orbit_10deg","small_4deg"}
old=np.load(N/"inputs/case111_native_CPU_v1/cameras.npz");new=np.load(Q/"component_cache/111/true4K_RGB_camera_component_v2/cameras.npz");H=np.asarray(json.loads((Q/"component_cache/111/true4K_RGB_camera_component_v2/INPUT_MANIFEST.json").read_text())["H_old_prepared1280_to_new_Original1280"])
cam_checks=[]
for row in cams:
 z=np.load(row["path"]);assert z["K"].shape==(49,3,3) and z["w2c_A"].shape==z["w2c_B"].shape==(49,4,4)
 np.testing.assert_array_equal(z["K"],old["K"]);np.testing.assert_array_equal(z["w2c_A"],old["w2c_A"])
 np.testing.assert_array_equal(H@z["K"],new["K"])
 if row["camera_version"]=="orbit_10deg":np.testing.assert_array_equal(z["w2c_B"],new["w2c_B"])
 else:assert not np.array_equal(z["w2c_B"],new["w2c_B"])
 row["new_full_K_known_H_transform_match"]=True;row["old_camera_bytes_retained"]=True;cam_checks.append(row)
alias=R/"inputs/case111_historical_mesh_HF_v3";assert not alias.exists();alias.symlink_to(out,target_is_directory=True)
result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),case="111",state="all_actual_original_move_replace_mesh_HF_folder_bytes_server_SHA_verified_and_two_real_cameras_extracted",HF_revision=head,historical_operation_count=1,historical_camera_versions=2,relative_ops_v1_exists=False,records=records,camera_checks=cam_checks,archive_SHA256=sha(archive),archive_bytes=archive.stat().st_size,old_camera_A_primary_B_preserved_exact=True,new_full_K_is_known_Original_crop_H_transform=True,small4_not_relabelled_orbit10=True,new_complete_groups=0,quality_accepted=False,GPU_used=False,Mac_payload_files=False,T9_writes=False,no6000D_execution=True,seconds=time.time()-start)
(out/"HF_IMPORT_MANIFEST.json").write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ["records","camera_checks"]}),flush=True)

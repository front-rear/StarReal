from pathlib import Path,PurePosixPath
import sys,os,json,datetime,hashlib,time,concurrent.futures,tarfile,fcntl,traceback,shutil
Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal")
U=Q/"all59_reconstruction_20261115_v1";N=Path("/tmp/starreal_uid1007_case19_cpu_v1")
sys.path[:0]=[str(Q.parent/"HF_ORIGINAL_STRICT500_v1/runtime/huggingface_py39_v1/python_deps"),str(N/"runtime/qh_cpu_py39_v1")]
from huggingface_hub import hf_hub_url
import requests,numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""
lock=open(U/"EXTRA5_ACTUAL_HF_HISTORY_MATERIALIZATION.lock","a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
root=Q/"inputs/extras_five_actual_HF_history_v1";root.mkdir(exist_ok=True)
state=U/"EXTRA5_ACTUAL_HF_HISTORY_MATERIALIZATION_FINITE_v1.json"
token=Path("/home/fangrui/.cache/lampedit-vlmdataset-upload/token").read_text().strip()
started=time.time();st=dict(pid=os.getpid(),uid=os.getuid(),boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),startticks=Path("/proc/self/stat").read_text().split()[21],state="running_finite_five_actual_original_folders",cases={},GPU_used=False,Mac_payload_files=False,T9_writes=False,no6000D_execution=True)
def dump(p,j):
 t=p.with_name(p.name+".tmp_"+str(os.getpid()));t.write_text(json.dumps(j,indent=2)+"\n");t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def verify(row,path):
 assert path.stat().st_size==row["size"],(path,"size")
 h=hashlib.sha256();g=hashlib.sha1(b"blob "+str(row["size"]).encode()+b"\0")
 with path.open("rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b);g.update(b)
 assert h.hexdigest()==row["LFS_SHA256"] if row["LFS_SHA256"] else g.hexdigest()==row["blob_id"],(path,"hash")
 return dict(HF_path=row["path"],local_path=str(path),bytes=path.stat().st_size,SHA256=h.hexdigest(),server_LFS_SHA256=row["LFS_SHA256"],server_blob_id=row["blob_id"])
def fetch(row,out,head):
 rel=PurePosixPath(row["path"]).relative_to("realcase/"+case);assert ".." not in rel.parts
 dest=out/"original_HF_folder"/str(rel);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():return dict(verify(row,dest),source="retained_independently_SHA_verified_prior_file")
 part=dest.with_name(dest.name+".part_"+str(os.getpid()));assert not part.exists()
 with requests.get(hf_hub_url("FrontRear/VLMdataset",row["path"],repo_type="dataset",revision=head),headers={"Authorization":"Bearer "+token},stream=True,timeout=(30,120)) as resp:
  resp.raise_for_status()
  with part.open("xb") as f:
   for b in resp.iter_content(4*1024**2):
    if b:f.write(b)
 rec=verify(row,part);part.replace(dest);rec["local_path"]=str(dest);rec["source"]="actual_pinned_HF_GET";return rec
dump(state,st)
try:
 for case in ["6374204","10302763","7505473","7505429","7578553"]:
  out=root/case;out.mkdir(exist_ok=True);pinned=json.loads((U/("CASE"+case+"_ACTUAL_HF_LAYOUT_20261116_v1.json")).read_text());head=pinned["HF_revision"];tree=pinned["files"]
  if (out/"HF_IMPORT_MANIFEST.json").exists():
   old=json.loads((out/"HF_IMPORT_MANIFEST.json").read_text());assert old["state"]=="all_actual_original_HF_folder_files_server_SHA_verified";st["cases"][case]={"state":old["state"],"files":len(old["records"])};dump(state,st);continue
  st["current_case"]=case;st["cases"][case]={"state":"materializing_actual_two_archives","expected_files":len(tree)};dump(state,st)
  dump(out/"PINNED_SERVER_TREE.json",pinned)
  archives=[r for r in tree if r["path"].endswith(".tar")];assert len(archives)==2
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:records=list(ex.map(lambda r:fetch(r,out,head),archives))
  handles=[];indices=[];members_meta=[];extracted=[]
  for r in records:
   a=Path(r["local_path"]);tf=tarfile.open(a);handles.append(tf);idx={m.name:m for m in tf.getmembers() if m.isfile()};indices.append((a,tf,idx))
   for name,m in idx.items():
    rel=PurePosixPath(name);assert not rel.is_absolute() and ".." not in rel.parts and not m.issym() and not m.islnk()
    members_meta.append({"archive":a.name,"archive_parent":a.parent.name,"member":name,"bytes":m.size})
    if m.size<=32*1024**2 and name.endswith((".npz",".json",".py",".md")):
     d=out/"extracted"/a.parent.name/str(rel);d.parent.mkdir(parents=True,exist_ok=True);data=tf.extractfile(m).read()
     if d.exists():assert d.read_bytes()==data
     else:d.write_bytes(data)
     extracted.append({"archive":str(a),"member":name,"path":str(d),"bytes":len(data),"SHA256":hashlib.sha256(data).hexdigest()})
  for row in tree:
   if row["path"].endswith(".tar"):continue
   rel=PurePosixPath(row["path"]).relative_to("realcase/"+case);dest=out/"original_HF_folder"/str(rel)
   if dest.exists():records.append(dict(verify(row,dest),source="retained_independently_SHA_verified_prior_file"));continue
   branch=rel.parts[0];within=str(PurePosixPath(*rel.parts[1:]));done=False
   for a,tf,idx in indices:
    if a.parent.name!=branch:continue
    choices=[m for n,m in idx.items() if (n==within or n.endswith("/"+within)) and m.size==row["size"]]
    if len(choices)!=1:continue
    mem=choices[0];dest.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(mem) as ff,dest.open("xb") as f:shutil.copyfileobj(ff,f,4*1024**2)
    rec=verify(row,dest);rec.update(source="exact_member_from_original_SHA_verified_archive_checked_against_independent_server_identity",archive_member=mem.name);records.append(rec);done=True;break
   if not done:records.append(fetch(row,out,head))
   st["cases"][case].update(state="materializing_original_folder_files",files_complete=len(records),expected_files=len(tree));dump(state,st)
  for tf in handles:tf.close()
  assert len(records)==len(tree) and len(set(r["HF_path"] for r in records))==len(tree)
  dump(out/"ARCHIVE_MEMBERS.json",members_meta);dump(out/"EXTRACTED_SMALL_INPUTS.json",extracted)
  res=dict(case=case,state="all_actual_original_HF_folder_files_server_SHA_verified",HF_revision=head,records=records,extracted_small_inputs=extracted,archive_members=len(members_meta),GPU_used=False,Mac_payload_files=False,T9_writes=False,quality_accepted=False,new_complete_groups=0)
  dump(out/"HF_IMPORT_MANIFEST.json",res);st["cases"][case]={"state":res["state"],"files":len(records),"archive_members":len(members_meta),"small_input_files":len(extracted)};dump(state,st)
 st.update(state="terminal0_all_five_actual_original_folders_verified",exit_code=0,seconds=time.time()-started);dump(state,st);print(json.dumps(st),flush=True)
except BaseException as e:
 st.update(state="terminal1_precise_exception_preserve_partial",exit_code=1,error=repr(e),traceback=traceback.format_exc(),seconds=time.time()-started);dump(state,st);raise

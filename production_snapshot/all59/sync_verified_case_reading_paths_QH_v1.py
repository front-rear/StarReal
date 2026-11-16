from pathlib import Path
import json,hashlib,datetime,os,argparse,shutil
Q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");U=Q/"all59_reconstruction_20261115_v1"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--case",required=True);a=ap.parse_args();c=a.case
 final=json.loads((U/f"CASE{c}_HF_FINAL_VERIFIED.json").read_text());assert final["state"]=="final_public_receipt_verified" and final["case"]==c
 d=json.loads((U/f"CASE{c}_PUBLICATION_PAYLOAD.json").read_text());receipt=json.loads((U/f"CASE{c}_HF_DATA_VERIFIED.json").read_text());assert receipt["current_cases"]==final["current_cases"]==d["current_cases"]
 current=json.loads((Q/"DATASET_CURRENT.json").read_text());assert current["current_cases"]<=d["current_cases"],"Do not roll back a newer canonical index"
 rows=[]
 for r in d["files"]:
  p=Path(r["local_path"]);expected=r["SHA256"];size=r["bytes"];commit=final["data_commit"]
  if r["HF_path"]==final["receipt_path"]:p=U/f"CASE{c}_HF_DATA_VERIFIED.json";expected=final["receipt_SHA256"];size=p.stat().st_size;commit=final["final_commit"]
  assert p.stat().st_size==size and sha(p)==expected
  rel=r["HF_path"].removeprefix("StarReal/");assert rel!=r["HF_path"] and ".." not in Path(rel).parts
  dst=Q/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists() and not dst.is_symlink() and sha(dst)!=expected:
   assert dst.stat().st_size<10*1024**2,"Preserve unexpected large existing canonical file"
   backup=U/"reading_entry_history"/("before_case"+c)/rel;backup.parent.mkdir(parents=True,exist_ok=True);assert not backup.exists();shutil.copy2(dst,backup);assert sha(backup)==sha(dst)
  if dst.resolve()!=p.resolve():
   tmp=dst.with_name(dst.name+".linktmp."+str(os.getpid()));assert not tmp.exists();tmp.symlink_to(p);os.replace(tmp,dst)
  assert dst.stat().st_size==size and sha(dst)==expected
  rows.append(dict(HF_path=r["HF_path"],QH_reading_path=str(dst),QH_original_file=str(p),bytes=size,SHA256=expected,large_payload_duplicated=False,verified_HF_commit=commit))
 out=dict(case=c,state="all_QH_reading_entries_synced_and_SHA_verified",at=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=rows,current_cases=final["current_cases"],final_commit=final["final_commit"],T9_changed=False,Mac_large_payloads_created=False)
 p=U/f"CASE{c}_QH_READING_ENTRIES_SYNCED_v1.json";assert not p.exists();p.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!="files"}))
if __name__=="__main__":main()

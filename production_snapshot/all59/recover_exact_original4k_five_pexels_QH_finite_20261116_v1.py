from pathlib import Path
import datetime,hashlib,json,os,time,urllib.request
q=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal");u=q/"all59_reconstruction_20261115_v1";src=json.loads((u/"provenance/EXTRA5_ORIGINAL_INPUT_LAYOUT_READONLY_DISCOVERY_20261115.json").read_text());out=q/"source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1";out.mkdir(parents=True,exist_ok=False);rows=[]
for row in src["extra_five_original_input_routes"]:
 c=row["case"];prov=row["original_4K_provenance"];name=Path(prov["historical_path"]).name;url="https://videos.pexels.com/video-files/"+c+"/"+name;expected=prov["SHA256"];r=dict(case=c,url=url,expected_original_SHA256=expected,state="candidate_NOT_admitted");start=time.time();tmp=out/(name+".partial."+str(os.getpid()))
 try:
  req=urllib.request.Request(url,headers={"User-Agent":"Mozilla/5.0"})
  with urllib.request.urlopen(req,timeout=45) as response:
   r.update(HTTP_status=response.status,content_length=response.headers.get("Content-Length"),content_type=response.headers.get("Content-Type"));assert response.status==200
   h=hashlib.sha256();num=0
   with tmp.open("xb") as f:
    while True:
     assert time.time()-start<300,"Finite300s origin retrieval deadline"
     b=response.read(4*1024**2)
     if not b:break
     num+=len(b);assert num<2*1024**3,"Unexpected oversized candidate";h.update(b);f.write(b)
  r.update(bytes=num,actual_SHA256=h.hexdigest())
  if h.hexdigest()==expected:
   dest=out/name;tmp.rename(dest);r.update(state="exact_original_4K_SHA_verified_recovered_on_QH",path=str(dest),formal_case_accepted=False)
  else:r.update(state="candidate_SHA_mismatch_preserved_NOT_source_replacement",path=str(tmp))
 except BaseException as e:r.update(state="candidate_retrieval_failed_no_source_replacement",error=repr(e),partial_path=str(tmp) if tmp.exists() else None)
 r["elapsed_seconds"]=time.time()-start;rows.append(r);(out/(c+"_RESULT.json")).write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
 result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,Mac_payload_files=False,T9_written=False,no6000D_execution_or_writes=True,quality_accepted=False)
 (out/"CURRENT.json").write_text(json.dumps(result,indent=2))
(out/"RESULT.json").write_text(json.dumps(result,indent=2))

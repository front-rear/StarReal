from pathlib import Path
import os,sys,json,hashlib,time,importlib.util,ast,copy,fcntl
os.environ["CUDA_VISIBLE_DEVICES"]=""
R=Path("/tmp/starreal_uid1007_case7505429_true4K_cpu_v2");U=Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal/all59_reconstruction_20261115_v1");CASE="7505429";os.environ["STARREAL_CASE"]=CASE
src=U/"build_case7505429_true4K_cached_conditions49_QH_v1.py";spec=importlib.util.spec_from_file_location("condition_producer_existing",src);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
np=mod.np;sha=mod.sha;put=mod.put;start=time.time();out=R/f"outputs/{CASE}/geometry_masks49_cached_QH_v1";guard=(U.parent/f"logs/case{CASE}_QH_conditions49.lock").open("a");fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
oldjob=json.loads((out/"JOB.json").read_text());assert oldjob["exit_code"]==1 and oldjob["frames_finished"]==49 and "source_hidden_geometry_policy" in oldjob["error"];assert not (out/"MANIFEST.json").exists()
(out/"JOB_v1_metadata_error_UNCHANGED.json").write_bytes((out/"JOB.json").read_bytes());snap=R/f"inputs/case{CASE}_native_CPU_v1";reviewpath=R/f"reviews/CASE{CASE}_FIVE_ROLE_ALL49_GEOMETRY_ACTUAL_REVIEW_v1.json";review=json.loads(reviewpath.read_text());cfg=json.loads((R/f"inputs/CASE{CASE}_GEOMETRY49_CONFIG_v1.json").read_text());assert "source_hidden_geometry_policy" not in review
oldreview=reviewpath.with_name(reviewpath.stem+"_BEFORE_POLICY_FIELD_v1.json");oldreview.write_bytes(reviewpath.read_bytes());review["source_hidden_geometry_policy"]=cfg["source_hidden_geometry_policy"];put(reviewpath,review)
backend=json.loads((R/f"inputs/CASE{CASE}_INPUT_BACKEND_QH_v1.json").read_text());geo=R/review["component_root"];result=json.loads((geo/"RESULT.json").read_text());asset_proof=json.loads(Path(cfg["assets_proof"]).read_text());byop={r["role"]:r["source_index"] for r in asset_proof["actual_exact_raw_SHA_matched"]}
specs=[("remove","remove",None,None),("move","move",None,"move")]+[(op+"_"+byop[op],op,byop[op],op+"_"+byop[op]) for op in ["insert","replace","move_replace"]]
jobs={name:dict(operation=op,asset_id=asset,cache_role=role,frames=[]) for name,op,asset,role in specs};before={};keys=["amodal","scene_occluded_estimate","visible_known_estimate","unobserved_scene_region","visible_with_unknown_estimate"]
def fields_metadata(p):
 before[str(p)]=sha(p)
 with np.load(p) as z:
  assert z["full_shape_hw"].tolist()==[704,1280];s=z["support"];x0,y0=map(int,z["origin_xy"]);h,w=s.shape
  for k in z.files:a=z[k];assert np.isfinite(a).all()
  return dict(support_pixels=int(s.sum()),roi_xyxy=[x0,y0,x0+w,y0+h])
for i in range(49):
 cachepath=geo/"visibility49"/f"{i:06d}.npz"
 with np.load(cachepath) as z:
  for name,d in jobs.items():
   folder=out/name;role=d["cache_role"];lf=fields_metadata(folder/"conditions/lamp_silhouette"/f"{i:06d}.npz");bf=fields_metadata(folder/"conditions/bbox_silhouette"/f"{i:06d}.npz")
   d["frames"].append(dict(frame=i,lamp=lf,bbox=bf,scene_occluded_pixels=int(z[role+"_B_scene_occluded_estimate"].sum()) if role else 0,unknown_scene_pixels=int(z[role+"_B_unobserved_scene_region"].sum()) if role else 0,visibility_cache_SHA256=sha(cachepath)))
assert len(before)==490
state=dict(case=CASE,state="recovering_manifest_only_all490_existing_fields_fully_read",pid=os.getpid(),frames_finished=49,GPU_used=False,quality_accepted=False)
def save():state["elapsed_seconds"]=time.time()-start;put(out/"JOB.json",state)
# Execute only unchanged original metadata assembly statements after frame49.
tree=ast.parse(src.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main");tr=next(n for n in main.body if isinstance(n,ast.Try));tail=[copy.deepcopy(n) for n in tr.body if n.lineno>=68];assert tail and tail[0].lineno==68;module=ast.Module(body=tail,type_ignores=[]);ast.fix_missing_locations(module)
env=dict(mod.__dict__);env.update(R=R,CASE=CASE,out=out,reviewpath=reviewpath,review=review,cfg=cfg,backend=backend,geo=geo,result=result,snap=snap,asset_proof=asset_proof,jobs=jobs,state=state,save=save)
exec(compile(module,str(src),"exec"),env)
assert before=={p:sha(p) for p in before};record=dict(case=CASE,state="terminal0_manifest_recovered_490_existing_conditions_SHA_unchanged",original_error_job="JOB_v1_metadata_error_UNCHANGED.json",fields_read=490,all_condition_file_SHA_unchanged=True,original_metadata_AST_executed=True,no_geometry_rerender_or_condition_rewrite=True,manifest_SHA256=sha(out/"MANIFEST.json"),GPU_used=False,quality_accepted=False,seconds=time.time()-start)
put(out/"MANIFEST_ONLY_RECOVERY_v2.json",record);print(json.dumps(record),flush=True)

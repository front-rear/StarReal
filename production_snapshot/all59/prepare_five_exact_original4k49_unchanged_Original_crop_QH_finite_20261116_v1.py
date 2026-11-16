import os,sys,pathlib,json,hashlib,datetime,time,ast,fcntl,traceback
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="8"
os.environ["MKL_NUM_THREADS"]="8"
N=pathlib.Path("/tmp/starreal_uid1007_case19_cpu_v1")
RUNTIME=N/"runtime/sam2_private_cpu_py312_v1"
sys.path[:0]=[str(RUNTIME/"deps"),str(RUNTIME/"sam2")]
import torch,torchvision,cv2,numpy as np
from PIL import Image,ImageDraw
torch.set_num_threads(8);cv2.setNumThreads(8)
Q=pathlib.Path("/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal")
U=Q/"all59_reconstruction_20261115_v1"
BASE=Q/"source_input_recovery/EXACT_ORIGINAL4K_PEXELS_20261116_v1"
OUT=BASE/"prepared49_from_actual_Original_crop_v1"
OUT.mkdir(exist_ok=True)
lk=open(OUT/"SOURCE_PREP_OWNER.lock","a+");fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(4*1024**2),b""):h.update(b)
 return h.hexdigest()
def write(p,d):
 t=p.with_name(p.name+".tmp");t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
src=pathlib.Path("/home/fangrui/LampEdit/source_snapshot/ROOMSDAY/LumiLens/diffsynth/trainers/utils.py")
s=src.read_text();tree=ast.parse(s)
cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="VideoDataset")
fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="crop_and_resize")
fn_source=ast.get_source_segment(s,fn)
namespace={"torchvision":torchvision}
mod=ast.Module(body=[ast.ClassDef(name="UnchangedOriginalCrop",bases=[],keywords=[],body=[fn],decorator_list=[])],type_ignores=[])
exec(compile(ast.fix_missing_locations(mod),str(src),"exec"),namespace)
crop=namespace["UnchangedOriginalCrop"]().crop_and_resize
(OUT/"Original_VideoDataset_crop_and_resize_UNMODIFIED.txt").write_text(fn_source)
recovered=json.loads((BASE/"RESULT.json").read_text())
sampling=json.loads((U/"provenance/EXTRA5_EXACT_ORIGINAL_LOWRES_SAMPLING_METADATA_20261116_v1.json").read_text())
rows=[];started=time.time()
aliases={"10302763":"108","6374204":"109","7505473":"111"}
def crop_matrix(w,h,tw,th):
 scale=max(tw/w,th/h);rw,rh=round(w*scale),round(h*scale)
 left,top=int(round((rw-tw)/2.)),int(round((rh-th)/2.))
 return np.array([[rw/w,0,-left],[0,rh/h,-top],[0,0,1]],np.float64),dict(input_wh=[w,h],scale=scale,resized_wh=[rw,rh],crop_xy=[left,top],output_wh=[tw,th])
for r in recovered["rows"]:
 cid=r["case"];d=OUT/cid
 if (d/"RESULT.json").exists():
  prev=json.loads((d/"RESULT.json").read_text());assert prev["state"]=="all49_exact_4K_original_frames_Original_crop_prepared_pending_camera_and_SAM_review";rows.append(prev);continue
 d.mkdir(exist_ok=False);(d/"RGB49").mkdir()
 oldmeta=next(x for x in sampling["rows"] if x["case"]==cid)["sampling.json"]
 meta=oldmeta["content"];p=pathlib.Path(r["path"]);assert sha(p)==r["expected_original_SHA256"]==meta["source_sha256"]
 ids=meta["frame_indices"];assert len(ids)==len(set(ids))==49 and ids==sorted(ids)
 cap=cv2.VideoCapture(str(p));w=int(cap.get(3));h=int(cap.get(4));fps=cap.get(5)
 assert [w,h]==meta["source_size_wh"] and abs(fps-meta["source_fps"])<1e-6
 M,transform=crop_matrix(w,h,1280,704);oldM=np.asarray(meta["resize_crop_matrix"],float);oldPreparedM=np.diag([2.,2.,1.])@oldM
 H=M@np.linalg.inv(oldPreparedM)
 writer=cv2.VideoWriter(str(d/"source_A_true4K_Original1280x704_49_FFV1.mkv"),cv2.VideoWriter_fourcc(*"FFV1"),10.,(1280,704));assert writer.isOpened()
 oldcap=None;alias=aliases.get(cid);cam_record=None;oldrgb=[]
 if alias:
  group="109_111" if alias in ["109","111"] else "106_108"
  op=N/("inputs/StarReal_native_readonly_SFTP_"+group+"_stream_v2")/alias/"source_A.mkv"
  oldcap=cv2.VideoCapture(str(op));assert oldcap.isOpened()
  z=np.load(N/("inputs/case"+alias+"_native_CPU_v1/cameras.npz"))
  raw=np.load(N/("inputs/case"+alias+"_native_CPU_v1/native_depth_K_E_subset.npz"))
  oldK=z["K"].copy();newK=H@oldK
  expectedK=raw["K"].astype(float).copy();expectedK[:,0,:]*=1280/raw["depth"].shape[2];expectedK[:,1,:]*=704/raw["depth"].shape[1]
  assert np.allclose(expectedK,oldK,atol=1e-8,rtol=1e-8)
  cameras={k:z[k] for k in z.files};cameras["K"]=newK
  np.savez_compressed(d/"cameras_known_sampling_homography_derived.npz",**cameras)
  rng=np.random.default_rng(0);rays=np.c_[rng.uniform(0,1280,200),rng.uniform(0,704,200),np.ones(200)]
  reproj=0.
  for fi in range(49):
   camr=(np.linalg.inv(oldK[fi])@rays.T).T
   po=(oldK[fi]@camr.T).T;pn=(newK[fi]@camr.T).T
   via=(H@po.T).T;reproj=max(reproj,float(np.max(np.abs(pn/pn[:,2:]-via/via[:,2:]))))
  assert reproj<1e-8
  cam_record=dict(alias=alias,path=str(d/"cameras_known_sampling_homography_derived.npz"),old_camera_K_retained=True,new_full_K_is_known_image_homography_derived=True,H_old_prepared1280_to_new_Original1280=H.tolist(),world_A_B_poses_preserved_exact=True,raw_depth_K_E_unchanged=True,new_Native_inference_run=False,max_ray_reprojection_error_pixels=reproj,quality_accepted=False)
 selected={n:i for i,n in enumerate(ids)};fr=[];sheet=Image.new("RGB",(1280,7*213),(18,18,18));draw=ImageDraw.Draw(sheet)
 for ni in range(ids[-1]+1):
  ok,bgr=cap.read();assert ok,("Original frame absent; no clamp permitted",cid,ni)
  if ni not in selected:continue
  i=selected[ni];im=crop(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)),704,1280)
  assert im.size==(1280,704);a=np.asarray(im)
  assert np.array_equal(a,np.asarray(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize(tuple(transform["resized_wh"]),Image.Resampling.BILINEAR).crop((transform["crop_xy"][0],transform["crop_xy"][1],transform["crop_xy"][0]+1280,transform["crop_xy"][1]+704))))
  dst=d/"RGB49"/f"{i:06d}.png";im.save(dst,compress_level=1);writer.write(cv2.cvtColor(a,cv2.COLOR_RGB2BGR))
  rr=dict(frame=i,original_frame=ni,original_timestamp_seconds=ni/fps,RGB_SHA256=hashlib.sha256(a.tobytes()).hexdigest(),PNG_SHA256=sha(dst),source_is_true_original4K=True)
  thumb=im.copy();thumb.thumbnail((256,181));sheet.paste(thumb,((i%5)*256,(i//5)*213+20)) if i<35 else None
  if oldcap:
   ok,old=oldcap.read();assert ok and old.shape==(704,1280,3)
   oldrgb=cv2.cvtColor(old,cv2.COLOR_BGR2RGB)
   warped=cv2.warpPerspective(oldrgb,H,(1280,704),flags=cv2.INTER_LINEAR)
   diff=np.abs(a.astype(np.int16)-warped.astype(np.int16))
   rr["old_prepared_comparison"]=dict(mean_abs=float(diff.mean()),p50=float(np.percentile(diff,50)),p95=float(np.percentile(diff,95)),same_pixels_fraction=float(np.mean(np.all(a==warped,axis=2))),not_exact_old_upscale=not np.array_equal(a,oldrgb))
   if i in [0,24,48]:
    panel=Image.new("RGB",(1280,704*2+40),(20,20,20));panel.paste(im,(0,20));panel.paste(Image.fromarray(warped),(0,744))
    pd=ImageDraw.Draw(panel);pd.text((5,3),f"{cid} frame {i}: exact original 4K -> unchanged Original 1280x704",(255,255,255));pd.text((5,725),"historical 640 RGB -> old prepared 1280, transformed by proven sampling H (comparison only)",(255,255,255));panel.save(d/f"true4K_vs_old640upscale_{i:02d}.jpg",quality=87)
  fr.append(rr)
 cap.release();writer.release()
 if oldcap:oldcap.release()
 assert len(fr)==49 and len({x["original_frame"] for x in fr})==49
 cap2=cv2.VideoCapture(str(d/"source_A_true4K_Original1280x704_49_FFV1.mkv"));decoded=[]
 for i in range(50):
  ok,bgr=cap2.read()
  if not ok:break
  decoded.append(hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest())
 cap2.release();assert decoded==[x["RGB_SHA256"] for x in fr]
 # All49 visible sheet 7 columns x7rows.
 sh=Image.new("RGB",(1280,7*122),(15,15,15));dd=ImageDraw.Draw(sh)
 for i in range(49):
  im=Image.open(d/"RGB49"/f"{i:06d}.png");im.thumbnail((182,101));x=(i%7)*182;y=(i//7)*122
  sh.paste(im,(x,y+17));dd.text((x+3,y),f"{i:02d} / original {ids[i]}",(255,255,255))
 sh.save(d/"all49_true4K_source_actual_review.jpg",quality=88)
 rec=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),case=cid,state="all49_exact_4K_original_frames_Original_crop_prepared_pending_camera_and_SAM_review",source_video=str(p),source_SHA256=sha(p),original_sampling_metadata_SHA256=oldmeta["SHA256"],output_wh=[1280,704],frames=49,FPS=10,original_crop_function_source=str(src),original_crop_module_SHA256=sha(src),original_crop_function_SHA256=hashlib.sha256(fn_source.encode()).hexdigest(),actual_Original_function_executed=True,PIL_implementation_pixel_identical_all49=True,transform=transform,old640_to4K_sampling_provenance_retained=True,H_old_prepared1280_to_new_Original1280=H.tolist(),camera_derivation=cam_record,frame_readbacks=fr,source_video_output=str(d/"source_A_true4K_Original1280x704_49_FFV1.mkv"),source_video_output_SHA256=sha(d/"source_A_true4K_Original1280x704_49_FFV1.mkv"),all49_FFV1_decoded_exact=True,GPU_used=False,Mac_payload_files=False,T9_writes=False,quality_accepted=False,formal_group_increment=0)
 write(d/"RESULT.json",rec);rows.append(rec);write(OUT/"CURRENT.json",dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=[x["case"] for x in rows],quality_accepted=False));print(json.dumps(dict(case=cid,state=rec["state"],camera_alias=alias,output_video_bytes=(d/"source_A_true4K_Original1280x704_49_FFV1.mkv").stat().st_size)),flush=True)
write(OUT/"RESULT.json",dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),state="five_original4K_49frame_inputs_prepared_all_exact_lossless_readback",cases=[x["case"] for x in rows],rows=[dict(case=x["case"],path=str(OUT/x["case"]/"RESULT.json"),camera_alias=aliases.get(x["case"])) for x in rows],seconds=time.time()-started,quality_accepted=False,formal_group_increment=0,Mac_payload_files=False,T9_writes=False))

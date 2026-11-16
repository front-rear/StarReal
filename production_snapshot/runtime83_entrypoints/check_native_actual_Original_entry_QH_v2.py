"""Actual Original input-boundary audit on Qinghai CPU; never neural inference."""
from pathlib import Path
import ast,gc,hashlib,json,os,sys,types
N=Path('/tmp/starreal_uid1007_case83_cpu_v2');Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal');CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal()
runtime=json.loads((N/'runtime/sam2_private_cpu_py312_v1/RESULT.json').read_text());sys.path[:0]=[str(N/'code'),runtime['dependencies_root']]
from einops import repeat
import numpy as np,torch
from PIL import Image
from realcase_condition_reader import RealcaseConditionReader
ORIGINAL=Q/'provenance/Original_wan_video_new_UNMODIFIED.py';BASE=Q/'provenance/Original_base_pipeline_UNMODIFIED.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ORIGINAL)=='5d93b5dd851bc11a4e5bb096310d2c1197570ecb285ff26be1b706170f089f94'
assert sha(BASE)=='04f3b66c17b8decb690be1d23b23915b42fdf91bc0d17de801f5ac5621f1f72d'
assert sha(N/'code/realcase_condition_reader.py')=='5e846b3be5269de422bc8c425da236610623460c38ace711efe7dca99708f23a'
def function(path,clsname,name,ns):
 cls=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.ClassDef) and x.name==clsname);node=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==name);before=ast.dump(ast.Module(body=node.body,type_ignores=[]));node.returns=None;node.decorator_list=[]
 for arg in node.args.args+node.args.kwonlyargs:arg.annotation=None
 assert ast.dump(ast.Module(body=node.body,type_ignores=[]))==before
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),ns);return ns[name]
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);captured={}
 def capture(**kw):captured.update(kw);return torch.zeros((1,1,1))
 ns=dict(torch=torch,np=np,repeat=repeat,prepare_camera_embeds=capture);pi=function(BASE,'BasePipeline','preprocess_image',ns);pv=function(BASE,'BasePipeline','preprocess_video',ns);process=function(ORIGINAL,'WanVideoUnit_WarpControl','process',ns)
 pipe=types.SimpleNamespace(device='cpu',torch_dtype=torch.float32,camera_encoder=types.SimpleNamespace(),vae=None);pipe.preprocess_image=types.MethodType(pi,pipe);pipe.preprocess_video=types.MethodType(pv,pipe);pipe.load_models_to_device=lambda names:None;unit=types.SimpleNamespace(onload_model_names=('vae','camera_encoder'))
 c=N/f'outputs/{CASE}/geometry_masks49_cached_QH_v1';j=N/f'outputs/{CASE}/two_pcf49_fixed_world_yellow_QH_v1';source=N/f'inputs/case{CASE}_source_video_HF_v1/source_A.mkv';native=json.loads((N/f'inputs/case{CASE}_native_CPU_v1/INPUT_MANIFEST.json').read_text());assert sha(source)==native['source_video_SHA256'];reader=RealcaseConditionReader(c,source,j);assert reader.manifest['case']==CASE
 assert json.loads((c/'INDEPENDENT_CONDITION_READBACK.json').read_text())['files_read']==490 and json.loads((j/'JOB.json').read_text())['exit_code']==0
 records=[]
 for family in ['lamp_silhouette','bbox_silhouette']:
  for variant,operation in reader.manifest['operations'].items():
   for level in reader.manifest['gamma_control_pairs_by_operation'][operation['operation']]:
    indices=[0,24,48];rows=[reader.condition_frame(variant,family,i,'purple',level) for i in indices];support=np.stack([x['support'] for x in rows]);masks=[x['color_mask_rgb'] for x in rows];warps=[Image.open(j/variant/'new_joint'/f'{i:06d}.png').convert('RGB') for i in indices];gamma=rows[0]['normalized_gamma_for_model'];captured.clear();process(unit,pipe,warp_video=warps,warp_mask=None,warped_target_color_mask=masks,relative_gamma=gamma)
    color=captured['warped_target_color_mask'];intensity=captured['warped_target_intentsity_mask'];assert color.shape==intensity.shape==(1,3,3,704,1280);np.testing.assert_array_equal((color.max(dim=1)[0][0]>-.95).numpy(),support);expected=np.where(support,np.float32(gamma),np.float32(-1))
    for channel in range(3):np.testing.assert_array_equal(intensity[0,channel].numpy(),expected)
    assert 'video' not in reader.provenance(variant,family,'purple',level);records.append(dict(case=CASE,variant=variant,family=family,level=level,frames=indices,relative_gamma=gamma,actual_original_entry_passed=True,actual_joint_used_as_warp=True));del rows,support,masks,warps,color,intensity,expected;captured.clear();gc.collect()
 assert len(records)==26
 signed=dict(case=CASE,state='actual_original_preprocessing_and_WarpControl_entry_passed',records=records,tests=26,signed_zero_and_absent_cases_checked=True,Original_source=str(ORIGINAL),Original_SHA256=sha(ORIGINAL),BasePipeline_source=str(BASE),BasePipeline_SHA256=sha(BASE),body_changed=False,annotations_removed_for_standalone_AST_execution=True,normalization_count=1,background_intensity_after_preprocess=-1,foreground_zero_is_not_absence=True,neural_encoder_intercepted_at_input=True,full_model_inference_run=False,quality_accepted=False,GPU_used=False)
 name=f'CASE{CASE}_NATIVE_FAST5_SIGNED_ORIGINAL_ENTRY_READBACK_v1.json'
 for root in [N,Q]:p=root/'logs'/name;assert not p.exists();p.write_text(json.dumps(signed,indent=2))
 hints=[]
 for family in ['lamp_silhouette','bbox_silhouette']:
  inputs=reader.model_condition_inputs('move',family,'source_hint','unchanged');assert inputs['relative_gamma']==inputs['relative_alpha']==0 and len(inputs['concat_video'])==len(inputs['warp_video'])==49;support=np.stack([reader.condition_frame('move',family,i,'source_hint','unchanged')['support'] for i in range(49)]);captured.clear();process(unit,pipe,warp_video=inputs['warp_video'],warp_mask=None,warped_target_color_mask=inputs['warped_target_color_mask'],relative_gamma=inputs['relative_gamma']);color=captured['warped_target_color_mask'];intensity=captured['warped_target_intentsity_mask'];np.testing.assert_array_equal((color.max(dim=1)[0][0]>-.95).numpy(),support);expected=np.where(support,np.float32(0),np.float32(-1))
  for channel in range(3):np.testing.assert_array_equal(intensity[0,channel].numpy(),expected)
  hints.append(dict(case=CASE,family=family,frames=49,source_hint_control=reader.appearance_reference['control_color_linear_rgb'],no_requested_recolor_or_brightness_change=True,foreground_intensity=0,background_intensity=-1,Original_entry_passed=True,source_color_is_measured=False));del inputs,support,color,intensity,expected;captured.clear();gc.collect()
 hint=dict(case=CASE,state='all49_pure_move_source_hint_controls_pass_actual_Original_entry',records=hints,full_model_inference_run=False,neural_encoder_intercepted_at_input=True,quality_accepted=False,GPU_used=False)
 for root in [N,Q]:p=root/'logs'/f'CASE{CASE}_NATIVE_FAST5_SOURCE_HINT_ORIGINAL_ENTRY_READBACK_v1.json';assert not p.exists();p.write_text(json.dumps(hint,indent=2))
 receipt=dict(case=CASE,state='actual_Original_input_boundary_verified_on_Qinghai_CPU',signed_tests=26,full49_source_hint_families=2,Original_SHA256=sha(ORIGINAL),BasePipeline_SHA256=sha(BASE),reader_source_SHA256=sha(N/'code/realcase_condition_reader.py'),Original_function_bodies_unchanged=True,reader_body_changed=False,normalization_count=1,background_minus1_foreground_zero_distinguished=True,GPU_used=False,no6000D_execution=True,full_model_inference_run=False,quality_accepted=False)
 for root in [N,Q]:p=root/'receipts'/f'CASE{CASE}_QH_ACTUAL_ORIGINAL_ENTRY_v2.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2))
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()

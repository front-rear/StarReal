"""Run official VideoCoF; observe actual completed LoRA additions without changing math."""
import ast,inspect,json,runpy,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from safetensors import safe_open
import videox_fun.utils.lora_utils as lu
args=sys.argv[1:];out=Path(args[args.index('--output_dir')+1]);out.mkdir(parents=True,exist_ok=True)
records=[];active=None
original=lu.merge_lora;source=inspect.getsource(original);tree=ast.parse(source)
def observe(layer,module,up,down,multiplier,alpha):
 active['completed_weight_additions'].append(dict(layer=layer,target_shape=list(module.weight.shape),
  up_shape=list(up.shape),down_shape=list(down.shape),multiplier=float(multiplier),alpha=float(alpha),
  up_max_abs=float(up.abs().max()),down_max_abs=float(down.abs().max())))
class Observer(ast.NodeTransformer):
 def __init__(self):self.count=0
 def visit_AugAssign(self,node):
  if ast.unparse(node.target)=='curr_layer.weight.data' and isinstance(node.op,ast.Add):
   self.count+=1
   call=ast.parse('__root_observe__(layer,curr_layer,weight_up,weight_down,multiplier,alpha)').body[0]
   return [node,ast.copy_location(call,node)]
  return node
visitor=Observer();tree=visitor.visit(tree);ast.fix_missing_locations(tree)
if visitor.count!=2:raise RuntimeError('Upstream LoRA merge structure changed; audit must be reviewed')
namespace=original.__globals__.copy();namespace['__root_observe__']=observe
exec(compile(tree,'<official_merge_lora_with_observers>','exec'),namespace)
observed_function=namespace['merge_lora']
def merge(pipe,lora_path,*a,**kw):
 global active
 with safe_open(lora_path,framework='pt',device='cpu') as f:keys=list(f.keys())
 active=dict(checkpoint=str(lora_path),expected_down_tensors=sum('.lora_down.' in k or '.lora_A.' in k for k in keys),
  completed_weight_additions=[],native_merge_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
  instrumentation='Observer inserted after both native weight += branches; update expressions unchanged')
 records.append(active)
 try:
  result=observed_function(pipe,lora_path,*a,**kw)
  active['completed_layers']=len(active['completed_weight_additions'])
  print('ACTUAL_LORA_ADDITIONS',str(lora_path),active['completed_layers'],'expected down tensors',active['expected_down_tensors'],flush=True)
  if active['expected_down_tensors'] and not active['completed_layers']:raise RuntimeError('No actual LoRA weight additions occurred')
  return result
 finally:(out/'LORA_APPLY_AUDIT.json').write_text(json.dumps(records,indent=2))
# Private compatibility path, after the existing native observer is constructed.
from fusionx_exact_loader_adapter_v1 import make_exact_fusionx_merge
_actual_adapter_path=Path(inspect.getsourcefile(make_exact_fusionx_merge)).resolve()
_actual_adapter_sha=hashlib.sha256(_actual_adapter_path.read_bytes()).hexdigest()
if _actual_adapter_sha!='ce48dac670210434a55efa4778c832952d253cad80380311bc85688210435064':raise RuntimeError('Private adapter code SHA mismatch')
from collections import Counter
_FUSIONX_EXACT_PATH=Path(args[args.index('--acceleration_lora_path')+1]).resolve()
_FUSIONX_SOURCE_SCHEMA='e2934f3cc25dadc329f585ce8c294fa99fd6cd35b2f4c62ca9885fe79ccf8bf7'
_fusionx_completed=[]
def _fusionx_observe(record):
 _fusionx_completed.append(record)
_fusionx_merge=make_exact_fusionx_merge(merge,_FUSIONX_EXACT_PATH,_FUSIONX_SOURCE_SCHEMA,lu.load_file,_fusionx_observe)
def _routed_merge(pipe,lora_path,*a,**kw):
 if Path(lora_path).resolve()!=_FUSIONX_EXACT_PATH:
  return merge(pipe,lora_path,*a,**kw)
 if _fusionx_completed:raise RuntimeError('Unexpected repeated FusionX application')
 failure=None
 try:
  return _fusionx_merge(pipe,lora_path,*a,**kw)
 except Exception as exc:
  failure={'type':type(exc).__name__,'message':str(exc)}
  raise
 finally:
  counts=dict(Counter(x['kind'] for x in _fusionx_completed))
  complete=failure is None and len(_fusionx_completed)==1052 and counts=={'lora':405,'diff_b':447,'diff':200}
  audit={'stage':'FULL_PINNED_FUSIONX_PATCHES_APPLIED' if complete else 'FUSIONX_APPLICATION_INCOMPLETE',
   'checkpoint':str(_FUSIONX_EXACT_PATH),'schema_fingerprint':_FUSIONX_SOURCE_SCHEMA,
   'adapter_path':str(_actual_adapter_path),'adapter_SHA256':_actual_adapter_sha,
   'private_entry_SHA256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'native_original_merge_source_SHA256':hashlib.sha256(source.encode()).hexdigest(),
   'applied_parameter_count':len(_fusionx_completed),'applied_kind_counts':counts,
   'source_tensor_keys_consumed':sum(len(x['source_keys']) for x in _fusionx_completed),
   'same_method_LoRA_routed_to_original_observer':True,'base_checkpoint_full_values_verified':False,
   'quality_accepted':False,'failure':failure,'applied_parameters':_fusionx_completed}
  (out/'FUSIONX_ACTUAL_COMPONENT_LOADING_v1.json').write_text(json.dumps(audit,indent=2))
  if failure is None and not complete:raise RuntimeError('FusionX actual patch coverage incomplete')
lu.merge_lora=_routed_merge


# Apply exactly the single production placement change to the verified upstream source.
upstream = Path.cwd() / 'fast_infer.py'
base = upstream.read_bytes()
if hashlib.sha256(base).hexdigest() != '7a03505ef28e83a0364553fc79002bd84227256e406c01140d76801b8601a689':
    raise RuntimeError('VideoCoF source checksum mismatch')
before = b'GPU_memory_mode = "sequential_cpu_offload"'
after = b'GPU_memory_mode = "model_full_load"'
if base.count(before) != 1:
    raise RuntimeError('VideoCoF upstream memory mode changed')
resident = base.replace(before, after)
if hashlib.sha256(resident).hexdigest() != '5da8866940827a421ef30f14b3b72c844f9afaf8d2e4d79de725a7cbf426e79c':
    raise RuntimeError('VideoCoF production placement checksum mismatch')
exec(compile(resident, str(upstream), 'exec'), {'__file__': str(upstream), '__name__': '__main__'})

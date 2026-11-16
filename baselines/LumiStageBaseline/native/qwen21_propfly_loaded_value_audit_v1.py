"""Read-only value audit used by the external native loader observer.

Current CPU use is restricted to tiny fake modules/checkpoints. A future Root
reviewed run may call this on actual native returned modules before offload.
No module loading, casting in place, key renaming, or sampling occurs here.
"""
from pathlib import Path
import contextlib,hashlib,importlib.util,json,math
import torch
from safetensors import safe_open

CORE_SHA256='4a59682ec5676d4749cdb4035f29a6c6a93706ee1d1c1e70d3e2309ac2b1c1dc'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_core(path,expected_sha256):
 if sha(path)!=expected_sha256:raise ValueError('Pinned observer core differs')
 spec=importlib.util.spec_from_file_location('qwen21_private_readonly_core',path)
 core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core);return core
def unique(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('Duplicate mapping key: '+k)
  d[k]=v
 return d
def unwrap_native_loading_info(value,component):
 """Future loader shim must restore native model-only return after capture."""
 if not isinstance(value,tuple) or len(value)!=2 or not isinstance(value[1],dict):
  raise ValueError('Expected exact native (model, loading_info) tuple: '+component)
 model,info=value
 for name in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']:
  if name not in info or info[name]:raise ValueError('Nonempty/absent native loading_info '+component+': '+name)
 return model,{'component':component,'actual_loading_info':info,'native_tuple_unwrapped':True,'model_identity_preserved':True}
def validate_state(module,expected_shapes,allowed_aliases=None):
 actual=module.state_dict(keep_vars=True);params=dict(module.named_parameters(remove_duplicate=False))
 aliases=allowed_aliases or {};resolved={}
 if set(expected_shapes)-set(actual):raise ValueError('Missing actual checkpoint keys')
 for key in set(actual)-set(expected_shapes):
  src=aliases.get(key)
  if src not in expected_shapes or key not in params or src not in params or params[key] is not params[src]:
   raise ValueError('Unexpected actual state, not exact explicit Parameter alias: '+key)
  if actual[key].shape!=actual[src].shape or actual[key].dtype!=actual[src].dtype:raise ValueError('Alias dtype/shape differs')
  resolved[key]={'source_checkpoint_key':src,'same_Parameter_object':True}
 for k,v in actual.items():
  if v.device.type=='meta' or v.layout!=torch.strided or v.is_quantized:raise ValueError('Meta/unsupported state: '+k)
  if k in expected_shapes and list(v.shape)!=expected_shapes[k]:raise ValueError('Shape mismatch: '+k)
 for k,v in module.named_buffers(remove_duplicate=False):
  if v.device.type=='meta':raise ValueError('Meta buffer: '+k)
 return actual,resolved
def compare_one(actual,source_slice,source_shape,source_dtype_bytes,core,chunk_bytes):
 ha,hb=hashlib.sha256(),hashlib.sha256();count=0;chunks=0
 maximum=max(1,chunk_bytes//max(source_dtype_bytes,actual.element_size()))
 for slices in core._bounded_slices(source_shape,maximum):
  src=source_slice(slices)
  expected=src.detach().to(device='cpu',dtype=actual.dtype).contiguous()
  got=(actual[slices] if source_shape else actual).detach().to('cpu').contiguous()
  a=got.reshape(-1).view(torch.uint8);b=expected.reshape(-1).view(torch.uint8)
  if max(a.numel(),b.numel())>chunk_bytes:raise ValueError('Chunk exceeds bound')
  if not torch.equal(a,b):raise ValueError('Actual loaded values differ after independent native dtype cast')
  ha.update(a.numpy().tobytes());hb.update(b.numpy().tobytes());count+=got.numel();chunks+=1
 if count!=math.prod(source_shape):raise ValueError('Element coverage differs')
 return {'elements_compared':count,'chunks_compared':chunks,'all_values_bitwise_equal_after_cast':True,'actual_cast_bytes_SHA256':ha.hexdigest(),'source_cast_bytes_SHA256':hb.hexdigest()}
def verify_sharded_module(module,component,folder,allowed_files,core,*,index_path=None,allowed_aliases=None,chunk_bytes=8<<20):
 """Every explicit pinned shard/header/index key and every actual value; no remap."""
 if chunk_bytes<8:raise ValueError('Invalid chunk size')
 folder=Path(folder);files=list(allowed_files)
 if not files or len(files)!=len(set(files)) or any(Path(f).name!=f for f in files):raise ValueError('Nonexact shard list')
 headers={f:core.read_safetensors_header(folder/f) for f in files}
 union={}
 for f,header in headers.items():
  for k,spec in header.items():
   if k in union:raise ValueError('Duplicate key across checkpoint shards: '+k)
   union[k]=(f,spec)
 if not union:raise ValueError('Empty checkpoint')
 if index_path is not None:
  wm=json.loads(Path(index_path).read_text(),object_pairs_hook=unique)['weight_map']
  if set(wm)!=set(union) or set(wm.values())!=set(files):raise ValueError('Index/header/file coverage differs')
  if any(wm[k]!=union[k][0] for k in union):raise ValueError('Index names incorrect shard')
 actual,aliases=validate_state(module,{k:z[1]['shape'] for k,z in union.items()},allowed_aliases)
 rows=[]
 with contextlib.ExitStack() as stack:
  handles={f:stack.enter_context(safe_open(str(folder/f),framework='pt',device='cpu')) for f in files}
  for k,(f,spec) in sorted(union.items()):
   if set(handles[f].keys())!=set(headers[f]):raise ValueError('Handle/header key coverage differs')
   h=handles[f];sl=h.get_slice(k);shape=spec['shape']
   if list(sl.get_shape())!=shape:raise ValueError('Header/slice shape differs')
   nbytes=core._SOURCE_ELEMENT_BYTES.get(spec['dtype'])
   if nbytes is None:raise ValueError('Unsupported source dtype')
   row=compare_one(actual[k],lambda s:sl[s] if shape else h.get_tensor(k),shape,nbytes,core,chunk_bytes)
   row.update(key=k,shard=f,shape=shape,source_dtype=spec['dtype'],actual_dtype=str(actual[k].dtype),actual_device=str(actual[k].device));rows.append(row)
 return {'component':component,'status':'ALL_ACTUAL_SHARDED_VALUES_VERIFIED','tensor_count':len(rows),'tensor_rows':rows,'actual_state_key_count':len(actual),'exact_explicit_aliases':aliases,'actual_values_changed':False}
def verify_native_captured_state(module,source_state,component,core,*,allowed_aliases=None,chunk_bytes=8<<20):
 """PropFly .bin: compare against ORIGINAL native torch.load return in memory.

 Caller must retain it unchanged, record strict=True loading return, identity
 and original pinned receipt. This function does not deserialize .bin again.
 """
 if not source_state or not all(torch.is_tensor(v) for v in source_state.values()):raise ValueError('Not exact native tensor state dict')
 actual,aliases=validate_state(module,{k:list(v.shape) for k,v in source_state.items()},allowed_aliases)
 rows=[]
 for k,src in sorted(source_state.items()):
  if src.device.type=='meta' or src.layout!=torch.strided or src.is_quantized:raise ValueError('Unsupported captured source')
  shape=list(src.shape)
  row=compare_one(actual[k],lambda s:src[s] if shape else src,shape,src.element_size(),core,chunk_bytes)
  row.update(key=k,shape=shape,source_dtype=str(src.dtype),actual_dtype=str(actual[k].dtype),actual_device=str(actual[k].device));rows.append(row)
 return {'component':component,'status':'ALL_ACTUAL_VALUES_EQUAL_UNCHANGED_NATIVE_CAPTURE','tensor_count':len(rows),'tensor_rows':rows,'exact_explicit_aliases':aliases,'actual_values_changed':False,'checkpoint_deserialized_by_observer':False}

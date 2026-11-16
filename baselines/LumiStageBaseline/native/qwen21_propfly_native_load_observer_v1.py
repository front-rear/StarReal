"""External native loader observer. It does not assign/cast/load any model state.

The original class loaders receive unchanged arguments except the explicitly
recorded metadata-only output_loading_info=True. Their native return contract
is restored: a caller asking for model gets that exact model, a caller asking
for a tuple retains that exact tuple. Original strict method state is passed
to the original load_state_dict exactly once, without cloning/renaming/filling.
"""
from pathlib import Path
import contextlib,hashlib,inspect,json,math
import torch
from qwen21_propfly_loaded_value_audit_v1 import (
 load_core,CORE_SHA256,unwrap_native_loading_info,verify_sharded_module,
 verify_native_captured_state,validate_state)

def jsonable(x):
 if isinstance(x,(str,int,float,bool)) or x is None:return x
 if isinstance(x,Path):return str(x)
 if isinstance(x,(tuple,list)):return [jsonable(t) for t in x]
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 return str(x)
def verify_after_native_offload(module,proof,core,chunk_bytes=8<<20,aliases=None):
 rows=proof['tensor_rows'];state,resolved=validate_state(module,{z['key']:z['shape'] for z in rows},aliases)
 reports=[]
 for z in rows:
  value=state[z['key']]
  if str(value.dtype)!=z['actual_dtype']:raise ValueError('Native offload changed dtype: '+z['key'])
  h=hashlib.sha256();count=0
  for sl in core._bounded_slices(z['shape'],max(1,chunk_bytes//value.element_size())):
   a=(value[sl] if z['shape'] else value).detach().to('cpu').contiguous()
   b=a.reshape(-1).view(torch.uint8)
   if b.numel()>chunk_bytes:raise ValueError('Offload CPU chunk exceeds bound')
   h.update(b.numpy().tobytes());count+=a.numel()
  if count!=math.prod(z['shape']) or h.hexdigest()!=z['source_cast_bytes_SHA256']:
   raise ValueError('Native offload changed actual values: '+z['key'])
  reports.append({'key':z['key'],'elements_compared':count,'dtype':str(value.dtype),'actual_device':str(value.device),'after_native_offload_SHA256':h.hexdigest(),'same_as_independent_checkpoint_cast_SHA256':True})
 return {'status':'NATIVE_OFFLOAD_ALL_ACTUAL_VALUES_UNCHANGED','tensor_rows':reports,'exact_aliases':resolved,'observer_called_model_to':False}

class NativeLoadObserver:
 def __init__(self,manifest,output,kind,classes,pipeline_class):
  self.m=manifest;self.output=Path(output);self.kind=kind;self.classes=classes;self.pipeline_class=pipeline_class
  self.core=load_core(Path(__file__).with_name('mfm_loaded_tensor_value_audit_v1.py'),CORE_SHA256)
  self.loaded={};self.restore=[];self.pipeline=None;self.captured_method=None;self.method_strict_count=0;self.method_source_load_count=0
  self.report={'stage':kind,'status':'STARTED_NOT_LOAD_PROOF','native_loader_events':[],'components':{},'actual_values_changed_by_observer':False,'observer_called_model_to':False,'observer_loaded_checkpoint_into_model':False,'sampler_or_writer_modified':False}
  self.report['actual_runtime_class_sources']=[{'role':c,'class':t.__name__,'path':inspect.getfile(t),'SHA256':hashlib.sha256(Path(inspect.getfile(t)).read_bytes()).hexdigest()} for c,t in [*classes.items(),('pipeline',pipeline_class)]]
 def save(self):
  name='QWEN21_ACTUAL_COMPONENT_LOADING_VALUES_v1.json' if self.kind=='qwen21' else 'PROPFly_ACTUAL_COMPONENT_LOADING_VALUES_v1.json'
  p=self.output/name;p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.private_tmp');q.write_text(json.dumps(jsonable(self.report),indent=2)+'\n');q.replace(p)
 def aliases(self,component):
  if self.kind=='propfly' and component=='text_encoder':
   return {'encoder.embed_tokens.weight':'shared.weight','shared.weight':'encoder.embed_tokens.weight'}
  return {}
 def location(self,component):
  if self.kind=='qwen21':
   base=Path(self.m['qwen']['local_model_dir']);allfiles=[z['file'] for z in self.m['qwen']['required_files']]
  else:
   base=Path(self.m['propfly']['base_model']);allfiles=[z['relative_path'] for z in self.m['observer']['propfly_base_runtime_files']]
  files=[str(Path(f).name) for f in allfiles if f.startswith(component+'/') and f.endswith('.safetensors')]
  indices=[base/f for f in allfiles if f.startswith(component+'/') and f.endswith('.safetensors.index.json')]
  if not files or len(indices)>1:raise ValueError('Missing/excess explicit native component shards: '+component)
  return base/component,files,indices[0] if indices else None
 def verify(self,module,component):
  folder,files,index=self.location(component)
  return verify_sharded_module(module,component,folder,files,self.core,index_path=index,allowed_aliases=self.aliases(component))
 def patch_classmethod(self,cls,name,replacement):
  own=name in cls.__dict__;old=cls.__dict__.get(name);original=getattr(cls,name)
  self.restore.append(lambda: setattr(cls,name,old) if own else delattr(cls,name))
  setattr(cls,name,classmethod(replacement));return original
 def observe_loader(self,component,cls):
  original=getattr(cls,'from_pretrained')
  def wrapper(actual_cls,*args,**kwargs):
   if component in self.loaded:raise ValueError('Unexpected duplicate native loader: '+component)
   native_kwargs=dict(kwargs);native_kwargs['output_loading_info']=True
   value=original(*args,**native_kwargs)
   module,info=unwrap_native_loading_info(value,component)
   if not isinstance(module,cls):raise ValueError('Native loader returned wrong model class: '+component)
   event={'component':component,'class':actual_cls.__name__,'native_original_args':jsonable(args),'native_original_kwargs':jsonable(kwargs),'metadata_flag_added':'output_loading_info=True','original_requested_output_loading_info':bool(kwargs.get('output_loading_info',False)),'actual_loading_info':info['actual_loading_info'],'actual_returned_model_id':id(module),'native_return_contract_restored':True}
   self.report['native_loader_events'].append(event);self.loaded[component]=module
   self.report['components'][component]={'loader_return_values':self.verify(module,component)}
   self.save()
   return value if kwargs.get('output_loading_info',False) else module
  self.patch_classmethod(cls,'from_pretrained',wrapper)
 def final_pipeline(self,pipe,kwargs):
  if set(self.loaded)!={'transformer','vae','text_encoder'}:raise ValueError('Native pipeline did not load all three components')
  self.pipeline=pipe
  for c,module in self.loaded.items():
   if getattr(pipe,c,None) is not module:raise ValueError('Pipeline changed returned component identity: '+c)
   if c in kwargs and kwargs[c] is not module:raise ValueError('Pipeline received wrong native shared component: '+c)
   if self.kind=='propfly' and c=='transformer':
    if self.captured_method is None or self.method_strict_count!=1:raise ValueError('Missing exact one native strict PropFly method load')
    proof=verify_native_captured_state(module,self.captured_method,c,self.core)
   else:proof=self.verify(module,c)
   self.report['components'][c]['final_pipeline_values']=proof
   self.report['components'][c]['same_native_returned_module_in_pipeline']=True
  self.report['status']='ALL_NATIVE_FINAL_PIPELINE_VALUES_VERIFIED_BEFORE_OFFLOAD';self.save()
 def install(self):
  for component,cls in self.classes.items():self.observe_loader(component,cls)
  original=getattr(self.pipeline_class,'from_pretrained')
  def pipeline_loader(cls,*args,**kwargs):
   pipe=original(*args,**kwargs);self.final_pipeline(pipe,kwargs);return pipe
  self.patch_classmethod(self.pipeline_class,'from_pretrained',pipeline_loader)
  old_offload=self.pipeline_class.enable_model_cpu_offload
  def offload(pipe,*args,**kwargs):
   if pipe is not self.pipeline:raise ValueError('Offload on different native pipeline object')
   result=old_offload(pipe,*args,**kwargs)
   for c,module in self.loaded.items():
    if getattr(pipe,c,None) is not module:raise ValueError('Offload changed native component identity')
    proof=self.report['components'][c]['final_pipeline_values']
    self.report['components'][c]['after_native_offload']=verify_after_native_offload(module,proof,self.core,aliases=self.aliases(c))
    self.report['components'][c]['native_offload_hook_class']=type(getattr(module,'_hf_hook',None)).__name__
   if self.kind=='propfly':
    # Only the observer's retained reference is released. The original native
    # dictionary was never edited, transferred wholesale, or passed again.
    self.captured_method=None
    self.report['native_method_capture_released_after_all_final_offload_values_verified_before_sampling']=True
   self.report['status']='ALL_NATIVE_LOAD_AND_FINAL_CAST_VALUES_OFFLOAD_VERIFIED';self.save();return result
  self.restore.append(lambda:setattr(self.pipeline_class,'enable_model_cpu_offload',old_offload));self.pipeline_class.enable_model_cpu_offload=offload
  if self.kind=='propfly':self.install_native_method_capture()
 def install_native_method_capture(self):
  original_load=torch.load;method=Path(self.m['propfly']['method_checkpoint']).resolve()
  def native_torch_load(*args,**kwargs):
   value=original_load(*args,**kwargs)
   path=Path(args[0]).resolve() if args and isinstance(args[0],(str,Path)) else None
   if path==method:
    self.method_source_load_count+=1
    if self.method_source_load_count!=1:raise ValueError('Duplicate native PropFly method torch.load')
    if not isinstance(value,dict):raise ValueError('Native method is not state dictionary')
    self.captured_method=value
    self.report['native_method_source']={'path':str(path),'original_torch_load_kwargs':jsonable(kwargs),'same_original_returned_dict_retained':True,'observer_deserialized_again':False,'native_state_dict_id':id(value)}
   return value
  self.restore.append(lambda:setattr(torch,'load',original_load));torch.load=native_torch_load
  original_state=torch.nn.Module.load_state_dict
  def native_state_load(module,state_dict,*args,**kwargs):
   is_target=module is self.loaded.get('transformer')
   if is_target:
    strict=kwargs.get('strict',args[0] if args else True)
    assign=kwargs.get('assign',args[1] if len(args)>1 else False)
    if strict is not True or assign is not False or state_dict is not self.captured_method:raise ValueError('Native PropFly method strict/state identity differs')
   result=original_state(module,state_dict,*args,**kwargs)
   if is_target:
    if result.missing_keys or result.unexpected_keys:raise ValueError('Native strict load returned missing/extra keys')
    self.method_strict_count+=1
    if self.method_strict_count!=1:raise ValueError('Repeated native method load')
    self.report['native_method_strict_load']={'strict':True,'assign':False,'exact_native_dict_passed_unchanged':True,'missing_keys':list(result.missing_keys),'unexpected_keys':list(result.unexpected_keys),'observer_called_extra_load_state_dict':False}
    self.report['method_post_strict_values']=verify_native_captured_state(module,state_dict,'transformer_method',self.core);self.save()
   return result
  self.restore.append(lambda:setattr(torch.nn.Module,'load_state_dict',original_state));torch.nn.Module.load_state_dict=native_state_load
 def __enter__(self):
  try:self.install()
  except BaseException:
   for action in reversed(self.restore):action()
   raise
  return self
 def __exit__(self,typ,value,tb):
  for action in reversed(self.restore):action()
  if typ is not None:self.report.update(status='FAILED_NATIVE_LOAD_OBSERVER',error_type=typ.__name__,error=str(value));self.save()
  elif self.report['status']!='ALL_NATIVE_LOAD_AND_FINAL_CAST_VALUES_OFFLOAD_VERIFIED':
   self.report['status']='FAILED_INCOMPLETE_NATIVE_OBSERVER';self.save();raise ValueError('Native observer did not observe complete offloaded pipeline')
  return False

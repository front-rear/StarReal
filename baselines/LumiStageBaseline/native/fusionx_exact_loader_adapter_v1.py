"""Private FusionX-format compatibility adapter; upstream VideoCoF stays intact.

The publisher uses ComfyUI's generic LoRA + diff/diff_b format. The CPU
admission report pins the exact 1457-key FusionX artifact and base schema.
Only an explicitly selected FusionX path is intercepted; method LoRA calls
retain the original function, arguments and return object.
"""
from pathlib import Path
import hashlib,json
import torch

PREFIX='diffusion_model.'

def schema_of(state):
    return {k:{'shape':list(v.shape),'dtype':str(v.dtype).removeprefix('torch.')} for k,v in state.items()}

def schema_fingerprint(schema):
    types={'float16':'F16','bfloat16':'BF16','float32':'F32','float64':'F64',
           'int64':'I64','int32':'I32','int16':'I16','int8':'I8','uint8':'U8','bool':'BOOL'}
    shape={k:{'shape':list(v['shape']),'dtype':types.get(v['dtype'],v['dtype'])}
           for k,v in schema.items() if k!='__metadata__'}
    return hashlib.sha256(json.dumps(shape,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def build_plan(source_schema,base_schema):
    """Validate all source keys and target shapes before any parameter write."""
    src={k:v for k,v in source_schema.items() if k!='__metadata__'}
    consumed=set();plan=[];targets=set()
    for key in sorted(src):
        if key in consumed:continue
        if not key.startswith(PREFIX):raise ValueError('Unsupported FusionX key prefix: '+key)
        leaf=key[len(PREFIX):]
        if leaf.endswith('.lora_up.weight'):
            down=key[:-len('.lora_up.weight')]+'.lora_down.weight'
            if down not in src:raise ValueError('Missing down tensor: '+key)
            continue
        if leaf.endswith('.lora_down.weight'):
            stem=leaf[:-len('.lora_down.weight')]
            up=PREFIX+stem+'.lora_up.weight'
            if up not in src:raise ValueError('Missing up tensor: '+key)
            ds,us=src[key]['shape'],src[up]['shape']
            if len(ds)!=2 or len(us)!=2 or us[1]!=ds[0]:raise ValueError('Invalid rank/dimensions: '+key)
            target=stem+'.weight';shape=[us[0],ds[1]];kind='lora';keys=[key,up]
        elif leaf.endswith('.diff_b'):
            target=leaf[:-len('.diff_b')]+'.bias';shape=src[key]['shape'];kind='diff_b';keys=[key]
        elif leaf.endswith('.diff'):
            target=leaf[:-len('.diff')]+'.weight';shape=src[key]['shape'];kind='diff';keys=[key]
        else:raise ValueError('Unsupported FusionX entry: '+key)
        if target not in base_schema:raise ValueError('Target parameter absent: '+target)
        if list(base_schema[target]['shape'])!=list(shape):raise ValueError('Target shape mismatch: '+target)
        if target in targets:raise ValueError('Ambiguous duplicate target: '+target)
        targets.add(target);consumed.update(keys)
        plan.append(dict(kind=kind,target=target,source_keys=keys,shape=list(shape)))
    if consumed!=set(src):raise ValueError('Unconsumed FusionX keys: '+str(sorted(set(src)-consumed)))
    return plan

def apply_fusionx_state(pipeline,state,multiplier,device='cpu',dtype=torch.float32,
                       audit_callback=None,expected_schema_fingerprint=None):
    """Apply the publisher's exact generic-format patches without name guessing.

    Native merge uses float32 arithmetic and per-layer return to original
    dtype/device. Keep that arithmetic for existing 400 blocks and the five
    previously unreachable Linear modules. diff/diff_b are additive deltas,
    as explicitly specified by ComfyUI's load_lora/calculate_weight.
    """
    if expected_schema_fingerprint is None:raise ValueError('Exact source schema pin is required')
    source=schema_of(state)
    if schema_fingerprint(source)!=expected_schema_fingerprint:raise ValueError('Source schema pin mismatch')
    model=pipeline.transformer
    # Match the native sequential-offload lifecycle; never create a new model.
    offloaded=model.device==torch.device(type='meta')
    offload_device=None
    if offloaded:
        offload_device=pipeline._offload_device
        pipeline.remove_all_hooks()
    try:
        params=dict(model.named_parameters())
        base={k:{'shape':list(v.shape)} for k,v in params.items()}
        plan=build_plan(source,base)
        for op in plan:
            param=params[op['target']]
            if param.is_meta:raise ValueError('Meta parameter was not restored: '+op['target'])
        # All shape/key checks precede the first write. No new bias/parameter,
        # resized module, RNG, model forward or inference setting is introduced.
        with torch.no_grad():
            for op in plan:
                param=params[op['target']];original_dtype=param.dtype;original_device=param.device
                before=param.detach().to(device=device,dtype=dtype,copy=True)
                if op['kind']=='lora':
                    down=state[op['source_keys'][0]].to(device=device,dtype=dtype)
                    up=state[op['source_keys'][1]].to(device=device,dtype=dtype)
                    delta=multiplier*1.0*torch.mm(up,down)  # no alpha keys in this pinned artifact
                else:
                    delta=multiplier*state[op['source_keys'][0]].to(device=device,dtype=dtype)
                expected=(before+delta).to(device=original_device,dtype=original_dtype)
                param.copy_(expected)
                exact=torch.equal(param.detach(),expected)
                if not exact:raise RuntimeError('Post-copy value mismatch: '+op['target'])
                if audit_callback is not None:
                    audit_callback(dict(**op,source_tensor_shapes=[list(state[k].shape) for k in op['source_keys']],
                        target_dtype=str(original_dtype),target_device=str(original_device),expected_equals_actual_after=exact,
                        arithmetic_dtype=str(dtype),strength=float(multiplier)))
                del before,delta,expected
        return pipeline
    finally:
        if offloaded:pipeline.enable_sequential_cpu_offload(device=offload_device)

def make_exact_fusionx_merge(original,fusionx_path,expected_schema_fingerprint,
                            source_loader,audit_callback=None):
    """Wrap only the explicit FusionX artifact. All other calls pass unchanged."""
    chosen=Path(fusionx_path).resolve()
    def merge(pipeline,lora_path,multiplier,device='cpu',dtype=torch.float32,state_dict=None,
              transformer_only=False,sub_transformer_name='transformer'):
        if Path(lora_path).resolve()!=chosen:
            return original(pipeline,lora_path,multiplier,device=device,dtype=dtype,state_dict=state_dict,
                            transformer_only=transformer_only,sub_transformer_name=sub_transformer_name)
        if transformer_only or sub_transformer_name!='transformer':
            raise ValueError('Only the reviewed native FusionX transformer call is supported')
        state=source_loader(lora_path) if state_dict is None else state_dict
        return apply_fusionx_state(pipeline,state,multiplier,device=device,dtype=dtype,
            audit_callback=audit_callback,expected_schema_fingerprint=expected_schema_fingerprint)
    return merge

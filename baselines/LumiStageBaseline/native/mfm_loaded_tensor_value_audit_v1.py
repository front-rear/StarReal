"""Read-only, bounded CPU comparison of actual returned tensors to safetensors.

This is an external observer helper. It never loads weights into a model, changes
values, fills missing keys, converts names, or executes a model forward.
"""
from pathlib import Path
import contextlib
import hashlib
import itertools
import json
import math
import struct

import torch
from safetensors import safe_open


class LoadedValueAdmissionError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate safetensors header key: ' + key)
        result[key] = value
    return result


def read_safetensors_header(path):
    """Header only; opening a real checkpoint is reserved for a future run."""
    path = Path(path)
    with path.open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('Incomplete safetensors prefix')
        size = struct.unpack('<Q', prefix)[0]
        if size > 64 << 20 or size < 2:
            raise ValueError('Unreasonable safetensors header length')
        raw = f.read(size)
        if len(raw) != size:
            raise ValueError('Incomplete safetensors header')
    header = json.loads(raw, object_pairs_hook=_unique_object)
    return {k: v for k, v in header.items() if k != '__metadata__'}


def validate_loaded_state(model, checkpoint, component, *, allowed_aliases=None,
                          expected_tensor_count=None):
    """Inspect the complete actual state and exact explicit alias policy."""
    aliases = dict(allowed_aliases or {})
    header = read_safetensors_header(checkpoint)
    state = model.state_dict(keep_vars=True)
    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    report = {
        'component': component, 'checkpoint': str(Path(checkpoint).resolve()),
        'checkpoint_tensor_count': len(header), 'actual_state_key_count': len(state),
        'missing_actual_checkpoint_keys': sorted(set(header) - set(state)),
        'unexpected_actual_state_keys': [], 'shape_mismatches': {},
        'meta_or_unsupported_state_keys': [], 'exact_aliases': {},
        'state_rows': [], 'nonpersistent_buffer_rows': [],
        'values_compared': False, 'actual_values_changed': False,
    }
    for key in sorted(set(state) - set(header)):
        source = aliases.get(key)
        if (source in header and key in params and source in params and
                params[key] is params[source] and
                list(state[key].shape) == list(state[source].shape) and
                state[key].dtype == state[source].dtype):
            report['exact_aliases'][key] = {
                'source_checkpoint_key': source, 'same_Parameter_object': True,
                'shape': list(state[key].shape), 'dtype': str(state[key].dtype),
            }
        else:
            report['unexpected_actual_state_keys'].append(key)
    for key, value in sorted(state.items()):
        report['state_rows'].append({
            'key': key, 'shape': list(value.shape), 'dtype': str(value.dtype),
            'device': str(value.device), 'parameter': key in params,
            'persistent_buffer': key in buffers,
        })
        if value.device.type == 'meta' or value.layout != torch.strided or value.is_quantized:
            report['meta_or_unsupported_state_keys'].append(key)
        if key in header and list(value.shape) != header[key]['shape']:
            report['shape_mismatches'][key] = {
                'checkpoint': header[key]['shape'], 'actual': list(value.shape)}
    for key in sorted(set(buffers) - set(state)):
        value = buffers[key]
        report['nonpersistent_buffer_rows'].append({
            'key': key, 'shape': list(value.shape), 'dtype': str(value.dtype),
            'device': str(value.device), 'checkpoint_value_claimed': False})
        if value.device.type == 'meta':
            report['meta_or_unsupported_state_keys'].append('nonpersistent_buffer:' + key)
    if not header:
        raise LoadedValueAdmissionError('Reject empty checkpoint', report)
    if expected_tensor_count is not None and len(header) != expected_tensor_count:
        report['expected_checkpoint_tensor_count'] = expected_tensor_count
        raise LoadedValueAdmissionError('Checkpoint header tensor count differs', report)
    if any(report[k] for k in ['missing_actual_checkpoint_keys', 'unexpected_actual_state_keys',
                              'shape_mismatches', 'meta_or_unsupported_state_keys']):
        raise LoadedValueAdmissionError('Actual state has incomplete or mismatched checkpoint coverage', report)
    return state, header, report


def _bounded_slices(shape, max_elements):
    """Non-overlapping rectangular blocks, including very large trailing axes."""
    if not shape:
        yield ()
        return
    if any(x == 0 for x in shape):
        return
    block = list(shape)
    while math.prod(block) > max_elements:
        axis = next((i for i, size in enumerate(block) if size > 1), None)
        if axis is None:
            raise ValueError('Cannot bound tensor slice')
        block[axis] = (block[axis] + 1) // 2
    for starts in itertools.product(*(range(0, size, step) for size, step in zip(shape, block))):
        yield tuple(slice(start, min(start + step, size))
                    for start, step, size in zip(starts, block, shape))


_SOURCE_ELEMENT_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
    'F16': 2, 'BF16': 2, 'I16': 2, 'U16': 2,
    'F32': 4, 'I32': 4, 'U32': 4, 'F64': 8, 'I64': 8, 'U64': 8,
}


def verify_loaded_model(model, checkpoint, component, *, allowed_aliases=None,
                        expected_tensor_count=None, chunk_bytes=8 << 20,
                        progress_callback=None):
    """Compare every checkpoint value after an independent native dtype cast.

    The current model is read only. No load_state_dict, .to(), copying into a
    parameter, or production backend/sampler substitution occurs here.
    """
    if chunk_bytes < 8:
        raise ValueError('chunk_bytes must hold at least one element')
    state, header, report = validate_loaded_state(
        model, checkpoint, component, allowed_aliases=allowed_aliases,
        expected_tensor_count=expected_tensor_count)
    report.update(comparison_chunk_bytes=chunk_bytes, tensor_rows=[],
        comparison='Every fixed safetensors value independently cast onCPU to the actual returned tensor dtype, then bounded CPU bytewise equality and SHA256. No parameter is changed.')
    with safe_open(str(checkpoint), framework='pt', device='cpu') as handle:
        if set(handle.keys()) != set(header):
            raise LoadedValueAdmissionError('Safetensors handle/header keys disagree', report)
        for key in sorted(header):
            tensor = state[key]
            spec = header[key]
            shape = spec['shape']
            source_element_bytes = _SOURCE_ELEMENT_BYTES.get(spec['dtype'])
            if source_element_bytes is None:
                raise LoadedValueAdmissionError('Unsupported source dtype: ' + spec['dtype'], report)
            maximum = max(1, chunk_bytes // max(source_element_bytes, tensor.element_size()))
            source_slice = handle.get_slice(key)
            if list(source_slice.get_shape()) != shape:
                raise LoadedValueAdmissionError('Safetensors shape differs from header: ' + key, report)
            digest_actual, digest_expected = hashlib.sha256(), hashlib.sha256()
            chunks, elements = 0, 0
            for slices in _bounded_slices(shape, maximum):
                source = source_slice[slices] if shape else handle.get_tensor(key)
                expected = source.to(device='cpu', dtype=tensor.dtype).contiguous()
                actual = (tensor[slices] if shape else tensor).detach().to('cpu').contiguous()
                a = actual.reshape(-1).view(torch.uint8)
                b = expected.reshape(-1).view(torch.uint8)
                if a.numel() > chunk_bytes or b.numel() > chunk_bytes:
                    raise LoadedValueAdmissionError('CPU chunk exceeded requested bound', report)
                equal = torch.equal(a, b)
                digest_actual.update(a.numpy().tobytes())
                digest_expected.update(b.numpy().tobytes())
                chunks += 1
                elements += actual.numel()
                if not equal:
                    report['failed_tensor'] = {'key': key, 'shape': shape,
                        'actual_dtype': str(tensor.dtype), 'source_dtype': spec['dtype'],
                        'failed_block': [[x.start, x.stop] for x in slices],
                        'actual_compared_prefix_sha256': digest_actual.hexdigest(),
                        'source_cast_compared_prefix_sha256': digest_expected.hexdigest()}
                    raise LoadedValueAdmissionError('Actual tensor values differ from fixed checkpoint: ' + key, report)
            if elements != math.prod(shape):
                raise LoadedValueAdmissionError('Did not visit every tensor element: ' + key, report)
            report['tensor_rows'].append({
                'key': key, 'shape': shape, 'source_dtype': spec['dtype'],
                'actual_dtype': str(tensor.dtype), 'actual_device': str(tensor.device),
                'chunks_compared': chunks, 'elements_compared': elements,
                'all_values_bitwise_equal_after_cast': True,
                'actual_cast_bytes_sha256': digest_actual.hexdigest(),
                'checkpoint_cast_bytes_sha256': digest_expected.hexdigest(),
            })
            if progress_callback:
                progress_callback({'component': component, 'completed_checkpoint_tensors': len(report['tensor_rows']),
                                   'total_checkpoint_tensors': len(header), 'last_key': key})
    report.update(status='ACTUAL_ALL_FIXED_CHECKPOINT_VALUES_BITWISE_VERIFIED',
                  values_compared=True, all_checkpoint_values_compared=True)
    return report

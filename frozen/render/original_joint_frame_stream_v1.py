"""Bounded-memory adapter: execute the unchanged Original joint_frames body.

Only file enumeration and frame IO are sliced. Original palette, blend, rounding,
wire precedence and native704 pixels remain unchanged. No scene renderer changes.
"""
from pathlib import Path
import ast, hashlib, json
import numpy as np
from PIL import Image


class OriginalJointStream:
    def __init__(self, original_path):
        self.original_path = Path(original_path)
        tree = ast.parse(self.original_path.read_text())
        joint = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'joint_frames')
        self.original_joint_AST_SHA256 = hashlib.sha256(ast.dump(joint, include_attributes=False).encode()).hexdigest()
        self.original_file_SHA256 = hashlib.sha256(self.original_path.read_bytes()).hexdigest()
        self.indices = None
        owner = self

        class SlicePath(type(Path())):
            def glob(self, pattern):
                paths = sorted((p for p in super().glob(pattern) if not p.name.startswith('._')), key=lambda p: int(p.stem))
                assert len(paths) == 49 and [int(p.stem) for p in paths] == list(range(49))
                assert owner.indices is not None
                return iter([paths[i] for i in owner.indices])

        ns = {'__name__': 'original_joint_stream_reference', 'Path': SlicePath, 'np': np, 'Image': Image, 'json': json}
        resolve = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resolve')
        exec(compile(ast.Module(body=[resolve, joint], type_ignores=[]), str(self.original_path), 'exec'), ns)
        self.ns = ns

        def frames(path, gray=False):
            data = []
            for i in self.indices:
                with Image.open(Path(path) / f'{i:06d}.png') as im:
                    assert im.size == (1280, 704), 'Never resize spatial fields in streaming adapter'
                    data.append(np.asarray(im.convert('L' if gray else 'RGB')).copy())
            value = np.stack(data)
            return value[..., None].astype(np.float32) / 255 if gray else value

        ns.update(frames=frames, background_frames=frames)

    def read(self, folder, indices):
        self.indices = list(indices)
        assert self.indices and len(set(self.indices)) == len(self.indices) and all(0 <= i < 49 for i in self.indices)
        value = self.ns['joint_frames'](str(folder), str(folder), str(folder))
        assert value.dtype == np.uint8 and value.shape == (len(self.indices), 704, 1280, 3)
        return value

    def frame(self, folder, index):
        return self.read(folder, [index])[0]

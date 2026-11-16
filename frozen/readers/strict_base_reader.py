"""Read published StarReal archives plus mandatory source-Remove supplements.

Paths are relative to a downloaded HF repository root. No downloads, extraction,
parameter rewrites or model inference occur. Gamma is returned as one scalar
normalized once; do not multiply it by another normalized spatial gamma tensor.
"""
from pathlib import Path, PurePosixPath
import hashlib, io, json, tarfile
import numpy as np
from PIL import Image

COMPLETION = 'StarReal/supplements/reference_and_remove_completion_v1'
FAMILIES = ('lamp_silhouette', 'bbox_silhouette')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def arrays(b):
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

class StarRealCompleteReader:
    def __init__(self, dataset_root, archive_path, verify_archive=True):
        self.dataset_root = Path(dataset_root)
        p = PurePosixPath(archive_path)
        if p.is_absolute() or '..' in p.parts or p.parts[:1] != ('StarReal',):
            raise ValueError('Use the exact relative HF StarReal archive path')
        self.archive_path = str(p)
        self.supplement_root = self.dataset_root / COMPLETION
        bindings = json.loads((self.supplement_root/'REMOVE_BINDINGS.json').read_text())
        aliases_path = Path(__file__).with_name('ARCHIVE_BINDING_ALIASES.json')
        aliases = json.loads(aliases_path.read_text())['aliases'] if aliases_path.exists() else {}
        alias = aliases.get(self.archive_path)
        self.binding_archive = alias['unchanged_source_reference_Remove_binding_archive'] if alias else self.archive_path
        rows = [x for x in bindings['rows'] if x['HF_source_archive'] == self.binding_archive]
        if len(rows) != 1: raise ValueError('No unique source-Remove binding')
        self.remove_binding = rows[0]
        self.archive = self.dataset_root / self.archive_path
        if verify_archive and digest(self.archive) != (alias['archive_SHA256'] if alias else rows[0]['archive_LFS_SHA256']):
            raise ValueError('Original case archive SHA mismatch')
        self.tf = tarfile.open(self.archive)
        all_files = [m for m in self.tf.getmembers() if m.isfile()]
        roots = [m.name for m in all_files if m.name.endswith('/MANIFEST.json')]
        self.prefix = min(roots, key=lambda s: s.count('/')).rsplit('/',1)[0]+'/'
        self.members = {m.name[len(self.prefix):]: m for m in all_files if m.name.startswith(self.prefix)}
        self.manifest = json.loads(self._get('MANIFEST.json'))
        self.controls = json.loads(self._get('conditions/MANIFEST.json'))
        self.source_appearance = json.loads(self._get('source_appearance_control_reference.json'))
        if self.manifest['frames'] != 49 or self.manifest['resolution'] != [1280,704]:
            raise ValueError('Unexpected source frame or spatial contract')
        self.references = json.loads((self.supplement_root/'OPERATION_REFERENCE_BINDINGS.json').read_text())
        self._remove_files = None
        if alias:
            if hashlib.sha256(self._get('conditions/cameras49.npz')).hexdigest() != alias['unchanged_source_cameras_SHA256']:
                raise ValueError('Source camera binding changed')
            if hashlib.sha256(self._get('source_appearance_control_reference.json')).hexdigest() != alias['unchanged_source_appearance_SHA256']:
                raise ValueError('Source appearance binding changed')

    def _get(self, rel):
        return self.tf.extractfile(self.members[rel]).read()

    def _remove(self):
        if self._remove_files is None:
            ref = self.remove_binding['new_condition_archive']
            path = self.supplement_root/ref['path']
            if path.stat().st_size != ref['bytes'] or digest(path) != ref['SHA256']:
                raise ValueError('Source-Remove completion archive mismatch')
            files = {}
            # One sequential gzip pass; retain only actual numeric conditions.
            with tarfile.open(path, 'r|gz') as t:
                for m in t:
                    if not m.isfile(): continue
                    root, sep, rel = m.name.partition('/')
                    if root != ref['extracted_root'] or not sep:
                        raise ValueError('Unexpected supplement extraction root')
                    if rel.endswith('.npz') or rel == 'MANIFEST.json':
                        files[rel] = t.extractfile(m).read()
            self._remove_files = files
        return self._remove_files

    def condition_frame(self, variant, family, frame, color=None, level=None, view='B'):
        if family not in FAMILIES or not 0 <= frame < 49 or view not in ('A','B'):
            raise ValueError('Explicit valid family, frame and A/B view required')
        if variant not in self.manifest['variants']: raise ValueError('Unknown variant')
        op = self.controls['operations'][variant]['operation']
        if op == 'remove':
            if color not in (None,'source_hint') or level not in (None,'removed'):
                raise ValueError('Remove uses the original source appearance and removed strength')
            rel = ('A_observed/' if view == 'A' else '') + family + f'/Intensity49/{frame:06d}.npz'
            z = arrays(self._remove()[rel]); rgb = z['color_linear_rgb']
            source = float(z['source_gamma_control']); target = float(z['target_gamma_control']); raw_field = z['raw_gamma']
            binding_role = 'original_source_before_removal'
        else:
            if view != 'B': raise ValueError('Existing target edit fields use B; do not relabel them A')
            if color is None or level is None: raise ValueError('Choose an explicit published color and intensity level')
            z = arrays(self._get(f'conditions/{variant}/conditions/{family}/{frame:06d}.npz'))
            if color == 'source_hint':
                rgb = z['support'][...,None] * np.asarray(self.source_appearance['control_color_linear_rgb'],np.float32)
            else:
                if color not in self.controls['colors_linear_rgb']: raise ValueError('Unknown requested color')
                rgb = z['color_'+color]
            if level not in self.controls['gamma_control_pairs_by_operation'][op]: raise ValueError('Unknown intensity level')
            source = float(z['source_gamma_control_'+level]); target = float(z['target_gamma_control_'+level]); raw_field = z['raw_gamma_'+level]
            binding_role = 'original_published_target_condition'
        support = z['support'].astype(bool); x,y = map(int,z['origin_xy']); H,W = map(int,z['full_shape_hw']); h,w = support.shape
        if (H,W) != (704,1280) or not (0 <= x <= x+w <= W and 0 <= y <= y+h <= H):
            raise ValueError('Malformed field origin or full shape')
        np.testing.assert_allclose(raw_field[support], target-source, atol=1e-7)
        if rgb.shape != (h,w,3): raise ValueError('Malformed Color field')
        mask = np.zeros((H,W),bool); mask[y:y+h,x:x+w] = support
        full = np.zeros((H,W,3),np.uint8); full[y:y+h,x:x+w] = np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)
        return dict(color_mask_rgb=Image.fromarray(full,'RGB'), support=mask,
                    source_gamma_control=source, target_gamma_control=target,
                    raw_gamma_target_minus_source=target-source,
                    normalized_gamma_for_model=(target-source)/1.1,
                    relative_alpha=(self.controls['alpha_control_reference_pair'][1]-self.controls['alpha_control_reference_pair'][0])/0.9,
                    normalization_count=1, operation=op, view='observed_A' if view=='A' else 'estimated_B',
                    condition_binding=binding_role, mask_family=family, edited_B_RGB=None)

    def joint_frame(self, variant, frame, version='new_joint'):
        if variant not in self.manifest['variants'] or version not in ('old_joint','new_joint') or not 0<=frame<49:
            raise ValueError('Explicit valid variant, joint version and frame required')
        direct = f'joints/{variant}/{version}/{frame:06d}.png'
        if direct in self.members:
            with Image.open(io.BytesIO(self._get(direct))) as im:
                if im.size != (1280,704): raise ValueError('Wrong joint resolution')
                return im.convert('RGB').copy()
        with Image.open(io.BytesIO(self._get(f'joints/shared_pcf/{frame:06d}.png'))) as im:
            base = np.asarray(im.convert('RGB')).copy()
        z=arrays(self._get(f'joints/{variant}/{version}/{frame:06d}.npz'));x,y=map(int,z['origin_xy']);patch=z['patch_rgb_uint8'];h,w=patch.shape[:2]
        if base.shape!=(704,1280,3) or patch.dtype!=np.uint8 or not (0<=x<=x+w<=1280 and 0<=y<=y+h<=704):
            raise ValueError('Malformed lossless joint patch')
        base[y:y+h,x:x+w]=patch
        return Image.fromarray(base,'RGB')

    def reference_image(self, variant):
        rows=[x for x in self.references['bindings'] if x['HF_source_archive']==self.binding_archive and x['variant']==variant]
        if len(rows)!=1: raise ValueError('No unique operation reference')
        ref=rows[0]['reference'];path=self.supplement_root/ref['path']
        if digest(path)!=ref['SHA256']: raise ValueError('Reference image SHA mismatch')
        with Image.open(path) as im:
            if im.mode=='RGBA':
                out=Image.new('RGB',im.size,(128,128,128));out.paste(im,mask=im.getchannel('A'));return out,rows[0]
            return im.convert('RGB').copy(),rows[0]

    def close(self): self.tf.close()
    def __enter__(self): return self
    def __exit__(self,*args): self.close()

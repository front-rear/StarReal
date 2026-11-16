"""Read reconstructed real-video conditions without inventing edited RGB truth.

Raw controls stay target-source on disk. This adapter applies Original scalar
normalization exactly once; it never also sends spatial raw_gamma as intensity.
All paths are supplied by the caller so a packaged case can be relocated.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image


class RealcaseConditionReader:
    def __init__(self, conditions_root, observed_frames, joint_root, appearance_reference=None):
        self.root = Path(conditions_root)
        self.observed_frames = Path(observed_frames)
        self.joint_root = Path(joint_root)
        self._observed_video_cache = None
        hint = (Path(appearance_reference) if appearance_reference is not None
                else self.root.parent/'source_appearance_control_reference.json')
        self.appearance_reference = json.loads(hint.read_text()) if hint.exists() else None
        self.manifest = json.loads((self.root / 'MANIFEST.json').read_text())
        assert self.manifest['frames'] == 49
        assert self.manifest['resolution'] == [1280, 704]
        assert self.manifest['edited_B_RGB'] is None

    def condition_frame(self, variant, family, frame, color, level):
        if family not in ('lamp_silhouette', 'bbox_silhouette'):
            raise ValueError(family)
        if not 0 <= frame < 49:
            raise IndexError(frame)
        with np.load(self.root / variant / 'conditions' / family / f'{frame:06d}.npz') as z:
            support = z['support']
            x, y = map(int, z['origin_xy'])
            shape = tuple(map(int, z['full_shape_hw']))
            assert shape == (704, 1280)
            h, w = support.shape
            assert 0 <= x <= x + w <= 1280 and 0 <= y <= y + h <= 704
            if color == 'source_hint':
                if self.appearance_reference is None:
                    raise ValueError('Source appearance reference is unavailable; do not invent measured color')
                rgb = support[...,None] * np.asarray(self.appearance_reference['control_color_linear_rgb'],np.float32)
            else:
                rgb = z['color_' + color]
            source = float(z['source_gamma_control_' + level])
            target = float(z['target_gamma_control_' + level])
            raw = target - source
            assert 0 <= source <= 1.1 and 0 <= target <= 1.1
            np.testing.assert_allclose(z['raw_gamma_' + level][support], raw, atol=1e-7)
            assert not z['raw_gamma_' + level][~support].any()
            full = np.zeros(shape + (3,), np.uint8)
            full[y:y+h, x:x+w] = np.rint(rgb.clip(0, 1) * 255).astype(np.uint8)
            mask = np.zeros(shape, bool)
            mask[y:y+h, x:x+w] = support
        return {
            'color_mask_rgb': Image.fromarray(full, 'RGB'),
            'support': mask,
            'source_gamma_control': source,
            'target_gamma_control': target,
            'raw_gamma_target_minus_source': raw,
            'normalized_gamma_for_model': raw / 1.1,
        }

    @staticmethod
    def _rgb(path):
        with Image.open(path) as image:
            assert image.size == (1280, 704), (str(path), image.size)
            return image.convert('RGB').copy()

    def _source_path(self, frame):
        candidates = [self.observed_frames / f'{frame:03d}.png',
                      self.observed_frames / f'{frame:06d}.png']
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(candidates)

    def observed_frame(self, frame):
        if self.observed_frames.is_dir():
            return self._rgb(self._source_path(frame))
        # Packaged observed RGB may be its original lossless source video.
        if self._observed_video_cache is None:
            import cv2
            cap = cv2.VideoCapture(str(self.observed_frames)); images = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                assert bgr.shape == (704,1280,3)
                images.append(Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)))
            fps=cap.get(5);cap.release()
            assert len(images)==49 and abs(fps-10)<1e-6
            self._observed_video_cache = images
        return self._observed_video_cache[frame].copy()

    def joint_frame(self, variant, joint_version, frame):
        folder = self.joint_root / variant / joint_version
        png = folder / f'{frame:06d}.png'
        if png.exists():
            return self._rgb(png)
        # Lossless factoring avoids copying identical full PCF backgrounds for
        # every operation. Reconstructed RGB is independently checked exactly.
        descriptor = json.loads((self.joint_root/'FACTORED_JOINTS.json').read_text())
        assert descriptor['schema']=='shared_PCF_plus_uint8_RGB_patch_v1'
        base = np.asarray(self._rgb(self.joint_root/'shared_pcf'/f'{frame:06d}.png')).copy()
        with np.load(folder/f'{frame:06d}.npz') as z:
            x,y=map(int,z['origin_xy']);patch=z['patch_rgb_uint8']
            h,w=patch.shape[:2]
            assert patch.dtype==np.uint8 and patch.shape==(h,w,3)
            assert 0<=x<=x+w<=1280 and 0<=y<=y+h<=704
            base[y:y+h,x:x+w]=patch
        return Image.fromarray(base,'RGB')

    def model_condition_inputs(self, variant, family, color, level,
                               joint_version='new_joint', frames=None):
        if joint_version not in ('old_joint', 'new_joint'):
            raise ValueError(joint_version)
        indices = list(range(49)) if frames is None else list(frames)
        rows = [self.condition_frame(variant, family, i, color, level) for i in indices]
        scalars = {r['normalized_gamma_for_model'] for r in rows}
        assert len(scalars) == 1, 'Original entry uses one relative scalar per clip'
        alpha_source, alpha_target = self.manifest['alpha_control_reference_pair']
        result = {
            'concat_video': [self.observed_frame(i) for i in indices],
            'warp_video': [self.joint_frame(variant, joint_version, i)
                           for i in indices],
            'warped_target_color_mask': [r['color_mask_rgb'] for r in rows],
            'relative_gamma': rows[0]['normalized_gamma_for_model'],
            'relative_alpha': (alpha_target - alpha_source) / 0.9,
        }
        # No `video` / target RGB B: these are conditioning-only real examples.
        # No `raw_gamma` tensor: Original WarpControl builds intensity from the
        # color-mask support plus the normalized scalar above.
        return result

    def provenance(self, variant, family, color, level):
        op = self.manifest['operations'][variant]['operation']
        source, target = self.manifest['gamma_control_pairs_by_operation'][op][level]
        rgb = (self.appearance_reference['control_color_linear_rgb'] if color=='source_hint'
               else self.manifest['colors_linear_rgb'][color])
        return {
            'variant': variant, 'operation': op, 'mask_family': family,
            'color_name': color, 'color_rgb_reference': rgb,
            'color_role': 'source appearance hint, not measured spectral emission' if color=='source_hint' else 'requested recolor',
            'source_gamma_control': source, 'target_gamma_control': target,
            'raw_gamma_target_minus_source': target - source,
            'model_relative_gamma': (target - source) / 1.1,
            'normalization_count': 1, 'control_references_are_measured_photometry': False,
            'edited_B_RGB': None, 'training_GT_available': False,
            'source_geometry_has_estimated_hidden_parts': True,
            'metric_scale_known': False,
        }

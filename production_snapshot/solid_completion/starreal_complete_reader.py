"""Strict single-color conditions; visibility diagnostics stay separate."""
from pathlib import Path
import json,numpy as np
from strict_base_reader import StarRealCompleteReader as _Base

class StarRealCompleteReader(_Base):
    def condition_frame(self,variant,family,frame,color=None,level=None,view='B'):
        op=self.controls['operations'][variant]['operation']
        if color is None:color='source_hint' if op in ('move','remove') else 'magenta'
        if level is None:level='removed' if op=='remove' else 'medium' if op=='insert' else 'unchanged'
        row=super().condition_frame(variant,family,frame,color,level,view)
        expected=np.asarray(self.source_appearance['control_color_linear_rgb'] if color=='source_hint' else self.controls['colors_linear_rgb'][color],np.float32)
        expected=np.rint(expected.clip(0,1)*255).astype(np.uint8)
        pixels=np.asarray(row['color_mask_rgb']);support=row['support']
        if np.any(pixels[~support]!=0) or np.any(pixels[support]!=expected):
            raise ValueError(f'Non-uniform Color condition: {self.archive_path} {variant} {family} frame{frame} {color}')
        row.update(color_name=color,intensity_level=level,foreground_RGB_uint8=expected.tolist(),single_color_foreground_verified=True,uncertainty_encoded_in_color=False)
        return row

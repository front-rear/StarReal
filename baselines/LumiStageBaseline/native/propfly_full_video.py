"""Run untouched official PropFly demo, then encode all native output PNGs.

The native GIF is capped to 25 frames even when num_frames=49. Keep it and add
an MP4 containing every generated frame, without resampling or model changes.
"""
import sys, runpy, json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image
args=sys.argv[1:]
out=Path(args[args.index('--output_dir')+1])
frames_dir=Path(args[args.index('--validation_image_dir')+1])
fps=10.0
manifest=frames_dir.parent/'INPUT_CONDITIONS.json'
if manifest.exists():
 data=json.loads(manifest.read_text())
 fps=float(data.get('fps',data.get('source_fps',10)))
runpy.run_path(str(Path.cwd()/'demo.py'),run_name='__main__')
frames=sorted((out/'output').glob('output_*.png'))
if not frames:raise RuntimeError('Official PropFly demo produced no output PNGs')
video=out/'all_generated_frames.mp4'
with imageio.get_writer(str(video),fps=fps,codec='libx264',quality=8,macro_block_size=1) as writer:
 for p in frames:writer.append_data(np.asarray(Image.open(p).convert('RGB')))
(out/'FULL_FRAME_VIDEO.json').write_text(json.dumps(dict(
 generated_png_frames=len(frames),mp4_fps=fps,output_mp4=str(video),
 native_gif_note='Official demo limits output.gif to 25 frames; GIF retained unchanged',
 method='Encode every native generated PNG in numerical order; no frame synthesis or resampling',
 input_manifest=str(manifest) if manifest.exists() else None),indent=2))

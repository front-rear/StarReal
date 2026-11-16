"""Decode every frame without temporal resampling and record RGB and container digests."""
import hashlib
import json
import subprocess
from fractions import Fraction
from common import require, sha


def video(path, expected_size=None, expected_frames=None):
    try:
        import imageio.v2 as imageio
    except ImportError:
        meta = json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
                          "-show_entries","stream=width,height,avg_frame_rate","-of","json",str(path)]))["streams"][0]
        size = [meta["width"], meta["height"]]
        fps = float(Fraction(meta["avg_frame_rate"]))
        frame_bytes = size[0] * size[1] * 3
        hashes = []
        cmd = ["ffmpeg","-v","error","-i",str(path),"-map","0:v:0","-fps_mode","passthrough",
               "-frames:v","50","-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as process:
            while True:
                frame = process.stdout.read(frame_bytes)
                if not frame:
                    break
                require(len(frame) == frame_bytes, "Truncated RGB24 frame")
                hashes.append(hashlib.sha256(frame).hexdigest())
            require(process.wait() == 0, "ffmpeg decode failed")
    else:
        with imageio.get_reader(str(path)) as reader:
            meta = reader.get_meta_data()
            fps, size = meta["fps"], list(meta["size"])
            hashes = []
            for pixels in reader:
                require(pixels.shape == (size[1], size[0], 3), "Variable shape or non-RGB frame")
                hashes.append(hashlib.sha256(pixels.tobytes()).hexdigest())
                require(len(hashes) <= 49, "More than 49 frames")
    require(abs(fps - 10) < 0.001 and len(hashes) == 49, f"Decoded {len(hashes)} frames at {fps} fps")
    if expected_size is not None:
        require(list(expected_size) == size, "Unexpected native resolution")
    if expected_frames is not None:
        require(hashes == expected_frames, "Paired source pixel sequence differs")
    return dict(sha256=sha(path), size=size, fps=10, decoded_frames=49, rgb_sha256=hashes)

"""Convert one verified StarReal operation and its real CoF sample into a portable request."""
import argparse
import os
import hashlib
import tempfile
from pathlib import Path
from media import video
from common import QWEN_CALL, bound, identifier, read, request, require, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("manifest", "cof-batch", "camera", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--geometry", type=Path)
    p.add_argument("--geometry-fields", type=Path, help="Explicit P/Q NPZ field mapping JSON")
    p.add_argument("--asset", type=Path)
    p.add_argument("--cof-pair-proof", type=Path, help="Required when frozen CoF input differs from source video")
    p.add_argument("--requested-rgb", nargs=3, type=float)
    p.add_argument("--camera-sha", required=True, help="SHA from the selected operation specification")
    p.add_argument("--geometry-sha", help="Required for separately bound fixed world geometry")
    a = p.parse_args()
    final = a.output.resolve()
    require(not final.exists(), "Use a new request directory")
    final.parent.mkdir(parents=True, exist_ok=True)
    dest = Path(tempfile.mkdtemp(prefix="preparing-", dir=final.parent))
    m = read(a.manifest)
    frozen = m["frozen_source_case"]
    samples = [r for r in read(a.cof_batch) if r.get("sample_id") == a.sample_id]
    require(len(samples) == 1, "Select exactly one actual VideoCoF sample")
    sample = samples[0]
    require(sample.get("request_manifest_SHA256") == sha(a.manifest), "CoF row bound to another request; use its actual manifest")
    require(sha(a.camera) == a.camera_sha, "Camera digest mismatch")
    if a.geometry:
        require(sha(a.geometry) == a.geometry_sha, "Geometry digest mismatch")
    require(m["native_call"] == {k:v for k,v in QWEN_CALL.items() if k != "num_images_per_prompt"},
            "Qwen native parameters differ")
    def ref(file, expected=None):
        path = Path(file).resolve()
        dig = sha(path)
        if expected:
            require(dig == expected, "Changed file " + path.name)
        return dict(path=os.path.relpath(path, dest), sha256=dig)
    conditions = [dict(**ref(z["path"], z["SHA256"]), role=z["role"]) for z in m["condition_images"]]
    source = ref(frozen["source_video"]["path"], frozen["source_video"]["SHA256"])
    cof_source = ref(sample["source_video_path"])
    frames = [dict(index=z["index"], **ref(z["path"], z["SHA256"]), rgb_sha256=z["RGB_SHA256"],
                   prop_rgb_sha256=z["native_PropFly832480_RGB_SHA256"]) for z in frozen["source49"]]
    require([r["index"] for r in frames] == list(range(49)), "Incorrect source indices")
    write(dest / "frames.json", frames)
    data = dict(case=identifier(str(m["case"])), operation=m["operation"], fps=frozen["source_fps"],
                frame_indices=frozen["source_indices"], source_video=source, videocof_video=cof_source,
                source_frames_index=ref(dest / "frames.json"), condition_images=conditions,
                camera=ref(a.camera), qwen_prompt=m["prompt"], propagation_caption=m["propagation_caption"],
                videocof_instruction=sample.get("text", sample.get("qwen_vl_72b_refined_instruction", sample["edit_instruction"])),
                original_manifest_sha256=sha(a.manifest), original_cof_batch_sha256=sha(a.cof_batch),
                original_cof_batch_position=read(a.cof_batch).index(sample), original_cof_sample_id=a.sample_id)
    if cof_source["sha256"] != source["sha256"]:
        require(sample.get("paired_original_source_SHA256") == source["sha256"],
                "CoF sample refers to a different original source")
        if a.cof_pair_proof:
            data["videocof_pair_proof"] = ref(a.cof_pair_proof)
        else:
            from PIL import Image
            cof_info = video(sample["source_video_path"])
            expected = []
            for frame in frames:
                with Image.open(dest / frame["path"]) as im:
                    rgb = im.convert("RGB").resize(tuple(cof_info["size"]), Image.Resampling.LANCZOS)
                    expected.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            require(expected == cof_info["rgb_sha256"], "Prepared CoF video differs from source LANCZOS RGB49")
            proof = dest / "paired_cof.json"
            write(proof, dict(source_sha256=source["sha256"], cof_sha256=cof_source["sha256"],
                  frame_indices=list(range(49)), cof_rgb_sha256=expected, resize="PIL.LANCZOS"))
            data["videocof_pair_proof"] = ref(proof)
    if a.geometry:
        data["geometry"] = ref(a.geometry)
    if a.geometry_fields:
        data["geometry_fields"] = read(a.geometry_fields)
    if m.get("asset_GLB_SHA256"):
        require(a.asset is not None, "Supply the original bound asset")
        data["asset"] = ref(a.asset, m["asset_GLB_SHA256"])
    if a.requested_rgb:
        data["requested_RGB"] = a.requested_rgb
    write(dest / "request.json", data)
    request(dest / "request.json")
    dest.rename(final)


if __name__ == "__main__":
    main()

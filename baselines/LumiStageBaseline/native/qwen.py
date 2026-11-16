"""The production Qwen pipeline call, with its original per-component loading observer."""
import argparse
import gc
import ctypes
import inspect
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import NATIVE, QWEN_CALL, VERSIONS, bound, observer_manifest, read, request, require, sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("requests", nargs="+", type=Path)
    args = p.parse_args()
    import torch
    from PIL import Image
    from diffusers import QwenImage21Pipeline
    from diffusers.models import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration
    from qwen21_propfly_native_load_observer_v1 import NativeLoadObserver

    requests = [(q.resolve(), request(q)) for q in args.requests]
    config = read(args.config)
    require(sha(inspect.getfile(QwenImage21Pipeline)) == VERSIONS["native_pipeline_SHA256"],
            "Official Qwen-Image-2.1 pipeline pin differs")
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "Expected offline model loading")
    torch.set_num_threads(1)
    classes = {"transformer": QwenImage21Transformer2DModel, "vae": AutoencoderKLQwenImage21,
               "text_encoder": Qwen3VLForConditionalGeneration}
    shared = args.output / "native_load"
    shared.mkdir(parents=True, exist_ok=False)
    with NativeLoadObserver(observer_manifest(config), shared, "qwen21", classes, QwenImage21Pipeline):
        pipe = QwenImage21Pipeline.from_pretrained(config["qwen_model"], torch_dtype=torch.bfloat16,
                                                   local_files_only=True)
        pipe.enable_model_cpu_offload()
    audit = shared / "QWEN21_ACTUAL_COMPONENT_LOADING_VALUES_v1.json"
    require(read(audit)["status"] == "ALL_NATIVE_LOAD_AND_FINAL_CAST_VALUES_OFFLOAD_VERIFIED", "Qwen load audit failed")
    for path, data in requests:
        dest = args.output / (data["case"] + "_" + data["operation"])
        dest.mkdir(exist_ok=False)
        images = []
        for ref in data["condition_images"]:
            with Image.open(bound(ref, path.parent)) as im:
                images.append(im.convert("RGB"))
        gc.collect()
        ctypes.CDLL(None).malloc_trim(0)
        result = pipe(prompt=data["qwen_prompt"], image=images, width=832, height=480, output_resolution=1024,
                      num_inference_steps=40, true_cfg_scale=1.0, num_images_per_prompt=1,
                      generator=torch.Generator(device="cuda").manual_seed(42), output_type="pil",
                      use_kv_cache=True).images[0]
        require(result.size == (832, 480), "Unexpected Qwen output dimensions")
        png = dest / "edited_firstframe_native.png"
        result.save(png)
        write(dest / "qwen_record.json", dict(request_sha256=sha(path), source_sha256=data["source_video"]["sha256"],
              asset_sha256=(data.get("asset") or {}).get("sha256"), condition_images=data["condition_images"],
              qwen_prompt=data["qwen_prompt"], png_sha256=sha(png), output_size=[832,480], native_call=QWEN_CALL,
              model_revision=VERSIONS["model_revision"], pipeline_sha256=VERSIONS["native_pipeline_SHA256"],
              entry_sha256=sha(__file__), audit_sha256=sha(audit), quality_accepted=None))


if __name__ == "__main__":
    main()

"""Validate one frozen scene, run the native baseline, or drain an explicitly listed queue."""
import argparse
import contextlib
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import uuid
from common import (NATIVE, ROOT, QWEN_CALL, VERSIONS, bound, frame_rows, observer_manifest,
                    read, request, require, sha, write)
from geometry import project
from media import video


def preflight(path):
    from PIL import Image
    path = Path(path).resolve()
    data = request(path)
    rows, frames = frame_rows(data, path.parent)
    for row, f in zip(rows, frames):
        with Image.open(f) as image:
            rgb = image.convert("RGB")
            require(hashlib.sha256(rgb.tobytes()).hexdigest() == row["rgb_sha256"], "Source pixels changed")
            require(hashlib.sha256(rgb.resize((832,480)).tobytes()).hexdigest() == row["prop_rgb_sha256"],
                    "Resized PropFly condition differs")
    src = video(bound(data["source_video"], path.parent))
    require(src["rgb_sha256"] == [r["rgb_sha256"] for r in rows], "Source frames differ from source video")
    cof = data.get("videocof_video", data["source_video"])
    if cof["sha256"] != data["source_video"]["sha256"]:
        require("videocof_pair_proof" in data, "A distinct prepared CoF input requires its paired RGB49 proof")
        proof = read(bound(data["videocof_pair_proof"], path.parent))
        require(proof["source_sha256"] == src["sha256"] and proof["cof_sha256"] == cof["sha256"]
                and proof["frame_indices"] == list(range(49)), "Paired CoF proof mismatch")
        video(bound(cof, path.parent), expected_frames=proof["cof_rgb_sha256"])
    dims = project(data, path.parent)
    return data, dict(source=src, K_AB_geometry=dims, conditions=[dict(role=r["role"], sha256=r["sha256"])
                      for r in data["condition_images"]], quality_accepted=None)


def normal_config(path):
    path = Path(path).resolve()
    cfg = read(path)
    return {k: str((path.parent / value).resolve()) for k,value in cfg.items()}


def verify_code(repo, name):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    require(commit == VERSIONS["repo_pins"][name], "Wrong upstream commit: " + name)
    subprocess.run(["git","-C",str(repo),"diff","--quiet","HEAD","--"], check=True)


def weight_check(cfg, stage=None):
    def check(root, items):
        for row in items:
            name = row.get("file")
            if not name:
                continue
            p = root / name
            require(p.stat().st_size == row["size"], "Wrong weight size: " + name)
            digest = row.get("LFS_SHA256") or row.get("sha256")
            if digest:
                require(sha(p) == digest, "Weight SHA256 differs: " + name)
            elif row.get("git_blob"):
                raw = p.read_bytes()
                expected = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()
                require(row["git_blob"] == expected, "Git-blob SHA differs: " + name)
    if stage in (None, "qwen"):
        check(Path(cfg["qwen_model"]), VERSIONS["qwen_files"])
    model_rows = []
    if stage in (None, "propfly"):
        model_rows.extend(VERSIONS["weights"])
    if stage in (None, "videocof"):
        model_rows.extend(VERSIONS["cof_weights"])
    for row in model_rows:
        name = row["repo"]
        if name in ("james16/PropFly", "XiangpengYang/VideoCoF", "MonsterMMORPG/Wan_GGUF"):
            field = {"james16/PropFly":"propfly_method", "XiangpengYang/VideoCoF":"cof_method",
                     "MonsterMMORPG/Wan_GGUF":"cof_fusionx"}[name]
            p = Path(cfg[field])
            require(p.stat().st_size == row["size"] and sha(p) == row["sha256"], "Method weight mismatch " + name)
        else:
            check(Path(cfg["propfly_base" if "VACE" in name else "cof_base"]), [row])


@contextlib.contextmanager
def exclusive(lockpath):
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    with lockpath.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield lock.fileno()


def verify_gate(request_path, data, png, gate):
    rec_path = png.parent / "qwen_record.json"
    rec = read(rec_path)
    require(rec["request_sha256"] == sha(request_path) and rec["source_sha256"] == data["source_video"]["sha256"]
            and rec["asset_sha256"] == (data.get("asset") or {}).get("sha256")
            and rec["condition_images"] == data["condition_images"] and rec["qwen_prompt"] == data["qwen_prompt"]
            and rec["native_call"] == QWEN_CALL and rec["entry_sha256"] == sha(NATIVE / "qwen.py")
            and rec["png_sha256"] == sha(png), "Qwen input/output record differs")
    audit = png.parent.parent / "native_load/QWEN21_ACTUAL_COMPONENT_LOADING_VALUES_v1.json"
    require(sha(audit) == rec["audit_sha256"] and read(audit)["status"] ==
            "ALL_NATIVE_LOAD_AND_FINAL_CAST_VALUES_OFFLOAD_VERIFIED", "Bound Qwen load audit differs")
    expected = dict(request_sha256=sha(request_path), png_sha256=sha(png), record_sha256=sha(rec_path),
                    propagation_entry_sha256=sha(NATIVE / "propfly_native_observed_runner_v1.py"))
    require(all(gate[k] == v for k,v in expected.items()) and gate.get("observed_png") is True
            and gate.get("decision") == "propagate" and type(gate.get("visible_firstframe_pass")) is bool
            and gate.get("visual_findings") and gate.get("reviewer"), "Missing actual visual observation/bound gate")
    return expected


def spawn(cmd, cwd, env, lock_fd):
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, pass_fds=(lock_fd,), check=True)


def execute(stage, requests, config, output, admission=None, png=None, review=None):
    requests = [Path(p).resolve() for p in requests]
    data = [preflight(p)[0] for p in requests]
    keys = [d["case"] + "_" + d["operation"] for d in data]
    require(len(keys) == len(set(keys)), "Duplicate case + operation in batch")
    hashes = [sha(r) for r in requests]
    authorization = read(admission)
    require(authorization.get("requests_sha256") == hashes and authorization.get("reviewed_inputs") is True
            and authorization.get("reviewer") and authorization.get("input_observations"),
            "Input contact sheet and request order require actual review")
    cfg = normal_config(config)
    out = Path(output).resolve()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
               PYTHONNOUSERSITE="1")
    with exclusive(Path(cfg["gpu_lock"])) as lock_fd:
        binding = dict(stage=stage, requests_sha256=hashes, admission_sha256=sha(admission),
                       config_sha256=sha(config), gate_sha256=sha(review) if review else None,
                       reference_sha256=sha(png) if png else None)
        completion = out / "complete.json"
        if completion.exists():
            saved = read(completion)
            require(saved["binding"] == binding, "Output belongs to a different run")
            for item in saved["artifacts"]:
                require(sha(out / item["path"]) == item["sha256"], "Saved result changed")
            return
        out.mkdir(parents=True, exist_ok=True)
        trial = out / ("attempt-" + uuid.uuid4().hex)
        trial.mkdir()
        result = []
        try:
            weight_check(cfg, stage)
            config_copy = trial / "runtime_config.json"
            write(config_copy, cfg)
            if stage == "qwen":
                spawn([cfg["qwen_python"], NATIVE / "qwen.py", config_copy, trial, *requests], ROOT, env, lock_fd)
                result = [trial / key / "edited_firstframe_native.png" for key in keys]
            elif stage == "propfly":
                require(len(requests) == 1 and png and review, "PropFly needs one actual PNG and its signed gate")
                png = Path(png).resolve()
                req, item = requests[0], data[0]
                verify_gate(req, item, png, read(review))
                repo = Path(cfg["propfly_repo"])
                verify_code(repo, "propfly")
                native_manifest = trial / "observer.json"
                write(native_manifest, observer_manifest(cfg))
                frames = trial / "frames"
                frames.mkdir()
                rows, paths = frame_rows(item, req.parent)
                for i, p in enumerate(paths):
                    (frames / f"{i:06d}.png").symlink_to(p)
                write(trial / "INPUT_CONDITIONS.json", {"fps":10})
                env["PYTHONPATH"] = str(repo)
                cmd = [cfg["propfly_python"], NATIVE / "propfly_native_observed_runner_v1.py", "--observer-manifest",
                       native_manifest, "--observer-manifest-sha", sha(native_manifest), "--pretrained_model_name_or_path",
                       cfg["propfly_base"], "--output_dir", trial, "--num_inference_steps",50,"--max_frame_num",240,
                       "--num_frames",49,"--guidance_scale",3,"--batch_frames",25,"--sample_stride",2,"--overlap",6,
                       "--height",480,"--width",832,"--validation_image_dir",frames,"--validation_ref_image",png,
                       "--prompt", item["propagation_caption"], "--transformer_path",cfg["propfly_method"]]
                spawn(cmd, repo, env, lock_fd)
                from PIL import Image
                outputs = sorted((trial / "output").glob("output_*.png"))
                conds = sorted((trial / "cond").glob("cond_*.png"))
                require(len(outputs) == len(conds) == 49, "Native PropFly did not produce all PNGs")
                for i, (native, cond) in enumerate(zip(outputs, conds)):
                    with Image.open(native) as im:
                        require(im.size == (832,480), "Incorrect native PNG size")
                    with Image.open(cond) as im:
                        require(hashlib.sha256(im.convert("RGB").tobytes()).hexdigest() == rows[i]["prop_rgb_sha256"],
                                "PropFly conditioning frame changed")
                result = [trial / "all_generated_frames.mp4"]
                write(trial / "decoded.json", video(result[0], [832,480]))
            elif stage == "videocof":
                repo = Path(cfg["cof_repo"])
                verify_code(repo, "videocof")
                samples, sizes = [], []
                for req, item in zip(requests, data):
                    source = bound(item.get("videocof_video", item["source_video"]), req.parent)
                    samples.append(dict(task_type=item["operation"], sample_id=item["case"], source_video_path=str(source),
                                        edit_instruction=item["videocof_instruction"]))
                    sizes.append(video(source)["size"])
                batch = trial / "cof_batch.json"
                write(batch,samples)
                cmd = [cfg["cof_python"],"-m","torch.distributed.run","--standalone","--nproc_per_node=1",
                       NATIVE / "videocof_audited_resident.py","--output_dir",trial,"--model_name",cfg["cof_base"],
                       "--videocof_path",cfg["cof_method"],"--enable_acceleration_lora","--acceleration_lora_path",
                       cfg["cof_fusionx"],"--num_frames",49,"--source_frames",49,"--reasoning_frames",4,
                       "--repeat_rope","--test_json",batch]
                spawn(cmd, repo, env, lock_fd)
                require(read(trial / "FUSIONX_ACTUAL_COMPONENT_LOADING_v1.json")["applied_kind_counts"] ==
                        {"lora":405,"diff_b":447,"diff":200}, "Partial FusionX patch")
                for sample, size in zip(samples, sizes):
                    native = trial / ("gen_" + sample["task_type"] + "_" + sample["sample_id"] + ".mp4")
                    write(native.with_suffix(".decoded.json"),video(native,size))
                    result.append(native)
            else:
                raise ValueError("Unknown stage")
            write(trial / "complete.json", dict(binding=binding, quality_accepted=None,
                  artifacts=[dict(path=str(x.relative_to(out)),sha256=sha(x)) for x in result]))
            os.replace(trial / "complete.json", completion)
        except BaseException as err:
            write(trial / "failure.json",dict(type=type(err).__name__,message=str(err),binding=binding))
            raise


def main():
    import json
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("dry-run", help="CPU schema + model interface check; no inference or files required")
    plan.add_argument("request", type=Path)
    check = sub.add_parser("check", help="Verify all bound bytes, temporal pairing and 3-D target")
    check.add_argument("request", type=Path)
    check.add_argument("--report", type=Path)
    weights = sub.add_parser("weights", help="Read and verify pinned local model files")
    weights.add_argument("config", type=Path)
    for stage in ("qwen", "propfly", "videocof"):
        run = sub.add_parser(stage)
        run.add_argument("requests", nargs="+", type=Path)
        run.add_argument("--config", required=True, type=Path)
        run.add_argument("--output", required=True, type=Path)
        run.add_argument("--admission", required=True, type=Path)
        if stage == "propfly":
            run.add_argument("--png", required=True, type=Path)
            run.add_argument("--review", required=True, type=Path)
    q = sub.add_parser("queue", help="Run a fixed ordered list once, using the configured exclusive GPU file lock")
    q.add_argument("list", type=Path)
    probe = sub.add_parser("decode")
    probe.add_argument("video", type=Path)
    a = p.parse_args()
    if a.action == "dry-run":
        d = request(a.request, False)
        print(json.dumps({"case":d["case"], "operation":d["operation"], "ordered_Qwen_inputs":
              [x["role"] for x in d["condition_images"]],"Qwen_native_call":QWEN_CALL,
              "PropFly_inputs":["source49","reviewed_Qwen_PNG","propagation_caption"],
              "VideoCoF_inputs":["source_video","text"],"GPU_started":False,"quality_accepted":None},
              ensure_ascii=False,indent=2))
    elif a.action == "check":
        _, report = preflight(a.request)
        if a.report:
            write(a.report, report)
        print("Verified frozen source49, condition hashes and A/B trajectory projections")
    elif a.action == "weights":
        weight_check(normal_config(a.config))
        print("Verified recorded local model fingerprints and sizes")
    elif a.action == "decode":
        print(json.dumps(video(a.video), indent=2))
    elif a.action == "queue":
        base = a.list.resolve().parent
        allowed = ("qwen","propfly","videocof")
        for n, row in enumerate(read(a.list)):
            require(row["stage"] in allowed, "Invalid queue stage")
            print(f"Queue row {n + 1}: {row['stage']}", flush=True)
            files = {k:base / row[k] for k in ("config","output","admission","png","review") if row.get(k)}
            execute(row["stage"], [base / q for q in row["requests"]], **files)
    else:
        execute(a.action, a.requests, a.config, a.output, a.admission, getattr(a,"png",None),
                getattr(a,"review",None))


if __name__ == "__main__":
    main()

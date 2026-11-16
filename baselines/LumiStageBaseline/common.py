"""Portable bindings shared by preflight, native entry points and study records."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / "native"
VERSIONS = json.loads((ROOT / "versions.json").read_text())
OPERATIONS = {"insert", "remove", "move", "replace", "relight", "camera_change", "move_replace"}
QWEN_CALL = dict(width=832, height=480, output_resolution=1024, num_inference_steps=40,
                 true_cfg_scale=1.0, num_images_per_prompt=1, seed=42, use_kv_cache=True)


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bound(ref, base):
    path = (base / ref["path"]).resolve()
    require(sha(path) == ref["sha256"], f"Input digest changed: {path.name}")
    return path


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.-]+", value)
            and value not in {".", ".."}, "Invalid case identifier")
    return value


def request(path, check_files=True):
    path = Path(path).resolve()
    data = read(path)
    base = path.parent
    identifier(data["case"])
    require(data["operation"] in OPERATIONS, "Unknown operation")
    require(data["frame_indices"] == list(range(49)) and data["fps"] == 10, "Expected frozen 49/10 source")
    conditions = data["condition_images"]
    require(conditions and all(isinstance(r["role"], str) and r["role"].strip() for r in conditions),
            "Every ordered Qwen condition needs a role")
    for k in ("qwen_prompt", "propagation_caption", "videocof_instruction"):
        require(isinstance(data[k], str) and data[k].strip(), "Missing " + k)
    if data["operation"] in {"insert", "replace", "move_replace"}:
        require(data.get("asset") is not None, "Assigned GLB required")
    if data["operation"] == "relight":
        color = data["requested_RGB"]
        require(len(color) == 3 and all(isinstance(v, (float, int)) and 0 <= v <= 1 for v in color),
                "Missing or invalid requested linear RGB")
    refs = [data["source_video"], data["source_frames_index"], data["camera"], *conditions]
    refs.extend(data[k] for k in ("asset", "geometry", "videocof_video", "videocof_pair_proof") if data.get(k))
    for ref in refs:
        require(isinstance(ref["path"], str) and ref["path"] and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
                "Invalid file binding")
        if check_files:
            bound(ref, base)
    return data


def frame_rows(data, base):
    rows = read(bound(data["source_frames_index"], base))
    require([r["index"] for r in rows] == list(range(49)), "Frozen frame order differs")
    root = (base / data["source_frames_index"]["path"]).resolve().parent
    paths = [bound(row, root) for row in rows]
    require(len(set(paths)) == 49, "Duplicate source frame paths")
    return rows, paths


def observer_manifest(cfg):
    native_sources = VERSIONS["propfly_native_sources"]
    repo = Path(cfg["propfly_repo"])
    source_paths = [repo / "demo.py", repo / "demo.sh", repo / "pipeline/pipeline_vace_video.py",
                    NATIVE / "propfly_full_video.py"]
    receipts = read(ROOT / "docs/production_sources.json")["copied_exactly"]
    helpers = [name for name in receipts if name not in ("fusionx_exact_loader_adapter_v1.py", "propfly_full_video.py")]
    return {
        "qwen": {"required_files": VERSIONS["qwen_files"], "local_model_dir": cfg["qwen_model"]},
        "propfly": {
            "base_model": cfg["propfly_base"], "method_checkpoint": cfg["propfly_method"],
            "full49_entry": str(NATIVE / "propfly_full_video.py"),
            "native_source_files": [{"path": str(p), "SHA256": native_sources[p.name]} for p in source_paths],
        },
        "observer": {
            "files": [{"path": str(NATIVE / n), "SHA256": receipts[n]} for n in helpers],
            "propfly_base_runtime_files": [{"relative_path": n} for n in VERSIONS["propfly_base_runtime_files"]],
        },
    }

"""Portable IO adapters; frozen production mathematics lives under frozen/."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

FROZEN = Path(__file__).resolve().parent / "_frozen"
def _load(name, relative):
    key = "_starreal_frozen_" + name
    if key in sys.modules:
        return sys.modules[key]
    path = FROZEN / relative
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module

def depth_renderer(accelerated=True, threads=2):
    renderer = _load("depth", "render/depth_fusion_fixed_world_yellow_overlap_v2.py")
    if not hasattr(renderer, "_numpy_trace_lattice"):
        renderer._numpy_trace_lattice = renderer.trace_lattice
    renderer.trace_lattice = renderer._numpy_trace_lattice
    if accelerated and (FROZEN / "render/pcf_trace_cpu.so").is_file():
        accel = _load("accel", "render/pcf_cpu_accelerator.py")
        renderer.trace_lattice = accel.CPUTrace(threads)
    return renderer

def render_pair(K, w2c, boxes, width=1280, height=704, aa=2,
                spacing=.5, far_cells=64., steps=4096,
                first_w2c_A=None, accelerated=True):
    """Render ordinary and opaque depth-aware PCF+bbox on identical cameras.

    boxes: fixed shared-world center, size (FULL extents), rotation, label1/2.
    first_w2c_A fixes the PCF coordinate origin once, never once per frame.
    World units are the caller's shared units, not automatically calibrated metres.
    """
    import cv2
    renderer = depth_renderer(accelerated)
    sys.path.insert(0, str(FROZEN / "ordinary"))
    ordinary = _load("bbox", "ordinary/bbox.py")
    A0 = np.eye(4) if first_w2c_A is None else np.asarray(first_w2c_A, float)
    K, w2c = np.asarray(K, float), np.asarray(w2c, float)
    C = np.linalg.inv(w2c @ np.linalg.inv(A0))
    normalized = []
    masks = {1: np.zeros((height, width), np.uint8),
             2: np.zeros((height, width), np.uint8)}
    wires = {k: v.copy() for k, v in masks.items()}
    for item in boxes:
        center = np.asarray(item["center"], float)
        size = np.asarray(item["size"], float)
        rotation = np.asarray(item.get("rotation", np.eye(3)), float)
        label = int(item["label"])
        if label not in (1, 2):
            raise ValueError("label must be source1 or target2")
        normalized.append(renderer.Box(
            (A0 @ np.r_[center, 1.])[:3], size / 2,
            A0[:3, :3] @ rotation, label))
        box = ordinary.OrientedBoundingBox(
            "box", center, size, rotation, np.array([255,255,255], np.uint8))
        lines, _, mask, _ = ordinary.render_bbox_conditions(
            [box], K, w2c, height, width, edge_width=3, fill_alpha=.28)
        wire = ((lines.max(-1)>100) & (mask>0)).astype(np.uint8)*255
        masks[label] |= mask
        wires[label] |= wire
    prepared = renderer.prepare_lattice(
        C, K, width, height, aa, spacing, far_cells, steps)
    background, _ = renderer.render(
        C, K, [], width, height, aa, spacing, far_cells, steps, prepared=prepared)
    # Exact unchanged Original joint_frames palette / rint / wire precedence.
    labels = (masks[1]>0).astype(np.uint8) + 2*(masks[2]>0).astype(np.uint8)
    palette = np.array([[0,0,0],[255,0,0],[0,0,255],[255,255,0]], np.uint8)
    ordinary_rgb = background.copy()
    occupied = labels>0
    ordinary_rgb[occupied] = np.rint(
        .42*background[occupied] + .58*palette[labels[occupied]]).astype(np.uint8)
    wire = (wires[1]>0) | (wires[2]>0)
    ordinary_rgb[wire] = palette[labels[wire]]
    depth_rgb, aux = renderer.render(
        C, K, normalized, width, height, aa, spacing, far_cells, steps,
        prepared=prepared)
    return ordinary_rgb, depth_rgb, aux

def geometry_module():
    return _load("geometry", "geometry/real_geometry.py")

def mesh_supports(vertices, faces, rotation, K, w2c, scene_depth=None,
                  width=1280, height=704):
    """Real mesh silhouette and separate full projected 3D OBB silhouette.

    Scene-occluded / unknown regions remain separate fields. Without a supplied
    scene depth, visibility is explicitly unknown, never observed ground truth.
    """
    g = geometry_module()
    raster = g.Rasterizer()
    vertices, faces = np.asarray(vertices,float), np.asarray(faces,np.int32)
    mesh_depth = raster.depth(vertices, faces, K, w2c, width, height)
    scene_depth = (np.full((height,width),np.inf,np.float32)
                   if scene_depth is None else np.asarray(scene_depth,np.float32))
    visibility = g.visibility(mesh_depth, scene_depth)
    center, size, rotation = g.bbox_for_mesh(vertices, np.asarray(rotation,float))
    import itertools
    signs = np.array(list(itertools.product((-1,1), repeat=3)),float)
    bv = (signs*size/2) @ rotation.T + center
    bf = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                   [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],np.int32)
    bd = raster.depth(bv, bf, K, w2c, width, height)
    return dict(lamp_silhouette=visibility["visible_with_unknown_estimate"],
                bbox_silhouette=np.isfinite(bd),
                bbox_scene_visible_estimate=g.visibility(bd,scene_depth)["visible_with_unknown_estimate"],
                **visibility)

def export_control_fields(supports, output, operation, frame=0, source_color_linear=None):
    """Export both exact frozen raw Color/gamma ROI families.

    Remove must receive ORIGINAL source supports. This function deliberately
    does not convert a missing target into an empty Remove source condition.
    Gamma stays raw signed target-source; actual model reader normalizes once.
    """
    import cv2
    import json
    if operation not in ("insert","remove","move","replace","move_replace"):
        raise ValueError(operation)
    sys.path.insert(0, str(FROZEN/"geometry"))
    sys.modules.setdefault("real_geometry", geometry_module())
    fields = _load("fields", "masks/build_case1_masks49.py")
    output = Path(output)
    if operation in ("move", "remove") and source_color_linear is None:
        raise ValueError("move/remove need actual source appearance control color")
    if source_color_linear is not None:
        source_color_linear=np.asarray(source_color_linear,np.float32)
        if source_color_linear.shape!=(3,) or not np.isfinite(source_color_linear).all() or np.any((source_color_linear<0)|(source_color_linear>1)):
            raise ValueError("source_color_linear must be 3 finite values in [0,1]")
    rows = {}
    for family in ("lamp_silhouette","bbox_silhouette"):
        support = np.asarray(supports[family],bool)
        if support.shape != (704,1280):
            raise ValueError("Production spatial fields must be native1280x704")
        if operation=="remove" and not support.any():
            raise ValueError("Remove needs actual original-source support")
        dest = output/family/f"{frame:06d}.npz"
        rows[family] = fields.fields(dest,support,fields.gamma_pairs(operation))
        if source_color_linear is not None:
            with np.load(dest,allow_pickle=False) as z: arrays={k:z[k] for k in z.files}
            arrays["color_source_hint"]=arrays["support"][...,None].astype(np.float32)*source_color_linear
            arrays["source_color_linear_rgb"]=source_color_linear
            np.savez_compressed(dest,**arrays)
        fields.png(output/family/"support"/f"{frame:06d}.png",support)
    output.mkdir(parents=True,exist_ok=True)
    (output/"COMPONENT_ONLY.json").write_text(json.dumps(dict(operation=operation,frame=frame,complete_case=False,requires_remaining=["49-frame A/B source conditions for Remove","actual source/reference binding","scene visibility review","complete reader","Quality before HF"]),indent=2))
    return rows

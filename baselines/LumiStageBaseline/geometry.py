"""Project fixed-world StarReal OBBs through all 49 A/B world-to-camera matrices."""
import itertools
from pathlib import Path
from common import bound, require, write


def projected_boxes(np, center, dimensions, rotation, K, w2c):
    center, dimensions, rotation = map(np.asarray, (center, dimensions, rotation))
    require(center.shape == dimensions.shape == (3,) and rotation.shape == (3, 3), "Invalid OBB")
    require(np.isfinite([*center, *dimensions, *rotation.flat]).all() and np.all(dimensions > 0), "Invalid OBB values")
    signs = np.array(list(itertools.product([-1, 1], repeat=3)))
    corners = signs * (dimensions / 2) @ rotation.T + center
    xyz = np.einsum("tij,nj->tni", w2c[:, :3, :3], corners) + w2c[:, None, :3, 3]
    pix = np.einsum("tij,tnj->tni", K, xyz)
    result = []
    for points, world in zip(pix, xyz):
        if (world[:, 2] <= 0).any():
            result.append(None)
        else:
            uv = points[:, :2] / points[:, 2, None]
            result.append(np.r_[uv.min(axis=0), uv.max(axis=0)].tolist())
    return result


def project(data, root):
    import numpy as np
    with np.load(bound(data["camera"], root), allow_pickle=False) as cam:
        K = cam["K"]
        if K.shape == (3, 3):
            K = np.broadcast_to(K, (49, 3, 3))
        paths = {key: cam["w2c_" + key] for key in ("A", "B")}
        require(K.shape == (49, 3, 3) and np.isfinite(K).all() and all(
            w.shape == (49, 4, 4) and np.isfinite(w).all() for w in paths.values()), "Invalid 49-frame camera pack")
        authority = data.get("geometry") or data["camera"]
        with np.load(bound(authority, root), allow_pickle=False) as geom:
            op = data["operation"]
            roles = ("Q",) if op == "insert" else ("P", "Q") if op in ("move", "replace", "move_replace") else ("P",)
            boxes = {}
            fields = data.get("geometry_fields", {})
            for name in roles:
                keys = fields.get(name, [name, name + "_size", name + "_rotation"])
                require(len(keys) == 3 and all(k in geom for k in keys), "Unbound geometry for " + name)
                boxes[name] = tuple(geom[k] for k in keys)
            if op == "replace":
                require(np.allclose(boxes["P"][0], boxes["Q"][0], atol=1e-6, rtol=0),
                        "Pure replacement target must be the source OBB center P; prepare an explicit pure-P packet")
            return {track: {name: projected_boxes(np, *box, K, w) for name, box in boxes.items()}
                    for track, w in paths.items()}

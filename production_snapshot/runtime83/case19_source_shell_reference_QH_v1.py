from pathlib import Path
import sys
R=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
sys.path[:0]=[str(R/'runtime/qh_cpu_py39_v1'),str(R/'inputs/case19_native_CPU_v1')]
import numpy as np
from real_geometry import unproject

def regularized_shell(mask, K, A, anchors, segments):
    h, w = mask.shape
    y, x = np.where(mask); ids = np.arange((h+1)*(w+1)).reshape(h+1, w+1)
    a, b, c, d = ids[y, x], ids[y, x+1], ids[y+1, x], ids[y+1, x+1]
    front_faces = np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)])
    used = np.unique(front_faces); mapping = np.full((h+1)*(w+1), -1, np.int32)
    mapping[used] = np.arange(len(used))
    uv = np.c_[used % (w+1) - .5, used // (w+1) - .5]
    nearest = np.full(len(uv), np.inf); z = np.zeros(len(uv)); thickness = np.zeros(len(uv))
    for left, right, pixels in segments:
        p, q = anchors[left], anchors[right]
        direction = q[:2] - p[:2]
        t = np.clip(((uv-p[:2]) @ direction) / max(float(direction@direction), 1e-12), 0, 1)
        distance = ((uv-p[:2]-t[:, None]*direction)**2).sum(1)
        take = distance < nearest
        # Perspective-correct segment interpolation from actual joint depths.
        z[take] = 1 / ((1-t[take])/p[2] + t[take]/q[2])
        thickness[take] = pixels*z[take]/K[0, 0]
        nearest[take] = distance[take]
    front = unproject(uv, z, K, A)
    back = unproject(uv, z+thickness, K, A)
    ff = mapping[front_faces].astype(np.int32); n = len(front)
    faces = [ff, ff[:, ::-1]+n]
    padded = np.pad(mask, 1)
    sides = [(~padded[y, x+1], a, b), (~padded[y+2, x+1], d, c),
             (~padded[y+1, x], c, a), (~padded[y+1, x+2], b, d)]
    for keep, left, right in sides:
        u, v = mapping[left[keep]], mapping[right[keep]]
        faces.extend([np.stack([u, v, v+n], 1), np.stack([u, v+n, u+n], 1)])
    return np.concatenate([front, back]), np.concatenate(faces).astype(np.int32), z

def rotation_on_plane(up, old):
    x = old[:, 0].copy(); x -= up * (x @ up); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up], 1)

def overlay(rgb, layer):
    img = rgb.copy()
    sel = layer['visible_known_estimate']
    img[sel] = np.rint(.45 * img[sel] + .55 * np.array([255, 0, 255])).astype(np.uint8)
    unknown = layer['unobserved_scene_region']
    img[unknown] = np.rint(.5 * img[unknown] + .5 * np.array([255, 255, 0])).astype(np.uint8)
    return img

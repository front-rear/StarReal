"""Explicit relative-world geometry and CPU asset/scene visibility.

No calibration or hidden-surface truth is implied. Both rasterized depths use
camera-z; PCF ray distance is a separate quantity and must be converted.
"""
import ctypes
from pathlib import Path

import cv2
import numpy as np


def project(vertices, K, w2c):
    camera = np.asarray(vertices) @ w2c[:3, :3].T + w2c[:3, 3]
    pixels = camera @ K.T
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = pixels[:, :2] / pixels[:, 2:3]
    return uv, camera[:, 2]


def unproject(uv, camera_z, K, w2c):
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    camera = rays * np.asarray(camera_z).reshape(-1, 1)
    return (camera - w2c[:3, 3]) @ w2c[:3, :3]


def translate_from_mount(vertices, source_mount, target_mount):
    """Same original fixture, all axes and physical dimensions unchanged."""
    return np.asarray(vertices) + np.asarray(target_mount) - np.asarray(source_mount)


def place_asset(vertices, local_mount, world_mount, rotation, scale):
    """Place a reviewed semantic mount, not the model's center, on support."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) <= 0:
        raise ValueError('proper rigid local-to-world rotation required')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('uniform positive scale required')
    return (np.asarray(vertices) - local_mount) * scale @ rotation.T + world_mount


def bbox_for_mesh(vertices, rotation):
    local = np.asarray(vertices) @ rotation
    lo, hi = local.min(0), local.max(0)
    return ((lo+hi)/2) @ rotation.T, hi-lo, rotation


def plane_intersection(pixel, K, w2c, normal, plane_d):
    origin = np.linalg.inv(w2c)[:3, 3]
    direction = w2c[:3, :3].T @ (np.linalg.inv(K) @ np.r_[pixel, 1.])
    denominator = float(np.asarray(normal) @ direction)
    if abs(denominator) < 1e-12:
        raise ValueError('ray parallel to reviewed support plane')
    t = -(float(np.asarray(normal) @ origin)+plane_d)/denominator
    if t <= 0:
        raise ValueError('support intersection behind camera')
    return origin+t*direction


def depth_surface(depth, K, w2c, valid_pixels=None, max_neighbor_ratio=None):
    """Triangulated predicted camera-z, preserving unobserved/discontinuous gaps."""
    h,w = depth.shape
    yy,xx = np.indices((h,w))
    vertices = unproject(np.c_[xx.ravel(), yy.ravel()], depth.ravel(), K, w2c)
    ids = np.arange(h*w).reshape(h,w)
    faces = np.concatenate([np.stack([ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1]],-1).reshape(-1,3),
                            np.stack([ids[:-1,1:],ids[1:,1:],ids[1:,:-1]],-1).reshape(-1,3)]).astype(np.int32)
    valid = np.isfinite(depth) & (depth>0)
    if valid_pixels is not None:
        valid &= valid_pixels
    keep = valid.ravel()[faces].all(1)
    if max_neighbor_ratio is not None:
        z=depth.ravel()[faces]
        keep &= z.max(1) <= max_neighbor_ratio*np.maximum(z.min(1),1e-20)
    return vertices, faces[keep]


def visible_surface_proxy(mask, depth, K, w2c):
    """One surface quad for each observed SAM pixel; no flat-depth homography.

    Geometry is estimated from the depth head. It is not a complete original
    fixture mesh. Hidden components must be supplied/labeled separately.
    """
    h,w=mask.shape
    full=cv2.resize(depth,(w,h),interpolation=cv2.INTER_LINEAR)
    # Interpolate at pixel boundaries so adjacent foreground quads share vertices.
    xx,yy=np.meshgrid(np.arange(w+1,dtype=np.float32)-.5,np.arange(h+1,dtype=np.float32)-.5)
    zz=cv2.remap(full,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    vertices=unproject(np.c_[xx.ravel(),yy.ravel()],zz.ravel(),K,w2c)
    ids=np.arange((h+1)*(w+1)).reshape(h+1,w+1)
    y,x=np.where(mask)
    a,b,c,d=ids[y,x],ids[y,x+1],ids[y+1,x],ids[y+1,x+1]
    faces=np.concatenate([np.stack([a,b,c],1),np.stack([b,d,c],1)]).astype(np.int32)
    used=np.unique(faces)
    mapping=np.full(len(vertices),-1,np.int32);mapping[used]=np.arange(len(used))
    return vertices[used],mapping[faces]


class Rasterizer:
    def __init__(self,library=None):
        library=library or Path(__file__).with_name('mesh_zbuffer.so')
        self.lib=ctypes.CDLL(str(library))
        array=np.ctypeslib.ndpointer
        self.lib.raster_depth.argtypes=[array(np.float64,flags='C_CONTIGUOUS'),ctypes.c_int64,
            array(np.int32,flags='C_CONTIGUOUS'),ctypes.c_int64,array(np.float64,flags='C_CONTIGUOUS'),
            ctypes.c_int,ctypes.c_int,ctypes.c_double,array(np.float32,flags='C_CONTIGUOUS')]
        self.lib.raster_depth.restype=ctypes.c_int

    def depth(self,vertices,faces,K,w2c,width=1280,height=704):
        vertices=np.asarray(vertices,dtype=np.float64)
        camera=np.ascontiguousarray(vertices@w2c[:3,:3].T+w2c[:3,3])
        faces=np.ascontiguousarray(faces,dtype=np.int32)
        K=np.ascontiguousarray(K,dtype=np.float64)
        if K.shape!=(3,3) or abs(K[0,1])+abs(K[1,0])>1e-12:
            raise ValueError('zero-skew pinhole K required')
        output=np.empty((height,width),np.float32)
        code=self.lib.raster_depth(camera,len(camera),faces,len(faces),K,width,height,1e-6,output)
        if code:
            raise RuntimeError(f'rasterizer returned {code}')
        return output


def visibility(asset_depth,scene_depth):
    amodal=np.isfinite(asset_depth)
    known=np.isfinite(scene_depth)
    # Epsilon is only floating-point tie handling, not a brightness/area gate.
    occluded=amodal & known & (scene_depth < asset_depth-1e-6)
    unknown=amodal & ~known
    return {'amodal':amodal,'scene_occluded_estimate':occluded,
            'visible_known_estimate':amodal & known & ~occluded,
            'unobserved_scene_region':unknown,
            'visible_with_unknown_estimate':amodal & ~occluded}


def camera_z_to_ray_distance(depth,K):
    yy,xx=np.indices(depth.shape)
    rays=np.c_[xx.ravel(),yy.ravel(),np.ones(depth.size)]@np.linalg.inv(K).T
    return depth*np.linalg.norm(rays,axis=1).reshape(depth.shape)

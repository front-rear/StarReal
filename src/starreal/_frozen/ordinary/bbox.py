"""Render ordinary world-space oriented bounding boxes as lamp proxies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

try:
    from .gaussian import camera_rays, quaternion_xyzw_to_matrix
except ImportError:  # Direct script execution.
    from gaussian import camera_rays, quaternion_xyzw_to_matrix


@dataclass(frozen=True)
class OrientedBoundingBox:
    identifier: str
    center_xyz: np.ndarray
    size_xyz: np.ndarray
    rotation_world: np.ndarray
    color_rgb: np.ndarray

    @property
    def half_size(self) -> np.ndarray:
        return 0.5 * self.size_xyz

    @property
    def vertices_world(self) -> np.ndarray:
        signs = np.array(
            [
                [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
            ],
            dtype=np.float64,
        )
        local = signs * self.half_size
        return local @ self.rotation_world.T + self.center_xyz


BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def load_boxes(path: Path | str) -> Tuple[List[OrientedBoundingBox], dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    default_color = document.get("color_rgb", [255, 48, 220])
    raw_boxes = document.get("boxes")
    if not isinstance(raw_boxes, list) or not raw_boxes:
        raise ValueError("Bounding-box JSON must contain a non-empty 'boxes' list.")
    boxes = []
    for index, item in enumerate(raw_boxes):
        center = np.asarray(item.get("center_xyz"), dtype=np.float64)
        size = np.asarray(item.get("size_xyz"), dtype=np.float64)
        color = np.asarray(item.get("color_rgb", default_color), dtype=np.float64)
        if center.shape != (3,) or size.shape != (3,) or color.shape != (3,):
            raise ValueError(
                f"Box {index}: center_xyz, size_xyz and color_rgb must each contain 3 values."
            )
        if not np.isfinite(center).all() or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError(f"Box {index}: center and positive size must be finite.")
        if not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 255):
            raise ValueError(f"Box {index}: color_rgb must be in [0,255].")
        boxes.append(
            OrientedBoundingBox(
                identifier=str(item.get("id", f"lamp_{index}")),
                center_xyz=center,
                size_xyz=size,
                rotation_world=quaternion_xyzw_to_matrix(
                    item.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
                ),
                color_rgb=np.rint(color).astype(np.uint8),
            )
        )
    return boxes, document


def _ray_box_depth(
    box: OrientedBoundingBox,
    rays_camera: np.ndarray,
    w2c: np.ndarray,
    near: float,
) -> np.ndarray:
    """Return camera-z at the first positive ray/OBB intersection."""

    c2w = np.linalg.inv(w2c)
    camera_center_world = c2w[:3, 3]
    rotation_c2w = c2w[:3, :3]
    origin_local = box.rotation_world.T @ (camera_center_world - box.center_xyz)
    rays_world = rays_camera @ rotation_c2w.T
    rays_local = rays_world @ box.rotation_world

    parallel_outside = (np.abs(rays_local) < 1e-12) & (
        np.abs(origin_local)[None, None, :] > box.half_size[None, None, :]
    )

    safe_direction = np.where(
        np.abs(rays_local) < 1e-12,
        np.where(rays_local >= 0, 1e-12, -1e-12),
        rays_local,
    )
    t_0 = (-box.half_size - origin_local) / safe_direction
    t_1 = (box.half_size - origin_local) / safe_direction
    t_near = np.max(np.minimum(t_0, t_1), axis=-1)
    t_far = np.min(np.maximum(t_0, t_1), axis=-1)
    first = np.where(t_near >= near, t_near, t_far)
    valid = (
        (t_far >= np.maximum(t_near, near))
        & (first >= near)
        & ~np.any(parallel_outside, axis=-1)
    )
    return np.where(valid, first, np.inf)


def _project_vertices(
    vertices_world: np.ndarray,
    intrinsics: np.ndarray,
    w2c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    vertices_camera = vertices_world @ w2c[:3, :3].T + w2c[:3, 3]
    valid = vertices_camera[:, 2] > 1e-6
    normalized = vertices_camera / np.maximum(vertices_camera[:, 2:3], 1e-6)
    pixels = normalized @ intrinsics.T
    return pixels[:, :2], valid


def render_bbox_conditions(
    boxes: Iterable[OrientedBoundingBox],
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    height: int,
    width: int,
    near: float = 0.05,
    depth_far: float = 20.0,
    edge_width: int = 4,
    fill_alpha: float = 0.38,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render filled OBB silhouettes, 12 box edges, encoded depth, and raw depth."""

    if near <= 0 or depth_far <= near:
        raise ValueError("Require 0 < near < depth_far.")
    rays = camera_rays(intrinsics, height, width)
    raw_depth = np.full((height, width), np.inf, dtype=np.float64)
    identity_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    box_mask = np.zeros((height, width), dtype=np.uint8)
    box_list = list(boxes)

    for box in box_list:
        depth = _ray_box_depth(box, rays, w2c, near)
        replace = depth < raw_depth
        if np.any(replace):
            raw_depth[replace] = depth[replace]
            identity_rgb[replace] = np.rint(
                box.color_rgb.astype(np.float64) * fill_alpha
            ).astype(np.uint8)
            box_mask[replace] = 255

    rectangle = (0, 0, width, height)
    for box in box_list:
        pixels, valid_vertices = _project_vertices(box.vertices_world, intrinsics, w2c)
        color = tuple(int(channel) for channel in box.color_rgb)
        for start_index, end_index in BOX_EDGES:
            if not (valid_vertices[start_index] and valid_vertices[end_index]):
                continue
            point_0 = tuple(np.rint(np.clip(pixels[start_index], -1e6, 1e6)).astype(int))
            point_1 = tuple(np.rint(np.clip(pixels[end_index], -1e6, 1e6)).astype(int))
            visible, clipped_0, clipped_1 = cv2.clipLine(rectangle, point_0, point_1)
            if visible:
                cv2.line(identity_rgb, clipped_0, clipped_1, color, edge_width, cv2.LINE_AA)
                cv2.line(box_mask, clipped_0, clipped_1, 255, edge_width, cv2.LINE_AA)

    valid_depth = np.isfinite(raw_depth)
    raw_depth_output = raw_depth.astype(np.float32)
    raw_depth_output[~valid_depth] = 0.0
    encoded = np.zeros((height, width), dtype=np.float64)
    encoded[valid_depth] = 1.0 - np.clip(
        (raw_depth[valid_depth] - near) / (depth_far - near), 0.0, 1.0
    )
    encoded_uint8 = np.rint(encoded * 255.0).astype(np.uint8)
    depth_rgb = np.repeat(encoded_uint8[..., None], 3, axis=-1)
    return identity_rgb, depth_rgb, box_mask, raw_depth_output

"""Analytic projection of lamp-position 3D Gaussian ellipsoids."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GaussianEllipsoid:
    """A world-space Gaussian used as a lamp placement proxy.

    `sigma_xyz` are the Gaussian standard deviations along local principal axes.
    The visible ellipsoid surface is the `level_sigma` Mahalanobis isosurface.
    """

    identifier: str
    mean_xyz: np.ndarray
    sigma_xyz: np.ndarray
    rotation_world: np.ndarray
    color_rgb: np.ndarray
    level_sigma: float
    projection_sigma: float

    @property
    def covariance_world(self) -> np.ndarray:
        return self.rotation_world @ np.diag(self.sigma_xyz**2) @ self.rotation_world.T


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("rotation_xyzw must contain [x,y,z,w].")
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("rotation_xyzw cannot be the zero quaternion.")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _parse_gaussian(item: dict, index: int, defaults: dict) -> GaussianEllipsoid:
    mean = np.asarray(item.get("mean_xyz"), dtype=np.float64)
    sigma = np.asarray(item.get("sigma_xyz"), dtype=np.float64)
    color = np.asarray(item.get("color_rgb", defaults["color_rgb"]), dtype=np.float64)
    if mean.shape != (3,) or sigma.shape != (3,) or color.shape != (3,):
        raise ValueError(
            f"Gaussian {index}: mean_xyz, sigma_xyz and color_rgb must each contain 3 values."
        )
    if not np.isfinite(mean).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError(f"Gaussian {index}: mean and positive sigma must be finite.")
    if not np.isfinite(color).all() or np.any(color < 0) or np.any(color > 255):
        raise ValueError(f"Gaussian {index}: color_rgb must be in [0,255].")

    level_sigma = float(item.get("level_sigma", defaults["level_sigma"]))
    projection_sigma = float(item.get("projection_sigma", defaults["projection_sigma"]))
    if level_sigma <= 0 or projection_sigma < level_sigma:
        raise ValueError(
            f"Gaussian {index}: require projection_sigma >= level_sigma > 0."
        )
    return GaussianEllipsoid(
        identifier=str(item.get("id", f"lamp_{index}")),
        mean_xyz=mean,
        sigma_xyz=sigma,
        rotation_world=quaternion_xyzw_to_matrix(
            item.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
        ),
        color_rgb=np.rint(color).astype(np.uint8),
        level_sigma=level_sigma,
        projection_sigma=projection_sigma,
    )


def load_gaussians(path: Path | str) -> Tuple[List[GaussianEllipsoid], dict]:
    """Load the documented Gaussian JSON format."""

    with Path(path).open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    defaults = {
        "level_sigma": float(document.get("level_sigma", 2.0)),
        "projection_sigma": float(document.get("projection_sigma", 3.0)),
        "color_rgb": document.get("color_rgb", [255, 96, 24]),
    }
    raw_gaussians = document.get("gaussians")
    if not isinstance(raw_gaussians, list) or not raw_gaussians:
        raise ValueError("Gaussian JSON must contain a non-empty 'gaussians' list.")
    gaussians = [_parse_gaussian(item, index, defaults) for index, item in enumerate(raw_gaussians)]
    return gaussians, document


def camera_rays(intrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return unnormalized camera rays [x/z,y/z,1], so ray parameter equals z depth."""

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    skew = float(intrinsics[0, 1])
    y = (yy - cy) / fy
    x = (xx - cx - skew * y) / fx
    return np.stack([x, y, np.ones_like(x)], axis=-1)


def render_gaussian_conditions(
    gaussians: Iterable[GaussianEllipsoid],
    intrinsics: np.ndarray,
    w2c: np.ndarray,
    height: int,
    width: int,
    near: float = 0.05,
    depth_far: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render placement RGB, encoded surface depth, alpha, and raw z depth.

    RGB uses fixed identity colors multiplied by projected Gaussian density. Depth
    is the first ray intersection with the `level_sigma` ellipsoid. The 8-bit
    depth image maps near to white and depth_far to black; invalid pixels are
    black and can be distinguished using the RGB/alpha support.
    """

    if near <= 0 or depth_far <= near:
        raise ValueError("Require 0 < near < depth_far.")
    rays = camera_rays(intrinsics, height, width)
    placement_rgb = np.zeros((height, width, 3), dtype=np.float64)
    placement_alpha = np.zeros((height, width), dtype=np.float64)
    surface_depth = np.full((height, width), np.inf, dtype=np.float64)
    color_order_depth = np.full((height, width), np.inf, dtype=np.float64)

    rotation_w2c = np.asarray(w2c[:3, :3], dtype=np.float64)
    translation_w2c = np.asarray(w2c[:3, 3], dtype=np.float64)
    for gaussian in gaussians:
        mean_camera = rotation_w2c @ gaussian.mean_xyz + translation_w2c
        covariance_camera = rotation_w2c @ gaussian.covariance_world @ rotation_w2c.T
        precision_camera = np.linalg.inv(covariance_camera)

        ray_precision = np.einsum("hwi,ij->hwj", rays, precision_camera)
        quadratic_a = np.einsum("hwi,hwi->hw", ray_precision, rays)
        ray_dot_mean = np.einsum("hwi,i->hw", ray_precision, mean_camera)
        mean_quadratic = float(mean_camera @ precision_camera @ mean_camera)
        closest_depth = ray_dot_mean / np.maximum(quadratic_a, 1e-12)
        closest_distance_squared = np.maximum(
            0.0, mean_quadratic - ray_dot_mean**2 / np.maximum(quadratic_a, 1e-12)
        )

        projected = (
            (closest_depth > near)
            & (closest_distance_squared <= gaussian.projection_sigma**2)
        )
        alpha = np.exp(-0.5 * closest_distance_squared)
        alpha *= projected

        # Keep the closest Gaussian's identity color at each pixel. This is
        # deterministic for overlapping lamp proxies and sufficient for the
        # initial single-lamp task.
        replace_color = projected & (closest_depth < color_order_depth)
        if np.any(replace_color):
            normalized_color = gaussian.color_rgb.astype(np.float64) / 255.0
            placement_rgb[replace_color] = alpha[replace_color, None] * normalized_color
            placement_alpha[replace_color] = alpha[replace_color]
            color_order_depth[replace_color] = closest_depth[replace_color]

        constant = mean_quadratic - gaussian.level_sigma**2
        discriminant = ray_dot_mean**2 - quadratic_a * constant
        root_delta = np.sqrt(np.maximum(discriminant, 0.0))
        root_near = (ray_dot_mean - root_delta) / np.maximum(quadratic_a, 1e-12)
        root_far = (ray_dot_mean + root_delta) / np.maximum(quadratic_a, 1e-12)
        first_positive_root = np.where(root_near >= near, root_near, root_far)
        intersects = (discriminant >= 0.0) & (first_positive_root >= near)
        replace_depth = intersects & (first_positive_root < surface_depth)
        surface_depth[replace_depth] = first_positive_root[replace_depth]

    raw_depth = surface_depth.astype(np.float32)
    valid_depth = np.isfinite(raw_depth)
    raw_depth[~valid_depth] = 0.0
    depth_value = np.zeros((height, width), dtype=np.float64)
    depth_value[valid_depth] = 1.0 - np.clip(
        (raw_depth[valid_depth] - near) / (depth_far - near), 0.0, 1.0
    )
    depth_uint8 = np.rint(depth_value * 255.0).astype(np.uint8)
    depth_rgb = np.repeat(depth_uint8[..., None], 3, axis=-1)
    placement_uint8 = np.rint(np.clip(placement_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    alpha_uint8 = np.rint(np.clip(placement_alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
    return placement_uint8, depth_rgb, alpha_uint8, raw_depth

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .compliance import LoadEvidence
from .geometry import CameraModel, wrap_axis_angle_deg


def _as_points(points: np.ndarray, minimum: int = 1) -> np.ndarray:
    value = np.asarray(points, dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or len(value) < minimum or not np.isfinite(value).all():
        raise ValueError(f"expected at least {minimum} finite 2D points")
    return value


def _pallet_basis(pallet_floor_corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    corners = _as_points(pallet_floor_corners, 4)
    if corners.shape != (4, 2):
        raise ValueError("pallet corners must be ordered FL, FR, BR, BL")
    front = corners[1] - corners[0]
    side = corners[3] - corners[0]
    width, depth = float(np.linalg.norm(front)), float(np.linalg.norm(side))
    if width < 1e-6 or depth < 1e-6:
        raise ValueError("degenerate pallet footprint")
    return corners[0], front / width, side / depth, width, depth


def footprint_metrics(pallet_floor_corners: np.ndarray, load_floor_points: np.ndarray) -> tuple[float, float]:
    """Return maximum planar overhang and load-centroid offset in metres.

    `load_floor_points` must describe the load contact footprint. Passing the top silhouette is
    geometrically invalid and callers should abstain when the contact edge is occluded.
    """
    load = _as_points(load_floor_points, 1)
    origin, x_axis, y_axis, width, depth = _pallet_basis(pallet_floor_corners)
    relative = load - origin
    local = np.c_[relative @ x_axis, relative @ y_axis]
    outside = np.c_[np.maximum(-local[:, 0], 0), np.maximum(local[:, 0] - width, 0),
                    np.maximum(-local[:, 1], 0), np.maximum(local[:, 1] - depth, 0)]
    overhang = float(outside.max())
    pallet_centre = np.array([width / 2, depth / 2])
    centroid_offset = float(np.linalg.norm(local.mean(axis=0) - pallet_centre))
    return overhang, centroid_offset


def box_alignment_angle_deg(box_floor_polygon: np.ndarray, pallet_theta_deg: float) -> float:
    """Smallest angle between a box principal edge and either pallet axis (0..45 degrees)."""
    polygon = _as_points(box_floor_polygon, 4)
    centred = polygon - polygon.mean(axis=0)
    covariance = centred.T @ centred
    values, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, int(np.argmax(values))]
    box_angle = math.degrees(math.atan2(principal[1], principal[0]))
    delta = abs(wrap_axis_angle_deg(box_angle - pallet_theta_deg)) % 90.0
    return float(min(delta, 90.0 - delta))


@dataclass(frozen=True)
class PinholeCamera:
    """Projection matrix mapping floor-frame XYZ metres to image pixels."""

    projection_world_to_image: np.ndarray

    def __post_init__(self) -> None:
        projection = np.asarray(self.projection_world_to_image, dtype=float)
        if projection.shape != (3, 4) or not np.isfinite(projection).all():
            raise ValueError("projection matrix must be finite 3x4")
        object.__setattr__(self, "projection_world_to_image", projection)

    def project(self, points_xyz: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xyz, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be Nx3")
        projected = (self.projection_world_to_image @ np.c_[points, np.ones(len(points))].T).T
        if np.any(np.abs(projected[:, 2]) < 1e-9):
            raise ValueError("point projects to infinity")
        return projected[:, :2] / projected[:, 2, None]

    def vertical_height(self, base_xy_m: np.ndarray, top_pixel: np.ndarray) -> tuple[float, float]:
        """Solve height on the world-vertical line through a known floor contact point.

        Returns `(height_m, algebraic_residual_px_like)`. A high residual signals that the chosen
        top point is not vertically above the base or calibration is inconsistent.
        """
        base = np.asarray(base_xy_m, dtype=float)
        pixel = np.asarray(top_pixel, dtype=float)
        if base.shape != (2,) or pixel.shape != (2,) or not np.isfinite(np.r_[base, pixel]).all():
            raise ValueError("base_xy_m and top_pixel must be finite 2-vectors")
        p = self.projection_world_to_image
        xyz1_at_zero = np.array([base[0], base[1], 0.0, 1.0])
        constants = p @ xyz1_at_zero
        u, v = pixel
        a = np.array([p[0, 2] - u * p[2, 2], p[1, 2] - v * p[2, 2]])[:, None]
        b = np.array([u * constants[2] - constants[0], v * constants[2] - constants[1]])
        height, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        residual = float(np.linalg.norm(a[:, 0] * height[0] - b))
        return float(height[0]), residual


def evidence_from_floor_geometry(
    camera: CameraModel,
    pallet_keypoints_px: np.ndarray,
    load_contact_px: np.ndarray | None,
    box_polygons_px: list[np.ndarray],
    pallet_theta_deg: float,
    visibility: float,
    **non_geometric: float | None,
) -> LoadEvidence:
    pallet_floor = camera.floor_points(_as_points(pallet_keypoints_px, 4))
    overhang = centroid = None
    if load_contact_px is not None:
        overhang, centroid = footprint_metrics(pallet_floor, camera.floor_points(load_contact_px))
    angles = tuple(box_alignment_angle_deg(camera.floor_points(poly), pallet_theta_deg) for poly in box_polygons_px)
    return LoadEvidence(overhang_m=overhang, centroid_offset_m=centroid, box_angles_deg=angles,
                        visibility=float(np.clip(visibility, 0, 1)), **non_geometric)


from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .models import Pose


@dataclass(frozen=True)
class CameraModel:
    """Pinhole camera and camera-to-floor homography (image px -> floor metres)."""

    homography_image_to_floor: np.ndarray
    reprojection_rmse_px: float = 0.0

    def __post_init__(self) -> None:
        h = np.asarray(self.homography_image_to_floor, dtype=float)
        if h.shape != (3, 3) or not np.isfinite(h).all():
            raise ValueError("homography must be a finite 3x3 matrix")
        if abs(np.linalg.det(h)) < 1e-12:
            raise ValueError("homography is singular")
        object.__setattr__(self, "homography_image_to_floor", h)

    def floor_points(self, image_points: np.ndarray) -> np.ndarray:
        points = np.asarray(image_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("image_points must be Nx2")
        homogeneous = np.c_[points, np.ones(len(points))]
        mapped = (self.homography_image_to_floor @ homogeneous.T).T
        if np.any(np.abs(mapped[:, 2]) < 1e-9):
            raise ValueError("point maps to infinity")
        return mapped[:, :2] / mapped[:, 2, None]


def wrap_axis_angle_deg(angle: float) -> float:
    """Return a directed pallet angle in [-180, 180). Face labels remove 180-deg ambiguity."""
    return (angle + 180.0) % 360.0 - 180.0


def pose_from_keypoints(
    keypoints_px: np.ndarray,
    camera: CameraModel,
    keypoint_confidences: np.ndarray | None = None,
    pixel_sigma: float = 2.0,
    max_position_sigma_m: float = 0.02,
    max_angle_sigma_deg: float = 3.0,
) -> Pose:
    """Estimate pose from ordered FL, FR, BR, BL pallet floor corners.

    FL->FR defines the labelled front face and therefore directed orientation.
    Uncertainty is propagated by a deterministic finite-difference Jacobian.
    """
    points = np.asarray(keypoints_px, dtype=float)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("keypoints must be four finite ordered Nx2 points")
    conf = np.ones(4) if keypoint_confidences is None else np.asarray(keypoint_confidences)
    if conf.shape != (4,):
        raise ValueError("keypoint_confidences must have four values")
    floor = camera.floor_points(points)
    centre = floor.mean(axis=0)
    front_axis = floor[1] - floor[0]
    if np.linalg.norm(front_axis) < 1e-6:
        raise ValueError("front keypoints are degenerate")
    theta = wrap_axis_angle_deg(math.degrees(math.atan2(front_axis[1], front_axis[0])))

    samples: list[tuple[float, float, float]] = []
    for idx in range(4):
        for dim in range(2):
            for sign in (-1.0, 1.0):
                shifted = points.copy()
                shifted[idx, dim] += sign * pixel_sigma
                sf = camera.floor_points(shifted)
                sc = sf.mean(axis=0)
                sa = sf[1] - sf[0]
                st = wrap_axis_angle_deg(math.degrees(math.atan2(sa[1], sa[0])))
                angle_delta = wrap_axis_angle_deg(st - theta)
                samples.append((sc[0], sc[1], theta + angle_delta))
    spread = np.std(np.asarray(samples), axis=0, ddof=1)
    calibration_m = camera.reprojection_rmse_px * pixel_sigma * 0.001
    sigma_x = float(math.hypot(spread[0], calibration_m))
    sigma_y = float(math.hypot(spread[1], calibration_m))
    sigma_theta = float(spread[2])
    min_conf = float(np.clip(conf.min(), 0.0, 1.0))
    reliable = (
        min_conf >= 0.5
        and sigma_x <= max_position_sigma_m
        and sigma_y <= max_position_sigma_m
        and sigma_theta <= max_angle_sigma_deg
    )
    reasons = []
    if min_conf < 0.5:
        reasons.append("low keypoint confidence")
    if max(sigma_x, sigma_y) > max_position_sigma_m:
        reasons.append("position uncertainty exceeds 2 cm")
    if sigma_theta > max_angle_sigma_deg:
        reasons.append("orientation uncertainty exceeds 3 degrees")
    return Pose(
        x_m=float(centre[0]), y_m=float(centre[1]), theta_deg=theta,
        sigma_x_m=sigma_x, sigma_y_m=sigma_y, sigma_theta_deg=sigma_theta,
        confidence=min_conf, reliable=reliable, reason="; ".join(reasons),
    )


def slot_error(pose: Pose, slot_x_m: float, slot_y_m: float, slot_theta_deg: float) -> dict[str, float | bool]:
    translation = math.hypot(pose.x_m - slot_x_m, pose.y_m - slot_y_m)
    rotation = abs(wrap_axis_angle_deg(pose.theta_deg - slot_theta_deg))
    epsilon = 1e-9
    return {
        "translation_error_m": translation,
        "rotation_error_deg": rotation,
        "compliant": pose.reliable and translation <= 0.02 + epsilon and rotation <= 3.0 + epsilon,
    }

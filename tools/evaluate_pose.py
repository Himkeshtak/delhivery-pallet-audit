from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pallet_audit.geometry import CameraModel, pose_from_keypoints, wrap_axis_angle_deg


def percentiles(values: list[float]) -> dict[str, float]:
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 95, 99)}


def evaluate(records: list[dict]) -> dict:
    translation, rotation, abstained = [], [], 0
    for row in records:
        camera = CameraModel(np.asarray(row["homography_image_to_floor"]), row.get("reprojection_rmse_px", 0))
        pose = pose_from_keypoints(np.asarray(row["keypoints_px"]), camera, np.asarray(row.get("confidences", [1]*4)))
        if not pose.reliable:
            abstained += 1
        gt = row["ground_truth"]
        translation.append(float(np.hypot(pose.x_m-gt["x_m"], pose.y_m-gt["y_m"])))
        rotation.append(abs(wrap_axis_angle_deg(pose.theta_deg-gt["theta_deg"])))
    return {"sample_count": len(records), "abstention_rate": abstained/max(len(records), 1),
            "translation_error_m": percentiles(translation), "rotation_error_deg": percentiles(rotation),
            "meets_bar_rate": float(np.mean((np.asarray(translation) <= .02) & (np.asarray(rotation) <= 3)))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("pose_metrics.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(evaluate(json.loads(args.input.read_text())), indent=2), encoding="utf-8")


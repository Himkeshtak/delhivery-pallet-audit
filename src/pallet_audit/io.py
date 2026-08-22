from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .compliance import LoadEvidence
from .geometry import CameraModel
from .pipeline import assess_pallet


def assess_request(payload: dict) -> dict:
    camera = CameraModel(np.asarray(payload["camera"]["homography_image_to_floor"], dtype=float),
                         float(payload["camera"].get("reprojection_rmse_px", 0.0)))
    evidence = LoadEvidence(**payload.get("evidence", {}))
    result = assess_pallet(
        payload["image_id"], payload["pallet_id"], np.asarray(payload["keypoints_px"], dtype=float),
        np.asarray(payload.get("keypoint_confidences", [1, 1, 1, 1]), dtype=float), camera, evidence,
        payload.get("model_version", "external-keypoints"),
    )
    return result.to_dict()


def run_file(input_path: str | Path, output_path: str | Path) -> None:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    Path(output_path).write_text(json.dumps(assess_request(payload), indent=2), encoding="utf-8")


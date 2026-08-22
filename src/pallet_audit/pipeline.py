from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .compliance import LoadEvidence, assess_sop, overall_verdict
from .geometry import CameraModel, pose_from_keypoints
from .models import Assessment


def assess_pallet(image_id: str, pallet_id: str, keypoints_px: np.ndarray,
                  keypoint_confidences: np.ndarray, camera: CameraModel,
                  evidence: LoadEvidence, model_version: str = "geometry-only-demo") -> Assessment:
    pose = pose_from_keypoints(keypoints_px, camera, keypoint_confidences)
    checks = assess_sop(evidence, pose)
    verdict, reasons = overall_verdict(checks, pose)
    return Assessment(
        schema_version="1.0", image_id=image_id, pallet_id=pallet_id, pose=pose,
        checks=checks, overall_verdict=verdict, reasons=reasons,
        provenance={"model_version": model_version, "assessed_at": datetime.now(timezone.utc).isoformat()},
    )


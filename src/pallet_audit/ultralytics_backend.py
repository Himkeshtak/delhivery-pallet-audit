"""Optional Ultralytics adapter. Importing pallet_audit does not require Ultralytics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PalletKeypointDetection:
    keypoints_px: np.ndarray
    confidences: np.ndarray
    detection_confidence: float
    box_xyxy: np.ndarray


class UltralyticsPoseBackend:
    def __init__(self, weights: str | Path, device: str | int | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('install the vision extra: pip install -e ".[vision]"') from exc
        self.model = YOLO(str(weights))
        self.device = device

    def predict(self, image: Any, confidence: float = 0.25) -> list[PalletKeypointDetection]:
        results = self.model.predict(image, conf=confidence, device=self.device, verbose=False)
        detections: list[PalletKeypointDetection] = []
        for result in results:
            if result.keypoints is None or result.boxes is None:
                continue
            xy = result.keypoints.xy.detach().cpu().numpy()
            kp_conf_tensor = result.keypoints.conf
            kp_conf = np.ones(xy.shape[:2]) if kp_conf_tensor is None else kp_conf_tensor.detach().cpu().numpy()
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            box_conf = result.boxes.conf.detach().cpu().numpy()
            for points, point_conf, box, score in zip(xy, kp_conf, boxes, box_conf):
                detections.append(PalletKeypointDetection(points.astype(float), point_conf.astype(float),
                                                           float(score), box.astype(float)))
        return detections


def mask_polygons(result: Any, class_names: set[str] | None = None) -> list[tuple[str, float, np.ndarray]]:
    """Extract named polygons without leaking framework tensors into the geometry layer."""
    if result.masks is None or result.boxes is None:
        return []
    polygons = result.masks.xy
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    output = []
    for polygon, class_id, score in zip(polygons, classes, scores):
        name = str(result.names[class_id])
        if class_names is None or name in class_names:
            output.append((name, float(score), np.asarray(polygon, dtype=float)))
    return output


"""End-to-end adapter from trained Ultralytics models to safe assessments.

The geometry and policy layers stay framework independent. This module is the
explicit boundary where model outputs are associated, validation gates are
applied, and unsupported negative evidence is suppressed.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .compliance import LoadEvidence
from .geometry import CameraModel, wrap_axis_angle_deg
from .pipeline import assess_pallet
from .ultralytics_backend import UltralyticsPoseBackend


@dataclass(frozen=True)
class VisualDetection:
    class_name: str
    confidence: float
    box_xyxy: np.ndarray
    polygon_px: np.ndarray | None


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apparent_alignment_angle_deg(polygon_px: np.ndarray, pallet_front_px: np.ndarray) -> float:
    """Return apparent box-to-pallet axis error in the image plane (0..45 degrees)."""
    polygon = np.asarray(polygon_px, dtype=float)
    front = np.asarray(pallet_front_px, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 4:
        raise ValueError("polygon_px must contain at least four 2D points")
    if front.shape != (2,) or np.linalg.norm(front) < 1e-6:
        raise ValueError("pallet_front_px must be a non-zero 2-vector")
    centred = polygon - polygon.mean(axis=0)
    values, vectors = np.linalg.eigh(centred.T @ centred)
    principal = vectors[:, int(np.argmax(values))]
    box_angle = np.degrees(np.arctan2(principal[1], principal[0]))
    pallet_angle = np.degrees(np.arctan2(front[1], front[0]))
    delta = abs(wrap_axis_angle_deg(float(box_angle - pallet_angle))) % 90.0
    return float(min(delta, 90.0 - delta))


def belongs_to_pallet(pallet_box: np.ndarray, detection_box: np.ndarray) -> bool:
    """Associate a load detection with the pallet column above a pallet footprint."""
    px1, py1, px2, py2 = np.asarray(pallet_box, dtype=float)
    dx1, dy1, dx2, dy2 = np.asarray(detection_box, dtype=float)
    width = max(px2 - px1, 1.0)
    height = max(py2 - py1, 1.0)
    overlap = max(0.0, min(px2 + 0.25 * width, dx2) - max(px1 - 0.25 * width, dx1))
    detection_width = max(dx2 - dx1, 1.0)
    vertical_ok = dy2 >= py1 - 5.0 * height and dy1 <= py2 + 0.5 * height
    return vertical_ok and overlap / detection_width >= 0.25


def _segmentation_detections(result: Any) -> list[VisualDetection]:
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    polygons = list(result.masks.xy) if result.masks is not None else []
    detections = []
    for index, (box, class_id, score) in enumerate(zip(boxes, classes, scores, strict=True)):
        polygon = None
        if index < len(polygons) and len(polygons[index]) >= 4:
            polygon = np.asarray(polygons[index], dtype=float)
        detections.append(VisualDetection(
            class_name=str(result.names[class_id]), confidence=float(score),
            box_xyxy=np.asarray(box, dtype=float), polygon_px=polygon,
        ))
    return detections


def _maximum_confidence(detections: list[VisualDetection], class_name: str) -> float | None:
    values = [item.confidence for item in detections if item.class_name == class_name]
    return max(values, default=None)


class TrainedAuditPipeline:
    """Run the trained pose/segmentation baselines and emit fail-safe assessments."""

    def __init__(self, pose_weights: str | Path, segment_weights: str | Path,
                 camera: CameraModel, device: str = "cpu", imgsz: int = 256) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError('install the vision extra: pip install -e ".[vision]"') from exc
        self.pose_weights = Path(pose_weights)
        self.segment_weights = Path(segment_weights)
        self.camera = camera
        self.device = device
        self.imgsz = imgsz
        self.pose_backend = UltralyticsPoseBackend(
            self.pose_weights, device=device, imgsz=imgsz
        )
        self.segment_model = YOLO(str(self.segment_weights))
        self.model_hashes = {
            "pose_sha256": file_sha256(self.pose_weights),
            "segment_sha256": file_sha256(self.segment_weights),
        }

    def predict(self, image: str | Path) -> list[dict[str, Any]]:
        image_path = Path(image)
        pose_started = time.perf_counter()
        pose_detections = self.pose_backend.predict(str(image_path), confidence=0.05)
        pose_ms = (time.perf_counter() - pose_started) * 1000.0

        segment_started = time.perf_counter()
        segment_result = self.segment_model.predict(
            str(image_path), conf=0.05, imgsz=self.imgsz,
            device=self.device, verbose=False, save=False,
        )[0]
        segment_ms = (time.perf_counter() - segment_started) * 1000.0
        segment_detections = _segmentation_detections(segment_result)
        image_hash = file_sha256(image_path)

        outputs = []
        for index, detection in enumerate(pose_detections, start=1):
            associated = [
                item for item in segment_detections
                if belongs_to_pallet(detection.box_xyxy, item.box_xyxy)
            ]
            experimental_box_angles = tuple(
                apparent_alignment_angle_deg(
                    item.polygon_px, detection.keypoints_px[1] - detection.keypoints_px[0]
                )
                for item in associated
                if item.class_name == "box" and item.polygon_px is not None
                and item.confidence >= 0.5
            )
            # Box mAP supports using wrap as positive presence evidence. A missing
            # detection remains unknown. Damage evidence is suppressed because both
            # damage classes scored zero AP on the held-out synthetic test.
            wrap_score = _maximum_confidence(associated, "stretch_wrap")
            evidence = LoadEvidence(
                # The raw angle evidence is retained below for evaluation, but it
                # is disabled here after a 46/60 synthetic false-failure gate.
                box_angles_deg=(),
                wrap_probability=(
                    wrap_score if wrap_score is not None and wrap_score >= 0.5 else None
                ),
                damage_probability=None,
                pallet_damage_probability=None,
                visibility=float(min(
                    detection.detection_confidence,
                    np.min(detection.confidences),
                )),
            )
            assessment = assess_pallet(
                image_id=image_path.name,
                pallet_id=f"pallet-{index:03d}",
                keypoints_px=detection.keypoints_px,
                keypoint_confidences=detection.confidences,
                camera=self.camera,
                evidence=evidence,
                model_version="pose-yolo11n+seg-yolo11n-synthetic-v1",
            )
            assessment.provenance.update({
                **self.model_hashes,
                "image_sha256": image_hash,
                "device": self.device,
                "imgsz": self.imgsz,
                "latency_ms": {"pose": pose_ms, "segment": segment_ms},
                "associated_detections": [
                    {"class": item.class_name, "confidence": item.confidence}
                    for item in associated
                ],
                "experimental_evidence": {
                    "box_angles_deg": list(experimental_box_angles),
                    "would_fail_sop3": (
                        max(experimental_box_angles, default=0.0) > 15.0
                    ),
                },
                "validation_gates": {
                    "box_alignment": (
                        "disabled: raw evidence falsely failed 46/60 "
                        "aligned synthetic scenes"
                    ),
                    "wrap_presence": "enabled only for positive detections: box mAP50-95=0.833",
                    "load_footprint": "disabled: load mask mAP50-95=0.006",
                    "visible_box_damage": "disabled: held-out AP=0",
                    "visible_pallet_damage": "disabled: held-out AP=0",
                },
                "claim_scope": "synthetic-v1 demonstration; not warehouse validated",
            })
            outputs.append(assessment.to_dict())
        return outputs

"""Evaluate trained pallet keypoints and metric pose on a frozen record set."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pallet_audit.geometry import CameraModel, pose_from_keypoints, wrap_axis_angle_deg


def distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {f"p{p}": None for p in (50, 90, 95, 99)}
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 95, 99)}


def evaluate(weights: Path, records_path: Path, data_root: Path,
             output_dir: Path, imgsz: int, device: str) -> dict:
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install vision dependencies with: pip install -e ".[vision]"') from exc
    records = json.loads(records_path.read_text(encoding="utf-8"))
    model = YOLO(str(weights))
    translations: list[float] = []
    rotations: list[float] = []
    keypoint_errors: list[float] = []
    detection_confidences: list[float] = []
    latencies: list[float] = []
    predictions = []
    missed = abstained = 0
    for record in records:
        image_path = data_root / record["image"]
        started = time.perf_counter()
        result = model.predict(str(image_path), imgsz=imgsz, device=device,
                               conf=.05, verbose=False)[0]
        latencies.append((time.perf_counter() - started) * 1000)
        if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
            missed += 1
            predictions.append({"image": record["image"], "detected": False})
            continue
        scores = result.boxes.conf.detach().cpu().numpy()
        best = int(np.argmax(scores))
        points = result.keypoints.xy[best].detach().cpu().numpy().astype(float)
        confidence_tensor = result.keypoints.conf
        confidences = (np.ones(len(points)) if confidence_tensor is None
                       else confidence_tensor[best].detach().cpu().numpy().astype(float))
        if points.shape != (4, 2):
            raise ValueError(f"expected four keypoints, got {points.shape}")
        camera = CameraModel(np.asarray(record["homography_image_to_floor"]),
                             float(record.get("reprojection_rmse_px", 0)))
        pose = pose_from_keypoints(points, camera, confidences)
        gt = record["ground_truth"]
        translation = float(np.hypot(pose.x_m - gt["x_m"], pose.y_m - gt["y_m"]))
        rotation = abs(wrap_axis_angle_deg(pose.theta_deg - gt["theta_deg"]))
        gt_points = np.asarray(record["keypoints_px"], dtype=float)
        per_keypoint = np.linalg.norm(points - gt_points, axis=1)
        translations.append(translation)
        rotations.append(rotation)
        keypoint_errors.extend(float(value) for value in per_keypoint)
        detection_confidences.append(float(scores[best]))
        if not pose.reliable:
            abstained += 1
        predictions.append({
            "image": record["image"], "detected": True,
            "detection_confidence": float(scores[best]), "keypoints_px": points.tolist(),
            "keypoint_confidences": confidences.tolist(), "pose": pose.__dict__,
            "ground_truth": gt, "translation_error_m": translation,
            "rotation_error_deg": rotation,
            "keypoint_errors_px": per_keypoint.tolist(),
        })
    paired = len(translations)
    joint = [t <= .02 and r <= 3 for t, r in zip(translations, rotations)]
    metrics = {
        "scope": "synthetic-v1 held-out test_shift; not warehouse performance",
        "weights": str(weights), "records": len(records), "detections": paired,
        "missed_detections": missed, "detection_recall": paired / max(len(records), 1),
        "abstentions_among_detections": abstained,
        "translation_error_m": distribution(translations),
        "rotation_error_deg": distribution(rotations),
        "keypoint_localization_error_px": distribution(keypoint_errors),
        "detection_confidence": distribution(detection_confidences),
        "joint_2cm_3deg_rate_all_images": sum(joint) / max(len(records), 1),
        "latency_ms_observed": distribution(latencies),
        "runtime": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                    "device": device, "imgsz": imgsz},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pose_test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "pose_test_predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("records", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pose_model"))
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.weights, args.records, args.data_root,
                              args.output_dir, args.imgsz, args.device), indent=2))


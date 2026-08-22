"""Run trained models and save one assessment JSON per detected pallet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pallet_audit.geometry import CameraModel
from pallet_audit.trained_pipeline import TrainedAuditPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="image file or directory")
    parser.add_argument("--pose-weights", type=Path,
                        default=Path("weights/pallet_pose_yolo11n_synthetic_v1.pt"))
    parser.add_argument("--segment-weights", type=Path,
                        default=Path("weights/load_seg_yolo11n_synthetic_v1.pt"))
    parser.add_argument("--calibration", type=Path,
                        default=Path("data/synthetic_calibration.json"))
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path("assessments"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=256)
    return parser.parse_args()


def image_paths(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path for path in source.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def main() -> None:
    args = parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    camera = CameraModel(
        np.asarray(calibration["homography_image_to_floor"], dtype=float),
        float(calibration.get("reprojection_rmse_px", 0.0)),
    )
    pipeline = TrainedAuditPipeline(
        args.pose_weights, args.segment_weights, camera,
        device=args.device, imgsz=args.imgsz,
    )
    images = image_paths(args.source)
    if not images:
        raise SystemExit(f"No images found at {args.source}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, object]] = []
    for image in images:
        assessments = pipeline.predict(image)
        files = []
        for assessment in assessments:
            output_path = args.output_dir / (
                f"{image.stem}_{assessment['pallet_id']}.json"
            )
            output_path.write_text(
                json.dumps(assessment, indent=2) + "\n", encoding="utf-8"
            )
            files.append(output_path.as_posix())
        index.append({
            "image": image.as_posix(),
            "detections": len(assessments),
            "assessment_files": files,
        })
    index_path = args.output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": len(images), "index": str(index_path)}, indent=2))


if __name__ == "__main__":
    main()

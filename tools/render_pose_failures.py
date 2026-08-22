"""Render the worst metric-pose cases with ground-truth and predicted keypoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def render(predictions_path: Path, data_root: Path, output_dir: Path,
           count: int = 3) -> list[dict]:
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    detected = [row for row in predictions if row.get("detected")]
    ranked = sorted(detected, key=lambda row: row["translation_error_m"] / .02
                    + row["rotation_error_deg"] / 3, reverse=True)[:count]
    output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for rank, row in enumerate(ranked, 1):
        image = Image.open(data_root / row["image"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        ground_truth = json.loads((data_root / "evaluation_pose_records.json").read_text(encoding="utf-8"))
        source = next(item for item in ground_truth if item["image"] == row["image"])
        gt_points = [tuple(point) for point in source["keypoints_px"]]
        predicted = [tuple(point) for point in row["keypoints_px"]]
        draw.line(gt_points + [gt_points[0]], fill=(40, 235, 80), width=3)
        draw.line(predicted + [predicted[0]], fill=(245, 55, 45), width=3)
        for point in gt_points:
            draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3), fill=(40, 235, 80))
        for point in predicted:
            draw.rectangle((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill=(245, 55, 45))
        message = (f"GT green / prediction red | t={row['translation_error_m']*100:.1f}cm "
                   f"r={row['rotation_error_deg']:.1f}deg")
        draw.rectangle((0, 0, image.width, 17), fill=(10, 10, 10))
        draw.text((4, 3), message, fill=(255, 255, 255))
        filename = f"worst_{rank}.jpg"
        image.save(output_dir / filename, quality=94)
        index.append({"rank": rank, "file": filename, "source_image": row["image"],
                      "translation_error_m": row["translation_error_m"],
                      "rotation_error_deg": row["rotation_error_deg"],
                      "keypoint_errors_px": row["keypoint_errors_px"]})
    (output_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/pose_model/failures"))
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(render(args.predictions, args.data_root, args.output_dir, args.count), indent=2))


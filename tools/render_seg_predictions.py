"""Render deterministic held-out segmentation examples and a machine-readable index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=3)
    return parser.parse_args()


def evenly_spaced(items: list[Path], count: int) -> list[Path]:
    if not items or count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    indexes = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)]
    return [items[index] for index in dict.fromkeys(indexes)]


def main() -> None:
    args = parse_args()
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install vision dependencies with: pip install -e ".[vision]"') from exc

    candidates = sorted(
        path for path in args.images.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    chosen = evenly_spaced(candidates, args.count)
    if not chosen:
        raise SystemExit(f"No images found in {args.images}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    index: list[dict[str, object]] = []
    for number, image_path in enumerate(chosen, start=1):
        result = model.predict(
            source=str(image_path), imgsz=args.imgsz, device=args.device,
            verbose=False, save=False,
        )[0]
        output_path = args.output / f"example_{number}.jpg"
        if not cv2.imwrite(str(output_path), result.plot()):
            raise RuntimeError(f"Could not write {output_path}")

        detections: list[dict[str, object]] = []
        if result.boxes is not None:
            for class_id, confidence in zip(
                result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist(), strict=True
            ):
                detections.append({
                    "class": result.names[int(class_id)],
                    "confidence": round(float(confidence), 6),
                })
        index.append({
            "source": image_path.as_posix(),
            "render": output_path.as_posix(),
            "detections": detections,
        })

    (args.output / "index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()

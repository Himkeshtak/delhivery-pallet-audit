"""Persist Ultralytics held-out AP and speed metrics as JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install vision dependencies with: pip install -e ".[vision]"') from exc
    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz,
                        device=args.device, workers=0, plots=False, verbose=False)
    output = {
        "scope": f"dataset={args.data}, split={args.split}; obey that dataset's claim scope",
        "weights": str(args.weights), "results": {key: float(value) for key, value in metrics.results_dict.items()},
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "runtime": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                    "device": args.device, "imgsz": args.imgsz},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()


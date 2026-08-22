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
    per_class: dict[str, dict[str, float]] = {}
    names = metrics.names
    for index, name in names.items():
        class_metrics: dict[str, float] = {}
        for prefix, component in (("box", getattr(metrics, "box", None)),
                                  ("mask", getattr(metrics, "seg", None)),
                                  ("pose", getattr(metrics, "pose", None))):
            if component is None:
                continue
            for metric_name, attribute in (("precision", "p"), ("recall", "r"),
                                           ("map50", "ap50"), ("map50_95", "maps")):
                values = getattr(component, attribute, None)
                if values is not None and len(values) > index:
                    class_metrics[f"{prefix}_{metric_name}"] = float(values[index])
        per_class[str(name)] = class_metrics
    output = {
        "scope": f"dataset={args.data}, split={args.split}; obey that dataset's claim scope",
        "weights": str(args.weights), "results": {key: float(value) for key, value in metrics.results_dict.items()},
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "per_class": per_class,
        "runtime": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                    "device": args.device, "imgsz": args.imgsz},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

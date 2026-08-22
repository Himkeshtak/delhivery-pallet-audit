from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--format", default="engine", choices=("engine", "onnx"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--data", help="required representative dataset config for INT8")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.int8 and not args.data:
        raise SystemExit("--data is required for representative INT8 calibration")
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install export dependencies with: pip install -e ".[vision]"') from exc
    model = YOLO(str(args.weights))
    exported = model.export(format=args.format, imgsz=args.imgsz, half=not args.int8,
                            int8=args.int8, data=args.data, batch=1, dynamic=False,
                            device=args.device, workspace=2)
    record = {"created_at": datetime.now(timezone.utc).isoformat(), "source": str(args.weights),
              "output": str(exported), "format": args.format, "imgsz": args.imgsz,
              "precision": "int8" if args.int8 else "fp16", "platform": platform.platform(),
              "ultralytics_version": ultralytics.__version__}
    Path(f"{exported}.provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


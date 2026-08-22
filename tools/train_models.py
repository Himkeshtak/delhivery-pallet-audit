"""Train a declared pose or segmentation baseline with explicit provenance."""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("pose", "segment"))
    parser.add_argument("--model", required=True, help="e.g. yolo11n-pose.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install training dependencies with: pip install -e ".[vision]"') from exc

    run_name = args.name or f"{args.task}-{Path(args.model).stem}"
    model = YOLO(args.model, task=args.task)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=args.project, name=run_name, seed=args.seed,
                deterministic=True, plots=True, patience=30)
    run_dir = Path(args.project) / run_name
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(), "task": args.task,
        "initial_weights": args.model, "data": str(Path(args.data).resolve()),
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "seed": args.seed, "platform": platform.platform(),
        "ultralytics_version": ultralytics.__version__,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


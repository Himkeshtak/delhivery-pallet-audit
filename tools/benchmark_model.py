"""Measure one exported or PyTorch model without claiming results from another device."""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sync_cuda(torch_module: object) -> None:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.synchronize()


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    return {f"p{p}_ms": float(np.percentile(samples_ms, p)) for p in (50, 90, 95, 99)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("-o", "--output", type=Path, default=Path("benchmark.json"))
    args = parser.parse_args()
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit('Install benchmark dependencies with: pip install -e ".[vision]"') from exc
    images = sorted(path for path in args.images.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"no images found under {args.images}")
    model = YOLO(str(args.weights))
    for index in range(args.warmup):
        model.predict(str(images[index % len(images)]), imgsz=args.imgsz, device=args.device, verbose=False)
    sync_cuda(torch)
    samples = []
    started = time.perf_counter()
    for index in range(args.iterations):
        sync_cuda(torch)
        before = time.perf_counter()
        model.predict(str(images[index % len(images)]), imgsz=args.imgsz, device=args.device, verbose=False)
        sync_cuda(torch)
        samples.append((time.perf_counter() - before) * 1000)
    elapsed = time.perf_counter() - started
    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(), "weights": str(args.weights),
        "device_argument": args.device, "platform": platform.platform(),
        "processor": platform.processor(), "python": platform.python_version(),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "image_count": len(images), "imgsz": args.imgsz, "warmup": args.warmup,
        "iterations": args.iterations, "latency": percentiles(samples),
        "end_to_end_fps": args.iterations / elapsed,
        "scope": "decode/preprocess/inference/postprocess through model.predict; excludes application serialization",
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


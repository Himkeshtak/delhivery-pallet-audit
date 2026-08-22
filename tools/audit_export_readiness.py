"""Record whether this workstation can produce and validate deployment exports."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    import torch

    cuda = bool(torch.cuda.is_available())
    tensorrt = installed("tensorrt")
    onnx = installed("onnx")
    record = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda else None,
        "modules": {
            "tensorrt": tensorrt,
            "onnx": onnx,
            "onnxruntime": installed("onnxruntime"),
        },
        "tensorrt_fp16_export_ready": cuda and tensorrt,
        "tensorrt_int8_export_ready": cuda and tensorrt,
        "quantization_accuracy_comparison_complete": False,
        "blocking_reasons": [
            "No CUDA device or TensorRT runtime is available on this workstation.",
            "No real representative calibration set exists for production INT8 calibration.",
            (
                "Accuracy must be compared on the target-scene frozen test set, "
                "not synthetic data alone."
            ),
        ],
        "claim_scope": "readiness audit only; not a Jetson benchmark",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()

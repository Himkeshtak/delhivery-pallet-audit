"""Combine component benchmarks into the documented model-cadence budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pose", type=Path)
    parser.add_argument("segment", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--segment-cadence", type=int, default=3)
    args = parser.parse_args()
    if args.segment_cadence < 1:
        raise SystemExit("--segment-cadence must be at least 1")
    pose = json.loads(args.pose.read_text(encoding="utf-8"))
    segment = json.loads(args.segment.read_text(encoding="utf-8"))
    pose_mean_ms = 1000.0 / float(pose["end_to_end_fps"])
    segment_mean_ms = 1000.0 / float(segment["end_to_end_fps"])
    cadence_mean_ms = pose_mean_ms + segment_mean_ms / args.segment_cadence
    result = {
        "scope": "derived from measured component mean service times; not a measured percentile",
        "pose_mean_service_ms": pose_mean_ms,
        "segment_mean_service_ms": segment_mean_ms,
        "segment_cadence_frames": args.segment_cadence,
        "derived_cadence_mean_service_ms_per_frame": cadence_mean_ms,
        "derived_cadence_fps": 1000.0 / cadence_mean_ms,
        "fifteen_fps_gate": 1000.0 / cadence_mean_ms >= 15.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Aggregate per-pallet assessment JSON files without weakening their verdicts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {f"p{value}": None for value in (50, 95, 99)}
    return {
        f"p{value}": float(np.percentile(values, value))
        for value in (50, 95, 99)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*_pallet-*.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    verdicts = Counter(record["overall_verdict"] for record in records)
    reliable = Counter(str(record["pose"]["reliable"]).lower() for record in records)
    checks: dict[str, Counter[str]] = defaultdict(Counter)
    pose_latency: list[float] = []
    segment_latency: list[float] = []
    combined_latency: list[float] = []
    experimental_alignment_failures = 0
    for record in records:
        for check in record["checks"]:
            checks[str(check["sop_id"])][check["verdict"]] += 1
        latency = record["provenance"].get("latency_ms", {})
        if "pose" in latency:
            pose_latency.append(float(latency["pose"]))
        if "segment" in latency:
            segment_latency.append(float(latency["segment"]))
        if "pose" in latency and "segment" in latency:
            combined_latency.append(float(latency["pose"]) + float(latency["segment"]))
        experimental = record["provenance"].get("experimental_evidence", {})
        experimental_alignment_failures += int(
            bool(experimental.get("would_fail_sop3", False))
        )

    summary = {
        "scope": "synthetic-v1 end-to-end demonstration; not warehouse performance",
        "assessments": len(records),
        "overall_verdict_counts": dict(sorted(verdicts.items())),
        "pose_reliability_counts": dict(sorted(reliable.items())),
        "per_sop_verdict_counts": {
            sop_id: dict(sorted(counts.items()))
            for sop_id, counts in sorted(checks.items(), key=lambda item: int(item[0]))
        },
        "experimental_gate_diagnostics": {
            "raw_box_alignment_would_fail": experimental_alignment_failures,
            "known_aligned_scenes": len(records),
            "accepted_for_decisions": False,
        },
        "observed_latency_ms_including_framework_overhead": {
            "pose": percentiles(pose_latency),
            "segment": percentiles(segment_latency),
            "pose_plus_segment": percentiles(combined_latency),
            "note": "first image includes model setup; use benchmark_model.py for steady state",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

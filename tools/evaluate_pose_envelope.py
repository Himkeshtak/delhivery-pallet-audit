"""Stratify trained pose errors by synthetic yaw without inventing range metadata."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def wilson_lower(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    """One-sided 95% Wilson lower confidence bound."""
    if total == 0:
        return 0.0
    observed = successes / total
    denominator = 1.0 + z * z / total
    centre = observed + z * z / (2.0 * total)
    spread = z * math.sqrt(observed * (1.0 - observed) / total + z * z / (4 * total * total))
    return (centre - spread) / denominator


def summarize(records: list[dict], low: float, high: float) -> dict:
    selected = [
        record for record in records
        if record.get("detected") and low <= abs(record["ground_truth"]["theta_deg"]) < high
    ]
    translation = [float(record["translation_error_m"]) for record in selected]
    rotation = [float(record["rotation_error_deg"]) for record in selected]
    successes = sum(t <= 0.02 and r <= 3.0 for t, r in zip(translation, rotation, strict=True))
    total = len(selected)
    return {
        "absolute_yaw_deg": [low, high],
        "samples": total,
        "translation_error_m_p95": (
            float(np.percentile(translation, 95)) if translation else None
        ),
        "rotation_error_deg_p95": (
            float(np.percentile(rotation, 95)) if rotation else None
        ),
        "joint_2cm_3deg_successes": successes,
        "joint_2cm_3deg_rate": successes / total if total else None,
        "joint_rate_one_sided_95pct_wilson_lower": wilson_lower(successes, total),
        "accepted_at_95pct_required_rate": (
            total > 0 and wilson_lower(successes, total) >= 0.95
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.predictions.read_text(encoding="utf-8"))
    bins = [summarize(records, low, high) for low, high in ((0, 15), (15, 30), (30, 45))]
    output = {
        "scope": "synthetic-v1 shifted test; not a physical operating envelope",
        "criterion": "one-sided 95% Wilson lower bound on joint success rate >=0.95",
        "yaw_bins": bins,
        "accepted_yaw_bins": [
            item["absolute_yaw_deg"] for item in bins
            if item["accepted_at_95pct_required_rate"]
        ],
        "range_evaluable": False,
        "range_reason": (
            "synthetic-v1 has no surveyed camera-to-pallet range and cannot support "
            "a short/mid/long range claim"
        ),
        "operational_envelope_declared": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

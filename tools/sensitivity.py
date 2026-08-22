"""Monte Carlo sensitivity of a floor homography to calibration perturbations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def simulate(height_m: float, tilt_deg: float, ranges: list[float], samples: int = 10000, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    output = []
    for distance in ranges:
        height_error = rng.normal(0, 0.01, samples)  # surveyed-height uncertainty: 1 cm
        tilt_error = np.deg2rad(rng.normal(0, 0.25, samples))
        # First-order ray/floor intersection sensitivity.
        elevation = math.atan2(height_m, distance)
        estimated = (height_m + height_error) / np.tan(elevation + tilt_error)
        error = np.abs(estimated - distance)
        output.append({"range_m": distance, "assumed_height_sigma_m": 0.01, "assumed_tilt_sigma_deg": .25,
                       "translation_error_m": {f"p{p}": float(np.percentile(error, p)) for p in (50, 90, 95, 99)},
                       "meets_2cm_rate": float(np.mean(error <= .02))})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=float, default=1.2)
    parser.add_argument("--tilt", type=float, default=20)
    parser.add_argument("--ranges", nargs="+", type=float, default=[1, 2, 3, 4, 5])
    parser.add_argument("-o", "--output", type=Path, default=Path("sensitivity.json"))
    args = parser.parse_args()
    args.output.write_text(json.dumps(simulate(args.height, args.tilt, args.ranges), indent=2), encoding="utf-8")


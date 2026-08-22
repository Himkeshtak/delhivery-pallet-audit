from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CheckResult, Pose, Verdict


@dataclass(frozen=True)
class LoadEvidence:
    overhang_m: float | None = None
    load_height_m: float | None = None
    box_angles_deg: tuple[float, ...] = ()
    size_inversion_score: float | None = None
    wrap_probability: float | None = None
    damage_probability: float | None = None
    centroid_offset_m: float | None = None
    pallet_damage_probability: float | None = None
    visibility: float = 1.0


def _threshold_check(sop_id: int, name: str, value: float | None, limit: float, confidence: float, unit_key: str, higher_bad: bool = True) -> CheckResult:
    if value is None or confidence < 0.5:
        return CheckResult(sop_id, name, Verdict.MANUAL, confidence, {}, "insufficient reliable evidence")
    failed = value > limit if higher_bad else value < limit
    return CheckResult(sop_id, name, Verdict.FAIL if failed else Verdict.PASS, confidence, {unit_key: value, "limit": limit})


def assess_sop(evidence: LoadEvidence, pose: Pose) -> list[CheckResult]:
    """Assess only claims supportable from a monocular side view; abstain otherwise."""
    quality = float(np.clip(min(evidence.visibility, pose.confidence), 0.0, 1.0))
    metric_quality = quality if pose.reliable else quality * 0.35
    max_rotation = max((abs(x) for x in evidence.box_angles_deg), default=None)
    return [
        _threshold_check(1, "box overhang", evidence.overhang_m, 0.03, metric_quality, "max_overhang_m"),
        _threshold_check(2, "load height", evidence.load_height_m, 1.8, metric_quality, "height_m"),
        _threshold_check(3, "column alignment", max_rotation, 15.0, quality, "max_box_rotation_deg"),
        CheckResult(4, "size ordering", Verdict.MANUAL, quality * 0.4,
                    {"visible_inversion_score": evidence.size_inversion_score or 0.0},
                    "rear and occluded boxes cannot be verified from one view"),
        _threshold_check(5, "stretch wrap", evidence.wrap_probability, 0.5, quality, "wrap_probability", higher_bad=False),
        CheckResult(6, "visible box damage", Verdict.FAIL if (evidence.damage_probability or 0) >= 0.6 else Verdict.MANUAL,
                    quality * (evidence.damage_probability or 0), {"damage_probability": evidence.damage_probability or 0},
                    "only visible faces are assessable"),
        _threshold_check(7, "load centring", evidence.centroid_offset_m, 0.10, metric_quality, "centroid_offset_m"),
        CheckResult(8, "pallet damage", Verdict.FAIL if (evidence.pallet_damage_probability or 0) >= 0.6 else Verdict.MANUAL,
                    quality * (evidence.pallet_damage_probability or 0), {"damage_probability": evidence.pallet_damage_probability or 0},
                    "hidden boards and rear stringers are not visible"),
    ]


def overall_verdict(checks: list[CheckResult], pose: Pose) -> tuple[Verdict, list[str]]:
    if not pose.reliable:
        return Verdict.MANUAL, ["pose is outside the validated uncertainty envelope"]
    failed = [c for c in checks if c.verdict == Verdict.FAIL and c.confidence >= 0.6]
    if failed:
        return Verdict.FAIL, [f"SOP-{c.sop_id} failed: {c.name}" for c in failed]
    manual = [c for c in checks if c.verdict in (Verdict.MANUAL, Verdict.NOT_VERIFIABLE)]
    if manual:
        return Verdict.MANUAL, [f"SOP-{c.sop_id} needs inspection: {c.name}" for c in manual]
    return Verdict.PASS, ["all reliably observable checks passed"]


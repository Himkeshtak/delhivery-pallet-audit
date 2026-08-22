from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    MANUAL = "manual_inspection"
    NOT_VERIFIABLE = "not_verifiable"


@dataclass(frozen=True)
class Pose:
    x_m: float
    y_m: float
    theta_deg: float
    sigma_x_m: float
    sigma_y_m: float
    sigma_theta_deg: float
    confidence: float
    reliable: bool
    reason: str = ""


@dataclass(frozen=True)
class CheckResult:
    sop_id: int
    name: str
    verdict: Verdict
    confidence: float
    measured: dict[str, float | str | bool] = field(default_factory=dict)
    reason: str = ""


@dataclass
class Assessment:
    schema_version: str
    image_id: str
    pallet_id: str
    pose: Pose
    checks: list[CheckResult]
    overall_verdict: Verdict
    reasons: list[str]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


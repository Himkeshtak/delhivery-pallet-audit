"""Research-backed model registry and runtime cadence.

Paper benchmark values are deliberately not encoded here: only measurements made on the
submission dataset belong in runtime decisions.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    role: str
    baseline: str
    inputs: str
    outputs: tuple[str, ...]
    cadence: int
    required: bool
    selection_gate: str


MODEL_STACK = (
    ModelSpec(
        role="pallet_pose",
        baseline="yolo11s-pose.pt",
        inputs="full RGB frame",
        outputs=("pallet_box", "fl", "fr", "br", "bl", "keypoint_confidence"),
        cadence=1,
        required=True,
        selection_gate="held-out joint 2 cm / 3 degree rate and abstention rate",
    ),
    ModelSpec(
        role="load_instances",
        baseline="yolo11n-seg.pt",
        inputs="pallet crop",
        outputs=("pallet_mask", "load_mask", "box_masks", "wrap_mask"),
        cadence=3,
        required=True,
        selection_gate="class-wise mask AP plus downstream SOP error distributions",
    ),
    ModelSpec(
        role="visible_damage",
        baseline="efficientad-s",
        inputs="rectified visible box and pallet-face crops",
        outputs=("anomaly_map", "anomaly_score"),
        cadence=10,
        required=False,
        selection_gate="defect AUROC/AUPRO at a fixed false-alarm operating point",
    ),
)


def runtime_schedule(frame_index: int) -> tuple[str, ...]:
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    return tuple(model.role for model in MODEL_STACK if frame_index % model.cadence == 0)


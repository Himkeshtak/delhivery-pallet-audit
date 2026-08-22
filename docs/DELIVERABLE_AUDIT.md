# Assignment deliverable audit

Audit date: 2026-08-22. `Complete` means an artefact exists and is reproducible. `Pending evidence` means code/protocol exists but the physical data or measured result does not. Synthetic results are never relabelled as warehouse results.

## Section 1 - Dataset and detection (30%)

| Deliverable | Status | Evidence / completion condition |
|---|---|---|
| Dataset in a standard format | Partial | COCO generator and YOLO converter exist. A committed, versioned training dataset is not yet present. |
| Annotation tooling | Complete | `tools/generate_synthetic.py`, `tools/coco_to_yolo_pose.py`, COCO keypoint convention and labelling guide. SAM 2 is specified for offline propagation. |
| Source, cost, counts, split, guideline, biases | Partial | `DATASET.md` covers all topics honestly; counts remain zero until dataset generation/acquisition completes. |
| Trained model and weights | Missing | Training scripts/configs exist; no trained checkpoint has been produced. |
| Detection/localisation held-out distributions | Missing | Evaluation tool exists; no model/test predictions yet. |
| Training decisions and reasoning | Complete | `README.md` and `docs/RESEARCH_AND_MODEL_SELECTION.md`. |
| Accuracy ceiling and improvements | Complete | `DATASET.md`. |

## Section 2 - Pose estimation (35%)

| Deliverable | Status | Evidence / completion condition |
|---|---|---|
| Camera calibration artefacts and reprojection error | Pending physical camera | Protocol exists in `docs/CALIBRATION.md`; target camera/board images are unavailable. A synthetic calibration can test code but cannot substitute for this measurement. |
| Pose method and reasoning | Complete | Calibrated floor homography, directed keypoints and propagated uncertainty in `geometry.py`. |
| Self-constructed evaluation | Partial | Survey protocol and distribution tool exist. Needs physical surveyed placements or declared synthetic benchmark records. |
| Translation/rotation error distributions | Missing | Needs trained predictions and ground truth. |
| Height/tilt sensitivity, short/long range | Partial | Monte Carlo tool exists; measured camera uncertainty is unavailable. |
| Usable envelope | Missing | Requires held-out range/yaw results. |
| Unreliable-pose output | Complete | Explicit `reliable=false`, uncertainty, reason and manual-inspection verdict. |

## Section 3 - Load analysis and SOP (25%)

| Deliverable | Status | Evidence / completion condition |
|---|---|---|
| Eight-check observability triage | Complete | README SOP table. |
| Implement verifiable subset | Complete as evidence layer | Metric overhang, height, box angle, centroid, wrap/damage evidence interfaces and fail-safe abstention. Learned mask/damage weights remain missing. |
| Per-check confidence | Complete | Every check emits confidence, measurements and reason. |
| Pass/fail/manual verdict and weighting | Complete | `compliance.py`; pose failure forces manual review and reliable high-confidence failures dominate. |

## Section 4 - Deployment and robustness (10%)

| Deliverable | Status | Evidence / completion condition |
|---|---|---|
| Measured latency on declared hardware | Missing | Honest benchmark harness exists; no trained weights have been timed. |
| Orin Nano change analysis | Complete | `docs/DEPLOYMENT.md`. |
| Quantisation accuracy cost | Missing | Export guard/tool exists; needs FP32/FP16/INT8 weights evaluated on the same test set. |
| Failure contract | Complete | Versioned assessment schema and explicit abstention. |
| Multi-frame use | Complete design | Cadenced runtime schedule and uncertainty-aware temporal policy documented. |

## Required output and submission material

| Deliverable | Status | Evidence / completion condition |
|---|---|---|
| One assessment per pallet | Complete | CLI, `schemas/assessment.schema.json`, example request/output path. |
| README required sections | Structurally complete | Real result plots and three actual worst-case images remain unavailable. |
| Dataset | Missing as committed artefact | Must be generated/acquired and versioned or linked with checksum. |
| Trained weights or link | Missing | Must train and publish checkpoints. |
| Five-minute screen recording | Missing, requires submitter | A recording runbook can be produced; the submitter must record the final trained pipeline and narration. |

## Current completion conclusion

The software design is complete, but the assignment is **not submission-complete**. The critical path is: dataset -> trained weights -> frozen-test distributions -> usable envelope -> latency/quantisation -> actual failure cases -> recording. Physical camera calibration and a real warehouse test set cannot be synthesized into truthful evidence.


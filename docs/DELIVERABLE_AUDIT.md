# Assignment deliverable audit

Audit date: 2026-08-22. `Complete` means an artefact exists and is reproducible. `Pending evidence` means code/protocol exists but the physical data or measured result does not. Synthetic results are never relabelled as warehouse results.

## Section 1 - Dataset and detection (30%)

| Deliverable | Status | Evidence / completion condition |
|---|---|---|
| Dataset in a standard format | Complete, synthetic-only | Committed `synthetic-v1`: 360 images, COCO keypoints and separate YOLO pose/segmentation labels. Real warehouse data remains absent. |
| Annotation tooling | Complete | `tools/generate_synthetic.py`, `tools/coco_to_yolo_pose.py`, COCO keypoint convention and labelling guide. SAM 2 is specified for offline propagation. |
| Source, cost, counts, split, guideline, biases | Complete for synthetic-v1 | `DATASET.md` and `data/dataset_manifest.json`; real count is explicitly zero. |
| Trained model and weights | Complete, synthetic-only | Committed YOLO11n pose and six-class segmentation checkpoints. Class-wise testing rejects the damage heads for operational use. |
| Detection/localisation held-out distributions | Complete, synthetic-only | `reports/pose_model/`: separate AP, pixel keypoint error, metric translation/rotation and predictions. |
| Training decisions and reasoning | Complete | `README.md` and `docs/RESEARCH_AND_MODEL_SELECTION.md`. |
| Accuracy ceiling and improvements | Complete | `DATASET.md`. |

## Section 2 - Pose estimation (35%)

| Deliverable | Status | Evidence / completion condition |
|---|---|---|
| Camera calibration artefacts and reprojection error | Pending physical camera | Protocol exists in `docs/CALIBRATION.md`; target camera/board images are unavailable. A synthetic calibration can test code but cannot substitute for this measurement. |
| Pose method and reasoning | Complete | Calibrated floor homography, directed keypoints and propagated uncertainty in `geometry.py`. |
| Self-constructed evaluation | Complete, synthetic-only | Sixty frozen shifted-test records include metric pose ground truth; physical surveyed placements remain required for real claims. |
| Translation/rotation error distributions | Complete, synthetic-only | p50/p90/p95/p99 in `reports/pose_model/pose_test_metrics.json`; real survey evaluation remains pending. |
| Height/tilt sensitivity, short/long range | Complete as assumed simulation | `reports/deployment/calibration_sensitivity.json` quantifies 1-5 m sensitivity under stated height/tilt-error assumptions; measured camera uncertainty remains unavailable. |
| Usable envelope | Complete negative result, synthetic-only | No synthetic yaw bin passes the one-sided 95% joint-success gate; range is not present in the data, so no operational envelope is declared. |
| Unreliable-pose output | Complete | Explicit `reliable=false`, uncertainty, reason and manual-inspection verdict. |

## Section 3 - Load analysis and SOP (25%)

| Deliverable | Status | Evidence / completion condition |
|---|---|---|
| Eight-check observability triage | Complete | README SOP table. |
| Implement verifiable subset | Complete as synthetic baseline | `run_trained_pipeline.py` connects both checkpoints to eight checks. Failed downstream gates suppress box-angle, load-mask and damage evidence instead of creating unsafe decisions. |
| Per-check confidence | Complete | Every check emits confidence, measurements and reason. |
| Pass/fail/manual verdict and weighting | Complete | `compliance.py`; pose failure forces manual review and reliable high-confidence failures dominate. |

## Section 4 - Deployment and robustness (10%)

| Deliverable | Status | Evidence / completion condition |
|---|---|---|
| Measured latency on declared hardware | Complete for available CPU | 1,000 warmed iterations per trained checkpoint with p50/p95/p99/FPS and hashes in `reports/deployment/`; no Jetson claim. |
| Orin Nano change analysis | Complete | `docs/DEPLOYMENT.md`. |
| Quantisation accuracy cost | Pending target runtime and real calibration set | Readiness audit proves CUDA/TensorRT and representative INT8 calibration data are absent; the required comparison protocol is documented without fabricated numbers. |
| Failure contract | Complete | Versioned assessment schema and explicit abstention. |
| Multi-frame use | Complete design | Cadenced runtime schedule and uncertainty-aware temporal policy documented. |

## Required output and submission material

| Deliverable | Status | Evidence / completion condition |
|---|---|---|
| One assessment per pallet | Complete | Trained pipeline emitted 60 separate schema-valid held-out assessment JSON files plus an index and summary in `reports/end_to_end/`. |
| README required sections | Complete for synthetic scope | Measured pose/segmentation results and three actual shifted-test pose failures are included. Real failures remain unavailable. |
| Dataset | Complete, synthetic-only | Versioned generator, manifest, COCO and YOLO artefacts are committed. Real validation is still required for operational claims. |
| Trained weights or link | Complete, synthetic-only | Synthetic YOLO11n pose and six-class load-segmentation weights are committed. The damage classes explicitly fail the acceptance gate. |
| Five-minute screen recording | Missing, requires submitter | A recording runbook can be produced; the submitter must record the final trained pipeline and narration. |

## Current completion conclusion

The software design is complete, but the assignment is **not submission-complete**. The critical path is: dataset -> trained weights -> frozen-test distributions -> usable envelope -> latency/quantisation -> actual failure cases -> recording. Physical camera calibration and a real warehouse test set cannot be synthesized into truthful evidence.

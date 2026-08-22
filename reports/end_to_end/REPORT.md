# End-to-end trained pipeline report

This report covers the committed `synthetic-v1` shifted test split only. It does
not establish real warehouse performance.

## Executed path

For every one of the 60 held-out images, the runner executed:

```text
image -> trained YOLO11n pose -> metric floor pose + uncertainty
      -> trained YOLO11n segmentation -> validation gates
      -> eight SOP checks -> fail-safe per-pallet assessment JSON
```

The runner produced 60 detections and 60 separate assessment files. All 60
validate against `schemas/assessment.schema.json`. Each includes the source image
hash, both checkpoint hashes, model version, inference device, evidence gates,
per-check confidence/reason, and inference timing.

## Verdict summary

| Output | Count |
|---|---:|
| `manual_inspection` | 60 |
| `fail` | 0 |
| `pass` | 0 |

This is the correct safety result for the current evidence. SOPs 1, 2, 4, 6, 7,
and 8 cannot be established from the validated model outputs. Three images have
high-confidence positive wrap detections, so SOP 5 passes on those images; lack
of a high-confidence wrap detection stays manual rather than becoming a false
failure.

## Validation gates caught unsafe evidence

The model's raw box-mask angles would have failed SOP 3 on 46 of 60 scenes even
though the generator's boxes are aligned with the pallet. That downstream false
failure rate rejects the feature despite aggregate box mask mAP50-95 of 0.464.
Raw angles and the hypothetical failure remain in every assessment's
`experimental_evidence`, but `box_angles_deg` is withheld from the policy.

Both damage classes are likewise suppressed because their held-out AP is zero.
A missing damage detection is **unknown**, not evidence of no damage. Load masks
are suppressed because their mask mAP50-95 is 0.006.

All 60 pose estimates pass the local uncertainty rule, while only 40% meet the
known 2 cm / 3 degree ground truth gate. This reveals overconfident uncertainty
under synthetic domain shift and is why no automatic overall pass is allowed.

## Reproduce

```bash
python tools/run_trained_pipeline.py data/pallet_pose/images/test \
  --pose-weights weights/pallet_pose_yolo11n_synthetic_v1.pt \
  --segment-weights weights/load_seg_yolo11n_synthetic_v1.pt \
  --calibration data/synthetic_calibration.json \
  --output-dir reports/end_to_end --device cpu --imgsz 256
python tools/summarize_assessments.py reports/end_to_end \
  -o reports/end_to_end/summary.json
python tools/validate_assessments.py reports/end_to_end
```

The summary is machine readable in `summary.json`; `index.json` maps each source
image to its per-pallet assessment file.

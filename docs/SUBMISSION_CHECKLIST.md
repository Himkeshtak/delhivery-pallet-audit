# Submission checklist

## Included in the GitHub repository

- [x] README with all five required sections
- [x] Dataset card with source/cost, exact class counts, split protocol, labelling
  guideline, biases, and accuracy ceiling
- [x] Standard COCO, YOLO pose, and YOLO segmentation dataset artefacts
- [x] Deterministic dataset generator, converter, validator, and statistics tool
- [x] Trained pose and segmentation weights with hashes and provenance
- [x] Separate detection AP, keypoint pixel error, and metric pose distributions
- [x] Three actual held-out worst cases with images and root causes
- [x] Calibration method, synthetic calibration fixture, and physical protocol
- [x] Pose sensitivity and a negative synthetic yaw-envelope result
- [x] Eight-check SOP triage, per-check confidence, safe verdict policy
- [x] One schema-valid assessment per detected held-out pallet
- [x] Measured CPU latency, target-hardware reasoning, and export-readiness audit
- [x] Multi-frame deployment policy and explicit failure contract
- [x] AI-tool disclosure
- [x] Five-minute recording script

## External evidence still required for an operational claim

- [ ] License-cleared target warehouse images and staged SOP violations
- [ ] Physical target-camera ChArUco/floor calibration and reprojection residuals
- [ ] Independently surveyed real pallet placements and real failure images
- [ ] TensorRT FP16/optional INT8 comparison and latency on Jetson Orin Nano 15 W
- [ ] Human-created five-minute screen recording and accessible submission link

These items require hardware, a warehouse/camera, or the submitter. They must not
be replaced by synthetic numbers or benchmarks copied from another machine.

## Final verification commands

```powershell
python tools\validate_training_dataset.py data
python tools\dataset_statistics.py data -o data\dataset_statistics.json
python tools\validate_assessments.py reports\end_to_end
python -m unittest discover -s tests -v
git diff --check
git status --short --branch
git log -1 --oneline
```

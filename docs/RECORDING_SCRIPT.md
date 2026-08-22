# Five-minute screen-recording runbook

The submitter must make the final recording because the assignment expects the
candidate to narrate and demonstrate work they understand. This runbook keeps the
recording within five minutes and avoids rerunning the 30-minute CPU training job.

## Before recording

1. Open the repository root, README, a terminal, and the three pose failure images.
2. Activate the environment with `pip install -e ".[vision,dev]"` already complete.
3. Use a 1920x1080 screen and increase terminal/editor text until it is legible.
4. Clear `tmp/demo-assessments` if it exists; `tmp/` is Git-ignored.
5. Start the recorder and verify the microphone level before the timed narration.

## Timed narration and actions

### 0:00-0:35 - Problem and claim scope

Show the README title and submission-status callout. Say:

> The system detects each pallet without fiducials, estimates directed floor pose,
> evaluates eight load SOP checks, and emits a fail-safe JSON assessment. The
> committed training and test data are synthetic, so I do not claim warehouse or
> Jetson performance.

### 0:35-1:15 - Dataset and training

Show `DATASET.md`, `data/dataset_statistics.json`, and both provenance files.
Run:

```powershell
python tools/validate_training_dataset.py data
Get-ChildItem weights\*.pt | Select-Object Name,Length
```

State: 360 images split 240/60/60, independent shifted test generation, ordered
four-corner pose labels, six segmentation classes, YOLO11n pose trained for 20
epochs and YOLO11n-seg for 12 epochs on the declared CPU.

### 1:15-2:10 - Execute the trained pipeline

Run one held-out image end to end:

```powershell
python tools/run_trained_pipeline.py data\pallet_pose\images\test\test_0000.jpg `
  --output-dir tmp\demo-assessments --device cpu --imgsz 256
Get-Content tmp\demo-assessments\test_0000_pallet-001.json
```

Point out metric `x_m`, `y_m`, directed `theta_deg`, uncertainty, eight per-check
verdicts/confidences/reasons, overall verdict, hashes, model version, and evidence
gates. Explain that unsupported evidence becomes `manual_inspection`.

### 2:10-3:00 - Results and failures

Show `reports/pose_model/REPORT.md`, `reports/load_seg_model/REPORT.md`, and the
three worst-case images in `reports/pose_model/failures/`. State the pose p50/p95
translation and rotation errors, 40% joint success, and the segmentation
class-wise failures. Root-cause glare shift, wrap/load edge competition, and
shallow apparent pallet depth. Emphasize that high AP did not imply 2 cm metric
accuracy.

### 3:00-3:45 - SOP safety decisions

Show `reports/end_to_end/REPORT.md` and `summary.json`. Explain that all 60 images
produce a schema-valid assessment and all route to manual inspection. Raw box-mask
angles would falsely fail 46 known-aligned scenes, so the validation gate stores
them as diagnostics but prevents them from affecting decisions. Damage non-
detections never become a pass.

### 3:45-4:25 - Deployment and robustness

Show `reports/deployment/REPORT.md`. State the actual CPU, 1,000-iteration p50/p95
results, derived 11.48 FPS cadence, negative yaw envelope, and height/tilt
sensitivity. Explain that no Jetson/TensorRT number is claimed because neither is
available; target measurement is a required external gate.

### 4:25-5:00 - Verification and limitations

Run:

```powershell
python tools/validate_assessments.py reports\end_to_end
python -m unittest discover -s tests -v
git status --short --branch
```

Finish by showing README sections 4 and 5. Name the remaining physical inputs:
real target-camera data, surveyed calibration, real frozen test placements, and
Orin Nano measurement. Disclose Codex usage and the caught error: treating absent
damage detections or unvalidated mask angles as confident evidence.

## Recording acceptance checklist

- Duration is between 4:45 and 5:15.
- Commands and outputs are readable at normal playback size.
- The JSON assessment and all three failure images appear on screen.
- The narration states synthetic scope and actual measured hardware.
- No tokens, credentials, unrelated files, or private notifications are visible.
- Upload the recording and add its accessible link to the submission message.

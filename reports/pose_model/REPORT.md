# YOLO11n pallet-pose training report

## Claim scope

All results are measured on `synthetic-v1`. The 60-image test split deliberately changes the background palette and adds stronger noise, blur and glare. These results validate the training/evaluation pipeline; they do not establish warehouse accuracy.

## Training

- Initial checkpoint: official `yolo11n-pose.pt`
- Output: `weights/pallet_pose_yolo11n_synthetic_v1.pt` (5.6 MB)
- Data: 240 train, 60 validation, 60 held-out test
- Four ordered keypoints: front-left, front-right, back-right, back-left
- 20 epochs, image size 256, batch 8, seed 17, deterministic mode
- Hardware: Intel Core Ultra 5 125U CPU, PyTorch 2.8.0+cpu
- Measured training time: 1,176.4 seconds (19.6 minutes)

Full epoch metrics and provenance are committed beside this report.

## Held-out results

| Metric | Result |
|---|---:|
| Detection recall | 1.000 |
| Box mAP50 / mAP50-95 | 0.995 / 0.966 |
| Pose mAP50 / mAP50-95 | 0.995 / 0.995 |
| Keypoint error p50 / p95 | 4.14 px / 7.94 px |
| Translation error p50 / p95 | 2.10 cm / 4.79 cm |
| Rotation error p50 / p95 | 1.41 deg / 3.68 deg |
| Joint <=2 cm and <=3 deg | 40.0% of all test images |
| Ultralytics CPU inference | 42.5 ms/image at 256 px |

The central finding is that high OKS-based pose AP does not imply compliance-grade metric pose. Several-pixel coherent keypoint bias becomes multiple centimetres after floor projection. The system therefore must retain metric evaluation and uncertainty-based abstention as first-class gates.

## Three worst cases

Green is ground truth and red is the prediction.

1. `worst_1.jpg`: 6.5 cm / 1.1 deg. Glare and the cool/dark test palette shift all four corners coherently; confidence alone does not expose the metric bias.
2. `worst_2.jpg`: 5.0 cm / 2.9 deg. The light wrap/load outline competes with the pallet rear edge, expanding and skewing the fitted footprint.
3. `worst_3.jpg`: 3.6 cm / 4.2 deg. A shallow apparent pallet depth makes orientation sensitive to asymmetric rear-corner error.

See `failures/index.json` for exact per-keypoint errors and the rendered evidence images.

## Required next experiment

Capture surveyed real placements from the target pillar camera. Fine-tune the checkpoint on real training sessions, freeze a different session/site for test, recalibrate confidence against metric error, and declare a usable range/yaw envelope only where the p95 bounds satisfy both tolerances.


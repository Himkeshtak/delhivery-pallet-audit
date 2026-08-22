# Deployment, sensitivity, and envelope report

All latency values below were measured on the available **Intel Core Ultra 5
125U CPU** with Python 3.13.5, PyTorch 2.8.0+cpu, Ultralytics 8.3.179, batch 1,
image size 256, confidence threshold 0.05, 20 warm-up iterations, and 1,000
timed iterations. They are not Jetson Orin Nano measurements.

## Steady-state model latency

| Model | p50 | p95 | p99 | Mean throughput |
|---|---:|---:|---:|---:|
| YOLO11n pose | 63.1 ms | 88.6 ms | 106.9 ms | 15.46 FPS |
| YOLO11n segmentation | 62.2 ms | 99.6 ms | 120.8 ms | 14.85 FPS |

The scope is decode, preprocess, inference, and postprocess through
`model.predict`; application serialization is excluded. Weight SHA-256 values,
software/platform provenance, and exact percentiles are in the two benchmark
JSON files.

From measured mean service times, running pose on every frame and segmentation
every third frame costs approximately 87.1 ms/frame, or **11.48 FPS**, before
tracking, geometry, and serialization. Therefore this CPU configuration fails a
15 FPS gate. The 60-image executed assessment path observed a 100.5 ms median
for pose plus segmentation; its first-run setup outlier is intentionally not
used as a steady-state benchmark.

## Calibration sensitivity

The Monte Carlo analysis assumes a 1.2 m camera, 1 cm camera-height uncertainty,
and 0.25 degree tilt-calibration uncertainty. Translation error p95 grows from
2.4 cm at 1 m to 5.1 cm at 2 m, 9.0 cm at 3 m, 14.1 cm at 4 m, and 20.6 cm at
5 m. Even the assumed 1 m case does not clear the 2 cm p95 requirement. These are
first-order assumptions, not a substitute for physical calibration perturbation.

## Synthetic yaw envelope

| Absolute yaw | Samples | Joint 2 cm / 3 deg | One-sided 95% lower bound | Accepted |
|---|---:|---:|---:|---:|
| 0-15 deg | 28 | 39.3% | 25.7% | No |
| 15-30 deg | 23 | 43.5% | 28.1% | No |
| 30-45 deg | 9 | 33.3% | 14.2% | No |

No yaw bin clears the declared 95% joint-success gate. The synthetic dataset has
no surveyed camera-to-pallet range, so a short/mid/long range envelope cannot be
reported. Accordingly, **no operational envelope is declared**.

## Quantization and Jetson gate

The readiness audit records no CUDA device and no TensorRT runtime on the current
machine. A production INT8 calibration set also does not exist. Exporting an
engine elsewhere and presenting it as measured here would be misleading, so no
FP16/INT8 accuracy delta or Jetson latency is claimed.

On the actual Orin Nano, the required next gate is TensorRT FP16 at batch 1 using
the same image resolution and frozen real test set. INT8 is accepted only after
calibration on representative target scenes and comparison of keypoint pixel
error, metric pose p95, joint 2 cm / 3 degree rate, class-wise mask AP, and SOP
verdict changes against FP32/FP16.

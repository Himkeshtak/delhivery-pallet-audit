# Deployment and robustness

## Latency measurement

No Jetson measurement is claimed. The available Intel Core Ultra 5 125U CPU was
measured at batch 1 and image size 256 after 20 warm-up iterations over 1,000
timed predictions per model at the runtime confidence threshold of 0.05. Pose
measured 63.1/88.6/106.9 ms p50/p95/p99 and 15.46 FPS mean throughput;
segmentation measured 62.2/99.6/120.8 ms and 14.85 FPS. Pose every frame plus
segmentation every third frame is approximately 11.48 FPS from measured mean
service time, before tracking and serialization. Exact
machine-readable records are in `reports/deployment/`.

For Orin Nano 15 W, export the selected model to TensorRT FP16 and profile batch=1. Achieving 15 FPS requires an end-to-end p95 below 66.7 ms, not merely model inference below that value. INT8 is acceptable only if the same held-out set shows tolerable change in keypoint and final pose distributions; calibration images must represent range, lighting, wrap, and occlusion.

## Failure contract

Downstream consumers receive `pose.reliable`, uncertainty, per-check confidence/reason, overall verdict, model/calibration version, and timestamp. Missing observations, invalid calibration, out-of-envelope range/view, conflicting temporal estimates, or stale frames produce `manual_inspection`. Health metrics track abstention rate, confidence drift, latency, and input-quality distributions.

## Temporal use

A stationary pallet provides repeated measurements, not independent ground truths. Associate tracks geometrically, reject moving/forklift-obscured frames, and fuse floor pose with an uncertainty-weighted robust estimator. Require temporal consistency before changing a verdict; preserve the worst supported SOP failure. Correlated calibration error does not average away, so the fused covariance includes a shared calibration term. Store representative evidence frames rather than every frame.

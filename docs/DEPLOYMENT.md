# Deployment and robustness

## Latency measurement

No Jetson measurement is claimed. On available hardware, benchmark decode, preprocessing, detector, load models, geometry, and serialization separately and end-to-end. Record CPU/GPU, power mode, software versions, input resolution, batch size, warm-up, synchronization, p50/p95/p99, and FPS over at least 1,000 frames.

For Orin Nano 15 W, export the selected model to TensorRT FP16 and profile batch=1. Achieving 15 FPS requires an end-to-end p95 below 66.7 ms, not merely model inference below that value. INT8 is acceptable only if the same held-out set shows tolerable change in keypoint and final pose distributions; calibration images must represent range, lighting, wrap, and occlusion.

## Failure contract

Downstream consumers receive `pose.reliable`, uncertainty, per-check confidence/reason, overall verdict, model/calibration version, and timestamp. Missing observations, invalid calibration, out-of-envelope range/view, conflicting temporal estimates, or stale frames produce `manual_inspection`. Health metrics track abstention rate, confidence drift, latency, and input-quality distributions.

## Temporal use

A stationary pallet provides repeated measurements, not independent ground truths. Associate tracks geometrically, reject moving/forklift-obscured frames, and fuse floor pose with an uncertainty-weighted robust estimator. Require temporal consistency before changing a verdict; preserve the worst supported SOP failure. Correlated calibration error does not average away, so the fused covariance includes a shared calibration term. Store representative evidence frames rather than every frame.


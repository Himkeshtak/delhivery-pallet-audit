# Data collection and model execution runbook

## Before capture

- Freeze camera model, lens, resolution, focus, exposure policy and mount; assign a `camera_id` and `calibration_id`.
- Print and survey the floor-frame diagram. Record camera height/tilt only as diagnostics; solve calibration from measured correspondences.
- Prepare distinct `capture_session` IDs for different days, light states and staging layouts.
- Confirm redistribution rights before mixing public images into the repository.

## Minimum capture matrix

Capture independent physical placements, not bursts counted as independent examples:

| Factor | Required bins |
|---|---|
| Range | 1-2 m, 2-3 m, 3-4 m, 4-5 m |
| Pallet yaw | every 10 degrees across the operational range |
| Occlusion | none, one corner, two corners, forklift/person, neighbouring load |
| Lighting | normal, dim, glare, hard shadow, mixed colour temperature |
| Pallet | wood/plastic and every operational dimension/entry-face type |
| Load | empty, low/high, dark/light cartons, transparent wrap, irregular outline |
| Violations | each visible SOP violation near threshold and clearly beyond threshold |

For pose ground truth, survey all four floor corners for at least 30 placements per range/yaw bin. Add repeated frames only for temporal evaluation.

## Annotation and conversion

1. Annotate in COCO keypoints with ordered `front_left, front_right, back_right, back_left` and COCO visibility values.
2. Put `capture_session` on every image record. Missing groups fall back to filenames and weaken leakage protection.
3. Convert with:

   ```bash
   python tools/coco_to_yolo_pose.py annotations.json images data/pallet_pose
   ```

4. Review `split_manifest.json`; no capture session may cross splits. Keep the test manifest frozen.
5. Use SAM 2 only to propose masks. A human corrects box boundaries, transparent wrap and contact footprints.

## Train in priority order

```bash
python tools/train_models.py pose --model yolo11n-pose.pt --data configs/pallet_pose.yaml
python tools/train_models.py pose --model yolo11s-pose.pt --data configs/pallet_pose.yaml
python tools/train_models.py segment --model yolo11n-seg.pt --data configs/load_seg.yaml
```

Choose pose weights by the output of `tools/evaluate_pose.py`, not training loss or COCO AP alone. Run sensitivity analysis before declaring a usable range.

## Export and measure

```bash
python tools/export_models.py runs/pose-yolo11s-pose/weights/best.pt --format engine
python tools/benchmark_model.py best.engine data/pallet_pose/images/test -o benchmark-jetson.json
```

For INT8, provide `--int8 --data configs/pallet_pose.yaml` and compare the same frozen held-out set. Record `nvpmodel`, `jetson_clocks`, thermals and power mode alongside the generated JSON. The model harness reports its actual platform and does not label a desktop run as Jetson.


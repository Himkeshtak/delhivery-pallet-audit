# Synthetic load segmentation model report

This report is limited to the committed `synthetic-v1` data. It is not evidence of
warehouse accuracy and must not be used to approve freight without real-data
validation.

## Reproducible training run

- Model: pretrained YOLO11n-seg, fine-tuned as six classes
- Classes: `pallet`, `load`, `box`, `stretch_wrap`, `visible_box_damage`,
  `visible_pallet_damage`
- Split: 240 train / 60 validation / 60 shifted held-out test images
- Run: 12 epochs, image size 256, batch 8, seed 17, deterministic mode
- Hardware: Intel Core Ultra 5 125U CPU; PyTorch 2.8.0+cpu; no CUDA
- Training elapsed time: 640.6 seconds
- Weight: `weights/load_seg_yolo11n_synthetic_v1.pt`

The command and runtime inputs are recorded in `training_provenance.json`; the
epoch history is in `training_results.csv`.

## Shifted held-out test result

| Metric | Box | Mask |
|---|---:|---:|
| Precision | 0.916 | 0.672 |
| Recall | 0.655 | 0.347 |
| mAP50 | 0.644 | 0.321 |
| mAP50-95 | 0.599 | 0.197 |

| Class | Box mAP50-95 | Mask mAP50-95 | Operational interpretation |
|---|---:|---:|---|
| pallet | 0.908 | 0.709 | Learned in the synthetic domain |
| load | 0.947 | 0.006 | Box learned; mask unusable |
| box | 0.904 | 0.464 | Moderate synthetic mask baseline |
| stretch_wrap | 0.833 | 0.000 | Detection only; mask unusable |
| visible_box_damage | 0.000 | 0.000 | Unsupported; always manual review |
| visible_pallet_damage | 0.000 | 0.000 | Unsupported; always manual review |

The aggregate metric hides total failure on both rare damage classes. The
runtime must therefore treat a missing damage detection as **unknown**, never as
proof that the pallet or boxes are undamaged. Full precision/recall/AP values
and measured validation timing are in `yolo_test_metrics.json`.

## Decision and next experiment

This checkpoint is retained as a reproducible baseline for pallet/box detection.
It is not accepted for metric load masks, wrap segmentation, or damage decisions.
The next training gate is a real, scene-split warehouse dataset with substantially
more positive damage examples, hard negatives, and mask QA. Class rebalancing or
damage crops should be tested only against the frozen real test set.

Rendered examples in `examples/` make the model's actual outputs inspectable.

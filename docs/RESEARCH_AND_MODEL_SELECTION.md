# Research review and model selection

Reviewed 2026-08-22. Reported paper numbers below are the authors' measurements on their own datasets/hardware; they are **not** results of this repository and are not directly comparable. The submission accepts only locally reproduced, held-out distributions.

## Decision summary

| Function | Chosen baseline | Why | Runtime policy |
|---|---|---|---|
| Pallet and directed geometry | YOLO11n/s-pose, four ordered floor corners | Directly matches pallet keypoint literature; one-stage, exportable, face labels resolve 180-degree ambiguity | Every frame; select n vs s by end-to-end metric error |
| Metric floor pose | Surveyed floor homography + uncertainty | The fixed camera and planar target make this more identifiable than learned monocular depth | Reject outside 2 cm/3-degree uncertainty envelope |
| Load/box/wrap evidence | YOLO11n-seg | Masks support contact footprint, visible centroid, wrap and box instances; lower cost than Mask2Former | Every third stable-track frame |
| Box alignment | PCA/minimum rectangle on projected box mask | Removes a separate OBB network; relative angle is the needed output | Abstain for small/occluded/near-square masks |
| Visible damage | EfficientAD-S | Fast nominal-only industrial anomaly baseline; rare damage labels are not required to start | Every tenth stable-track frame, positive evidence only |
| Annotation acceleration | SAM 2 | Strong promptable image/video propagation | Offline only; human accepts/corrects every propagated label |

## Most relevant pallet research

### Geometry-aware keypoints are the closest match

Zhou and Lu et al. (2025), *Unmanned forklift pallet positioning algorithm based on an improved human pose estimation model*, modify YOLO11s-pose to detect 12 structural pallet keypoints, then use visually weighted topology constraints and EPnP. The authors report under-19 mm horizontal and under-2.9-degree angular error on their embedded setup. This is unusually close to the assignment tolerance, but the camera is forklift-facing and the dataset/protocol differ. It supports the **architecture choice**, not a performance claim. [Paper/abstract](https://nyaspubs.onlinelibrary.wiley.com/doi/10.1111/nyas.70001)

Ye et al. (2026), *A Hypergraph Computing and Knowledge-Enhanced Framework for Forklift Pallet Pose Estimation*, adds high-order structural constraints and uncertainty-aware optimization for occlusion. The reported 18 mm/1.6-degree errors and Orin Nano throughput are promising, but the custom hypergraph modules are not released here and reproducing them within the assignment window would be risky. We adopt its topology/uncertainty principles through ordered keypoints, confidence gates and geometry tests. [Paper](https://nyaspubs.onlinelibrary.wiley.com/doi/10.1111/nyas.70219)

Mueller et al. (2025), *Pallet Detection and Localisation From Synthetic Data*, uses Unity domain randomization and side-face geometry. Its reported detection is strong, but the 4.2 cm position and 8.2-degree rotation errors miss this assignment's tolerances. Synthetic data is therefore a pretraining/rare-case supplement, never the only held-out evaluation source. [Paper](https://arxiv.org/abs/2503.22965)

Knitt et al. (2022), *Estimating the Pose of a Euro Pallet with an RGB Camera based on Synthetic Training Data*, trains NVIDIA DOPE from synthetic pallet renders and confirms RGB-only feasibility, while the preferred dataset's mean position error remains below 20 cm rather than 2 cm. It reinforces the domain-gap risk. [Paper](https://arxiv.org/abs/2210.06001) and [official DOPE code](https://github.com/NVlabs/Deep_Object_Pose)

Miura et al. (2026), *Real-Time 6DoF Pallet Pose Estimation with Monocular Metric Depth*, combines Depth Anything V2, a ground-normal constraint, and RGB-depth fusion. The authors report 3.88 cm translation and 1.65-degree rotation error, with 78.1% success under a looser 5 cm/3-degree gate up to 5 m. This is a valuable **ablation** for cameras that move, but it does not clear this fixed-camera assignment's 2 cm bar. [Paper](https://www.scitepress.org/Papers/2026/146268/146268.pdf)

## General pose research

PVNet's pixel-wise voting recovers occluded keypoints and passes their uncertainty into PnP. This is the best research upgrade if direct keypoint regression fails under occlusion, but it costs custom implementation and deployment work. [PVNet, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Peng_PVNet_Pixel-Wise_Voting_Network_for_6DoF_Pose_Estimation_CVPR_2019_paper.html)

ZebraPose encodes dense surface correspondences and solves PnP. It fits a known CAD object but assumes a stable object surface model; mixed pallet types and loads weaken that assumption. [ZebraPose, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Su_ZebraPose_Coarse_To_Fine_Surface_Encoding_for_6DoF_Object_Pose_CVPR_2022_paper.html)

FoundationPose is a strong novel-object 6D estimator with CAD or reference views, but its main formulation uses RGB-D. This assignment supplies one monocular RGB view, so it is not the primary baseline. It becomes attractive if a depth camera is permitted in the live requirement change. [FoundationPose, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.pdf)

RTMPose is a credible lightweight keypoint alternative with strong deployment work, but its standard top-down design needs a detector. Benchmark it if one-stage pallet pose misses small/occluded corners. [RTMPose](https://arxiv.org/abs/2303.07399)

## Segmentation and temporal evidence

Mask2Former is a high-quality universal segmentation research baseline, but its transformer stack is less aligned with the 15 W deployment budget than a nano instance segmenter. [Mask2Former, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf)

SAM 2 adds streaming-memory video segmentation and needs fewer interactions than its predecessor. Use it to propagate human-corrected masks across a capture clip and to prototype temporal consistency, not as evidence that an automatically propagated label is correct. [SAM 2 paper](https://arxiv.org/abs/2408.00714) and [Meta project](https://ai.meta.com/research/sam2/)

## Damage and anomaly research

PatchCore is highly sample-efficient and localizes unexpected patches from nominal examples using a coreset memory bank. It is the accuracy-oriented research baseline. [PatchCore, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html)

EfficientAD uses a lightweight student-teacher model plus an autoencoder for local and logical anomalies and is explicitly designed for low latency. That makes EfficientAD-S the deployment baseline; PatchCore remains the comparison. The threshold must be selected on warehouse validation images at a fixed false-positive rate. [EfficientAD, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond-Level_Latencies_WACV_2024_paper.html) and [Anomalib implementation](https://github.com/open-edge-platform/anomalib)

An anomaly score can support a **positive visible-damage failure**. It cannot prove that occluded faces are undamaged, so low scores remain manual inspection for SOP 6 and 8.

## Why not use learned monocular depth as ground truth?

Depth Anything V2 is a strong and efficient depth foundation model with metric variants, but metric scale is domain-dependent. The pallet-specific metric-depth paper still reports translation error above this assignment's tolerance. The stationary calibrated camera supplies a more constrained route: survey the floor once, project detected contact geometry into the declared floor frame, and quantify calibration sensitivity. [Depth Anything V2](https://arxiv.org/abs/2406.09414)

## Implementation map

- `src/pallet_audit/ultralytics_backend.py`: optional pose/mask adapter.
- `src/pallet_audit/evidence.py`: floor footprint, box alignment, and vertical-line height geometry.
- `src/pallet_audit/model_stack.py`: explicit model roles and temporal cadence.
- `tools/coco_to_yolo_pose.py`: capture-session-grouped conversion.
- `tools/train_models.py`: deterministic pose/segment training with provenance.
- `tools/export_models.py`: FP16/INT8 export with an INT8 calibration-data guard.
- `configs/`: dataset and baseline declarations.

## Experiment order and stop/go gates

1. **Calibration-only gate:** p95 survey-point back-projection must be comfortably below 2 cm across the floor. If not, no model can rescue the system.
2. **Pose n/s baseline:** train both on identical groups. Continue only if held-out p95 corner error yields a usable joint 2 cm/3-degree region; report abstentions.
3. **Occlusion upgrade:** if hidden corners dominate failures, compare RTMPose and a PVNet-style voting head before increasing model size.
4. **Segmentation:** evaluate mask AP separately, then SOP measurement error in centimetres/degrees. AP alone is insufficient.
5. **Anomaly:** compare EfficientAD-S against PatchCore at the same crop resolution and fixed false-positive rate. Never tune on the test set.
6. **Temporal fusion:** measure accuracy and latency at cadences 1/3/10. Shared calibration bias must not be averaged away.
7. **Export:** compare PyTorch FP32, TensorRT FP16, and optionally INT8 on the same held-out records. Refit confidence thresholds after quantization only on validation data.
8. **Target measurement:** accept 15 FPS only from end-to-end p95 measurements on the actual Orin Nano 15 W mode. NVIDIA documents 15 W as the default Orin Nano development-kit mode; paper or desktop numbers are context only. [NVIDIA power-mode documentation](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)


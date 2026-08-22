# Dataset card

## Sources and rationale

No third-party images are committed because license and access were not supplied. The planned dataset has three auditable sources: (1) target-camera captures from multiple warehouse sessions for geometry and deployment realism; (2) permissively licensed public pallet/box datasets for appearance breadth; (3) staged failure cases for rare SOP violations. Every image receives `source`, `license`, `capture_session`, `site`, `camera_id`, and consent fields in a manifest.

The cost is deliberate: target captures require operations access and surveying, staged violations require safe setup, and license validation may exclude attractive public data. This costs speed but avoids data leakage, illegal redistribution, and a misleading ceiling based on unrelated web imagery.

The included generator creates COCO-keypoints smoke-test data. It has zero real examples, zero domain validity, and must not be used for benchmark claims.

## Counts

Current committed counts: 0 real images, 0 real pallet instances, 0 real box/damage/wrap annotations. Synthetic count is chosen by the generator CLI and reported in its `annotations.json`. Target minimum before training: 2,000 images, 4,000 pallets, at least 500 examples per reliably trained binary attribute, and at least 100 staged positives for each rare damage category. Counts must be regenerated from annotations, not copied into prose.

## Split protocol

Split 70/15/15 by `(site, capture_session, camera_id)` group with no group crossing splits. The held-out test set should use a different session and, preferably, a different site/camera. Near-duplicate video frames stay in one group. Stratify groups for range, yaw, occlusion, illumination, pallet type, load height, and SOP failures. Freeze the test manifest before model selection.

## One-page labelling guideline

1. Draw a tight visible pallet bounding box; do not include load or shadow.
2. Label four floor-plane corners in order: front-left, front-right, back-right, back-left, viewed in the pallet's physical labelled-front frame. Use pallet markings/entry-face geometry to resolve the front; mark `front_unknown=true` if it cannot be resolved.
3. Keypoint state uses COCO convention: 0 absent/unknown, 1 labelled but occluded, 2 visible. Do not guess an invisible corner unless the physical edge interpolation is unambiguous; state 1 records the inference.
4. Segment visible box faces and load silhouette. Keep individual touching boxes separate where an edge is discernible.
5. For overhang and centroid, annotate the visible load contact footprint. Set `rear_extent_unknown=true` when the rear side is hidden.
6. Annotate height at the highest load point, excluding loose wrap. Record top occlusion.
7. Box orientation is the dominant horizontal edge relative to labelled pallet axes. Ignore printing/text direction.
8. Size inversion is positive only when a supported upper box footprint is demonstrably larger than the supporting lower box in the visible stack.
9. Wrap is positive only with visible film evidence; glare-only ambiguous cases are `unknown`, not negative.
10. Damage uses localized polygons and type (`crushed`, `torn`, `broken_board`, `split_stringer`). Hidden regions are unknown.
11. QA: double-label 10%; adjudicate ordering disagreements and any corner delta above 5 px. Track inter-annotator pixel error and Cohen's kappa for attributes.

## Known biases and gaps

Public imagery overrepresents clean product shots and frontal views. Fixed-camera captures can overfit one floor texture and illumination. Rare severe damage is safety-constrained and underrepresented. Transparent wrap, dark cartons, unusual pallets, people/forklifts, night illumination, motion blur, rain/dust, and rear-side violations are expected gaps. A single camera has structural unobservability, which more training cannot remove.

## Accuracy ceiling

With roughly 2,000 target images, a small pose model may localize visible corners well, but the ±2 cm metric ceiling will likely be set by surveyed calibration, occlusion, and long-range projection rather than detector AP. More adjacent video frames add little. The ceiling rises through independent sites/cameras, precise surveyed ground truth, more rare conditions, temporal fusion, and a second viewpoint. It does not rise by assigning confident labels to invisible geometry.


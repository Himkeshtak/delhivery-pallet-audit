# Calibration and pose ground truth protocol

1. Lock focus, resolution, zoom, mount, and image preprocessing. Any change creates a new calibration ID.
2. Capture at least 30 ChArUco frames spanning the full image, depths, and board tilts. Estimate intrinsics/distortion; report per-frame and overall reprojection RMSE. Reject calibration only by a documented rule, not to improve the headline.
3. Survey at least 12 non-collinear floor points across the usable area with a tape/laser referenced to a declared origin. Fit the undistorted image-to-floor homography with RANSAC, then refit on inliers. Store intrinsics, distortion, homography, image size, timestamps, units, and point residuals in a versioned JSON artefact.
4. Ground truth pallet pose with four surveyed floor corners or a rigid jig measured independently of the vision result. Repeat at short/mid/long range and yaw bins, including partial occlusion. The labelled entry face defines directed angle.
5. Report translation and rotation distributions separately plus joint 2 cm/3 degree pass rate and abstention rate. Bootstrap 95% confidence intervals by physical placement, not frame.
6. Run `tools/sensitivity.py` with survey-derived height/tilt uncertainty. Validate Monte Carlo predictions by deliberately perturbing stored calibration parameters and re-running held-out placements.

Calibration failure is explicit: wrong image dimensions, unknown calibration ID, singular projection, excessive residual, or pose uncertainty outside the envelope yields `manual_inspection`, never a zero pose.


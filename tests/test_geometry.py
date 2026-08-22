import unittest

import numpy as np

from pallet_audit.geometry import CameraModel, pose_from_keypoints, slot_error


class GeometryTests(unittest.TestCase):
    def test_identity_pose_and_orientation(self):
        pose = pose_from_keypoints(np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]]), CameraModel(np.eye(3)), pixel_sigma=.001)
        self.assertAlmostEqual(pose.x_m, .6)
        self.assertAlmostEqual(pose.y_m, .5)
        self.assertAlmostEqual(pose.theta_deg, 0)
        self.assertTrue(pose.reliable)

    def test_directed_front_face_preserves_180_degrees(self):
        pose = pose_from_keypoints(np.array([[1.2, 1], [0, 1], [0, 0], [1.2, 0]]), CameraModel(np.eye(3)), pixel_sigma=.001)
        self.assertAlmostEqual(pose.theta_deg, -180)

    def test_low_confidence_abstains(self):
        pose = pose_from_keypoints(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), CameraModel(np.eye(3)), np.array([.9, .4, .9, .9]))
        self.assertFalse(pose.reliable)
        self.assertIn("low keypoint", pose.reason)

    def test_slot_thresholds_are_inclusive(self):
        pose = pose_from_keypoints(np.array([[0, 0], [1, 0], [1, 1], [0, 1]]), CameraModel(np.eye(3)), pixel_sigma=.001)
        self.assertTrue(slot_error(pose, .52, .5, 3)["compliant"])


if __name__ == "__main__":
    unittest.main()

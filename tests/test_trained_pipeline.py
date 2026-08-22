import unittest

import numpy as np

from pallet_audit.trained_pipeline import apparent_alignment_angle_deg, belongs_to_pallet


class TrainedPipelineTests(unittest.TestCase):
    def test_apparent_box_alignment_relative_to_pallet(self):
        angle = np.deg2rad(12)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ])
        rectangle = np.array([[-2, -1], [2, -1], [2, 1], [-2, 1]]) @ rotation.T
        self.assertAlmostEqual(
            apparent_alignment_angle_deg(rectangle, np.array([1.0, 0.0])),
            12.0,
            places=6,
        )

    def test_load_above_pallet_is_associated(self):
        pallet = np.array([100, 180, 220, 220])
        load = np.array([110, 80, 210, 185])
        distant = np.array([260, 80, 310, 185])
        self.assertTrue(belongs_to_pallet(pallet, load))
        self.assertFalse(belongs_to_pallet(pallet, distant))


if __name__ == "__main__":
    unittest.main()

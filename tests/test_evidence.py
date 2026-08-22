import unittest

import numpy as np

from pallet_audit.evidence import PinholeCamera, box_alignment_angle_deg, footprint_metrics


class EvidenceTests(unittest.TestCase):
    def test_footprint_overhang_and_centroid(self):
        pallet = np.array([[0, 0], [1.2, 0], [1.2, 1.0], [0, 1.0]])
        load = np.array([[-.04, .2], [1.0, .2], [1.0, .8], [-.04, .8]])
        overhang, centroid = footprint_metrics(pallet, load)
        self.assertAlmostEqual(overhang, .04)
        self.assertAlmostEqual(centroid, .12)

    def test_box_angle_relative_to_pallet_axis(self):
        angle = np.deg2rad(12)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rectangle = np.array([[-1, -.25], [1, -.25], [1, .25], [-1, .25]]) @ rotation.T
        self.assertAlmostEqual(box_alignment_angle_deg(rectangle, 0), 12, places=6)

    def test_vertical_height_from_known_base(self):
        projection = np.array([[100, 0, 0, 0], [0, 100, -100, 200], [0, 0, 0, 1]], dtype=float)
        camera = PinholeCamera(projection)
        top = camera.project(np.array([[0, 0, 1.5]]))[0]
        height, residual = camera.vertical_height(np.array([0, 0]), top)
        self.assertAlmostEqual(height, 1.5)
        self.assertAlmostEqual(residual, 0)


if __name__ == "__main__":
    unittest.main()


import unittest

import numpy as np

from pallet_audit.compliance import LoadEvidence, assess_sop, overall_verdict
from pallet_audit.geometry import CameraModel, pose_from_keypoints
from pallet_audit.models import Verdict


def good_pose():
    return pose_from_keypoints(np.array([[0, 0], [1.2, 0], [1.2, 1], [0, 1]]), CameraModel(np.eye(3)), pixel_sigma=.001)


class ComplianceTests(unittest.TestCase):
    def test_high_confidence_failure_drives_overall_failure(self):
        evidence = LoadEvidence(overhang_m=.05, load_height_m=1.5, box_angles_deg=(2,), wrap_probability=.9,
                                damage_probability=.05, centroid_offset_m=.03, pallet_damage_probability=.05)
        checks = assess_sop(evidence, good_pose())
        verdict, reasons = overall_verdict(checks, good_pose())
        self.assertEqual(checks[0].verdict, Verdict.FAIL)
        self.assertEqual(verdict, Verdict.FAIL)
        self.assertIn("SOP-1", reasons[0])

    def test_missing_evidence_never_silently_passes(self):
        checks = assess_sop(LoadEvidence(), good_pose())
        self.assertTrue(all(c.verdict != Verdict.PASS for c in checks))


if __name__ == "__main__":
    unittest.main()

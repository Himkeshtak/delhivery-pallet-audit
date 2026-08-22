import unittest

from pallet_audit.model_stack import runtime_schedule


class ModelStackTests(unittest.TestCase):
    def test_temporal_cadence(self):
        self.assertEqual(runtime_schedule(0), ("pallet_pose", "load_instances", "visible_damage"))
        self.assertEqual(runtime_schedule(1), ("pallet_pose",))
        self.assertEqual(runtime_schedule(3), ("pallet_pose", "load_instances"))
        self.assertEqual(runtime_schedule(10), ("pallet_pose", "visible_damage"))

    def test_negative_frame_rejected(self):
        with self.assertRaises(ValueError):
            runtime_schedule(-1)


if __name__ == "__main__":
    unittest.main()


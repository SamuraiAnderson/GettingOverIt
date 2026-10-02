"""Check that pose analysis distinguishes rigid transforms from articulation."""
import copy
import unittest

import numpy as np

from analyze_start_pose import difference


def pose():
    state = np.zeros((7, 6))
    state[:, :2] = [(0, 0), (0, .35), (0, .35), (.2, .4), (1, .2), (2, -.1), (2, -.1)]
    return {"bodyNames": ["Player", "Hub", "Slider", "Handle", "PoleMiddle", "Tip", "Cursor"],
            "bodies": state.flatten().tolist(), "motors": [0, 10000, 0, 10000, 0, .2]}


class StartPoseChecks(unittest.TestCase):
    def test_whole_body_translation_rotation_removed(self):
        a = pose()
        b = copy.deepcopy(a)
        state = np.array(b["bodies"]).reshape(7, 6)
        angle = np.deg2rad(35)
        rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        state[:, :2] = state[:, :2] @ rotate.T + [4, -2]
        state[:, 2] += 35
        b["bodies"] = state.flatten().tolist()
        result = difference(a, b)
        self.assertGreater(result["max_world_position_distance_m"], 1)
        self.assertLess(result["max_position_distance_after_aligning_player_m"], 1e-12)

    def test_articulation_survives_alignment(self):
        a, b = pose(), pose()
        b["bodies"][30] += .1  # tip x
        b["motors"][5] += .1
        result = difference(a, b)
        self.assertAlmostEqual(result["tip_distance_after_aligning_player_m"], .1)
        self.assertAlmostEqual(result["slider_translation_difference_m"], .1)

    def test_angle_wrap_does_not_wrap_angular_velocity(self):
        a, b = pose(), pose()
        a["bodies"][2], b["bodies"][2] = 359.99, -.01
        b["bodies"][5] = 360
        result = difference(a, b)
        self.assertAlmostEqual(result["bodies"][0]["angle_delta_deg"], 0)
        self.assertEqual(result["bodies"][0]["angular_velocity_delta_deg_s"], 360)


if __name__ == "__main__":
    unittest.main()

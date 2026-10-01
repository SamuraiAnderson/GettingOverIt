"""Helpers for the offline policy-window renderer."""
import unittest

import numpy as np

from src.tests.evaluation.present.render_policy_window import guidance_for_state, relative_xy


class PolicyWindowHelperTests(unittest.TestCase):
    def test_player_is_the_origin_of_the_window(self):
        player = np.array([12.0, -3.5])
        np.testing.assert_allclose(relative_xy(player, player), [0.0, 0.0])
        np.testing.assert_allclose(relative_xy([12.2, -3.9], player), [0.2, -0.4])

    def test_guidance_index_follows_the_state_not_the_previous_action(self):
        guidance = np.array([0, 0, 1, -1], dtype=np.int8)
        self.assertEqual(guidance_for_state(guidance, 0), 0)
        self.assertEqual(guidance_for_state(guidance, 2), 1)
        self.assertEqual(guidance_for_state(guidance, 3), -1)
        self.assertEqual(guidance_for_state(guidance, 99), -1)

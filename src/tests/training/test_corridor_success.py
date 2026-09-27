"""到达门槛 + 容差保持；原严格 dwell 成功仍单独保留。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.config import TrainConfig
from training.dataset import Trajectory
from training.local_evaluation import score_corridor_episode
from training.reference_corridor import CorridorSegment, ReferenceCorridor


def _config():
    return TrainConfig(d_model=16, nhead=2, num_layers=1, context_len=3,
                       num_agents=2, random_deploy=False, deploy_exclusion_check=False,
                       contact_heads_enabled=False)


def _route(*pairs):
    segments = []
    for a, b in pairs:
        points = np.array([a, b], dtype=np.float64)
        segments.append(CorridorSegment(points, np.ones(1)))
    return ReferenceCorridor(segments, _config())


def _walk(xs, ys=None):
    xs = np.asarray(xs, dtype=np.float32)
    states = np.zeros((len(xs), 33), dtype=np.float32)
    states[:, 0] = xs
    states[:, 1] = 5.0 if ys is None else ys
    return Trajectory(states, np.zeros((len(xs) - 1, 2)), contact=np.zeros((len(xs), 5)))


def _score(xs, goal=5.0, dwell=3, hold_steps=3, ys=None, **kwargs):
    return score_corridor_episode(_walk(xs, ys), _route(([0.0, 5.0], [20.0, 5.0])),
                                 goal, dwell, hold_steps=hold_steps, **kwargs)[0]


class CorridorSuccessTests(unittest.TestCase):
    def test_arrival_is_first_threshold_crossing(self):
        metrics = _score([0, 1, 2, 3, 4, 5, 5, 5])
        self.assertEqual(metrics["arrival_steps"], 5)
        self.assertTrue(metrics["success"])
        self.assertTrue(metrics["hold_success"])
        self.assertAlmostEqual(metrics["terminal_progress_m"], 5.0)

    def test_strict_success_still_needs_consecutive_dwell(self):
        metrics = _score([0, 1, 2, 3, 4, 5, 4.8, 5], dwell=3)
        self.assertEqual(metrics["arrival_steps"], 5)
        self.assertFalse(metrics["success"])

    def test_hold_allows_one_outside_frame_in_twenty(self):
        xs = [0, 1, 2, 3, 4] + [5] * 20
        ys = [5] * 25
        ys[10] = 7.0  # 1 of 20 hold frames outside (19/20 = 0.95)
        metrics = score_corridor_episode(
            _walk(xs, ys), _route(([0.0, 5.0], [20.0, 5.0])), 5, 1, hold_steps=20)[0]
        self.assertEqual(metrics["arrival_steps"], 5)
        self.assertTrue(metrics["hold_success"])

    def test_hold_rejects_too_much_time_outside(self):
        traj = _walk([0, 1, 2, 3, 4, 5, 5, 5], ys=[5, 5, 5, 5, 5, 5, 7.0, 5])
        metrics = score_corridor_episode(
            traj, _route(([0.0, 5.0], [20.0, 5.0])), 5, 1, hold_steps=3)[0]
        self.assertEqual(metrics["arrival_steps"], 5)
        self.assertFalse(metrics["hold_success"])

    def test_hold_rejects_drop_below_floor(self):
        metrics = _score([0, 1, 2, 3, 4, 5, 4.6, 5], hold_steps=3)
        self.assertEqual(metrics["arrival_steps"], 5)
        self.assertFalse(metrics["hold_success"])

    def test_hold_requires_full_window_and_return_to_goal(self):
        metrics = _score([0, 1, 2, 3, 4, 5, 5], hold_steps=3)
        self.assertTrue(metrics["hold_incomplete"])
        self.assertFalse(metrics["hold_success"])
        metrics = _score([0, 1, 2, 3, 4, 5, 5, 4.75], hold_steps=3)
        self.assertFalse(metrics["hold_incomplete"])
        self.assertFalse(metrics["hold_success"])

    def test_later_verified_segment_counts(self):
        route = _route(([0.0, 5.0], [3.0, 5.0]), ([3.0, 5.0], [13.0, 5.0]))
        xs = [0, 1, 2, 3, 3.2, 4, 5, 6, 6, 6]
        metrics = score_corridor_episode(_walk(xs), route, 5, 1, hold_steps=3)[0]
        self.assertIsNotNone(metrics["arrival_steps"])
        self.assertTrue(metrics["continued_on_later_segment"])
        self.assertGreaterEqual(metrics["terminal_progress_m"], 5.0)
        self.assertFalse(metrics["success"])

    def test_gap_is_unconfirmed_not_a_fall(self):
        route = _route(([0.0, 5.0], [6.0, 5.0]), ([10.0, 5.0], [20.0, 5.0]))
        xs = [0, 1, 2, 3, 4, 5, 6, 6, 6]
        metrics = score_corridor_episode(_walk(xs), route, 5, 1, hold_steps=3)[0]
        self.assertEqual(metrics["arrival_steps"], 5)
        self.assertTrue(metrics["reference_unconfirmed"])
        self.assertFalse(metrics["hold_success"])
        self.assertFalse(metrics["fell"])

    def test_hold_rejects_water_not_a_height_dip(self):
        xs = [0, 1, 2, 3, 4, 5, 5, 5]
        metrics = _score(xs, hold_steps=3, ys=[5, 5, 5, 5, 5, 5, 3.5, 5])
        self.assertTrue(metrics["hold_success"])
        self.assertFalse(metrics["watered"])
        wet = _score(xs, hold_steps=3, ys=[5, 5, 5, 5, 5, 5, -11.0, 5])
        self.assertFalse(wet["hold_success"])
        self.assertTrue(wet["watered"])

    def test_max_retreat_after_arrival(self):
        metrics = _score([0, 1, 2, 3, 4, 5, 6.5, 5.2], hold_steps=3)
        self.assertAlmostEqual(metrics["max_retreat_after_arrival_m"], 1.3, places=5)
        self.assertIsNone(_score([0, 1, 2, 3], hold_steps=3)["max_retreat_after_arrival_m"])

    def test_summarize_leads_with_median_and_separates_incomplete(self):
        from training.local_evaluation import format_eval_line, summarize
        short = _score([0, 1, 2, 3, 4, 5, 5], hold_steps=3)
        held = _score([0, 1, 2, 3, 4, 5, 5, 5], hold_steps=3)
        summary = summarize([{**short, "mode": "deterministic"}, {**held, "mode": "deterministic"}])
        det = summary["deterministic"]
        self.assertEqual(det["arrival_count"], 2)
        self.assertEqual(det["hold_success_count"], 1)
        self.assertEqual(det["hold_incomplete_count"], 1)
        self.assertGreater(det["credited_progress_median_m"], 0)
        line = format_eval_line(det)
        self.assertIn("credited 中位", line)
        self.assertIn("未观察完 1", line)
        self.assertNotIn("严格 1 秒", line)


if __name__ == "__main__":
    unittest.main()

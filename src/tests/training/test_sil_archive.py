"""Endpoint (t, D) SIL archive: dominance, ingest, sampling, gate. No game."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset
from training.sil_archive import (
    bucket_compress,
    credited_distance,
    dedup_td,
    dominates,
    equal_traj_window_index,
    gate_eligible_indices,
    ingest_td_pareto,
    pareto_front,
    sparse_cap,
    td_point,
)


def _traj(t: int, distance: float, *, x: float = 0.0, y: float = 0.0,
          iteration: int = 0, score: float = 1.0, extra_steps: int = 0) -> Trajectory:
    length = t + extra_steps
    progress = np.zeros(length, dtype=np.float32)
    progress[t - 1] = np.float32(distance)
    states = np.zeros((length + 1, 33), dtype=np.float32)
    states[:, 0] = x
    states[:, 1] = y
    return Trajectory(
        raw_states=states,
        actions=np.zeros((length, 2), dtype=np.float32),
        score=score,
        iteration=iteration,
        corridor_progress=progress,
    )


class SILArchiveTests(unittest.TestCase):
    def test_dominance_includes_one_equal_dimension(self):
        self.assertTrue(dominates((100, 6.0), (100, 4.0)))
        self.assertTrue(dominates((90, 6.0), (100, 6.0)))
        self.assertTrue(dominates((100, 6.0), (377, 6.0)))
        self.assertFalse(dominates((100, 4.0), (377, 6.0)))
        self.assertFalse(dominates((377, 6.0), (100, 4.0)))
        self.assertFalse(dominates((100, 4.0), (100, 4.0)))

    def test_front_keeps_fast_short_and_slow_far(self):
        a, b = _traj(100, 4.0, iteration=1), _traj(377, 6.0, iteration=1)
        kept = pareto_front([a, b])
        self.assertEqual({td_point(t) for t in kept}, {(100, 4.0), (377, 6.0)})
        winner = _traj(100, 6.0, iteration=2)
        kept = pareto_front([a, b, winner])
        self.assertEqual([td_point(t) for t in kept], [(100, 6.0)])

    def test_exact_points_are_deduped(self):
        old = _traj(100, 4.0, iteration=1, score=1.0)
        new = _traj(100, 4.0, iteration=3, score=1.0)
        kept = dedup_td([old, new])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].iteration, 3)

    def test_reject_zero_progress_and_bad_length(self):
        zero = _traj(10, 0.0)
        self.assertIsNone(td_point(zero))
        broken = _traj(10, 2.0)
        broken.corridor_progress = broken.corridor_progress[:-1]
        self.assertIsNone(td_point(broken))

    def test_ingest_skips_buckets_when_pool_is_small(self):
        cfg = TrainConfig()
        cfg.sil_archive_mode = "td_pareto"
        cfg.sil_td_bucket_enable_at = 16
        cfg.sil_td_behavior_slots = 0
        incoming = [_traj(100, 4.0), _traj(377, 6.0), _traj(200, 3.0, x=20.0)]
        kept, stats = ingest_td_pareto([], incoming, [1, 1, 1], 1, cfg)
        self.assertEqual(stats["sil_archive_bucketed"], 0.0)
        self.assertEqual({td_point(t) for t in kept}, {(100, 4.0), (377, 6.0)})

    def test_bucket_keeps_faster_near_duplicate_and_farthest(self):
        close = [_traj(100, 4.01), _traj(103, 4.04), _traj(400, 8.0)]
        kept = bucket_compress(close, 0.10)
        points = [td_point(t) for t in kept]
        self.assertEqual([p[0] for p in points], [100, 400])
        self.assertAlmostEqual(points[0][1], 4.01, places=2)
        self.assertNotIn(103, [p[0] for p in points])

    def test_sparse_cap_protects_fastest_and_farthest(self):
        trajs = [_traj(20 + 20 * i, 1.0 + 0.2 * i, x=float(i)) for i in range(8)]
        kept = sparse_cap(trajs, 3)
        points = {td_point(t) for t in kept}
        self.assertIn(td_point(_fastest_like(trajs)), points)
        self.assertIn(td_point(max(trajs, key=lambda t: td_point(t)[1])), points)
        self.assertEqual(len(kept), 3)

    def test_behavior_slots_fill_small_front(self):
        cfg = TrainConfig()
        cfg.sil_td_behavior_slots = 2
        cfg.sil_td_bucket_enable_at = 16
        frontish = [_traj(100, 4.0, x=0.0), _traj(377, 6.0, x=1.0)]
        extra = _traj(200, 3.0, x=20.0, score=9.0)
        kept, stats = ingest_td_pareto([], frontish + [extra], [1, 1, 9], 1, cfg)
        self.assertEqual(stats["sil_archive_pareto"], 2.0)
        self.assertEqual(stats["sil_archive_behavior"], 1.0)
        roles = {t.archive_role for t in kept}
        self.assertEqual(roles, {"pareto", "behavior"})

    def test_equal_window_budget_does_not_drop_raw_steps(self):
        short = _traj(100, 4.0)
        long = _traj(377, 6.0)
        index = equal_traj_window_index([short, long], 0)
        short_n = sum(1 for ti, _ in index if ti == 0)
        long_n = sum(1 for ti, _ in index if ti == 1)
        self.assertEqual(short_n, long_n)
        self.assertEqual(short_n, 100)
        self.assertEqual(len(long.raw_states), 378)

    def test_gate_is_per_trajectory(self):
        pool = [_traj(100, 4.0), _traj(377, 6.0)]
        self.assertEqual(gate_eligible_indices(pool, 5.0, 0.0), [1])
        self.assertEqual(gate_eligible_indices(pool, 2.0, 0.0), [0, 1])
        self.assertEqual(gate_eligible_indices(pool, 6.5, 0.0), [])
        self.assertGreater(credited_distance(pool[1]), 5.0)

    def test_dataset_dispatch_and_default_mode(self):
        cfg = TrainConfig()
        self.assertEqual(cfg.sil_archive_mode, "map_elites")
        cfg.sil_archive_mode = "td_pareto"
        cfg.sil_td_behavior_slots = 0
        ds = TrajectoryDataset(cfg, np.ones((8, 8), dtype=bool), np.zeros((8, 8), np.float32))
        ds.ingest_sil_archive(
            [_traj(100, 4.0), _traj(377, 6.0)], [1.0, 1.0], iteration=1)
        self.assertEqual(len(ds.trajectories), 2)
        counts = {}
        for ti, _tau in ds._index:
            counts[ti] = counts.get(ti, 0) + 1
        self.assertEqual(counts[0], counts[1])


def _fastest_like(trajs):
    return min(trajs, key=lambda t: (td_point(t)[0], -td_point(t)[1]))


if __name__ == "__main__":
    unittest.main()

"""Endpoint (t, D) SIL archive: dominance, ingest, sampling, gate. No game."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.dataset import SampledWindowView, Trajectory, TrajectoryDataset
from training.main_ppo import configure_sil_sampling
from training.ppo_trainer import PPOTrainer
from training.reference_corridor import PROGRESS_MODE_V2
from training.sil_archive import (
    allocate_quotas,
    bucket_compress,
    build_td_sampling_plan,
    coverage_area,
    credited_distance,
    dedup_td,
    dominates,
    equal_traj_window_index,
    full_candidate_index,
    recover_v2_termination,
    gate_eligible_indices,
    ingest_td_pareto,
    pareto_front,
    sparse_cap,
    stratified_taus,
    td_point,
    uniform_stratified_weights,
)
import torch


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


def _traj_v2(t: int, distance: float, *, extra_steps: int = 0,
             contact: np.ndarray | None = None) -> Trajectory:
    cap = t - 1
    length = t + extra_steps
    states = np.zeros((length + 1, 33), dtype=np.float32)
    actions = np.zeros((length, 2), dtype=np.float32)
    events = np.zeros(length, dtype=bool)
    events[cap] = True
    confirmed = np.zeros(length + 1, dtype=np.float64)
    confirmed[cap + 1:] = distance
    progress = np.zeros(length, dtype=np.float32)
    progress[cap] = np.float32(distance)
    rewards = np.zeros(length, dtype=np.float64)
    rewards[cap] = distance
    return Trajectory(
        raw_states=states,
        actions=actions,
        corridor_progress=progress,
        corridor_credit_events=events,
        corridor_confirmed_progress=confirmed,
        step_rewards=rewards,
        terminated=np.zeros(length, dtype=bool),
        termination_known=True,
        progress_mode=PROGRESS_MODE_V2,
        contact=contact,
    )


class CoverageSamplingTests(unittest.TestCase):
    def test_v2_missing_termination_is_filled_without_rewriting_known_rows(self):
        missing = _traj_v2(6, 3.0)
        missing.terminated = None
        missing.termination_known = False
        self.assertTrue(recover_v2_termination(missing))
        self.assertTrue(missing.termination_known)
        self.assertEqual(missing.terminated.shape, (6,))
        self.assertFalse(bool(missing.terminated.any()))
        self.assertFalse(recover_v2_termination(missing))

        known = _traj_v2(4, 2.0)
        known.terminated = np.zeros(4, dtype=bool)
        known.terminated[-1] = True
        known.termination_known = True
        self.assertFalse(recover_v2_termination(known))
        self.assertTrue(bool(known.terminated[-1]))

        legacy = _traj(5, 1.0)
        legacy.termination_known = False
        legacy.terminated = None
        self.assertFalse(recover_v2_termination(legacy))
        self.assertFalse(legacy.termination_known)
        self.assertIsNone(legacy.terminated)

    def test_area_is_conserved_and_empty_pool_skips(self):
        points = [(10, 4.0), (30, 6.0), (20, 5.0), None]
        weights, envelope = coverage_area(points, 100)
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue(np.all(weights >= 0.0))
        self.assertAlmostEqual(float(weights.sum()), envelope, places=8)
        self.assertAlmostEqual(float(weights[np.isfinite(weights)].sum() / envelope), 1.0, places=8)
        self.assertEqual(weights[3], 0.0)
        skipped = build_td_sampling_plan(
            [], mode="coverage_share", horizon=100, k=4, window_budget=32,
            seed=1, iteration=1,
        )
        self.assertTrue(skipped["skip"])
        self.assertEqual(skipped["index"], [])

    def test_dominated_trajectory_keeps_less_positive_area(self):
        points = [(10, 6.0), (20, 4.0)]
        weights, _envelope = coverage_area(points, 100)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], 0.0)

    def test_horizon_boundary_and_distance_scale(self):
        at_horizon, _ = coverage_area([(10, 1.0)], 10)
        past_horizon, _ = coverage_area([(11, 1.0)], 10)
        self.assertGreater(at_horizon[0], 0.0)
        self.assertEqual(past_horizon[0], 0.0)
        base, _ = coverage_area([(10, 1.0), (20, 2.0)], 100)
        scaled, _ = coverage_area([(10, 100.0), (20, 200.0)], 100)
        np.testing.assert_allclose(base / base.sum(), scaled / scaled.sum(), atol=1e-12)
        short_h, _ = coverage_area([(80, 5.0), (20, 1.0)], 50)
        long_h, _ = coverage_area([(80, 5.0), (20, 1.0)], 100)
        self.assertEqual(short_h[0], 0.0)
        self.assertGreater(long_h[0], 0.0)
        self.assertGreater(long_h.sum(), short_h.sum())

    def test_quota_matches_budget_and_long_run_frequency(self):
        probs = np.array([0.1, 0.2, 0.7])
        rng = np.random.default_rng(7)
        totals = np.zeros(3, dtype=np.float64)
        rounds = 4000
        budget = 50
        floors = np.floor(budget * probs)
        ceilings = np.ceil(budget * probs)
        for _ in range(rounds):
            quotas = allocate_quotas(probs, budget, rng)
            self.assertEqual(int(quotas.sum()), budget)
            self.assertTrue(np.all(quotas >= floors))
            self.assertTrue(np.all(quotas <= ceilings))
            totals += quotas
        frequency = totals / (rounds * budget)
        sigma = np.sqrt(probs * (1.0 - probs) / rounds)
        self.assertTrue(np.all(np.abs(frequency - probs) < 3.0 * sigma + 1e-6))

    def test_short_trajectory_does_not_collapse_long_sampling(self):
        pool = [_traj_v2(1, 0.001), _traj_v2(476, 20.0), _traj_v2(800, 24.0)]
        legacy = equal_traj_window_index(pool, 0)
        self.assertTrue(all(tau == 0 for _ti, tau in legacy))
        full = full_candidate_index(pool, 1200)
        self.assertEqual(sum(1 for ti, _tau in full if ti == 2), 800)
        self.assertGreater(max(tau for ti, tau in full if ti == 2), 400)
        for mode in ("balanced_stratified", "coverage_share"):
            plan = build_td_sampling_plan(
                pool, mode=mode, horizon=1200, k=8, window_budget=24,
                seed=11, iteration=3,
            )
            long_taus = [tau for ti, tau in plan["index"] if ti == 2]
            self.assertGreater(max(long_taus), 400)
            self.assertTrue(any(tau > 0 for tau in long_taus))
            self.assertTrue(all(0 <= tau <= 799 for tau in long_taus))

    def test_single_draw_is_not_fixed_at_the_origin(self):
        draws = [stratified_taus(range(801), 1, np.random.default_rng(seed))[0]
                 for seed in range(40)]
        self.assertGreater(len(set(draws)), 1)
        self.assertTrue(any(tau != 0 for tau in draws))

    def test_sampled_view_preserves_window_identity(self):
        cfg = TrainConfig(context_len=4, patch_size=8, sparse_geometry=False)
        traj = _traj_v2(6, 3.0)
        ds = TrajectoryDataset(cfg, np.ones((8, 8), dtype=bool), np.zeros((8, 8), np.float32))
        ds.set_trajectories([traj])
        view = SampledWindowView(ds, [(0, 0), (0, 5)])
        for idx, (_ti, tau) in enumerate(view._index):
            got = view[idx]
            expected = ds.encode_window(traj, tau)
            self.assertEqual(len(got), len(expected))
            self.assertTrue(torch.equal(got[4], expected[4]))
        self.assertTrue(all(0 <= tau <= 5 for _ti, tau in view._index))

    def test_phase_filter_removes_empty_trajectory_before_area(self):
        contact = np.zeros((2, 5), dtype=np.float32)
        contact[:, 0] = 1.0
        kept = _traj_v2(1, 1.0, contact=contact)
        airborne = _traj_v2(40, 5.0, contact=np.zeros((41, 5), dtype=np.float32))
        plan = build_td_sampling_plan(
            [airborne, kept], mode="coverage_share", horizon=100, k=4,
            window_budget=16, seed=5, iteration=2, phase_enabled=True,
            phase_window=1, airborne_sample=0.0,
        )
        self.assertFalse(plan["skip"])
        self.assertEqual(plan["record"]["trajectories"][0]["reason"], "empty_after_phase")
        self.assertAlmostEqual(plan["record"]["trajectories"][1]["prob"], 1.0)
        self.assertTrue(all(ti == 1 for ti, _tau in plan["index"]))
        self.assertEqual(plan["metrics"]["sil_sampling_n_empty"], 1.0)

    def test_zero_budget_skips_without_touching_global_rng(self):
        pool = [_traj_v2(5, 1.0), _traj_v2(8, 2.0)]
        np.random.seed(123)
        before = np.random.get_state()
        plan = build_td_sampling_plan(
            pool, mode="balanced_stratified", horizon=20, k=4, window_budget=0,
            seed=9, iteration=1,
        )
        self.assertTrue(plan["skip"])
        self.assertEqual(plan["skip_reason"], "zero_update_budget")
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])

    def test_same_seed_reproduces_plan_and_epoch_changes_it(self):
        pool = [_traj_v2(30, 2.0), _traj_v2(50, 4.0)]
        kwargs = dict(mode="coverage_share", horizon=80, k=6, window_budget=12, seed=19, iteration=4)
        first = build_td_sampling_plan(pool, epoch=0, **kwargs)
        second = build_td_sampling_plan(pool, epoch=0, **kwargs)
        other_epoch = build_td_sampling_plan(pool, epoch=1, **kwargs)
        self.assertEqual(first["index"], second["index"])
        self.assertNotEqual(first["index"], other_epoch["index"])

    def test_uniform_weights_ignore_dominance(self):
        points = [(10, 6.0), (20, 4.0)]
        weights = uniform_stratified_weights(points, 100)
        np.testing.assert_allclose(weights, [1.0, 1.0])

    def test_resume_keeps_legacy_until_a_new_run_dir_asks(self):
        cfg = TrainConfig()
        old = {"config": {"sil_archive_mode": "map_elites", "sil_td_windows_per_traj": 0}}
        args = argparse_namespace(sil_td_sampling_mode=None, run_dir=None, seed=47)
        configure_sil_sampling(cfg, args, old)
        self.assertEqual(cfg.sil_td_sampling_mode, "legacy_min_length")

        fresh = TrainConfig(sil_archive_mode="td_pareto", steps_per_rollout=1200,
                            reference_progress_mode=PROGRESS_MODE_V2)
        switched = argparse_namespace(
            sil_td_sampling_mode="coverage_share", run_dir="runs/new", seed=47,
            sil_td_sampling_horizon_steps=None, sil_td_windows_per_traj=None,
            sil_td_sampling_seed=None,
        )
        configure_sil_sampling(fresh, switched, old)
        self.assertEqual(fresh.sil_td_sampling_mode, "coverage_share")
        self.assertEqual(fresh.sil_td_sampling_horizon_steps, 1200)
        self.assertEqual(fresh.sil_td_windows_per_traj, 32)
        self.assertEqual(fresh.sil_td_sampling_seed, 47)

        saved = {"config": {
            "sil_td_sampling_mode": "coverage_share",
            "sil_td_sampling_horizon_steps": 1200,
            "sil_td_windows_per_traj": 32,
            "sil_td_sampling_seed": 47,
            "sil_archive_mode": "td_pareto",
            "reference_progress_mode": PROGRESS_MODE_V2,
            "steps_per_rollout": 1200,
        }}
        restored = TrainConfig(**saved["config"])
        conflict = argparse_namespace(
            sil_td_sampling_mode="balanced_stratified", run_dir="runs/other", seed=47,
            sil_td_sampling_horizon_steps=None, sil_td_windows_per_traj=None,
            sil_td_sampling_seed=None,
        )
        with self.assertRaises(ValueError):
            configure_sil_sampling(restored, conflict, saved)
        inplace = TrainConfig(sil_archive_mode="td_pareto",
                              reference_progress_mode=PROGRESS_MODE_V2)
        with self.assertRaises(ValueError):
            configure_sil_sampling(inplace, argparse_namespace(
                sil_td_sampling_mode="coverage_share", run_dir=None, seed=47,
            ), old)

    def test_sampling_update_covers_later_actions_without_global_rng(self):
        cfg = TrainConfig(
            d_model=16, nhead=2, num_layers=1, context_len=4, patch_size=8,
            num_agents=1, random_deploy=False, deploy_exclusion_check=False,
            contact_heads_enabled=False, sparse_geometry=False,
            sil_archive_mode="td_pareto", sil_td_sampling_mode="coverage_share",
            sil_td_sampling_horizon_steps=1200, sil_td_windows_per_traj=8,
            sil_td_sampling_seed=47, sil_batch_size=8, sil_max_batches=2,
            sil_num_workers=0, sil_epochs=1, sil_advantage_weighting_enabled=False,
            sil_phase_weighting_enabled=False, reference_progress_mode=PROGRESS_MODE_V2,
        )
        pool = [_traj_v2(1, 0.001), _traj_v2(476, 20.0), _traj_v2(800, 24.0)]
        ds = TrajectoryDataset(cfg, np.ones((8, 8), dtype=bool), np.zeros((8, 8), np.float32))
        ds.set_trajectories(pool)
        self.assertGreater(len(ds), 3)
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")
        np.random.seed(5)
        torch.manual_seed(5)
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state()
        metrics = trainer.sil_update(model, ds, budget_updates=4, iteration=7)
        np.testing.assert_array_equal(numpy_before[1], np.random.get_state()[1])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        self.assertGreater(metrics["sil_updates"], 0)
        self.assertLessEqual(metrics["sil_updates"], 2)
        self.assertLessEqual(metrics["sil_sampling_actual_windows"], 16)
        long = next(row for row in trainer.last_sil_sampling["trajectories"] if row["t"] == 800)
        self.assertGreater(long["max_tau"], 400)
        self.assertGreater(long["n_last_third"], 0)

    def test_v2_return_does_not_follow_the_sampled_endpoint(self):
        cfg = TrainConfig(
            d_model=16, nhead=2, num_layers=1, context_len=4, patch_size=8,
            contact_heads_enabled=False, sparse_geometry=False, gamma=0.99,
        )
        traj = _traj_v2(6, 3.0)
        ds = TrajectoryDataset(cfg, np.ones((8, 8), dtype=bool), np.zeros((8, 8), np.float32))
        ds.set_trajectories([traj])
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")
        early = SampledWindowView(ds, [(0, 0), (0, 1)])
        late = SampledWindowView(ds, [(0, 4), (0, 5)])
        _values_a, returns_a, _frac_a = trainer._sil_pool_values(model, early)
        _values_b, returns_b, _frac_b = trainer._sil_pool_values(model, late)
        self.assertAlmostEqual(float(returns_a[0]), float(trainer._sil_pool_values(model, early)[1][0]))
        full = trainer._sil_pool_values(model, SampledWindowView(ds, [(0, tau) for tau in range(6)]))[1]
        self.assertAlmostEqual(float(returns_a[0]), float(full[0]), places=6)
        self.assertAlmostEqual(float(returns_b[-1]), float(full[5]), places=6)


def argparse_namespace(**kwargs):
    from argparse import Namespace
    defaults = {
        "sil_td_sampling_mode": None,
        "sil_td_sampling_horizon_steps": None,
        "sil_td_windows_per_traj": None,
        "sil_td_sampling_seed": None,
        "run_dir": None,
        "seed": 0,
    }
    defaults.update(kwargs)
    return Namespace(**defaults)


if __name__ == "__main__":
    unittest.main()

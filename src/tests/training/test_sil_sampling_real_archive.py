"""Numerical benchmark for the iter-123 elite pool. No game and no training."""
from __future__ import annotations

import math
import sys
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset
from training.model import expand_state_dict_for_dynamics_dim
from training.ppo_trainer import PPOTrainer
from training.reference_corridor import PROGRESS_MODE_V2
from training.run_logging import load_elites
from training.sil_archive import (
    build_td_sampling_plan,
    coverage_area,
    pareto_front,
    recover_v2_termination,
    td_point,
)

ARCHIVE = ROOT / "runs/confirmed_endpoint_v2_formal_20260928_S_s47/elites/elites_iter_0123.npz"
CHECKPOINT = ROOT / "runs/confirmed_endpoint_v2_formal_20260928_S_s47/checkpoints/ppo_iter_0123.pt"
# Doc §2.4 percentages, as probabilities. Absolute error must stay within 1e-6.
DOC_PROBS = {
    (800, 24.356533): 0.164344,
    (476, 20.174842): 0.215006,
    (191, 13.892033): 0.108402,
    (194, 9.894296): 0.041392,
    (1026, 17.119965): 0.021355,
    (1, 0.000778): 0.00000455,
}
DOC_AREA = 21412.80462645653


def _trajectory(row: dict) -> Trajectory:
    return Trajectory(
        raw_states=row["raw_states"],
        actions=row["actions"],
        score=row["score"],
        iteration=row["iteration"],
        corridor_progress=row["corridor_progress"],
        corridor_credit_events=row["corridor_credit_events"],
        corridor_confirmed_progress=row["corridor_confirmed_progress"],
        progress_mode=row["progress_mode"] or PROGRESS_MODE_V2,
        trajectory_valid=True,
    )


class RealArchiveSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trajs = [_trajectory(row) for row in load_elites(ARCHIVE)]
        cls.points = [td_point(traj) for traj in cls.trajs]
        cls.weights, cls.area = coverage_area(cls.points, 1200)
        cls.probs = cls.weights / cls.weights.sum()

    def test_pool_shape_and_doc_probabilities(self):
        self.assertEqual(len(self.trajs), 19)
        self.assertTrue(all(point is not None for point in self.points))
        self.assertEqual(len(pareto_front(self.trajs)), 17)
        self.assertAlmostEqual(self.area, DOC_AREA, delta=1e-8)
        self.assertAlmostEqual(float(self.weights.sum()), self.area, places=8)
        self.assertAlmostEqual(float(self.probs.sum()), 1.0, places=12)
        for (t_cap, distance), expected in DOC_PROBS.items():
            matches = [idx for idx, point in enumerate(self.points)
                       if point[0] == t_cap and abs(point[1] - distance) < 1e-4]
            self.assertEqual(len(matches), 1, msg=f"missing {(t_cap, distance)}")
            self.assertAlmostEqual(float(self.probs[matches[0]]), expected, delta=1e-6)

    def test_long_trajectories_receive_later_windows(self):
        plan = build_td_sampling_plan(
            self.trajs, mode="coverage_share", horizon=1200, k=32,
            window_budget=19 * 32, seed=47, iteration=123, phase_enabled=False,
        )
        self.assertFalse(plan["skip"])
        self.assertEqual(plan["metrics"]["sil_sampling_actual_windows"], 19 * 32)
        by_t = {row["t"]: row for row in plan["record"]["trajectories"]}
        for t_cap in (476, 800):
            row = by_t[t_cap]
            self.assertGreater(row["max_tau"], 0)
            self.assertGreater(row["n_first_third"], 0)
            self.assertGreater(row["n_middle_third"], 0)
            self.assertGreater(row["n_last_third"], 0)
        self.assertTrue(any(tau > 0 for _ti, tau in plan["index"]))

    def test_recovered_pool_runs_advantage_weighted_sil(self):
        rows = load_elites(ARCHIVE)
        trajs = []
        for row in rows:
            traj = _trajectory(row)
            traj.contact = row["contact"]
            traj.step_rewards = row["step_rewards"]
            traj.terminated = row["terminated"]
            traj.termination_known = row["termination_known"]
            traj.guidance_index = None
            trajs.append(traj)
        self.assertGreater(sum(not traj.termination_known for traj in trajs), 0)
        self.assertEqual(sum(recover_v2_termination(traj) for traj in trajs), 11)
        self.assertTrue(all(traj.termination_known for traj in trajs))

        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        cfg = TrainConfig()
        for field in fields(cfg):
            if field.name in checkpoint.get("config", {}) and field.name not in ("game_root", "port"):
                setattr(cfg, field.name, checkpoint["config"][field.name])
        cfg.sil_archive_mode = "td_pareto"
        cfg.sil_td_sampling_mode = "coverage_share"
        cfg.sil_td_sampling_horizon_steps = 1200
        cfg.sil_td_windows_per_traj = 32
        cfg.sil_td_sampling_seed = 47
        cfg.sil_num_workers = 0
        cfg.sil_advantage_weighting_enabled = True
        cfg.contact_heads_enabled = "actor_mean_contact.weight" in checkpoint["model"]
        model = ActorCritic(cfg).eval()
        model.load_state_dict(expand_state_dict_for_dynamics_dim(
            checkpoint["model"], model.state_dict()))
        grid = np.zeros((8, 8), np.float32)
        dataset = TrajectoryDataset(
            cfg, np.ones((8, 8), dtype=bool), grid, min_gx=0, min_gy=0,
            solid_polygons=None, tip_local=None, body_local=None,
        )
        dataset.set_trajectories(trajs)
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")
        metrics = trainer.sil_update(model, dataset, budget_updates=40, iteration=123)
        self.assertTrue(math.isfinite(metrics["sil_adv_pos_frac"]))
        self.assertGreater(metrics["sil_updates"], 0)
        long = next(row for row in trainer.last_sil_sampling["trajectories"] if row["t"] == 800)
        self.assertGreater(long["max_tau"], long["cap"] / 2)


if __name__ == "__main__":
    unittest.main()

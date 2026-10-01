"""Delayed exploration scale and the log-prob that PPO must reuse."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.dataset import DYNAMICS_DIM
from training.delayed_exploration import OPEN_STEPS, noise_scale, scales_for_agents


class DelayedExplorationTests(unittest.TestCase):
    def test_schedule_keeps_a_quiet_prefix_then_ramps(self):
        self.assertEqual(noise_scale(0, 0, 0.15, 50), 1.0)
        self.assertEqual(noise_scale(199, 200, 0.15, 50), 0.15)
        self.assertEqual(noise_scale(200, 200, 0.15, 50), 0.15)
        self.assertAlmostEqual(noise_scale(225, 200, 0.15, 50), 0.15 + 0.85 * 0.5)
        self.assertEqual(noise_scale(250, 200, 0.15, 50), 1.0)
        self.assertEqual(noise_scale(600, 600, 0.15, 50), 0.15)
        self.assertEqual(tuple(OPEN_STEPS), (0, 0, 200, 200, 400, 400, 600, 600))

    def test_eight_players_keep_distinct_scales_at_step_200(self):
        cfg = TrainConfig(num_agents=8, steps_per_rollout=1200, delayed_exploration_enabled=True)
        scales = scales_for_agents(cfg, range(8), 200)
        self.assertEqual(list(scales[:2]), [1.0, 1.0])
        self.assertEqual(list(scales[2:4]), [0.15, 0.15])
        self.assertTrue((scales[4:] == 0.15).all())
        disabled = scales_for_agents(TrainConfig(num_agents=8), range(8), 200)
        self.assertTrue((disabled == 1).all())

    def test_log_prob_uses_the_scale_from_collection(self):
        torch.manual_seed(0)
        cfg = TrainConfig()
        cfg.ou_phi = 0.85
        model = ActorCritic(cfg).eval()
        dynamics = torch.randn(4, cfg.context_len, DYNAMICS_DIM)
        patches = torch.randn(4, cfg.context_len, 4, cfg.patch_size, cfg.patch_size)
        history = torch.randn(4, cfg.context_len, 2)
        valid = torch.ones(4, cfg.context_len)
        previous = torch.randn(4, 2)
        scale = torch.tensor([1.0, 0.15, 0.5, 1.0])
        sampled = model.get_action_and_value(
            dynamics, patches, history, valid, previous, cfg.ou_phi, noise_scale=scale,
        )
        matched, _, _ = model.evaluate_actions(
            dynamics, patches, history, sampled.action, valid, previous, cfg.ou_phi,
            raw_actions=sampled.raw_action, noise_scale=scale,
        )
        torch.testing.assert_close(sampled.log_prob, matched, atol=1e-5, rtol=1e-5)
        full, _, _ = model.evaluate_actions(
            dynamics, patches, history, sampled.action, valid, previous, cfg.ou_phi,
            raw_actions=sampled.raw_action,
        )
        self.assertGreater((matched - full).abs().max().item(), 1e-3)

    def test_zero_scale_is_rejected(self):
        model = ActorCritic(TrainConfig()).eval()
        with self.assertRaises(ValueError):
            model._scaled_std(torch.ones(2, 2), torch.zeros(2))


if __name__ == "__main__":
    unittest.main()

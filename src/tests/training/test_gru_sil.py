"""Recurrent SIL: histories, returns, selective gradients, rollback and CLI contracts."""
import contextlib
import copy
import io
import json
import sys
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.config import TrainConfig
from training.dataset import TrajectoryDataset
from training.decision_buffer import DecisionRolloutBuffer
from training.experimental_policy import create_actor_critic
from training.experiment_options import ppo_config
from training.gru_sil import (conditional_kl, discounted_returns, gru_sil_update,
                              rollout_distributions, sequence_features, sequence_predictions, validate_gru_sil)
from training.main_ppo import main
from training.ppo_trainer import PPOTrainer
from src.tests.training.test_elite_archive import episode


class GRUSILTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        self.cfg = replace(TrainConfig.ppo_mainline(), d_model=16, nhead=2, num_layers=1,
                           reference_trajectory="test", recurrent_seq_len=3, gru_sil_enabled=True,
                           gru_sil_seed=7, lr=1e-4, weight_decay=1e-4)
        self.traj = episode(t=8, tail=3)
        self.traj.actions[:] = [30., -20.]
        self.traj.step_rewards[:] = .5
        terrain = np.ones((100, 100), np.float32)
        self.encoder = TrajectoryDataset(self.cfg, terrain, terrain, reference_maps=[terrain])
        self.frames = self.encoder.encode_sequence(self.traj)
        self.model = create_actor_critic(self.cfg).eval()
        with torch.no_grad():
            self.model.critic_head[-1].weight.zero_()
            self.model.critic_head[-1].bias.fill_(-2.)
        self.trainer = PPOTrainer(self.cfg)
        self.trainer.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr,
                                                   weight_decay=self.cfg.weight_decay)
        self.archive = SimpleNamespace(trajectories=[self.traj])
        n = len(self.traj.actions)
        ep = dict(self.frames, decision_steps=np.arange(n), noise_scale=np.ones(n, np.float32),
                  noise_prev=np.zeros((n, 2), np.float32))
        self.buffer = DecisionRolloutBuffer({"noise_phi": .85})
        self.buffer.episodes = [ep]

    def update(self, budget=16, iteration=51):
        return gru_sil_update(self.trainer, self.model, self.archive, self.encoder, self.buffer,
                              budget_updates=budget, iteration=iteration)

    def assert_state_equal(self, left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for k in left:
                self.assert_state_equal(left[k], right[k])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_sequence_matches_shared_window_builders_and_previous_executed_action(self):
        np.testing.assert_array_equal(self.frames["act_history"][0], [0, 0])
        np.testing.assert_array_equal(self.frames["act_history"][1:], self.traj.actions)
        fields = ("dynamics", "patches", "act_history", None, None,
                  "contour_tokens", "contour_mask", "relation_tokens", "relation_mask")
        for t in range(len(self.traj.raw_states)):
            window = self.encoder.encode_window(self.traj, t, need_action=False)
            for k, name in enumerate(fields):
                if name:
                    np.testing.assert_array_equal(window[k][-1].numpy(), self.frames[name][t])

    def test_current_weight_prefix_matches_full_replay_and_is_detached(self):
        full = sequence_features(self.model, self.frames, 0, 10)
        chunk = sequence_features(self.model, self.frames, 7, 10)
        torch.testing.assert_close(chunk, full[7:], rtol=1e-5, atol=1e-6)
        old = chunk.detach().clone()
        with torch.no_grad():
            self.model.action_proj.weight.add_(.2)
        calls = []
        original = self.model.recurrent_features
        def traced(*args, **kwargs):
            calls.append((args[0].shape[1], torch.is_grad_enabled(), kwargs.get("hidden")))
            return original(*args, **kwargs)
        with patch.object(self.model, "recurrent_features", side_effect=traced):
            rebuilt = sequence_features(self.model, self.frames, 7, 10)
        self.assertEqual([(n, grad) for n, grad, h in calls], [(7, False), (3, True)])
        self.assertFalse(calls[-1][2].requires_grad)
        torch.testing.assert_close(rebuilt, sequence_features(self.model, self.frames, 0, 10)[7:],
                                   rtol=1e-5, atol=1e-6)
        self.assertGreater(float((rebuilt.detach()-old).abs().max()), 1e-4)
        rebuilt.sum().backward()
        self.assertGreater(float(self.model.gru.weight_hh_l0.grad.abs().sum()), 0)

    def test_returns_include_tail_and_distinguish_terminal_from_truncation(self):
        np.testing.assert_allclose(discounted_returns([1, 2], [False, False], 5, .5), [3.25, 4.5])
        np.testing.assert_allclose(discounted_returns([1, 2], [False, True], 5, .5), [2, 2])
        self.traj.step_rewards[8:] = -100
        self.traj.terminated[-1] = True
        before = copy.deepcopy(self.model.state_dict())
        metrics = self.update()
        self.assertEqual(metrics["gru_sil_updates"], 0)
        self.assertEqual(metrics["gru_sil_positive_frames"], 0)
        self.assert_state_equal(before, self.model.state_dict())
        self.assertTrue(all(row["bootstrap"] == 0 for row in self.trainer.last_gru_sil_sampling["chunks"]))

    def test_positive_chunks_update_gru_keep_std_fixed_and_obey_budget(self):
        std = self.model.actor_log_std.detach().clone()
        before = self.model.gru.weight_hh_l0.detach().clone()
        baseline = rollout_distributions(self.model, self.buffer)
        metrics = self.update(budget=8)
        self.assertEqual(metrics["gru_sil_budget"], 2)
        self.assertEqual(metrics["gru_sil_updates"], 2)
        self.assertGreater(float((before-self.model.gru.weight_hh_l0.detach()).abs().sum()), 0)
        torch.testing.assert_close(std, self.model.actor_log_std, rtol=0, atol=0)
        self.assertIsNone(self.model.actor_log_std.grad)
        self.assertNotIn(self.model.actor_log_std, self.trainer.optimizer.state)
        kl = conditional_kl(baseline, rollout_distributions(self.model, self.buffer))
        self.assertAlmostEqual(float(kl.mean()), metrics["gru_sil_kl"], places=7)
        self.assertLessEqual(float(kl.max()), self.cfg.gru_sil_max_frame_kl)
        chunks = self.trainer.last_gru_sil_sampling["chunks"]
        self.assertTrue(all(row["end"] <= 8 and row["start"] < row["end"] for row in chunks))
        self.assertTrue(all(row["bootstrap"] != 0 for row in chunks))

    def test_negative_advantage_and_zero_budget_leave_everything_unchanged(self):
        for budget in (0, 3, 16):
            self.traj.step_rewards[:] = -1
            before = copy.deepcopy((self.model.state_dict(), self.trainer.optimizer.state_dict()))
            metrics = self.update(budget)
            self.assertEqual(metrics["gru_sil_updates"], 0)
            self.assert_state_equal(before, (self.model.state_dict(), self.trainer.optimizer.state_dict()))

    def test_kl_rejection_restores_parameters_and_populated_optimizer_exactly(self):
        self.assertGreater(self.update(8)["gru_sil_updates"], 0)
        for group in self.trainer.optimizer.param_groups:
            group["lr"] = 10.
        before = copy.deepcopy((self.model.state_dict(), self.trainer.optimizer.state_dict()))
        metrics = self.update(16, iteration=52)
        self.assertEqual(metrics["gru_sil_updates"], 0)
        self.assertEqual(metrics["gru_sil_rejected_steps"], 1)
        self.assert_state_equal(before, (self.model.state_dict(), self.trainer.optimizer.state_dict()))

    def test_sampling_is_iteration_deterministic_without_consuming_global_rng(self):
        self.traj.step_rewards[:] = -1
        np.random.seed(19)
        expected = np.random.random(4)
        np.random.seed(19)
        self.update()
        first = copy.deepcopy(self.trainer.last_gru_sil_sampling)
        np.testing.assert_array_equal(np.random.random(4), expected)
        self.update()
        self.assertEqual(first, self.trainer.last_gru_sil_sampling)

    def test_nonfinite_update_restores_model_and_optimizer_before_raising(self):
        self.update(8)
        before = copy.deepcopy((self.model.state_dict(), self.trainer.optimizer.state_dict()))
        step = self.trainer.optimizer.step
        def corrupt():
            step()
            with torch.no_grad():
                self.model.actor_mean.bias.fill_(float("nan"))
        with patch.object(self.trainer.optimizer, "step", side_effect=corrupt):
            with self.assertRaises(FloatingPointError):
                self.update()
        self.assert_state_equal(before, (self.model.state_dict(), self.trainer.optimizer.state_dict()))

    def test_ou_kl_accounts_for_low_noise_scale(self):
        old = rollout_distributions(self.model, self.buffer)
        with torch.no_grad():
            self.model.actor_mean.bias.add_(.02)
        normal = conditional_kl(old, rollout_distributions(self.model, self.buffer)).mean()
        self.buffer.episodes[0]["noise_scale"][:] = .25
        with torch.no_grad():
            self.model.actor_mean.bias.sub_(.02)
        old_low = rollout_distributions(self.model, self.buffer)
        with torch.no_grad():
            self.model.actor_mean.bias.add_(.02)
        low = conditional_kl(old_low, rollout_distributions(self.model, self.buffer)).mean()
        self.assertAlmostEqual(float(low / normal), 16, delta=.002)

    def test_cli_enables_separate_path_and_resume_preserves_settings(self):
        argv = ["main_ppo", "--dry-run", "--run-dir", "runs/test", "--reference-trajectory", "--gru-sil"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            main()
        data = json.loads(output.getvalue())
        self.assertTrue(data["gru_sil_enabled"])
        self.assertTrue(data["elite_archive_enabled"])
        self.assertFalse(data["sil_enabled"])
        restored = ppo_config({"config": asdict(self.cfg), "model": {}})
        self.assertTrue(restored.gru_sil_enabled)
        self.assertEqual(restored.gru_sil_seed, 7)
        for fields in ({"policy_backbone": "transformer"}, {"event_features": True},
                       {"action_segment_steps": 2}, {"sil_enabled": True}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_gru_sil(replace(self.cfg, **fields))
        with patch.object(sys, "argv", argv + ["--no-elite-archive"]), self.assertRaises(ValueError):
            main()


if __name__ == "__main__":
    unittest.main()

"""Recurrent SIL: histories, returns, selective gradients, rollback and CLI contracts."""
import contextlib
import copy
import io
import json
import math
import sys
import tempfile
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
                              rollout_distributions, sequence_features, sequence_predictions, validate_gru_sil,
                              GRUSILSamplingPlan)
from training.gru_sil import EVENT_SAMPLING_MODE, bounded_target_distribution
from training.climb_diagnostics import launch_event_windows
from training.main_ppo import main
from training.ppo_trainer import PPOTrainer
from src.tests.training.test_elite_archive import episode


class GRUSILTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        self.cfg = replace(TrainConfig.ppo_mainline(), d_model=16, nhead=2, num_layers=1,
                           reference_trajectory="test", recurrent_seq_len=3, gru_sil_enabled=True,
                           gru_sil_max_updates=4, gru_sil_budget_ratio=.25,
                           gru_sil_seed=7, lr=1e-4, weight_decay=1e-4)
        self.traj = episode(t=8, tail=3)
        self.traj.actions[:] = [30.125, -20.875]
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

    def test_fractional_actions_remain_ordered_and_targets_use_previous_action_history(self):
        n=len(self.traj.actions)
        self.traj.actions[:,0]=np.linspace(-31.125,42.875,n)
        self.traj.actions[:,1]=np.linspace(9.375,-20.625,n)
        frames=self.encoder.encode_sequence(self.traj)
        start,end=3,8
        np.testing.assert_array_equal(frames["act_history"][start:end],self.traj.actions[start-1:end-1])
        raw=torch.atanh(torch.as_tensor(self.traj.actions[start:end])/self.cfg.action_scale)
        reconstructed=torch.tanh(raw)*self.cfg.action_scale
        torch.testing.assert_close(reconstructed,torch.as_tensor(self.traj.actions[start:end]),atol=1e-5,rtol=1e-6)

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

    def test_expanded_budget_performs_16_updates_and_preserves_first_four_draws(self):
        for group in self.trainer.optimizer.param_groups:
            group["lr"] = 1e-5
        initial = copy.deepcopy((self.model.state_dict(), self.trainer.optimizer.state_dict()))
        old = self.update(budget=16)
        old_chunks = copy.deepcopy(self.trainer.last_gru_sil_sampling["chunks"])
        self.assertEqual(old["gru_sil_updates"], 4)
        self.model.load_state_dict(initial[0])
        self.trainer.optimizer.load_state_dict(initial[1])
        self.cfg.gru_sil_max_updates = 16
        self.cfg.gru_sil_budget_ratio = 1.0
        new = self.update(budget=16)
        self.assertEqual(new["gru_sil_budget"], 16)
        self.assertEqual(new["gru_sil_attempts"], 16)
        self.assertEqual(new["gru_sil_updates"], 16)
        self.assertEqual(self.trainer.last_gru_sil_sampling["chunks"][:4], old_chunks)
        self.assertGreater(new["gru_sil_positive_frames"], old["gru_sil_positive_frames"])
        self.assertEqual(new["gru_sil_rejected_steps"], 0)
        self.assertLessEqual(new["gru_sil_kl"], self.cfg.gru_sil_target_kl)
        self.assertLessEqual(new["gru_sil_frame_kl_max"], self.cfg.gru_sil_max_frame_kl)

    def test_expanded_budget_still_obeys_ppo_updates_and_negative_advantage_gate(self):
        self.cfg.gru_sil_max_updates = 16
        self.cfg.gru_sil_budget_ratio = 1.0
        self.assertEqual(self.update(budget=0)["gru_sil_budget"], 0)
        small = self.update(budget=3)
        self.assertEqual(small["gru_sil_budget"], 3)
        self.assertEqual(small["gru_sil_updates"], 3)
        self.traj.step_rewards[:] = -100
        before = copy.deepcopy((self.model.state_dict(), self.trainer.optimizer.state_dict()))
        blocked = self.update(budget=100)
        self.assertEqual(blocked["gru_sil_budget"], 16)
        self.assertEqual(blocked["gru_sil_updates"], 0)
        self.assert_state_equal(before, (self.model.state_dict(), self.trainer.optimizer.state_dict()))

    def test_phase_sampling_preserves_recurrent_update_budget_and_records_weights(self):
        self.cfg.gru_sil_sampling_mode = "phase_balanced_v1"
        result = self.update(budget=8)
        self.assertEqual(result["gru_sil_updates"], 2)
        record = self.trainer.last_gru_sil_sampling
        self.assertEqual(record["sampling_mode"], "phase_balanced_v1")
        for row in record["chunks"]:
            self.assertEqual(sum(row["phase_counts"].values()), row["end"]-row["start"])
            self.assertEqual(sum(row["positive_by_phase"].values()), row["positive_frames"])
            self.assertGreater(sum(row["advantage_sum_by_phase"].values()), 0)
            self.assertIsNotNone(row["selected_phase"])
            self.assertLessEqual(row["end"], row["cap"])

    def test_event_weighting_preserves_update_budget_and_audits_each_draw(self):
        self.cfg.gru_sil_sampling_mode = EVENT_SAMPLING_MODE
        result = self.update(budget=8)
        self.assertEqual(result["gru_sil_updates"], 2)
        record = self.trainer.last_gru_sil_sampling
        self.assertEqual(record["weighting"]["version"], EVENT_SAMPLING_MODE)
        for row in record["chunks"]:
            weight = row["sampling_weight"]
            self.assertGreater(weight["probability"], 0)
            self.assertGreaterEqual(weight["probability_ratio"], .5-1e-12)
            self.assertLessEqual(weight["probability_ratio"], 4+1e-12)
            self.assertEqual(sum(row["phase_counts"].values()), row["end"]-row["start"])

    def test_dynamics_weighting_audits_joint_and_conditional_probabilities(self):
        self.cfg.gru_sil_sampling_mode = "quality_dynamics_v3"
        result = self.update(budget=8)
        self.assertEqual(result["gru_sil_updates"], 2)
        record = self.trainer.last_gru_sil_sampling
        self.assertEqual(record["weighting"]["version"], "quality_dynamics_v3")
        for row in record["chunks"]:
            weight = row["sampling_weight"]
            self.assertAlmostEqual(weight["probability"], weight["trajectory_probability"]*weight["conditional_chunk_probability"])
            self.assertLessEqual(weight["core_frames"]+weight["context_frames"]+weight["free_flight_frames"], row["end"]-row["start"])

    def test_kl_rejection_restores_parameters_and_populated_optimizer_exactly(self):
        self.cfg.gru_sil_max_updates = 16
        self.cfg.gru_sil_budget_ratio = 1.0
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
        self.assertEqual(data["gru_sil_sampling_mode"], "quality_dynamics_v3")
        self.assertEqual(data["gru_sil_cache_mb"], 1024)
        self.assertEqual(data["gru_sil_max_updates"], 16)
        self.assertEqual(data["gru_sil_budget_ratio"], 1.0)
        restored = ppo_config({"config": asdict(self.cfg), "model": {}})
        self.assertTrue(restored.gru_sil_enabled)
        self.assertEqual(restored.gru_sil_seed, 7)
        for fields in ({"policy_backbone": "transformer"}, {"event_features": True},
                       {"action_segment_steps": 2}, {"sil_enabled": True}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_gru_sil(replace(self.cfg, **fields))
        with patch.object(sys, "argv", argv + ["--no-elite-archive"]), self.assertRaises(ValueError):
            main()

    def test_cache_cli_resume_override_and_size_validation(self):
        saved = asdict(self.cfg)
        saved.pop("gru_sil_cache_mb")
        self.assertEqual(ppo_config({"config": saved}).gru_sil_cache_mb, 1024)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/"source.pt"
            saved["gru_sil_cache_mb"] = 256
            torch.save({"config": saved, "model": {}}, checkpoint)
            base = ["main_ppo", "--dry-run", "--resume", str(checkpoint), "--run-dir", "runs/test"]
            for extra, expected in (([], 256), (["--gru-sil-cache-mb", "0"], 0),
                                    (["--gru-sil-cache-mb", "512"], 512)):
                output = io.StringIO()
                with patch.object(sys,"argv",base+extra), contextlib.redirect_stdout(output): main()
                self.assertEqual(json.loads(output.getvalue())["gru_sil_cache_mb"], expected)
        for size in (-1, 1.5, float("nan"), True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                validate_gru_sil(replace(self.cfg, gru_sil_cache_mb=size))

    def test_budget_resume_preserves_old_values_until_explicitly_overridden(self):
        saved = asdict(self.cfg)
        for field in ("gru_sil_max_updates", "gru_sil_budget_ratio"):
            saved.pop(field)
        legacy = ppo_config({"config": saved})
        self.assertEqual((legacy.gru_sil_max_updates, legacy.gru_sil_budget_ratio), (4, .25))
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/"source.pt"
            torch.save({"config": asdict(self.cfg), "model": {}}, checkpoint)
            base = ["main_ppo", "--dry-run", "--resume", str(checkpoint), "--run-dir", "runs/test"]
            cases = (([], (4,.25)), (["--gru-sil-max-updates","16"], (16,.25)),
                     (["--gru-sil-budget-ratio","1"], (4,1.)),
                     (["--gru-sil-max-updates","16","--gru-sil-budget-ratio","1"], (16,1.)))
            for extra, expected in cases:
                output = io.StringIO()
                with patch.object(sys,"argv",base+extra), contextlib.redirect_stdout(output): main()
                resolved = json.loads(output.getvalue())
                self.assertEqual((resolved["gru_sil_max_updates"],resolved["gru_sil_budget_ratio"]), expected)
                self.assertEqual(resolved["gru_sil_sampling_mode"],self.cfg.gru_sil_sampling_mode)
                self.assertEqual(resolved["gru_sil_target_kl"],self.cfg.gru_sil_target_kl)
        for values in ({"gru_sil_max_updates":0}, {"gru_sil_max_updates":1.5},
                       {"gru_sil_max_updates":True}, {"gru_sil_budget_ratio":0},
                       {"gru_sil_budget_ratio":1.1}, {"gru_sil_budget_ratio":float("nan")},
                       {"gru_sil_budget_ratio":float("inf")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_gru_sil(replace(self.cfg, **values))

    def test_warm_cache_keeps_inputs_but_rebuilds_current_weight_predictions(self):
        with patch.object(self.encoder, "encode_sequence", wraps=self.encoder.encode_sequence) as encode:
            first = self.update(budget=8, iteration=51)
            self.assertEqual(first["gru_sil_cache_misses"], 1)
            old = copy.deepcopy(self.model.state_dict())
            second = self.update(budget=8, iteration=52)
            self.assertEqual(second["gru_sil_cache_misses"], 0)
            self.assertEqual(second["gru_sil_cache_hits"], second["gru_sil_budget"])
            self.assertEqual(encode.call_count, 1)
        self.assertGreater(float((old["gru.weight_hh_l0"]-self.model.gru.weight_hh_l0.detach()).abs().sum()), 0)

    def test_disabled_cache_drops_round_inputs_after_failed_kl_audit(self):
        self.cfg.gru_sil_cache_mb = 0
        old = copy.deepcopy((self.model.state_dict(), self.trainer.optimizer.state_dict()))
        with patch("training.gru_sil.conditional_kl", side_effect=RuntimeError("audit failed")):
            with self.assertRaises(RuntimeError): self.update(budget=8)
        self.assertFalse(self.encoder._gru_sil_observation_cache._round_local)
        self.assert_state_equal(old, (self.model.state_dict(), self.trainer.optimizer.state_dict()))

    def test_old_resume_requires_explicit_sampler_change_and_keeps_seed(self):
        saved = asdict(self.cfg); saved.pop("gru_sil_sampling_mode")
        saved["training_seed"] = 7
        self.assertEqual(ppo_config({"config": saved}).gru_sil_sampling_mode, "trajectory_uniform")
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/"source.pt"
            torch.save({"config": saved, "model": {}}, checkpoint)
            base = ["main_ppo", "--dry-run", "--resume", str(checkpoint), "--run-dir", "runs/test"]
            for extra, expected in (([], "trajectory_uniform"),
                                    (["--gru-sil-sampling-mode", "quality_dynamics_v3"], "quality_dynamics_v3")):
                output = io.StringIO()
                with patch.object(sys, "argv", base+extra), contextlib.redirect_stdout(output):
                    main()
                data = json.loads(output.getvalue())
                self.assertEqual(data["gru_sil_sampling_mode"], expected)
                self.assertEqual(data["training_seed"], 7)


class GRUSILSamplingTests(unittest.TestCase):
    def trajectories(self):
        trajs = [episode(t=95, tail=5), episode(t=160, tail=5)]
        for t in trajs:
            t.raw_states[:, 33] = -40
            t.raw_states[64:, 33] = -30
            t.raw_states[:, 24] = -2
            t.contact[:] = 1
            t.contact[32:60] = 0
            t.raw_states[32:60, 3] = 2
            t.raw_states[32:60, 34] = np.linspace(0, 1, 28)
        trajs[1].raw_states[96:, 23:25] = [-30, 2]
        return trajs

    def test_legacy_rng_draws_are_exactly_preserved(self):
        plan = GRUSILSamplingPlan(self.trajectories(), [0, 1], 32, "trajectory_uniform")
        old_rng, new_rng = np.random.default_rng(7), np.random.default_rng(7)
        for _ in range(100):
            ti = int(old_rng.choice([0, 1])); cap = [95, 160][ti]
            a = int(old_rng.integers(math.ceil(cap/32))) * 32
            self.assertEqual(plan.draw(new_rng), (ti, a, min(a+32, cap), cap, None, "trajectory_uniform"))

    def test_mixture_probabilities_and_actual_draws_obey_contiguous_caps(self):
        trajs = self.trajectories()
        modes = ("trajectory_uniform", "phase_balanced_v1", "phase_mixture_v1")
        plans = [GRUSILSamplingPlan(trajs, [0, 1], 32, mode) for mode in modes]
        old, phase, mixed = [p.probabilities() for p in plans]
        self.assertEqual(old.keys(), mixed.keys()); self.assertEqual(phase.keys(), mixed.keys())
        self.assertAlmostEqual(sum(mixed.values()), 1.)
        for key in mixed:
            self.assertAlmostEqual(mixed[key], .5*(old[key]+phase[key]))
        from collections import Counter
        counts, branches = Counter(), Counter()
        rng = np.random.default_rng(7)
        for _ in range(10000):
            ti, a, b, cap, selected_phase, branch = plans[-1].draw(rng)
            self.assertTrue(0 <= a < b <= cap <= len(trajs[ti].actions))
            self.assertLessEqual(b-a, 32)
            counts[(ti, a, b, cap)] += 1; branches[branch] += 1
            self.assertEqual(selected_phase is None, branch == "trajectory_uniform")
        self.assertAlmostEqual(branches["phase_balanced"]/10000, .5, delta=.015)
        self.assertLess(max(abs(counts[k]/10000-p) for k,p in mixed.items()), .015)

    def test_event_window_recovers_supported_push_across_chunk_boundary(self):
        traj = episode(t=150, tail=20)
        traj.raw_states[:] = 0
        traj.raw_states[:, 33] = -40
        traj.contact[:] = 0
        traj.contact[50:100, 3] = 1
        traj.raw_states[80:100, 3] = 2
        traj.raw_states[100:141, 34] = np.linspace(0, 2, 41)
        traj.raw_states[100:141, 3] = 2
        traj.raw_states[141:, 34] = 2
        events = launch_event_windows(traj.raw_states, traj.contact, 150)
        self.assertEqual(len(events), 1)
        # Rising contact starts at 80; 12 context states plus the entering action.
        self.assertEqual((events[0]["start"], events[0]["end"]), (67, 140))
        self.assertEqual(events[0]["release_state"], 100)
        self.assertEqual(launch_event_windows(traj.raw_states, traj.contact, 120)[0]["end"], 120)
        self.assertEqual(launch_event_windows(traj.raw_states, traj.contact, 99), [])
        plan = GRUSILSamplingPlan([traj], [0], 32, EVENT_SAMPLING_MODE)
        detail = plan.chunk_details[(0, 64, 96, 150)]
        self.assertGreater(detail["event_probability"], 0)
        self.assertEqual(detail["event_overlaps"][0]["overlap_steps"], 29)

    def test_soft_membership_preserves_a_minority_phase(self):
        traj = episode(t=96, tail=5)
        traj.raw_states[:] = 0
        traj.raw_states[:, 33] = -40
        traj.contact[:] = 1
        # Just one elevated-contact frame in the middle chunk, no majority change.
        traj.raw_states[40, 23:25] = [-30, 2]
        new = GRUSILSamplingPlan([traj], [0], 32, EVENT_SAMPLING_MODE)
        old = GRUSILSamplingPlan([traj], [0], 32, "phase_mixture_v1")
        self.assertAlmostEqual(old.probabilities()[(0, 32, 64, 96)], 1/3)
        self.assertGreater(new.probabilities()[(0, 32, 64, 96)], 1/3)
        self.assertEqual(new.distribution_summary["event_count"], 0)
        for chunk, row in new.chunk_details.items():
            self.assertAlmostEqual(row["event_probability"], row["base_probability"])

    def test_rare_event_mass_is_bounded_and_redistributed(self):
        base = np.array([.001, .099, .4, .5])
        target = np.array([.9, .05, .025, .025])
        bounded = bounded_target_distribution(target, base)
        mixed = .5*(base+bounded)
        self.assertAlmostEqual(bounded.sum(), 1)
        self.assertAlmostEqual(mixed.sum(), 1)
        self.assertAlmostEqual(mixed[0]/base[0], 4.)
        self.assertTrue(np.all(mixed/base >= .5))
        self.assertTrue(np.all(mixed/base <= 4.+1e-12))
        self.assertTrue(np.all(bounded[1:] > target[1:]))

    def test_event_groups_do_not_let_many_approach_jumps_dilute_tree_events(self):
        first = self.trajectories()[0]
        trajs = [copy.deepcopy(first) for _ in range(3)]
        trajs[2].raw_states[:, 33] = -30
        plan = GRUSILSamplingPlan(trajs, [0,1,2], 32, EVENT_SAMPLING_MODE)
        self.assertEqual(plan.distribution_summary["event_groups"], {"approach_only": 2, "tree_connected": 1})
        event_mass = [sum(row["event_probability"] for chunk, row in plan.chunk_details.items() if chunk[0] == i)
                      for i in range(3)]
        np.testing.assert_allclose(event_mass, [.25, .25, .5], rtol=0, atol=1e-12)

    def test_event_draws_match_marginals_are_reproducible_and_respect_short_cap(self):
        from collections import Counter
        plan = GRUSILSamplingPlan(self.trajectories(), [0, 1], 32, EVENT_SAMPLING_MODE)
        rng, repeated = np.random.default_rng(47), np.random.default_rng(47)
        counts = Counter()
        for _ in range(20000):
            draw = plan.draw(rng)
            self.assertEqual(draw, plan.draw(repeated))
            ti, a, b, cap, phase, branch = draw
            self.assertTrue(0 <= a < b <= cap and b-a <= 32)
            counts[(ti,a,b,cap)] += 1
        self.assertGreater(counts[(0,64,95,95)], 0)
        for chunk, p in plan.probabilities().items():
            self.assertAlmostEqual(counts[chunk]/20000, p, delta=.012)
            self.assertTrue(.5-1e-12 <= p/plan.chunk_details[chunk]["base_probability"] <= 4+1e-12)
        self.assertAlmostEqual(sum(plan.probabilities().values()), 1.)


if __name__ == "__main__":
    unittest.main()

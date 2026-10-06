"""Temporal windows: probability, recurrent continuity, archival and RNG contracts."""
import copy
import argparse
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.config import TrainConfig
from training.decision_buffer import DecisionRolloutBuffer
from training.experimental_policy import create_actor_critic
from training.exploration_windows import add_arguments, apply_arguments, configure_precision, make_rng, sample_plan, scales_for_plan, validate
from training.experiment_options import ppo_config
from training.main_ppo import save_checkpoint
from training.policy_audit import audit_policy
from training.random_state import capture_random_state, restore_random_state, seed_training
from training.rollout import RolloutWorker
from src.tests.training.test_policy_experiments import state
from src.tests.training.test_training_consistency import assert_same


class EightAgentEnv:
    def reset(self):
        self.step_index = 0
        return np.stack([state()] * 8)

    def step(self, actions):
        self.step_index += 1
        return np.stack([state(self.step_index)] * 8), np.zeros(8, bool)


class ExplorationWindowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_training(7)
        self.cfg = replace(TrainConfig.ppo_mainline(), d_model=16, nhead=2, num_layers=1,
            sparse_geometry=False, recurrent_seq_len=8, num_agents=8, steps_per_rollout=24,
            random_deploy=False, deploy_exclusion_check=False, reference_trajectory=None,
            training_seed=7, exploration_windows_enabled=True, exploration_window_min_steps=2,
            exploration_window_max_steps=8, exploration_window_suffix_min_steps=4,
            exploration_window_floor=.15)

    def worker(self, cfg=None):
        cfg = cfg or self.cfg
        worker = RolloutWorker(cfg)
        worker.env = EightAgentEnv()
        worker._ppo_patches = lambda *args: np.zeros(
            (8, cfg.patch_channels, cfg.patch_size, cfg.patch_size), np.float32)
        worker._compute_contact_batch = lambda obs, prev: (np.zeros((8, 5), np.float32), prev)
        return worker

    def test_schedule_bounds_coverage_and_reassignment(self):
        cfg = replace(self.cfg, steps_per_rollout=1200, exploration_window_min_steps=32,
                      exploration_window_max_steps=480, exploration_window_suffix_min_steps=120)
        rng = make_rng(cfg)
        seen = [set() for _ in range(8)]
        global_before = capture_random_state()
        for _ in range(100):
            plan = sample_plan(cfg, 1200, rng)
            self.assertEqual(sorted(r["kind"] for r in plan["agents"]),
                             ["full", "low", "suffix", "suffix"] + ["window"] * 4)
            lengths = sorted(r["stop"]-r["start"] for r in plan["agents"] if r["kind"] == "window")
            for j, length in enumerate(lengths):
                self.assertGreaterEqual(length, round(32 * (480/32)**(j/4)))
                self.assertLessEqual(length, round(32 * (480/32)**((j+1)/4)))
            for i, row in enumerate(plan["agents"]):
                seen[i].add(row["kind"])
                self.assertEqual(i, row["agent"])
                self.assertTrue(0 <= row["start"] <= row["stop"] <= 1200)
                if row["kind"] == "suffix":
                    self.assertGreaterEqual(row["stop"]-row["start"], 120)
                for step in (max(0, row["start"]-1), row["start"], min(1199, row["stop"])):
                    expected = 1. if row["start"] <= step < row["stop"] else .15
                    self.assertEqual(scales_for_plan(plan, [i], step)[0], np.float32(expected))
        self.assertTrue(all(len(kinds) == 4 for kinds in seen))
        assert_same(self, global_before, capture_random_state())

    def test_invalid_combinations_fail_before_game(self):
        for kwargs in ({"num_agents": 7}, {"policy_backbone": "transformer"},
                       {"action_segment_steps": 2}, {"delayed_exploration_enabled": True},
                       {"phase_noise_enabled": True}, {"exploration_window_floor": -.1},
                       {"exploration_window_peak": float("nan")},
                       {"exploration_window_min_steps": 9}, {"exploration_window_max_steps": 25},
                       {"exploration_window_suffix_min_steps": 25},
                       {"exploration_window_transition_steps": 1},
                       {"exploration_window_anchor_scale": 0.},
                       {"exploration_window_anchor_scale": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.worker(replace(self.cfg, **kwargs))

    def test_legacy_checkpoint_only_restores_soft_anchor_scale_and_cli_overrides_it(self):
        saved = {"exploration_window_action_limit": 2.5}
        for cfg in (TrainConfig(**saved), ppo_config({"config": saved})):
            self.assertEqual(cfg.deterministic_anchor_scale, 2.5)
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            apply_arguments(cfg, parser.parse_args(["--exploration-window-anchor-scale", "4"]))
            self.assertEqual(cfg.deterministic_anchor_scale, 4.)
            self.assertNotIn("--exploration-window-action-limit", parser.format_help())

    def test_disabled_and_deterministic_do_not_consume_window_stream(self):
        rng = make_rng(self.cfg)
        before = copy.deepcopy(rng.bit_generator.state)
        self.assertIsNone(sample_plan(replace(self.cfg, exploration_windows_enabled=False), 24, rng))
        self.assertIsNone(sample_plan(self.cfg, 24, rng, deterministic=True))
        self.assertEqual(before, rng.bit_generator.state)
        model, worker = create_actor_critic(self.cfg).eval(), self.worker()
        before_all = capture_random_state(worker._rng, worker._exploration_rng)
        buffer, _ = worker.collect_ppo(model, 24, deterministic=True)
        assert_same(self, before_all, capture_random_state(worker._rng, worker._exploration_rng))
        for episode in buffer.episodes:
            np.testing.assert_array_equal(episode["noise_scale"], 1.)

    def test_all_one_schedule_is_exact_legacy_sampling(self):
        cfg = replace(self.cfg, exploration_window_floor=1.)
        model = create_actor_critic(cfg).eval()
        worker = self.worker(cfg)
        before = capture_random_state()
        enabled, _ = worker.collect_ppo(model, 24)
        rng_after = capture_random_state()
        restore_random_state(before)
        cfg.exploration_windows_enabled = False
        disabled, _ = worker.collect_ppo(model, 24)
        assert_same(self, enabled.episodes, disabled.episodes)
        assert_same(self, rng_after, capture_random_state())

    def test_live_ou_gru_continuity_scale_archive_and_probability(self):
        model, worker = create_actor_critic(self.cfg).eval(), self.worker()
        before_weights = copy.deepcopy(model.state_dict())
        ou_calls, recurrent_calls = [], []
        original_sample, original_recurrent = model.sample_features, model.recurrent_features

        def sample(*args, **kwargs):
            result = original_sample(*args, **kwargs)
            ou_calls.append((args[1].clone(), result.noise.clone()))
            return result

        def recurrent(*args, **kwargs):
            result = original_recurrent(*args, **kwargs)
            if args[0].shape[:2] == (8, 1):
                recurrent_calls.append((kwargs["hidden"].clone(), result[1].clone()))
            return result

        with patch.object(model, "sample_features", side_effect=sample), patch.object(
                model, "recurrent_features", side_effect=recurrent):
            buffer, _ = worker.collect_ppo(model, 24)
        self.assertEqual(len(ou_calls), 24)
        self.assertEqual(len(recurrent_calls), 24)
        for calls in (ou_calls, recurrent_calls):
            for previous, current in zip(calls, calls[1:]):
                torch.testing.assert_close(previous[1], current[0], rtol=0, atol=0)
        plan = buffer.collection_metadata["exploration_windows"]
        self.assertEqual(buffer.collection_metadata["episode_agent_ids"], list(range(8)))
        for i, ep in enumerate(buffer.episodes):
            np.testing.assert_array_equal(ep["noise_scale"],
                [scales_for_plan(plan, [i], int(t))[0] for t in ep["decision_steps"]])
        before_rng = capture_random_state(worker._rng, worker._exploration_rng)
        audit = audit_policy(model, buffer)
        self.assertEqual(audit["audit_samples"], 192)
        assert_same(self, before_rng, capture_random_state(worker._rng, worker._exploration_rng))
        assert_same(self, before_weights, model.state_dict())
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rollout.npz"
            buffer.save(path)
            loaded = DecisionRolloutBuffer.load(path)
            assert_same(self, buffer.collection_metadata, loaded.collection_metadata)
            assert_same(self, buffer.episodes, loaded.episodes)
            audit_policy(model, loaded)
            loaded.episodes[0]["noise_scale"] *= 2
            with self.assertRaisesRegex(ValueError, "replay failed"):
                audit_policy(model, loaded)
        self.cfg.exploration_window_floor = .2
        with self.assertRaisesRegex(ValueError, "settings changed"):
            audit_policy(model, buffer)

    def test_checkpoint_restores_next_windows_and_actions_exactly(self):
        model, worker = create_actor_critic(self.cfg).eval(), self.worker()
        worker.collect_ppo(model, 24)
        with tempfile.TemporaryDirectory() as root:
            save_checkpoint(model, None, 1, root, self.cfg, deployment_rng=worker._rng,
                            exploration_rng=worker._exploration_rng)
            saved = torch.load(Path(root) / "ppo_iter_0001.pt", weights_only=False)
            expected, _ = worker.collect_ppo(model, 24)
            expected_rng = capture_random_state(worker._rng, worker._exploration_rng)
            other = self.worker()
            restore_random_state(saved["rng_state"], other._rng, other._exploration_rng)
            actual, _ = other.collect_ppo(model, 24)
            assert_same(self, expected.collection_metadata, actual.collection_metadata)
            assert_same(self, expected.episodes, actual.episodes)
            assert_same(self, expected_rng, capture_random_state(other._rng, other._exploration_rng))

    def test_old_checkpoint_without_window_rng_keeps_independent_seed(self):
        rng = make_rng(self.cfg)
        before = copy.deepcopy(rng.bit_generator.state)
        old = capture_random_state()
        del old["exploration_windows"]
        restore_random_state(old, exploration_rng=rng)
        self.assertEqual(before, rng.bit_generator.state)
        self.assertFalse(TrainConfig.ppo_mainline().exploration_windows_enabled)

    def test_precision_opt_in_and_disabled_legacy_defaults(self):
        original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            configure_precision(replace(self.cfg, exploration_windows_enabled=False))
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            create_actor_critic(self.cfg)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = original


if __name__ == "__main__":
    unittest.main()

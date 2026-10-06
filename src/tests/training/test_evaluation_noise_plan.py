"""Fixed frozen interventions keep the production sampler and cannot train."""
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.tests.training import test_exploration_windows as fixtures
from training.decision_buffer import DecisionRolloutBuffer
from training.evaluation_noise_plan import fixed_plan
from training.experimental_policy import create_actor_critic
from training.gru_sil import gru_sil_update
from training.policy_audit import audit_policy
from training.ppo_trainer import PPOTrainer


class FixedNoisePlanTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.ExplorationWindowTests()
        fixture.setUp()
        self.cfg = replace(fixture.cfg, exploration_window_floor=0.)
        self.worker = fixture.worker(self.cfg)
        self.model = create_actor_critic(self.cfg).eval()
        self.spec = {"version": 1, "horizon": 24, "agents": [
            {"agent": i, "condition": str(i), "start": a, "stop": b}
            for i, (a, b) in enumerate([(0, 0), (0, 24), (3, 7), (4, 9),
                                      (6, 12), (8, 16), (12, 18), (15, 21)])]}

    def test_schedule_continuity_audit_archive_and_no_training(self):
        before = copy.deepcopy(self.worker._exploration_rng.bit_generator.state)
        ou, gru = [], []
        sample, recurrent = self.model.sample_features, self.model.recurrent_features
        def observed_sample(*args, **kwargs):
            result = sample(*args, **kwargs)
            ou.append((args[1].clone(), result.noise.clone()))
            return result
        def observed_recurrent(*args, **kwargs):
            result = recurrent(*args, **kwargs)
            if args[0].shape[:2] == (8, 1):
                gru.append((kwargs["hidden"].clone(), result[1].clone()))
            return result
        with patch.object(self.model, "sample_features", side_effect=observed_sample), patch.object(
                self.model, "recurrent_features", side_effect=observed_recurrent):
            buffer, _ = self.worker.collect_ppo(self.model, 24, evaluation_noise_plan=self.spec)
        self.assertEqual(before, self.worker._exploration_rng.bit_generator.state)
        for calls in (ou, gru):
            self.assertEqual(len(calls), 24)
            for previous, current in zip(calls, calls[1:]):
                torch.testing.assert_close(previous[1], current[0], rtol=0, atol=0)
        for row, ep in zip(self.spec["agents"], buffer.episodes):
            expected = (np.arange(24) >= row["start"]) & (np.arange(24) < row["stop"])
            np.testing.assert_array_equal(ep["noise_scale"], expected)
            np.testing.assert_array_equal(ep["likelihood_valid"], expected)
        audit = audit_policy(self.model, buffer)
        self.assertEqual(audit["audit_samples"], 192)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout.npz"
            buffer.save(path)
            buffer = DecisionRolloutBuffer.load(path)
        self.assertTrue(buffer.collection_metadata["evaluation_only"])
        audit_policy(self.model, buffer)
        trainer = PPOTrainer(self.cfg)
        trainer.device = torch.device("cpu")
        for update in (trainer.update, trainer.update_critic_only):
            with self.assertRaisesRegex(ValueError, "Evaluation-only"):
                update(self.model, buffer)
        self.assertIsNone(trainer.optimizer)
        self.assertIsNone(trainer.critic_optimizer)
        with self.assertRaisesRegex(ValueError, "Evaluation-only"):
            gru_sil_update(trainer, self.model, None, None, buffer, budget_updates=16, iteration=1)

    def test_invalid_plans_rejected_before_reset(self):
        malformed = []
        for field, value in (("agent", 1), ("start", -1), ("stop", 25), ("start", 1.5),
                             ("start", True), ("condition", "")):
            spec = copy.deepcopy(self.spec)
            spec["agents"][0][field] = value
            malformed.append(spec)
        malformed += [{**self.spec, "horizon": 25}, {**self.spec, "agents": []}]
        with patch.object(self.worker.env, "reset") as reset:
            for spec in malformed:
                with self.assertRaises(ValueError):
                    self.worker.collect_ppo(self.model, 24, evaluation_noise_plan=spec)
            reset.assert_not_called()
        for kwargs in ({"deterministic": True}, {"initial_offsets": np.ones((8, 2))}):
            with self.assertRaises(ValueError):
                fixed_plan(self.spec, self.cfg, 24, **kwargs)


if __name__ == "__main__":
    unittest.main()

"""Frozen interventions preserve real action history and recurrent continuity."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.dataset import Trajectory
from training.decision_buffer import DecisionRolloutBuffer
from training.experimental_policy import create_actor_critic
from training.gru_sil import _step, gru_sil_update
from training.policy_audit import audit_policy
from training.ppo_trainer import PPOTrainer
from training.prefix_handoff import PrefixHandoff
from src.tests.training.test_exploration_windows import ExplorationWindowTests


class PrefixHandoffTests(unittest.TestCase):
    def setUp(self):
        fixture = ExplorationWindowTests()
        fixture.setUp()
        self.cfg = replace(fixture.cfg, exploration_windows_enabled=False)
        self.worker = fixture.worker(self.cfg)
        self.model = create_actor_critic(self.cfg).eval()
        self.steps = 9
        states = np.zeros((10, 35), np.float32)
        actions = np.tile(np.array([61., -37.], np.float32), (9, 1))
        self.prefix = PrefixHandoff(Trajectory(states, actions), [0, 1, 2, 3, 4, 7, 8, 9])

    def test_executed_history_continuous_gru_and_control(self):
        baseline, original = self.worker.collect_ppo(self.model, self.steps, deterministic=True)
        buffer, trajectories = self.worker.collect_ppo(self.model, self.steps,
            deterministic=True, evaluation_prefix=self.prefix)
        np.testing.assert_array_equal(original[0].actions, trajectories[0].actions)
        for agent, (ep, traj) in enumerate(zip(buffer.episodes, trajectories)):
            cut = self.prefix.cuts[agent]
            np.testing.assert_array_equal(traj.actions[:cut], self.prefix.trajectory.actions[:cut])
            np.testing.assert_array_equal(traj.actions[cut:], ep["actions"][cut:])
            np.testing.assert_array_equal(ep["act_history"][1:], traj.actions)
            np.testing.assert_array_equal(ep["act_history"][0], np.zeros(2))
            with torch.no_grad():
                features, _ = _step(self.model, ep, 0, self.steps, None)
                dist, _ = self.model.distribution_from_features(features[0])
                predicted = self.model._squash_action(dist.mean).numpy()
            np.testing.assert_allclose(predicted, ep["actions"], atol=1e-4, rtol=1e-5)
        self.assertGreater(np.max(abs(buffer.episodes[3]["actions"]-baseline.episodes[3]["actions"])), .01)
        audit_policy(self.model, buffer)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "policy_probes.npz"
            buffer.save(path)
            loaded = DecisionRolloutBuffer.load(path)
            self.assertEqual(buffer.collection_metadata, loaded.collection_metadata)
            np.testing.assert_array_equal(loaded.episodes[3]["act_history"], buffer.episodes[3]["act_history"])
        trainer = PPOTrainer(self.cfg)
        trainer.device = torch.device("cpu")
        with self.assertRaisesRegex(ValueError, "Evaluation rollouts"):
            trainer.update(self.model, buffer)
        trainer.config.reference_trajectory = "fixture"
        with self.assertRaisesRegex(ValueError, "deterministic evaluation"):
            gru_sil_update(trainer, self.model, None, type("Encoder", (), {})(), buffer,
                           budget_updates=1, iteration=1)

    def test_invalid_interventions_fail_before_reset(self):
        for cfg, deterministic, offsets in (
            (self.cfg, False, None), (replace(self.cfg, policy_backbone="transformer"), True, None),
            (replace(self.cfg, action_segment_steps=2), True, None),
            (replace(self.cfg, random_deploy=True), True, None),
            (self.cfg, True, np.ones((8, 2)))):
            with self.assertRaises(ValueError):
                self.prefix.validate(cfg, self.steps, deterministic, offsets)
        for cuts in ([0]*7, [-1]*8, [10]*8, [.5]*8):
            with self.assertRaises(ValueError):
                PrefixHandoff(self.prefix.trajectory, cuts).validate(self.cfg, self.steps, True)

    def test_state_comparison_excludes_clock_and_checks_executed_actions(self):
        demo = self.prefix.trajectory
        actual = Trajectory(demo.raw_states.copy(), demo.actions.copy())
        actual.raw_states[:, 28] += 10000
        actual.raw_states[8, 33] += .25
        report = self.prefix.compare(actual, 6, actual.actions)
        self.assertEqual(report["observable_errors"]["body_position_m"]["at_handoff"], .25)
        self.assertEqual(report["observable_errors"]["body_velocity_m_s"]["prefix_max"], 0.)
        actual.actions[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Executed prefix"):
            self.prefix.compare(actual, 6, actual.actions)


if __name__ == "__main__":
    unittest.main()

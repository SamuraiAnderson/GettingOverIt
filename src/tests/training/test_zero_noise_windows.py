"""Zero-noise windows: explicit densities, recurrent context, learning and rollback."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.config import TrainConfig
from training.dataset import TrajectoryDataset
from training.decision_buffer import DecisionRolloutBuffer, evaluated_batches
from training.elite_archive import EliteTrajectoryArchive
from training.experimental_policy import create_actor_critic
from training.gru_sil import conditional_kl, deterministic_action_drift, gru_sil_update, rollout_distributions
from training.main_ppo import save_checkpoint
from training.policy_audit import audit_policy
from training.ppo_buffer import PPORolloutBuffer
from training.ppo_trainer import PPOTrainer
from training.random_state import capture_random_state, restore_random_state, seed_training
from training.reference_fingerprint import reference_fingerprint
from training.rollout import RolloutWorker
from src.tests.training.test_elite_archive import episode
from src.tests.training.test_exploration_windows import EightAgentEnv
from src.tests.training.test_training_consistency import assert_same


class ZeroNoiseWindowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_training(7)
        self.cfg = replace(TrainConfig.ppo_mainline(), d_model=16, nhead=2, num_layers=1,
            sparse_geometry=False, recurrent_seq_len=3, ppo_batch_size=3, ppo_epochs=1,
            num_agents=8, steps_per_rollout=9, random_deploy=False, deploy_exclusion_check=False,
            reference_trajectory="fixture", training_seed=7, gru_sil_seed=7, lr=1e-5,
            exploration_windows_enabled=True, exploration_window_floor=0.,
            exploration_window_min_steps=2, exploration_window_max_steps=5,
            exploration_window_suffix_min_steps=2, gru_sil_max_updates=2, gru_sil_budget_ratio=1.)
        self.model = create_actor_critic(self.cfg).eval()
        with torch.no_grad():
            self.model.critic_head[-1].weight.zero_()
            self.model.critic_head[-1].bias.fill_(-2.)
        self.worker = self.make_worker()

    def make_worker(self):
        worker = RolloutWorker(replace(self.cfg, reference_trajectory=None))
        worker.env = EightAgentEnv()
        worker._rng = np.random.default_rng(7)
        worker._ppo_patches = lambda *args: np.zeros(
            (8, self.cfg.patch_channels, self.cfg.patch_size, self.cfg.patch_size), np.float32)
        worker._compute_contact_batch = lambda obs, previous: (np.zeros((8, 5), np.float32), previous)
        return worker

    def collect(self):
        return self.worker.collect_ppo(self.model, 9)[0]

    def trainer(self):
        trainer = PPOTrainer(self.cfg)
        trainer.device = torch.device("cpu")
        return trainer

    def test_zero_outputs_mean_and_ou_keeps_evolving_with_identical_draws(self):
        features = torch.randn(8, self.cfg.d_model)
        previous = torch.randn(8, 2)
        scales = torch.tensor([0., 1.] * 4)
        rng = torch.get_rng_state()
        mixed = self.model.sample_features(features, previous, .85, noise_scale=scales)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        full = self.model.sample_features(features, previous, .85, noise_scale=torch.ones(8))
        torch.testing.assert_close(mixed.noise, full.noise, rtol=0, atol=0)
        torch.testing.assert_close(after, torch.get_rng_state(), rtol=0, atol=0)
        mean = self.model.distribution_from_features(features)[0].mean
        torch.testing.assert_close(mixed.raw_action[scales == 0], mean[scales == 0], rtol=0, atol=0)
        self.assertTrue(bool((mixed.action[scales == 0] != 0).any()))
        self.assertTrue(bool((mixed.noise[scales == 0] != 0).any()))
        lp, entropy, _ = self.model.evaluate_features(features, mixed.raw_action, previous, .85, noise_scale=scales)
        torch.testing.assert_close(lp, mixed.log_prob)
        self.assertTrue(bool((lp[scales == 0] == 0).all()))
        self.assertTrue(bool((entropy[scales == 0] == 0).all()))

    def test_full_archive_probability_and_deterministic_action_audits(self):
        buffer = self.collect()
        audit = audit_policy(self.model, buffer)
        self.assertEqual(audit["audit_samples"], 72)
        self.assertGreater(audit["audit_stochastic_samples"], 0)
        self.assertGreater(audit["audit_deterministic_samples"], 0)
        self.assertEqual(audit["audit_stochastic_samples"] + audit["audit_deterministic_samples"], 72)
        for ep in buffer.episodes:
            np.testing.assert_array_equal(ep["decision_steps"], np.arange(9))
            np.testing.assert_array_equal(ep["likelihood_valid"], ep["noise_scale"] > 0)
            self.assertEqual(len(ep["dynamics"]), 10)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "mixed.npz"
            buffer.save(path)
            with np.load(path) as data:
                self.assertEqual(json.loads(str(data["metadata"]))["version"], 4)
            loaded = PPORolloutBuffer.load(path)
            assert_same(self, buffer.episodes, loaded.episodes)
            audit_policy(self.model, loaded)
        ep = next(e for e in buffer.episodes if np.any(e["noise_scale"] == 0))
        row = np.flatnonzero(ep["noise_scale"] == 0)[0]
        ep["log_probs"][row] = 1.
        with self.assertRaisesRegex(ValueError, "sentinel"):
            audit_policy(self.model, buffer)
        ep["log_probs"][row] = 0.
        ep["raw_actions"][row] += .1
        ep["actions"][row] = np.tanh(ep["raw_actions"][row]) * self.cfg.action_scale
        with self.assertRaisesRegex(ValueError, "deterministic-action replay"):
            audit_policy(self.model, buffer)

    def test_zero_frame_advantages_do_not_change_stochastic_normalization(self):
        buffer = self.collect()
        def stochastic_advantages():
            return torch.cat([b.advantages[b.likelihood_valid] for b, *_ in evaluated_batches(
                self.model, buffer, 3, rng=np.random.default_rng(4))])
        before = stochastic_advantages()
        for ep in buffer.episodes:
            ep["advantages"][ep["noise_scale"] == 0] = 1e10
        torch.testing.assert_close(before, stochastic_advantages(), rtol=0, atol=0)
        self.assertAlmostEqual(float(before.mean()), 0., places=5)

    def test_ppo_uses_only_stochastic_actor_rows_and_records_mean_action_drift(self):
        trainer, buffer = self.trainer(), self.collect()
        metrics = trainer.update(self.model, buffer)
        self.assertEqual(metrics["ppo_deterministic_guard_rejected"], 0)
        self.assertGreater(metrics["policy_updates"], 0)
        self.assertLess(metrics["policy_updates"], metrics["n_updates"])
        self.assertTrue(np.isfinite(metrics["deterministic_action_max_delta"]))
        quiet = [row for row in trainer.last_update_trace if row["stochastic_samples"] == 0 and row["applied"]]
        self.assertTrue(quiet)
        self.assertTrue(all(row["policy_loss"] == 0 and row["pre_step_kl"] == 0 for row in quiet))

    def test_ppo_large_action_drift_keeps_weights_optimizer_and_update_counts(self):
        trainer = self.trainer()
        trainer.update(self.model, self.collect())
        buffer = self.collect()
        before = copy.deepcopy((self.model.state_dict(), trainer.optimizer.state_dict(), trainer.checkpoint_state()))
        with patch("training.gru_sil.deterministic_action_drift", return_value={
                "deterministic_action_max_delta": 2., "deterministic_action_rms_delta": 1.}):
            metrics = trainer.update(self.model, buffer)
        self.assertGreater(metrics["policy_updates"], 0)
        self.assertEqual(metrics["n_updates"], metrics["ppo_attempted_updates"])
        self.assertEqual(metrics["policy_updates"], metrics["ppo_attempted_policy_updates"])
        self.assertEqual(metrics["ppo_deterministic_guard_rejected"], 0)
        self.assertEqual(metrics["deterministic_action_max_delta"], 2.)
        self.assertTrue(any(not torch.equal(v, before[0][k]) for k, v in self.model.state_dict().items()))
        self.assertGreater(trainer._global_step, before[2]["global_step"])
        self.assertTrue(any(s["step"] > before[1]["state"][k]["step"]
                            for k, s in trainer.optimizer.state_dict()["state"].items()))
        self.assertTrue(any(row["applied"] for row in trainer.last_update_trace))
        self.assertFalse(any(row.get("rolled_back", False) for row in trainer.last_update_trace))

    def test_sil_kl_excludes_zero_rows_and_drift_measures_them(self):
        buffer = self.collect()
        base = rollout_distributions(self.model, buffer)
        active_count = sum(int(e["likelihood_valid"].sum()) for e in buffer.episodes)
        self.assertEqual(len(conditional_kl(base, base)), active_count)
        with torch.no_grad():
            self.model.actor_mean.bias.add_(.1)
        self.assertGreater(deterministic_action_drift(self.model, buffer)["deterministic_action_max_delta"], 1.)

    def test_ppo_audit_exception_restores_the_update_transaction(self):
        trainer = self.trainer()
        trainer.update(self.model, self.collect())
        buffer = self.collect()
        before = copy.deepcopy((self.model.state_dict(), trainer.optimizer.state_dict(), trainer.checkpoint_state()))
        with patch("training.gru_sil.deterministic_action_drift", side_effect=RuntimeError("audit interrupted")):
            with self.assertRaisesRegex(RuntimeError, "audit interrupted"):
                trainer.update(self.model, buffer)
        assert_same(self, before, (self.model.state_dict(), trainer.optimizer.state_dict(), trainer.checkpoint_state()))

    def test_ppo_nonfinite_parameters_still_restore_the_update_transaction(self):
        trainer = self.trainer()
        trainer.update(self.model, self.collect())
        buffer = self.collect()
        before = copy.deepcopy((self.model.state_dict(), trainer.optimizer.state_dict(), trainer.checkpoint_state()))
        update = trainer._update

        def corrupt(*args, **kwargs):
            result = update(*args, **kwargs)
            with torch.no_grad():
                self.model.actor_mean.bias.fill_(float("nan"))
            return result

        with patch.object(trainer, "_update", side_effect=corrupt):
            with self.assertRaisesRegex(FloatingPointError, "Non-finite parameters"):
                trainer.update(self.model, buffer)
        assert_same(self, before, (self.model.state_dict(), trainer.optimizer.state_dict(), trainer.checkpoint_state()))

    def sil_fixture(self):
        trainer, buffer = self.trainer(), self.collect()
        trainer.update(self.model, buffer)
        terrain = np.ones((100, 100), np.float32)
        encoder = TrajectoryDataset(self.cfg, terrain, terrain, reference_maps=[terrain])
        traj = episode(t=8, tail=3)
        traj.step_rewards[:] = .5
        traj.actions[:] = [30., -20.]
        archive = SimpleNamespace(trajectories=[traj])
        return trainer, buffer, encoder, archive

    def test_sil_large_action_drift_keeps_update_when_stochastic_kl_passes(self):
        trainer, buffer, encoder, archive = self.sil_fixture()
        before = copy.deepcopy((self.model.state_dict(), trainer.optimizer.state_dict()))
        with patch("training.gru_sil.deterministic_action_drift", return_value={
                "deterministic_action_max_delta": 2., "deterministic_action_rms_delta": 1.}):
            metrics = gru_sil_update(trainer, self.model, archive, encoder, buffer,
                                    budget_updates=16, iteration=1)
        self.assertGreater(metrics["gru_sil_updates"], 0)
        self.assertEqual(metrics["gru_sil_deterministic_rejections"], 0)
        self.assertEqual(metrics["gru_sil_deterministic_action_max_delta"], 2.)
        self.assertTrue(trainer.last_gru_sil_sampling["chunks"][0]["accepted"])
        self.assertTrue(any(not torch.equal(v, before[0][k]) for k, v in self.model.state_dict().items()))

    def test_sil_stochastic_kl_rejection_still_restores_model_and_optimizer(self):
        trainer, buffer, encoder, archive = self.sil_fixture()
        before = copy.deepcopy((self.model.state_dict(), trainer.optimizer.state_dict()))
        with patch("training.gru_sil.conditional_kl", return_value=torch.tensor([1.])):
            metrics = gru_sil_update(trainer, self.model, archive, encoder, buffer,
                                    budget_updates=16, iteration=1)
        self.assertEqual(metrics["gru_sil_updates"], 0)
        self.assertEqual(metrics["gru_sil_rejected_steps"], 1)
        self.assertEqual(trainer.last_gru_sil_sampling["chunks"][0]["rejection_reason"], "stochastic_kl")
        assert_same(self, before, (self.model.state_dict(), trainer.optimizer.state_dict()))

    def test_zero_noise_ppo_sil_resume_is_tensor_exact(self):
        trainer = self.trainer()
        terrain = np.ones((100, 100), np.float32)
        encoder = TrajectoryDataset(self.cfg, terrain, terrain, reference_maps=[terrain])
        with tempfile.TemporaryDirectory() as root:
            archive = EliteTrajectoryArchive(self.cfg, Path(root) / "elites", reference_fingerprint(self.cfg))
            traj = episode(t=8, tail=3)
            traj.step_rewards[:] = .5
            traj.actions[:] = [30., -20.]
            archive.ingest([traj], 1)

            def train_round(iteration):
                buffer = self.collect()
                ppo = trainer.update(self.model, buffer)
                sil = gru_sil_update(trainer, self.model, archive, encoder, buffer,
                                     budget_updates=int(ppo["policy_updates"]), iteration=iteration)
                self.assertGreater(ppo["policy_updates"], 0)
                self.assertGreater(sil["gru_sil_updates"], 0)
                return copy.deepcopy((buffer.episodes, buffer.collection_metadata, ppo,
                    {k: v for k, v in sil.items() if not k.startswith("gru_sil_cache_")},
                    self.model.state_dict(), trainer.optimizer.state_dict(), trainer.checkpoint_state(),
                    capture_random_state(self.worker._rng, self.worker._exploration_rng)))

            train_round(1)
            save_checkpoint(self.model, trainer.optimizer, 1, root, self.cfg, trainer=trainer,
                            deployment_rng=self.worker._rng, exploration_rng=self.worker._exploration_rng)
            saved = torch.load(Path(root) / "ppo_iter_0001.pt", weights_only=False)
            expected = train_round(2)
            self.model = create_actor_critic(self.cfg).eval()
            self.model.load_state_dict(saved["model"])
            trainer = self.trainer()
            trainer.set_pending_optimizer_state(saved["optimizer"])
            trainer.restore_checkpoint_state(saved["trainer_state"], self.model)
            self.worker = self.make_worker()
            restore_random_state(saved["rng_state"], self.worker._rng, self.worker._exploration_rng)
            assert_same(self, expected, train_round(2))


if __name__ == "__main__":
    unittest.main()

"""Cross-component contracts: randomness, recurrent replay and checkpoint continuation."""
import contextlib
import copy
import io
import json
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.config import TrainConfig
from training.dataset import TrajectoryDataset
from training.elite_archive import EliteTrajectoryArchive
from training.experimental_policy import create_actor_critic
from training.experiment_options import ppo_config
from training.gru_sil import gru_sil_update
from training.main_ppo import main, save_checkpoint
from training.policy_audit import audit_policy
from training.ppo_trainer import PPOTrainer
from training.random_state import capture_random_state, restore_random_state, seed_training
from training.reference_fingerprint import reference_fingerprint
from training.rollout import RolloutWorker
from src.tests.training.test_elite_archive import episode
from src.tests.training.test_policy_experiments import Env


def assert_same(test, a, b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a:
            assert_same(test, a[k], b[k])
    elif isinstance(a, (tuple, list)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            assert_same(test, x, y)
    else:
        test.assertEqual(a, b)


class TrainingConsistencyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_training(7)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = replace(TrainConfig.ppo_mainline(), d_model=16, nhead=2, num_layers=1,
                           sparse_geometry=False, recurrent_seq_len=3, ppo_batch_size=3,
                           num_agents=2, deploy_exclusion_check=False, ppo_epochs=1,
                           lr=1e-5, reference_trajectory="fixture", gru_sil_enabled=True,
                           training_seed=7, gru_sil_seed=7, max_grad_norm_adaptive=True,
                           ppo_target_kl_adaptive=True, feature_step_seconds=.02)
        self.encoder = self.make_encoder()

    def make_encoder(self):
        terrain = np.ones((100, 100), np.float32)
        return TrajectoryDataset(self.cfg, terrain, terrain, reference_maps=[terrain])

    def model(self):
        model = create_actor_critic(self.cfg).eval()
        with torch.no_grad():
            model.critic_head[-1].weight.zero_()
            model.critic_head[-1].bias.fill_(-2.)
        return model

    def worker(self):
        # Deterministic fixture physics exercises the actual online collector.
        worker = RolloutWorker(replace(self.cfg, reference_trajectory=None))
        worker.env = Env()
        worker._rng = np.random.default_rng(7)
        worker._ppo_patches = lambda obs, *args: np.stack([self.encoder._build_patch(s) for s in obs])
        worker._compute_contact_batch = lambda obs, prev: (np.zeros((2, 5), np.float32), prev)
        return worker

    def trainer(self):
        trainer = PPOTrainer(self.cfg)
        trainer.device = torch.device("cpu")
        return trainer

    def archive(self):
        return EliteTrajectoryArchive(self.cfg, self.root / "elites", reference_fingerprint(self.cfg))

    def round(self, model, trainer, worker, archive, iteration):
        buffer, _ = worker.collect_ppo(model, 9)
        metrics = trainer.update(model, buffer)
        sil = gru_sil_update(trainer, model, archive, self.encoder, buffer,
                             budget_updates=metrics["n_updates"], iteration=iteration)
        self.assertGreater(sil["gru_sil_updates"], 0)
        # Cache counters/timers intentionally differ for warm, cold and disabled
        # caches; every learning metric and sampled chunk must still be exact.
        sil = {k: v for k, v in sil.items() if not k.startswith("gru_sil_cache_")}
        return copy.deepcopy((buffer.episodes, metrics, sil, trainer.last_gru_sil_sampling,
                              model.state_dict(), trainer.optimizer.state_dict(), trainer.checkpoint_state()))

    def test_continuous_and_resumed_ppo_gru_sil_are_tensor_exact(self):
        model, trainer, worker, archive = self.model(), self.trainer(), self.worker(), self.archive()
        traj = episode(t=8, tail=3)
        traj.step_rewards[:] = .5
        traj.actions[:] = [30., -20.]
        archive.ingest([traj], 1)
        self.round(model, trainer, worker, archive, 1)
        save_checkpoint(model, trainer.optimizer, 1, str(self.root), self.cfg,
                        trainer=trainer, deployment_rng=worker._rng)
        archive.save(1)
        continuous = self.round(model, trainer, worker, archive, 2)
        self.assertEqual(self.encoder._gru_sil_observation_cache.stats["misses"], 0)
        continuous_rng = capture_random_state(worker._rng)
        saved = torch.load(self.root / "ppo_iter_0001.pt", map_location="cpu", weights_only=False)
        model2, trainer2, worker2, archive2 = self.model(), self.trainer(), self.worker(), self.archive()
        model2.load_state_dict(saved["model"])
        trainer2.set_pending_optimizer_state(saved["optimizer"])
        trainer2.restore_checkpoint_state(saved["trainer_state"], model2)
        archive2.restore(self.root / "elites/elites_iter_0001.npz")
        self.encoder = self.make_encoder()  # A restarted process has no observation cache.
        restore_random_state(saved["rng_state"], worker2._rng)
        resumed = self.round(model2, trainer2, worker2, archive2, 2)
        self.assertEqual(self.encoder._gru_sil_observation_cache.stats["misses"], 1)
        assert_same(self, continuous, resumed)
        assert_same(self, continuous_rng, capture_random_state(worker2._rng))

    def test_cache_enabled_and_disabled_ppo_sil_rounds_are_tensor_and_rng_exact(self):
        results = []
        for cache_mb in (0, 1024):
            seed_training(7)
            self.cfg.gru_sil_cache_mb = cache_mb
            self.encoder = self.make_encoder()
            model, trainer, worker, archive = self.model(), self.trainer(), self.worker(), self.archive()
            traj = episode(t=8, tail=3)
            traj.step_rewards[:] = .5
            traj.actions[:] = [30., -20.]
            archive.ingest([traj], 1)
            rows = [self.round(model, trainer, worker, archive, i) for i in (1, 2, 3)]
            results.append((rows, capture_random_state(worker._rng)))
            cache = self.encoder._gru_sil_observation_cache
            self.assertEqual(cache.stats["misses"], int(cache_mb == 0))
            self.assertFalse(cache._round_local)
        assert_same(self, *results)

    def test_audit_and_deterministic_collection_leave_training_rng_unchanged(self):
        model, worker = self.model(), self.worker()
        buffer, _ = worker.collect_ppo(model, 9)
        before = capture_random_state(worker._rng)
        audit_policy(model, buffer, max_samples=3)
        audit_policy(model, buffer, max_samples=None)
        assert_same(self, before, capture_random_state(worker._rng))
        worker.collect_ppo(model, 9, deterministic=True)
        assert_same(self, before, capture_random_state(worker._rng))

    def test_all_random_streams_restore_and_checkpoint_capture_is_neutral(self):
        deployment = np.random.default_rng(19)
        state = capture_random_state(deployment)
        def draws():
            return (random.random(), np.random.randn(5), torch.randn(5), deployment.normal(size=5),
                    torch.randn(5, device="cuda").cpu() if torch.cuda.is_available() else None)
        expected = draws()
        restore_random_state(state, deployment)
        save_checkpoint(self.model(), None, 1, str(self.root), self.cfg, deployment_rng=deployment)
        # Model construction intentionally consumed Torch RNG; compare at the saved boundary.
        saved = torch.load(self.root / "ppo_iter_0001.pt", weights_only=False)
        after_save = draws()
        restore_random_state(saved["rng_state"], deployment)
        assert_same(self, after_save, draws())
        restore_random_state(state, deployment)
        assert_same(self, expected, draws())

    def test_warmup_optimizer_and_adapters_survive_and_lr_override_applies(self):
        model, trainer, worker = self.model(), self.trainer(), self.worker()
        buffer, _ = worker.collect_ppo(model, 9)
        trainer.update_critic_only(model, buffer)
        trainer._kl_adapter.target = .024
        trainer._grad_norm_adapter._norms = [.2, .3]
        trainer._grad_norm_adapter._frozen_norm = .45
        saved = copy.deepcopy(trainer.checkpoint_state())
        other = self.trainer()
        other.restore_checkpoint_state(saved, model)
        assert_same(self, saved, other.checkpoint_state())
        other.config = replace(self.cfg, lr=2e-5)
        other.restore_checkpoint_state(saved, model)
        self.assertEqual(other.critic_optimizer.param_groups[0]["lr"], 2e-5)
        opt = torch.optim.AdamW(model.parameters(), lr=9e-4)
        other.set_pending_optimizer_state(copy.deepcopy(opt.state_dict()))
        other.update(model, buffer)
        self.assertEqual(other.optimizer.param_groups[0]["lr"], 2e-5)

    def test_fresh_exploration_matches_config_but_loading_preserves_learned_std(self):
        for backbone, segment in (("gru", 1), ("gru", 2), ("transformer", 1)):
            cfg = replace(self.cfg, policy_backbone=backbone, action_segment_steps=segment,
                          ppo_init_log_std=-2)
            model = create_actor_critic(cfg)
            torch.testing.assert_close(model.actor_log_std, torch.full_like(model.actor_log_std, -2.))
            with torch.no_grad():
                model.actor_log_std.fill_(-1.234)
            loaded = create_actor_critic(cfg)
            loaded.load_state_dict(model.state_dict())
            assert_same(self, model.actor_log_std, loaded.actor_log_std)

    def test_gru_rejects_silent_phase_noise_and_ou_has_unit_stationary_variance(self):
        with self.assertRaisesRegex(ValueError, "phase_noise"):
            create_actor_critic(replace(self.cfg, phase_noise_enabled=True))
        # Test the implemented sampling law, including the first OU action.
        model = self.model()
        n, phi = 30000, .85
        previous = torch.randn(n, 2)
        mean, std = model._ou_conditional(torch.zeros_like(previous), torch.ones_like(previous), previous, phi)
        current = torch.distributions.Normal(mean, std).sample()
        self.assertAlmostEqual(float(current.var()), 1., delta=.03)
        self.assertAlmostEqual(float((current * previous).mean()), phi, delta=.03)

    def test_resume_cli_preserves_seed_and_delayed_noise_unless_explicitly_changed(self):
        cfg = replace(self.cfg, num_agents=8, steps_per_rollout=1200, delayed_exploration_enabled=True)
        save_checkpoint(self.model(), None, 1, str(self.root), cfg)
        argv = ["main_ppo", "--resume", str(self.root / "ppo_iter_0001.pt"),
                "--run-dir", str(self.root / "next"), "--dry-run"]
        def run(extra):
            output = io.StringIO()
            with patch.object(sys, "argv", argv + extra), contextlib.redirect_stdout(output):
                main()
            return json.loads(output.getvalue())
        result = run([])
        self.assertEqual(result["training_seed"], 7)
        self.assertEqual(result["rng_resume_mode"], "restore")
        self.assertTrue(result["delayed_exploration_enabled"])
        with self.assertRaisesRegex(ValueError, "reset-rng"):
            run(["--seed", "8"])
        result = run(["--seed", "8", "--reset-rng", "--no-delayed-exploration"])
        self.assertEqual(result["rng_resume_mode"], "seed")
        self.assertFalse(result["delayed_exploration_enabled"])

    def test_archive_scores_and_terminal_spatial_inputs_preserve_contracts(self):
        archive = self.archive()
        traj = episode()
        archive.ingest([traj], 1)
        archive.trajectories[0].score = 1.0000000000123
        archive.save(1)
        restored = self.archive()
        restored.restore(self.root / "elites/elites_iter_0001.npz")
        self.assertEqual(archive.trajectories[0].score, restored.trajectories[0].score)
        traj.terminated[-1] = True
        frames = self.encoder.encode_sequence(traj)
        self.assertFalse(frames["patches"][-1].any())
        self.assertTrue(frames["patches"][:-1].any())

    def test_capacity_compressed_archive_preserves_sampling_order_on_resume(self):
        archive = self.archive()
        archive.ingest([episode(t=i + 2, distance=(i + 1) * .2, tail=1) for i in range(35)], 1)
        self.assertEqual(len(archive.trajectories), self.cfg.elite_archive_capacity)
        before = [(len(t.actions), t.score, t.archive_role) for t in archive.trajectories]
        archive.save(1)
        restored = self.archive()
        restored.restore(self.root / "elites/elites_iter_0001.npz")
        self.assertEqual(before, [(len(t.actions), t.score, t.archive_role) for t in restored.trajectories])


if __name__ == "__main__":
    unittest.main()

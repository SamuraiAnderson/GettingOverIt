"""P0 probability, temporal alignment, replay, and actual optimizer regressions."""
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.dataset import DYNAMICS_DIM, build_dynamics
from training.policy_audit import audit_policy
from training.ppo_buffer import PPORolloutBuffer
from training.ppo_trainer import PPOTrainer
from training.rollout import RolloutWorker
from training.contact_features import TerrainContactIndex, _min_distance_to_env, raycast_down_to_polygons


def config(**kwargs):
    return TrainConfig(d_model=16, nhead=2, num_layers=1, context_len=3,
                       num_agents=2, random_deploy=False, deploy_exclusion_check=False,
                       contact_heads_enabled=False, **kwargs)


class Env:
    def reset(self):
        self.index = 0
        return self.obs()

    def obs(self):
        obs = np.zeros((2, 33), dtype=np.float32)
        obs[:, 1] = 5
        obs[:, 0] = self.index * .1
        obs[:, 23] = obs[:, 0] + 1
        obs[:, 24] = 5
        return obs

    def step(self, actions):
        self.index += 1
        return self.obs(), np.array([self.index == 2, False])


class PPOContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        np.random.seed(23)

    def test_boundary_log_probability_and_stable_jacobian(self):
        model = ActorCritic(config()).eval()
        with torch.no_grad():
            model.actor_mean.weight.zero_()
            model.actor_mean.bias.copy_(torch.tensor([20., -20.]))
        cfg = config()
        ps = cfg.patch_size
        d = torch.randn(6, 3, DYNAMICS_DIM)
        p = torch.randn(6, 3, 4, ps, ps)
        a = torch.randn(6, 3, 2)
        mask = torch.tensor([[0., 0., 1.]] * 6)
        z = torch.randn(6, 2)
        for phi in [0., .85, .999]:
            for boost in [0., .7]:
                model.exploration_boost = boost
                av = model.get_action_and_value(d, p, a, mask, z, phi)
                lp, _, _ = model.evaluate_actions(d, p, a, av.action, mask, z, phi,
                                                   raw_actions=av.raw_action)
                self.assertTrue(torch.isfinite(lp).all())
                torch.testing.assert_close(lp, av.log_prob, atol=1e-5, rtol=1e-5)
                self.assertTrue(torch.all(av.action.abs() == 100))
        raw = torch.tensor([[-40., 40.]], requires_grad=True)
        jac = model._log_prob_squash(torch.zeros(1), raw)
        jac.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(jac.item(), 100)  # epsilon clipping used to flatten this.

    def collect(self):
        cfg = config(ppo_batch_size=4, ppo_epochs=1)
        model = ActorCritic(cfg)
        worker = RolloutWorker(cfg)
        worker.env = Env()
        calls = []

        def contact(obs, prev):
            calls.append(obs[0, 0])
            # History-sensitive proxy: duplicate updates become visible immediately.
            row = np.zeros((2, 5), dtype=np.float32)
            row[:, 1] = len(calls) * .1
            return row, prev

        worker._compute_contact_batch = contact
        buffer, trajectories = worker.collect_ppo(model, 4)
        return cfg, model, worker, buffer, trajectories, calls

    def test_one_contact_update_per_frame_terminal_and_reset(self):
        cfg, model, worker, buffer, trajectories, calls = self.collect()
        self.assertEqual(len(calls), 5)  # initial + four new frames; no bootstrap mutation
        self.assertEqual(buffer.size, 6)
        self.assertEqual([len(t.actions) for t in trajectories], [2, 4])
        np.testing.assert_array_equal(buffer._valid_masks[0], [0, 0, 1])
        np.testing.assert_array_equal(buffer._act_histories[0], np.zeros((3, 2)))
        for i, trajectory in enumerate(trajectories):
            base = 0 if i == 0 else 2
            for t in range(len(trajectory.actions)):
                expected = build_dynamics(trajectory.raw_states[t], trajectory.contact[t])
                np.testing.assert_array_equal(buffer._dynamics[base + t][-1], expected)
                if t:
                    np.testing.assert_array_equal(buffer._act_histories[base + t][-1], trajectory.actions[t - 1])
        again, _ = worker.collect_ppo(model, 1)
        np.testing.assert_array_equal(again._valid_masks[0], [0, 0, 1])
        self.assertEqual(audit_policy(model, buffer)["audit_samples"], 6)

    def test_replay_persists_raw_and_context_and_update_moves_weights(self):
        cfg, model, worker, buffer, _, _ = self.collect()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.npz"
            buffer.save(path)
            replay = PPORolloutBuffer.load(path)
            np.testing.assert_array_equal(replay._patches, buffer._patches)
            audit_policy(model, replay)
            model.exploration_boost = .2
            with self.assertRaisesRegex(ValueError, "settings changed"):
                audit_policy(model, replay)
            model.exploration_boost = 0.
            before = model.actor_mean.weight.detach().clone()
            metrics = PPOTrainer(cfg).update(model, replay)
            self.assertGreater(metrics["n_updates"], 0)
            self.assertTrue(np.isfinite(list(metrics.values())).all())
            self.assertFalse(torch.equal(before, model.actor_mean.weight.cpu()))
        buffer.reset()
        self.assertEqual(buffer.size, 0)
        self.assertEqual(len(buffer._raw_actions), 0)

    def test_batch_forward_and_frame_storage_are_equivalent(self):
        cfg, model, _, buffer, _, _ = self.collect()
        batch = next(buffer.get_batches(8))
        with torch.no_grad():
            joint, value = model(batch.dynamics, batch.patches, batch.act_history, batch.valid_mask,
                                 **batch.sparse_kwargs())
            for i in range(len(batch.actions)):
                sparse = {k: None if v is None else v[i:i+1] for k, v in batch.sparse_kwargs().items()}
                dist, v = model(batch.dynamics[i:i+1], batch.patches[i:i+1],
                                batch.act_history[i:i+1], batch.valid_mask[i:i+1], **sparse)
                torch.testing.assert_close(dist.mean[0], joint.mean[i], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(v[0], value[i], atol=1e-5, rtol=1e-5)
        self.assertEqual(
            buffer.patch_storage_bytes(),
            buffer.size * cfg.patch_channels * cfg.patch_size * cfg.patch_size * 4,
        )

    def test_cached_geometry_matches_uncached_geometry(self):
        rng = np.random.default_rng(1)
        polygons = [np.array([[x, y], [x+1, y], [x+1, y+1], [x, y+1]])
                    for x, y in rng.uniform(-5, 5, (40, 2))]
        index = TerrainContactIndex(polygons)
        for x, y in rng.uniform(-5, 5, (40, 2)):
            query = np.array([[x, y], [x+.3, y], [x+.3, y+.3], [x, y+.3]])
            self.assertEqual(_min_distance_to_env(query, polygons), _min_distance_to_env(query, index))
            self.assertEqual(raycast_down_to_polygons(x, y, polygons, 20),
                             raycast_down_to_polygons(x, y, index, 20))

    def test_geometry_migration_preserves_checkpoint_inputs(self):
        from training.geometry_profile import configure_geometry
        cfg = config()
        self.assertFalse(configure_geometry(cfg))
        self.assertEqual(cfg.tip_angle_offset, -180.)
        old = {"model": {}, "config": {"tip_angle_offset": -90., "body_angle_offset": -90.}}
        self.assertFalse(configure_geometry(cfg, old))
        self.assertEqual((cfg.tip_angle_offset, cfg.body_angle_offset), (-90., -90.))
        self.assertTrue(configure_geometry(cfg, old, "native-pose-v1"))
        self.assertEqual((cfg.tip_angle_offset, cfg.body_angle_offset), (-180., -84.49833))
        calibrated = {"config": {"tip_angle_offset": -180., "body_angle_offset": -84.49833}}
        self.assertFalse(configure_geometry(cfg, calibrated, "native-pose-v1"))
        self.assertFalse(configure_geometry(cfg, {"model": {}}))
        self.assertEqual((cfg.tip_angle_offset, cfg.body_angle_offset), (-90., -90.))
        with self.assertRaisesRegex(ValueError, "Unknown geometry"):
            configure_geometry(cfg, profile="unknown")

    def test_evaluation_needs_progress_and_dwell_and_rejects_shortcut(self):
        from training.local_evaluation import score_corridor_episode, summarize
        from training.dataset import Trajectory
        from training.reference_corridor import CorridorSegment, ReferenceCorridor
        cfg = config(reference_anchor_id="player_origin")  # This fixture deliberately supplies legacy33.
        route = ReferenceCorridor([CorridorSegment(np.array([[0., 5.], [20., 5.]]), np.ones(1))], cfg)
        states = np.zeros((8, 33), dtype=np.float32)
        states[:, 0] = [0, 1, 2, 3, 4, 5, 5, 5]
        states[:, 1] = 5
        trajectory = Trajectory(states, np.zeros((7, 2)), contact=np.zeros((8, 5)))
        metrics, trace = score_corridor_episode(trajectory, route, 5, 3)
        self.assertEqual(metrics["arrival_steps"], 5)
        self.assertTrue(metrics["success"])
        self.assertEqual(trace[0]["contact_enter"], [0, 0, 0])
        self.assertEqual(summarize([{**metrics, "mode": "deterministic"}])["deterministic"]["success_rate"], 1)
        self.assertEqual(summarize([{**metrics, "mode": "deterministic"}])["deterministic"]["arrival_count"], 1)
        states[1:, 0] = 6  # jump exceeds max-step; no credit
        self.assertFalse(score_corridor_episode(trajectory, route, 5, 1)[0]["success"])

    def test_diagnostic_wire_packet_and_known_flags(self):
        import json
        import struct
        from env import GoiEnv
        packet = {"version": 1, "agents": [{"control": [0] * 6, "controlKnown": [False] * 6,
                                           "contact": [0] * 3, "contactKnown": [False] * 3}]}
        payload = json.dumps(packet).encode()
        wire = bytearray(b"\xff" + struct.pack("<I", len(payload)) + payload)
        env = GoiEnv(num_agents=1)
        sent = []
        env._send_all = sent.append
        def recv(n):
            result = bytes(wire[:n])
            del wire[:n]
            return result
        env._recv_exact = recv
        self.assertEqual(env.observation_diagnostics(), packet)
        self.assertEqual(sent, [b"D"])
        self.assertEqual(len(wire), 0)
        wire.extend(b"\x00")
        with self.assertRaises(RuntimeError):
            env.observation_diagnostics()


if __name__ == "__main__":
    unittest.main()

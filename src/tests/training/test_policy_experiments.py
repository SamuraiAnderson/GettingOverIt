"""Behavioral regressions for relation/event/geometry/GRU/option PPO candidates."""
from dataclasses import asdict
import sys
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.config import TrainConfig
from training.dataset import DYNAMICS_DIM
from training.experimental_policy import create_actor_critic
from training.experimental_rollout import segment_action
from training.policy_features import FeatureBuilder, SurfaceIndex, relations, feature_layout, scale_report
from training.decision_buffer import duration_gae
from training.policy_audit import audit_policy
from training.ppo_buffer import PPORolloutBuffer
from training.ppo_trainer import PPOTrainer
from training.rollout import RolloutWorker


def config(**kwargs):
    return TrainConfig(d_model=16, nhead=2, num_layers=1, context_len=4,
        num_agents=2, random_deploy=False, deploy_exclusion_check=False,
        contact_heads_enabled=False, ppo_epochs=1, ppo_batch_size=4, recurrent_seq_len=3,
        feature_step_seconds=.02, **kwargs)


def state(t=0):
    s = np.zeros(33, np.float32)
    for pair, offset in ((0, 0), (5, 0), (10, .2), (15, .5), (19, 1), (23, 2), (29, 3)):
        s[pair:pair+2] = (t*.1+offset, 5.)
    s[30] += 1
    s[2:4], s[21:23], s[25:27], s[31:33] = (1, 0), (1, 1), (1, 2), (3, 4)
    s[28] = t*.02
    return s


class Env:
    def __init__(self):
        self.index = 0
        self.steps = 0

    def reset(self):
        self.index = 0
        return np.stack([state(), state()])

    def step(self, actions):
        self.index += 1
        self.steps += 1
        return np.stack([state(self.index)] * 2), np.array([self.index == 6, False])

    def observation_diagnostics(self):
        active = 2 <= self.index < 6
        row = {"physicsContact": [int(active), 0, 0, 0], "physicsKnown": [True]*4}
        return {"version": 2, "aligned": True, "sequence": self.index + 100,
                "fixedDeltaTime": .02, "stepFrames": 1, "agents": [row, row]}


class PolicyExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        np.random.seed(42)

    def collect(self, **overrides):
        cfg = config(**overrides)
        model = create_actor_critic(cfg).eval()
        worker = RolloutWorker(cfg)
        worker.env = Env()
        patch_rng = np.random.default_rng(9)
        # Nonzero images exercise CNN gradients; an empty-terrain fixture would
        # give zero first-convolution weight gradients regardless of correctness.
        worker._ppo_patches = lambda *args: patch_rng.normal(
            0, .1, (2, cfg.patch_channels, cfg.patch_size, cfg.patch_size)
        ).astype(np.float32)
        calls = []
        def contact(obs, previous):
            calls.append(obs[0, 0])
            return np.zeros((2, 5), np.float32), previous
        worker._compute_contact_batch = contact
        buffer, trajectories = worker.collect_ppo(model, 9)
        return cfg, model, worker, buffer, trajectories, calls

    def test_relations_have_units_translation_invariance_and_degenerate_masks(self):
        cfg = config()
        r = relations(state(), cfg)
        np.testing.assert_allclose(r[:4], [1/3.5]*4)
        np.testing.assert_allclose(r[4:8], [.2]*4)
        self.assertAlmostEqual(float(r[10]), .1)
        np.testing.assert_allclose(r[12:], [0, -1, 1, 1])
        shifted = state()
        for pair in (0, 5, 10, 15, 19, 23, 29):
            shifted[pair:pair+2] += [50, -20]
        np.testing.assert_allclose(relations(shifted, cfg), r)
        shifted[23:25] = shifted[19:21]
        degenerate = relations(shifted, cfg)
        self.assertEqual(degenerate[14], 0)
        self.assertEqual(degenerate[10], 0)
        self.assertTrue(np.isfinite(degenerate).all())

    def test_surface_distance_normal_winding_velocity_and_missing(self):
        polygon = np.array([[-5., -1.], [5., -1.], [5., 0.], [-5., 0.]])
        first, reverse = SurfaceIndex([polygon]), SurfaceIndex([polygon[::-1]])
        expected = [.25/3, 0, 1, -.3, -.2, 1]
        np.testing.assert_allclose(first.query([0, .25], np.array([2, -3]), 3, 10), expected)
        np.testing.assert_allclose(reverse.query([0, .25], np.array([2, -3]), 3, 10), expected)
        self.assertLess(first.query([0, -.1], np.zeros(2), 3, 10)[0], 0)
        np.testing.assert_array_equal(first.query([20, 20], np.zeros(2), 3, 10), np.zeros(6))

    def test_feature_builder_uses_first_33_of_anchor35(self):
        cfg = config(relation_features=True)
        raw33 = state()
        raw35 = np.concatenate([raw33, [1.0, 2.0]]).astype(np.float32)
        a = FeatureBuilder(cfg).step(raw33)[0]
        b = FeatureBuilder(cfg).step(raw35)[0]
        np.testing.assert_array_equal(a, b)

    def test_diagnostic_masks_distinguish_native_contact_from_surface_availability(self):
        cfg = config(relation_features=True, event_features=True, geometry_features=True, trend_features=True)
        # Native tip contact is known while there is no nearby exported surface.
        extra, _ = FeatureBuilder(cfg).step(state(), [1, 0, 0], [1, 1, 1])
        report = scale_report(extra[None], cfg)
        self.assertEqual(len(report), 75)
        self.assertEqual(report["events.tip_known"]["rms"], 1.)
        self.assertEqual(report["geometry.tip_known"]["rms"], 0.)

    def test_shared_online_offline_events_trends_reset_and_unknown(self):
        cfg = config(relation_features=True, event_features=True, trend_features=True)
        builder = FeatureBuilder(cfg)
        states = np.stack([state(t) for t in range(40)])
        contacts = np.zeros((40, 3)); contacts[:5, 0] = 1; contacts[8:, 0] = 1
        known = np.ones((40, 3), dtype=bool); known[7:9, 0] = False
        online = np.stack([builder.step(s, contacts[t], known[t])[0] for t, s in enumerate(states)])
        np.testing.assert_array_equal(builder.sequence(states, contacts, known), online)
        layout, _ = feature_layout(cfg)
        events = online[:, layout["events"]]
        self.assertEqual(events[0, 1], 0)  # reset while already in contact: no fake enter
        self.assertEqual(events[5, 2], 1)
        self.assertEqual(events[9, 1], 0)  # unknown interval: no inferred edge
        self.assertTrue((online[:4, layout["trends"]][:, :10] == 0).all())
        self.assertEqual(online[4, layout["trends"]][9], 1)
        builder.reset()
        self.assertEqual(builder.step(state(), [1, 0, 0], [1, 1, 1])[0][layout["events"]][1], 0)

    def test_gru_streaming_equals_sequence_and_reset_mask(self):
        cfg = config(policy_backbone="gru", relation_features=True)
        model = create_actor_critic(cfg).eval()
        width = DYNAMICS_DIM + feature_layout(cfg)[1]
        dyn = torch.randn(2, 8, width)
        patches = torch.randn(2, 8, 4, cfg.patch_size, cfg.patch_size)
        actions = torch.randn(2, 8, 2)
        with torch.no_grad():
            full, h_full = model.recurrent_features(dyn, patches, actions)
            hidden, streamed = None, []
            for t in range(8):
                out, hidden = model.recurrent_features(dyn[:, t:t+1], patches[:, t:t+1], actions[:, t:t+1], hidden)
                streamed.append(out)
            torch.testing.assert_close(full, torch.cat(streamed, 1), atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(hidden, h_full)
            reset = torch.zeros(2, 8); reset[:, 4] = 1
            masked, _ = model.recurrent_features(dyn, patches, actions, reset_mask=reset)
            fresh, _ = model.recurrent_features(dyn[:, 4:], patches[:, 4:], actions[:, 4:])
            torch.testing.assert_close(masked[:, 4:], fresh)
            valid = torch.ones(2, 8); valid[:, :4] = 0
            padded, _ = model.recurrent_features(dyn, patches, actions, valid_mask=valid)
            torch.testing.assert_close(padded[:, 4:], fresh)

    def test_event_branch_has_gradient_paths_to_actor_and_critic(self):
        cfg = config(event_branch=True)
        model = create_actor_critic(cfg).eval()
        width = DYNAMICS_DIM + feature_layout(cfg)[1]
        dyn = torch.randn(2, 4, width)
        dist, value = model(dyn, torch.randn(2, 4, 4, cfg.patch_size, cfg.patch_size), torch.zeros(2, 4, 2))
        (dist.mean.sum() + value.sum()).backward()
        self.assertGreater(model.event_to_backbone.weight.grad.abs().sum(), 0)
        self.assertGreater(model.event_to_heads.weight.grad.abs().sum(), 0)
        self.assertGreater(model.event_encoder[0].weight.grad.abs().sum(), 0)

    def test_variable_duration_gae_uses_elapsed_physics_steps(self):
        g = .9
        rewards = [1+g*2+g*g*3, 4]
        advantage, returns = duration_gae(rewards, [.1, .2], [0, 1], [3, 1], 999, g, 1.)
        np.testing.assert_allclose(returns, [1+g*2+g*g*3+g**3*4, 4], atol=1e-6)
        a, _ = duration_gae([2], [1], [0], [3], 5, g, .95)
        self.assertAlmostEqual(float(a[0]), 2 + g**3*5 - 1, places=5)

    def test_segment_interpolation_respects_bounds(self):
        params = np.array([100., -100., -100., 100.], np.float32)
        np.testing.assert_array_equal(segment_action(params, 0, 4), params[:2])
        np.testing.assert_array_equal(segment_action(params, 3, 4), params[2:])
        self.assertTrue(np.all(np.abs(segment_action(params, 1, 4)) <= 100))

    def test_transformer_segments_interrupt_terminal_and_replay(self):
        cfg, model, worker, buffer, trajectories, calls = self.collect(
            relation_features=True, geometry_features=True, trend_features=True, event_branch=True,
            normalize_action_history=True, action_segment_steps=4)
        self.assertEqual(worker.env.steps, 9)
        self.assertEqual(len(calls), 10)
        self.assertEqual([len(t.actions) for t in trajectories], [6, 9])
        self.assertEqual(buffer.environment_steps, 15)
        np.testing.assert_array_equal(buffer.episodes[0]["durations"], [2, 4])
        np.testing.assert_array_equal(buffer.episodes[1]["durations"], [2, 4, 3])
        self.assertEqual(buffer.size, 5)  # one probability per decision, not fifteen samples
        self.assertEqual(buffer.episodes[0]["actions"].shape[1], 4)
        audit_policy(model, buffer)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sequence.npz"
            buffer.save(path)
            restored = PPORolloutBuffer.load(path)
            self.assertEqual(restored.environment_steps, 15)
            audit_policy(model, restored)
        self.assertGreater(worker.last_profile["rollout_contact_interruptions"], 0)

    def test_gru_full_prefix_replay_and_real_optimizer_update(self):
        cfg, model, worker, buffer, _, _ = self.collect(policy_backbone="gru", relation_features=True,
            event_branch=True, action_segment_steps=4, normalize_action_history=True)
        audit_policy(model, buffer, batch_size=1)  # every noninitial decision needs burn-in
        before = model.actor_mean.weight.detach().clone()
        cnn_before = next(model.spatial_encoder.parameters()).detach().clone()
        metrics = PPOTrainer(cfg).update(model, buffer)
        self.assertGreater(metrics["n_updates"], 0)
        self.assertFalse(torch.equal(before, model.actor_mean.weight.detach().cpu()))
        self.assertFalse(torch.equal(cnn_before, next(model.spatial_encoder.parameters()).detach().cpu()))
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        with torch.no_grad():
            expected = {}
            device = next(model.parameters()).device
            for e, episode in enumerate(buffer.episodes):
                inputs = [torch.tensor(episode[name], device=device)[None] for name in ("dynamics", "patches", "act_history")]
                features, _ = model.recurrent_features(*inputs)
                indices = torch.as_tensor(episode["decision_steps"], device=device)
                lp, _, _ = model.evaluate_features(features[0, indices],
                    torch.tensor(episode["raw_actions"], device=device),
                    torch.tensor(episode["noise_prev"], device=device), buffer.policy_context["noise_phi"])
                expected.update({(e, i): p for i, p in enumerate(lp)})
            for batch, lp, _, _ in buffer.evaluated_batches(model, 1):
                torch.testing.assert_close(lp, torch.stack([expected[row] for row in batch.row_ids]), atol=1e-5, rtol=1e-5)

    def test_per_step_feature_variant_and_checkpoint_migration(self):
        base = create_actor_critic(config()).eval()
        cfg = config(relation_features=True, event_branch=True)
        variant = create_actor_critic(cfg).eval()
        variant.initialize_from(base.state_dict())
        dyn = torch.randn(2, 4, DYNAMICS_DIM)
        patches = torch.randn(2, 4, 4, cfg.patch_size, cfg.patch_size)
        actions = torch.randn(2, 4, 2)
        with torch.no_grad():
            a, v = base(dyn, patches, actions)
            b, w = variant(torch.cat([dyn, torch.randn(2, 4, feature_layout(cfg)[1])], -1), patches, actions)
        torch.testing.assert_close(a.mean, b.mean)
        torch.testing.assert_close(v, w)
        _, model, worker, buffer, _, _ = self.collect(relation_features=True)
        self.assertEqual(buffer.size, buffer.environment_steps)
        audit_policy(model, buffer)

    def test_incompatible_resume_requires_explicit_initialization(self):
        from training.main_ppo import parse_args
        from training.experiment_options import apply_experiment_arguments
        args = parse_args(["--features", "relations", "--policy-backbone", "gru"])
        with self.assertRaisesRegex(ValueError, "use --init-from"):
            apply_experiment_arguments(config(), args, {"config": asdict(config()), "model": {}})
        cfg = config()
        apply_experiment_arguments(cfg, args)
        self.assertTrue(cfg.relation_features)
        self.assertEqual(cfg.policy_backbone, "gru")
        self.assertFalse(cfg.sil_enabled)


if __name__ == "__main__":
    unittest.main()

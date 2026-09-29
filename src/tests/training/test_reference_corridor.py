"""Offline behavioral checks for corridor rewards and the actual PPO collection path."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.dataset import build_patch
from training.reference_corridor import CorridorSegment, ReferenceCorridor
from training.reward import RewardNormalizer, step_reward
from training.rollout import RolloutWorker, _build_patches_batch


def route(*parts, **kwargs):
    config = TrainConfig(
        reference_corridor_radius=1.0,
        reference_progress_mode=kwargs.pop("reference_progress_mode", "legacy_credit_v1"),
        **kwargs,
    )
    segments = [CorridorSegment(np.asarray(p, dtype=float), np.ones(len(p) - 1)) for p in parts]
    return ReferenceCorridor(segments, config)


def write_route(path, parts, **meta):
    payload = {"segments": [[{"x": p[0], "y": p[1], "source": "observed"} for p in part]
                            for part in parts], **meta}
    path.write_text(json.dumps(payload), encoding="utf-8")


class CorridorRewardTests(unittest.TestCase):
    def setUp(self):
        self.route = route([(0, 0), (20, 0)])

    def test_progress_inside_beats_outside_without_centerline_bias(self):
        center = self.route.tracker(np.array([0, 0])).step(np.array([1, 0]))
        lateral = self.route.tracker(np.array([0, .8])).step(np.array([1, .8]))
        outside = self.route.tracker(np.array([0, 2])).step(np.array([1, 2]))
        self.assertEqual(center.reward, 1)
        self.assertEqual(center.reward, lateral.reward)
        self.assertEqual(outside.progress, 0)
        self.assertLess(outside.reward, 0)
        self.assertLessEqual(outside.deviation_penalty, .01)

    def test_stationary_lateral_and_backtracking_cannot_farm_reward(self):
        t = self.route.tracker(np.array([0, 0]))
        self.assertEqual(t.step(np.array([1, 0])).reward, 1)
        for point in [[1, .8], [1, 0], [0, 0], [1, 0], [1, 0]]:
            self.assertEqual(t.step(np.array(point)).reward, 0)
        self.assertEqual(t.step(np.array([2, 0])).reward, 1)

    def test_outside_detour_is_not_paid_on_reentry(self):
        t = self.route.tracker(np.array([0, 0]))
        for point in [[1, 2], [2, 2], [2, 0]]:
            self.assertEqual(t.step(np.array(point)).progress, 0)
        self.assertEqual(t.step(np.array([2.5, 0])).progress, .5)
        self.assertEqual(t.total_progress, .5)

    def test_agents_and_deployment_have_independent_high_water_marks(self):
        a, b = self.route.tracker(np.array([0, 0])), self.route.tracker(np.array([8, 0]))
        self.assertEqual(b.step(np.array([8, 0])).reward, 0)
        self.assertEqual(a.step(np.array([1, 0])).reward, 1)
        self.assertEqual(b.step(np.array([9, 0])).reward, 1)
        self.assertEqual(self.route.tracker(np.array([9, 0])).step(np.array([9, 0])).reward, 0)

    def test_jump_is_not_rewarded_now_or_next_step(self):
        t = self.route.tracker(np.array([0, 0]))
        self.assertEqual(t.step(np.array([10, 0])).reward, 0)
        self.assertEqual(t.step(np.array([10, 0])).reward, 0)
        self.assertEqual(t.step(np.array([11, 0])).reward, 1)

    def test_coarse_endpoint_accepts_small_coordinate_offset_without_bonus(self):
        r = route([(0, 0), (4, 0)], [(10, 0), (14, 0)])
        t = r.tracker(np.array([2, .5]))
        result = t.step(np.array([3.5, .5]))
        self.assertEqual(result.progress, 1.5)
        self.assertTrue(t.waiting)
        self.assertEqual(t.step(np.array([6, .5])).reward, 0)

    def test_crossing_cannot_skip_a_loop_or_creep_while_stationary(self):
        r = route([(0, 0), (4, 0), (4, 4), (0, 4), (0, 0), (0, -4)])
        t = r.tracker(np.array([0, 0]))
        for _ in range(50):
            self.assertEqual(t.step(np.array([0, 0])).progress, 0)
        self.assertEqual(t.step(np.array([0, -.5])).progress, 0)
        self.assertLess(t.progress, 1)

    def test_gaps_have_no_line_reward_or_penalty_and_only_next_start_connects(self):
        r = route([(0, 0), (2, 0)], [(8, 0), (12, 0)], reference_max_step=20)
        t = r.tracker(np.array([1, 0]))
        self.assertEqual(t.step(np.array([2, 0])).progress, 1)
        self.assertTrue(t.waiting)
        self.assertEqual(t.guidance_index, 1)
        for point in [[5, 0], [12, 0]]:
            result = t.step(np.array(point))
            self.assertTrue(result.gap)
            self.assertEqual(result.reward, 0)
            self.assertEqual(t.segment_index, 0)
        self.assertEqual(t.step(np.array([8, 0])).reward, 0)
        self.assertEqual(t.segment_index, 1)
        self.assertEqual(t.step(np.array([9, 0])).reward, 1)
        t.step(np.array([12, 0]))
        self.assertIsNone(t.guidance_index)
        self.assertEqual(t.step(np.array([13, 0])).reward, 0)

    def test_reference_reward_bypasses_old_rewards_and_channel_normalization(self):
        cfg = self.route.config
        cfg.reference_trajectory = "test.json"
        prev = np.zeros(33)
        curr = prev.copy()
        curr[0] = 1
        class ForbiddenMap:
            def query(self, *args):
                raise AssertionError("Old efficiency reward must not run")
        normalizer = RewardNormalizer()
        reward, _ = step_reward(prev, curr, ForbiddenMap(), cfg, 0, normalizer,
                                corridor=self.route.tracker(prev))
        self.assertEqual(reward, 1)
        curr[1] = -11
        reward, _ = step_reward(prev, curr, ForbiddenMap(), cfg, 0,
                                corridor=self.route.tracker(prev))
        self.assertEqual(reward, cfg.water_penalty)
        with self.assertRaises(ValueError):
            step_reward(prev, curr, None, cfg, 0)

    def test_legacy_reward_is_unchanged_when_disabled(self):
        cfg = TrainConfig()
        prev, curr = np.zeros(33), np.zeros(33)
        curr[:2] = [1, 1]
        reward, high = step_reward(prev, curr, None, cfg, 0)
        self.assertEqual(reward, 1 + cfg.progress_wx)
        self.assertEqual(high, 1)

    def test_file_simplification_weights_validation_and_disconnected_parts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "route.json"
            payload = {"segments": [[
                {"x": 0, "y": 0, "frame": 10, "source": "observed"},
                {"x": 1, "y": .01, "frame": 11, "source": "interpolated"},
                {"x": 2, "y": 0, "frame": 12, "source": "observed"}],
                [{"x": 10, "y": 0}],
                [{"x": 15, "y": 0}, {"x": 20, "y": 0}]]}
            path.write_text(json.dumps(payload))
            r = ReferenceCorridor.from_file(path, TrainConfig(reference_progress_mode="legacy_credit_v1"))
            self.assertEqual(len(r.segments), 2)
            self.assertEqual(r.metadata["points"], 4)
            self.assertEqual(r.tracker(np.array([0, 0])).step(np.array([1, 0])).reward, .5)
            payload["segments"][0][1]["frame"] = 13
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                ReferenceCorridor.from_file(path, TrainConfig())
            write_route(path, [[(0, 0), (0, 0)]])
            with self.assertRaises(ValueError):
                ReferenceCorridor.from_file(path, TrainConfig())
        with self.assertRaises(ValueError):
            ReferenceCorridor(self.route.segments, TrainConfig(reference_corridor_radius=0))

    def test_guidance_keeps_shape_and_shared_observation_builder(self):
        r = route([(0, 0), (4, 0)], [(14, 0), (18, 0)])
        cfg = r.config
        maps = r.guidance_maps((50, 50), -20, -20, 1)
        self.assertGreater(maps[0][20, 23], maps[0][20, 20])
        self.assertEqual(maps[0][20, 29], 0)  # No drawn bridge across missing interval.
        obs = np.zeros((2, 33), dtype=np.float32)
        terrain = np.zeros((50, 50), dtype=bool)
        online = _build_patches_batch(obs, None, cfg, -20, -20, terrain,
                                      guidance_arrays={0: maps[0], 1: maps[1]})
        expected = build_patch(obs[0], terrain, maps[0], cfg, -20, -20)
        np.testing.assert_array_equal(online[0], expected)
        self.assertEqual(online.shape, (2, cfg.patch_channels, cfg.patch_size, cfg.patch_size))
        self.assertFalse(np.array_equal(online[0, 0], online[1, 0]))

    def test_cli_resume_preserves_reference_semantics_and_can_disable(self):
        from training.main_ppo import configure_reference, parse_args, save_checkpoint
        args = parse_args(["--reference-trajectory", "--reference-corridor-radius", "2"])
        cfg = TrainConfig()
        configure_reference(cfg, args, {})
        self.assertTrue(Path(cfg.reference_trajectory).is_file())
        self.assertEqual(cfg.reference_corridor_radius, 2)
        self.assertFalse(cfg.random_deploy)
        self.assertFalse(cfg.sil_enabled)
        with tempfile.TemporaryDirectory() as temp:
            save_checkpoint(torch.nn.Linear(2, 2), None, 1, temp, cfg, {"sha256": "example"})
            saved = torch.load(Path(temp) / "ppo_iter_0001.pt", weights_only=False)
            restored = TrainConfig()
            configure_reference(restored, parse_args([]), saved)
            self.assertEqual(restored.reference_trajectory, cfg.reference_trajectory)
            self.assertEqual(restored.reference_corridor_radius, 2)
            disabled = TrainConfig()
            configure_reference(disabled, parse_args(["--no-reference-trajectory"]), saved)
            self.assertIsNone(disabled.reference_trajectory)
            self.assertTrue(disabled.sil_enabled)

    def test_actual_video_route_loads_as_separate_coarse_segments(self):
        path = ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json"
        r = ReferenceCorridor.from_file(path, TrainConfig())
        self.assertEqual(len(r.segments), 6)  # One isolated detection is not a line.
        self.assertLess(r.metadata["points"], r.metadata["raw_points"])
        self.assertFalse(r.metadata["training_ready"])
        for segment in r.segments:
            t = r.tracker(segment.points[0])
            self.assertTrue(np.isfinite(t.step(segment.points[0]).reward))

    def test_live_camera_tas_starts_on_live_pot(self):
        from training.reference_anchor import LIVE_CAMERA_REFERENCE
        payload = json.loads(LIVE_CAMERA_REFERENCE.read_text(encoding="utf-8"))
        start = np.array([payload["segments"][0][0]["x"], payload["segments"][0][0]["y"]], dtype=float)
        live = np.array(payload["spawn_check"]["live_detected_pot"], dtype=float)
        self.assertLess(np.linalg.norm(start - live), 0.02)
        self.assertGreater(payload["spawn_check"]["dist_old_to_live_pot_m"], 0.2)
        r = ReferenceCorridor.from_file(LIVE_CAMERA_REFERENCE, TrainConfig())
        self.assertGreaterEqual(len(r.segments), 1)
        self.assertEqual(payload["status"], "live_camera_reprojection_v1")

    def test_game_world_tas_locks_to_unity_pot(self):
        from training.reference_anchor import DEFAULT_REFERENCE_TRAJECTORY
        payload = json.loads(DEFAULT_REFERENCE_TRAJECTORY.read_text(encoding="utf-8"))
        start = np.array([payload["segments"][0][0]["x"], payload["segments"][0][0]["y"]], dtype=float)
        unity = np.array(payload["spawn_check"]["unity_transform_point_b"], dtype=float)
        self.assertEqual(payload["status"], "game_world_spawn_lock_v1")
        self.assertLess(np.linalg.norm(start - unity), 1e-9)
        self.assertLess(payload["spawn_check"]["dist_to_unity_b_m"], 1e-9)
        # Live detector residual is measurement, not the lock target.
        self.assertLess(payload["spawn_check"]["dist_to_live_detected_m"], 0.03)
        r = ReferenceCorridor.from_file(DEFAULT_REFERENCE_TRAJECTORY, TrainConfig())
        self.assertGreaterEqual(len(r.segments), 1)

    def test_credit_ablation_survives_checkpoint_resume_without_default_reset(self):
        from training.main_ppo import configure_credit, configure_reference, parse_args, save_checkpoint
        cfg = TrainConfig()
        args = parse_args(["--gamma", ".998", "--gae-lambda", ".995",
                           "--reference-trajectory", "--reference-deviation-weight", ".001"])
        configure_credit(cfg, args, {})
        configure_reference(cfg, args, {})
        with tempfile.TemporaryDirectory() as temp:
            save_checkpoint(torch.nn.Linear(2, 2), None, 1, temp, cfg)
            saved = torch.load(Path(temp) / "ppo_iter_0001.pt", weights_only=False)
        restored = TrainConfig()
        configure_credit(restored, parse_args([]), saved)
        configure_reference(restored, parse_args([]), saved)
        self.assertEqual((restored.gamma, restored.gae_lambda, restored.reference_deviation_weight),
                         (.998, .995, .001))
        with self.assertRaisesRegex(ValueError, "fresh run"):
            configure_credit(restored, parse_args(["--gamma", ".99"]), saved)

    def test_invalid_credit_and_penalty_fail_before_game_launch(self):
        from training.main_ppo import configure_credit, configure_reference, parse_args
        for flag, value in [("--gamma", "nan"), ("--gamma", "0"),
                            ("--gae-lambda", "1.1"), ("--gae-lambda", "-0.1")]:
            with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                configure_credit(TrainConfig(), parse_args([flag, value]), {})
        for value in ("nan", "inf", "-0.01"):
            with self.subTest(penalty=value), self.assertRaises(ValueError):
                configure_reference(TrainConfig(), parse_args(["--reference-deviation-weight", value]), {})


class PPOCollectionTests(unittest.TestCase):
    def test_multi_agent_rollout_uses_corridor_and_stops_terminal_agent(self):
        class Env:
            def reset(self):
                self.index = 0
                return np.zeros((2, 33), dtype=np.float32)

            def step(self, actions):
                self.index += 1
                obs = np.zeros((2, 33), dtype=np.float32)
                obs[:, 0] = self.index
                if self.index == 2:
                    obs[0, 1] = -11
                return obs, np.zeros(2, dtype=bool)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
                self.config = TrainConfig()
                self.exploration_boost = 0.0

            def get_action_and_value(self, *args, **kwargs):
                n = len(args[0])
                return SimpleNamespace(action=torch.zeros(n, 2), log_prob=torch.zeros(n),
                                       value=torch.zeros(n), noise=torch.zeros(n, 2),
                                       raw_action=torch.zeros(n, 2))

            def forward(self, *args, **kwargs):
                return torch.zeros(1, 2), torch.zeros(1)

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "route.json"
            write_route(path, [[(0, 0), (10, 0)]])
            cfg = TrainConfig(
                reference_trajectory=str(path), num_agents=2, context_len=2,
                random_deploy=False, deploy_exclusion_check=False,
                reference_progress_mode="legacy_credit_v1",
                reference_anchor_id="player_origin",
                reference_anchor_verified=False,
                reference_anchor_calibration_sha256=None,
                tcp_protocol="legacy33",
            )
            worker = RolloutWorker(cfg)
            env = Env()
            worker.env = env
            worker._reset_and_deploy = env.reset
            worker._get_stable_agents = lambda obs: {0, 1}
            worker._compute_contact_batch = lambda obs, prev: (np.zeros((2, 5), dtype=np.float32), prev)
            worker.set_terrain_info(np.ones((50, 50), dtype=bool), -20, -20)
            buffer, trajs = worker.collect_ppo(Model(), 3)
            self.assertEqual(buffer.size, 5)
            np.testing.assert_allclose(buffer._rewards, [1, cfg.water_penalty, 1, 1, 1])
            self.assertEqual([len(t.actions) for t in trajs], [2, 3])
            self.assertEqual(buffer._dones, [False, True, False, False, False])
            self.assertEqual(worker.last_corridor_metrics["corridor_progress_m"], 4)
            self.assertTrue(np.isfinite(buffer._returns).all())
            self.assertGreater(float(buffer._patches[0][-1, 0].max()), 0)
            again, _ = worker.collect_ppo(Model(), 1)
            np.testing.assert_allclose(again._rewards, [1, 1])


class ConfirmedEndpointTests(unittest.TestCase):
    def test_outside_detour_is_caught_up_at_the_next_event(self):
        corridor = route([(0, 0), (20, 0)], reference_progress_mode="confirmed_endpoint_v2")
        tracker = corridor.tracker(np.array([0.0, 0.0]))
        for point in ([2.0, 2.0], [2.0, 0.0]):
            step = tracker.step(np.array(point))
            self.assertEqual(step.legacy_credit, 0.0)
            self.assertEqual(step.confirmed_progress, 0.0)
        paid = tracker.step(np.array([2.5, 0.0]))
        self.assertAlmostEqual(paid.legacy_credit, 0.5)
        self.assertTrue(paid.credit_event)
        self.assertAlmostEqual(paid.confirmed_progress, 2.5)
        self.assertAlmostEqual(paid.delta_d, 2.5)
        self.assertAlmostEqual(paid.reward, 2.5)
        self.assertAlmostEqual(tracker.total_progress, 0.5)
        self.assertAlmostEqual(sum(tracker_delta(corridor, [[0, 0], [2, 2], [2, 0], [2.5, 0]])), 2.5)

    def test_jump_invalidates_the_trajectory(self):
        corridor = route([(0, 0), (20, 0)], reference_progress_mode="confirmed_endpoint_v2")
        tracker = corridor.tracker(np.array([0.0, 0.0]))
        jumped = tracker.step(np.array([10.0, 0.0]))
        self.assertTrue(jumped.invalid)
        self.assertEqual(jumped.reward, 0.0)
        nxt = tracker.step(np.array([11.0, 0.0]))
        self.assertTrue(nxt.invalid)
        self.assertEqual(nxt.reward, 0.0)
        self.assertEqual(tracker.confirmed_d, 0.0)

    def test_no_event_cap_is_not_the_first_action(self):
        from training.reference_corridor import trajectory_endpoint
        from training.dataset import Trajectory
        corridor = route(
            [(0, 0), (20, 0)], reference_progress_mode="confirmed_endpoint_v2",
            reference_anchor_id="player_origin", reference_anchor_verified=False,
            reference_anchor_calibration_sha256=None, tcp_protocol="legacy33",
        )
        tracker = corridor.tracker(np.array([0.0, 0.0]))
        tracker.step(np.array([0.0, 0.0]))
        played = replay_states(corridor, [[0, 0], [0, 0]])
        self.assertEqual(int(played["cap"]), -1)
        traj = Trajectory(
            raw_states=np.zeros((2, 33), np.float32),
            actions=np.zeros((1, 2), np.float32),
            corridor_progress=played["legacy_credit"].astype(np.float32),
            corridor_credit_events=played["credit_events"],
            corridor_confirmed_progress=played["confirmed_progress"],
            progress_mode="confirmed_endpoint_v2",
        )
        endpoint = trajectory_endpoint(traj)
        self.assertEqual(endpoint.cap, -1)
        self.assertEqual(endpoint.distance, 0.0)

    def test_replay_matches_online_steps(self):
        corridor = route(
            [(0, 0), (20, 0)], reference_progress_mode="confirmed_endpoint_v2",
            reference_anchor_id="player_origin", reference_anchor_verified=False,
            reference_anchor_calibration_sha256=None, tcp_protocol="legacy33",
        )
        points = [[0, 0], [1, 0], [2, 2], [2.5, 0]]
        online = []
        tracker = corridor.tracker(np.array(points[0], dtype=float))
        for point in points[1:]:
            online.append(tracker.step(np.array(point, dtype=float)).confirmed_progress)
        played = replay_states(corridor, points)
        np.testing.assert_allclose(played["confirmed_progress"][1:], online)


def tracker_delta(corridor, points):
    tracker = corridor.tracker(np.array(points[0], dtype=float))
    return [tracker.step(np.array(point, dtype=float)).delta_d for point in points[1:]]


def replay_states(corridor, points):
    from training.reference_corridor import replay_corridor_result
    states = np.zeros((len(points), 33), np.float32)
    for i, point in enumerate(points):
        states[i, 0], states[i, 1] = point
    return replay_corridor_result(corridor, states)


if __name__ == "__main__":
    unittest.main()

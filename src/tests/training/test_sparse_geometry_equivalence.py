"""Bitwise contracts against the frozen, pre-optimization scalar builder."""
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training import sparse_geometry as sg
from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset
from training.rollout import RolloutWorker
from tests.training import _sparse_geometry_reference as reference


BOX = np.array([[-.1, -.1], [.1, -.1], [.1, .1], [-.1, .1]])
TERRAIN = [
    np.array([[-2., -.2], [2., -.2], [2., 0.], [-2., 0.]]),
    BOX + [.4, 0.], BOX + [.4, 0.],  # equal distance candidates
    np.array([[4., -1.], [4., 1.], [4.4, 1.], [4.4, -1.]]),
    np.array([[1., 1.], [1., 1.], [2., 1.], [1., 2.]]),  # zero-length edge
]
FIELDS = ("contour_tokens", "contour_mask", "relation_tokens", "relation_mask")


def state_at(x=0., y=0., dimension=33):
    state = np.zeros(dimension, np.float32)
    state[:2] = x, y
    state[5:7] = x, y + .35
    state[19:21] = x + .25, y
    state[23:25] = x + .5, y
    return state


def make_cache(polygons=TERRAIN):
    return sg.SparseGeometryCache.build(polygons, BOX, BOX, BOX + [0., -.2])


class SparseGeometryEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(TrainConfig.ppo_mainline(), body_angle_offset=0., tip_angle_offset=0.)

    def assert_bytes(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def assert_frame(self, state, cache, config=None):
        config = config or self.cfg
        expected = reference.build_sparse_geometry(state, cache, config)
        actual = sg.build_sparse_geometry(state, cache, config)
        for a, b in zip(actual, expected):
            self.assert_bytes(a, b)

    def test_scalar_equivalence_across_poses_order_winding_and_capacities(self):
        rng = np.random.default_rng(20261004)
        for polygons in ([], TERRAIN, list(reversed(TERRAIN)), [p[::-1] for p in TERRAIN]):
            cache = make_cache(polygons)
            for size, k in ((128, 8), (5, 1), (0, 0)):
                cfg = replace(self.cfg, sparse_max_contour_tokens=size, sparse_relation_k=k)
                for t in range(24):
                    state = state_at(*rng.uniform(-1., 1., 2), dimension=33 if t % 2 else 35)
                    state[2:4] = rng.normal(size=2)
                    state[23:25] = state[:2] + rng.uniform(-2., 2., 2)
                    state[19:21] = (state[:2] + state[23:25]) / 2
                    state[25:27] = rng.normal(size=2)
                    with self.subTest(capacity=(size, k), frame=t):
                        self.assert_frame(state, cache, cfg)

    def test_prepared_distance_retains_closest_points_and_strict_ties(self):
        rng = np.random.default_rng(13)
        for n in (3, 4, 12, 37):
            for t in range(100):
                poly = rng.normal(size=(n, 2))
                a, b = rng.normal(size=(2, 2))
                if t % 7 == 0:
                    poly[1] = poly[0]
                if t % 11 == 0:
                    b = a.copy()
                expected = reference._polygon_to_segment(poly, a, b)
                actual = sg._prepared_polygon_to_segment(sg._prepare_polygon_edges(poly), a, b)
                self.assertEqual(actual[0], expected[0])
                self.assert_bytes(actual[1], expected[1])
                self.assert_bytes(actual[2], expected[2])
        for poly in (np.array([]), np.zeros((0, 2)), np.ones((1, 2)), BOX):
            a, b = np.zeros(2), np.ones(2)
            expected = reference._polygon_to_segment(poly, a, b)
            actual = sg._polygon_to_segment(poly, a, b)
            self.assertEqual(actual[0], expected[0])
            self.assert_bytes(actual[1], expected[1])
            self.assert_bytes(actual[2], expected[2])

    def test_window_edges_and_config_changes_never_reuse_frame_membership(self):
        # Exactly representable boundary and its adjacent float32 positions.
        cache = make_cache([BOX + [5., 0.], BOX + [-3., 0.]])
        xs = (np.float32(1.), np.nextafter(np.float32(1.), np.float32(2.)),
              np.nextafter(np.float32(1.), np.float32(0.)), np.float32(100.))
        for width in (8., .25, 20., 8.):
            cfg = replace(self.cfg, sparse_window_meters=width)
            for x in xs:
                self.assert_frame(state_at(x), cache, cfg)
        # Terrain endpoints/midpoints exactly on a window boundary.
        cache = make_cache([np.array([[5., -1.], [5., 1.], [6., 1.], [6., -1.]])])
        for x in xs[:3]:
            self.assert_frame(state_at(x), cache, replace(self.cfg, sparse_window_meters=8.))

    def test_lazy_static_cache_rebuild_and_query_copies(self):
        cache = make_cache([BOX, BOX + [40., 40.]])
        self.assertFalse(cache._edges)
        self.assertFalse(cache._midpoints)
        sg.build_sparse_geometry(state_at(), cache, self.cfg)
        self.assertEqual(set(cache._midpoints), {0})
        mids = cache._midpoints[0]
        self.assert_frame(state_at(.2), cache)
        self.assertIs(cache._midpoints[0], mids)
        self.assert_frame(state_at(40., 40.), cache)
        self.assertEqual(set(cache._midpoints), {0, 1})
        # Replaced geometry has a separate static cache, not stale midpoints.
        fresh = make_cache([BOX + [.5, 0.]])
        self.assert_frame(state_at(), fresh)
        self.assertFalse(np.array_equal(fresh._midpoints[0], mids))
        nearby = fresh.index.nearby
        with patch.object(fresh.index, "nearby", side_effect=lambda *args: [p.copy() for p in nearby(*args)]):
            self.assert_frame(state_at(), fresh)
        cloned = copy.deepcopy(fresh)
        self.assert_frame(state_at(), cloned)
        self.assertIsNone(fresh.polygon_index(BOX + [99., 99.]))

    def test_zero_capacity_and_missing_parts_preserve_contract(self):
        for parts in ((None, None, None), (BOX[:1], BOX[:2], None), (BOX, None, BOX)):
            cache = sg.SparseGeometryCache.build(TERRAIN, *parts)
            self.assert_frame(state_at(), cache)
            self.assert_frame(state_at(), cache, replace(self.cfg, sparse_geometry=False))
        self.assert_frame(state_at(), None)
        cache = make_cache()
        for state in (np.zeros(32), np.zeros((1, 33)), np.full(33, np.nan)):
            with self.assertRaises(ValueError):
                sg.build_sparse_geometry(state, cache, self.cfg)

    def test_only_retained_terrain_and_relation_tokens_are_materialized(self):
        cfg = replace(self.cfg, sparse_max_contour_tokens=14, sparse_relation_k=1)
        cache = make_cache()
        with patch.object(sg, "_contour_token", wraps=sg._contour_token) as contours, \
                patch.object(sg, "_relation_token", wraps=sg._relation_token) as relations:
            _, cm, _, rm = sg.build_sparse_geometry(state_at(), cache, cfg)
        self.assertEqual(int(cm.sum()), 14)
        self.assertEqual(contours.call_count, 14)
        self.assertEqual(relations.call_count, int(rm.sum()))
        self.assertEqual(relations.call_count, 3)

    def test_offline_full_sequence_and_online_batch_share_exact_inputs(self):
        cfg = replace(self.cfg, patch_size=8)
        ds = TrajectoryDataset(cfg, np.ones((8, 8), np.float32), None,
                               solid_polygons=TERRAIN, tip_local=BOX, body_local=[BOX, BOX])
        states = np.stack([state_at(x) for x in (0., .1, .7, 1.2)])
        traj = Trajectory(raw_states=states, actions=np.ones((3, 2), np.float32),
                          contact=np.zeros((4, 5), np.float32))
        with patch.object(sg, "build_sparse_geometry", reference.build_sparse_geometry):
            expected = ds.encode_sequence(traj)
        actual = ds.encode_sequence(traj)
        for name in expected:
            self.assert_bytes(actual[name], expected[name])
        worker = object.__new__(RolloutWorker)
        worker.config = cfg
        worker._sparse_cache = ds._sparse_geometry_cache()
        for name, value in zip(FIELDS, worker._ppo_sparse(states)):
            self.assert_bytes(value, expected[name])
        traj.terminated = np.array([False, False, True])
        terminal = ds.encode_sequence(traj)
        for name in ("patches", *FIELDS):
            self.assertFalse(np.any(terminal[name][-1]))
        for name in ("dynamics", "act_history"):
            self.assert_bytes(terminal[name], expected[name])


if __name__ == "__main__":
    unittest.main()

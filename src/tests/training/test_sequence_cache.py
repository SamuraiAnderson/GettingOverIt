"""Cache contracts: exact observations, mutation invalidation, bounded retention."""
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.config import TrainConfig
from training.dataset import TrajectoryDataset
from training.sequence_cache import SILObservationCache, TRAJECTORY_INPUTS
from src.tests.training.test_elite_archive import episode


class SequenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(TrainConfig.ppo_mainline(), patch_size=8,
                           sparse_max_contour_tokens=16, sparse_relation_k=2)
        self.traj = episode(t=4, tail=2)
        self.encoder = self.encoder_for(self.cfg)
        self.cache = SILObservationCache()
        self.limit = 1024**2

    @staticmethod
    def encoder_for(config):
        terrain = np.ones((30,30), np.float32)
        polygon = np.array([[-1.,0.],[2.,0.],[2.,2.],[-1.,2.]])
        part = np.array([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
        return TrajectoryDataset(config,terrain.copy(),terrain.copy(),min_gx=-10,min_gy=-10,
            solid_polygons=[polygon],tip_local=part.copy(),body_local=[part.copy(),part.copy()],
            reference_maps=[terrain.copy(),np.zeros_like(terrain)])

    def begin(self, trajectories=None, limit=None):
        self.cache.begin_round(self.encoder, [self.traj] if trajectories is None else trajectories,
                               self.limit if limit is None else limit)

    def assert_frames_equal(self, left, right):
        self.assertEqual(set(left),set(right))
        for name in left: np.testing.assert_array_equal(left[name],right[name])

    def test_same_content_reuses_across_rounds_and_pool_indices_and_is_read_only(self):
        self.begin()
        cached=self.cache.get(self.encoder,self.traj)
        self.assert_frames_equal(cached,self.encoder.encode_sequence(self.traj))
        clone=copy.deepcopy(self.traj)
        self.begin([episode(distance=9),clone])
        with patch.object(self.encoder,'encode_sequence',side_effect=AssertionError('unexpected rebuild')):
            self.assertIs(self.cache.get(self.encoder,clone),cached)
        self.assertEqual(self.cache.stats['hits'],1)
        self.assertEqual(self.cache.stats['misses'],0)
        with self.assertRaises(ValueError): cached['patches'][0,0,0,0]=42
        with self.assertRaises(TypeError): cached['patches']=None

    def test_each_trajectory_input_mutation_invalidates_and_terminal_frame_is_blank(self):
        for name in TRAJECTORY_INPUTS:
            with self.subTest(name=name):
                self.begin()
                self.cache.get(self.encoder,self.traj)
                changed=copy.deepcopy(self.traj)
                array=getattr(changed,name)
                if name=='terminated': array[-1]=True
                else: array.flat[0]+=1
                self.begin([changed])
                actual=self.cache.get(self.encoder,changed)
                self.assertEqual(self.cache.stats['misses'],1)
                self.assert_frames_equal(actual,self.encoder.encode_sequence(changed))
                if name=='terminated':
                    self.assertFalse(actual['patches'][-1].any())
                    self.assertFalse(actual['contour_mask'][-1].any())

    def test_configuration_maps_and_geometry_changes_invalidate_derived_geometry_too(self):
        changes = [
            lambda: setattr(self.cfg,'wide_patch_resolution',self.cfg.wide_patch_resolution*2),
            lambda: setattr(self.cfg,'sparse_max_contour_tokens',24),
            lambda: setattr(self.cfg,'feature_position_scale',2.),
            lambda: setattr(self.cfg,'reference_guidance_sigma',3.),
            lambda: self.encoder.reference_maps[0].fill(.25),
            lambda: self.encoder.solid_polygons[0].__iadd__([1.,0.]),
            lambda: self.encoder.tip_local.__imul__(2.),
            lambda: self.encoder.body_local[0].__imul__(2.),
            lambda: self.encoder.terrain_mask.fill(0),
            lambda: setattr(self.encoder,'_min_gx',-20),
        ]
        self.begin(); self.cache.get(self.encoder,self.traj)
        for change in changes:
            change()
            self.begin()
            self.assertEqual(self.cache.stats['invalidations'],1)
            self.assertIsNone(self.encoder._sparse_cache)
            fresh=copy.copy(self.encoder); fresh._sparse_cache=None
            actual=self.cache.get(self.encoder,self.traj)
            self.assert_frames_equal(actual,fresh.encode_sequence(self.traj))
            self.assertEqual(self.cache.stats['misses'],1)

    def test_training_settings_rewards_and_unused_efficiency_do_not_invalidate(self):
        self.begin(); original=self.cache.get(self.encoder,self.traj)
        self.cfg.lr*=2
        self.cfg.gamma=.9
        self.cfg.gru_sil_max_updates=8
        self.cfg.gru_sil_sampling_mode='trajectory_uniform'
        self.traj.score+=10
        self.traj.step_rewards+=1
        self.encoder.update_efficiency_arr(np.zeros((30,30),np.float32))
        self.begin()
        self.assertIs(self.cache.get(self.encoder,self.traj),original)
        # A non-reference sequence really uses the efficiency map and must miss.
        plain=replace(self.traj,guidance_index=None)
        self.begin([plain]); old=self.cache.get(self.encoder,plain)
        self.encoder.efficiency_arr.fill(.5)
        self.begin([plain]); new=self.cache.get(self.encoder,plain)
        self.assertEqual(self.cache.stats['invalidations'],1)
        self.assertFalse(np.array_equal(old['patches'],new['patches']))

    def test_lru_byte_limit_pool_removal_and_capacity_shrink(self):
        trajectories=[copy.deepcopy(self.traj) for _ in range(3)]
        for i,traj in enumerate(trajectories): traj.actions+=i
        size=sum(v.nbytes for v in self.encoder.encode_sequence(self.traj).values())
        self.begin(trajectories,2*size)
        a=self.cache.get(self.encoder,trajectories[0])
        self.cache.get(self.encoder,trajectories[1])
        self.assertIs(self.cache.get(self.encoder,trajectories[0]),a)
        self.cache.get(self.encoder,trajectories[2])
        self.assertEqual(self.cache.bytes,2*size)
        self.assertEqual(self.cache.stats['evictions'],1)
        self.cache.get(self.encoder,trajectories[1])
        self.assertEqual(self.cache.stats['misses'],4)
        self.begin(trajectories,size)
        self.assertEqual(self.cache.bytes,size)
        self.begin([],size)
        self.assertEqual(self.cache.bytes,0)
        self.assertEqual(self.cache.stats['obsolete'],1)

    def test_oversized_entries_bypass_and_disabled_cache_releases_between_rounds(self):
        self.begin(limit=1)
        result=self.cache.get(self.encoder,self.traj)
        self.assertEqual(self.cache.bytes,0)
        self.assertEqual(self.cache.stats['oversized'],1)
        self.assert_frames_equal(result,self.encoder.encode_sequence(self.traj))
        self.begin(limit=0)
        result=self.cache.get(self.encoder,self.traj)
        self.assertIs(self.cache.get(self.encoder,self.traj),result)
        self.assertEqual(self.cache.bytes,0)
        self.cache.end_round()
        self.assertFalse(self.cache._round_local)
        self.begin(limit=0)
        self.assertIsNot(self.cache.get(self.encoder,self.traj),result)
        self.assertEqual(self.cache.stats['misses'],1)

    def test_failed_encoding_is_never_cached(self):
        self.begin()
        with patch.object(self.encoder,'encode_sequence',side_effect=RuntimeError('broken input')):
            with self.assertRaises(RuntimeError): self.cache.get(self.encoder,self.traj)
        self.assertEqual(self.cache.bytes,0)
        self.assert_frames_equal(self.cache.get(self.encoder,self.traj),self.encoder.encode_sequence(self.traj))


if __name__=='__main__': unittest.main()

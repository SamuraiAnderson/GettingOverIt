"""Dual-pool quotas, learning boundaries, persistence and acceptance accounting."""
import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from src.tests.training import test_gru_sil as fixture
from src.tests.training.test_elite_archive import episode
from training.gru_sil import GRUSILSamplingPlan, RecoverySamplingPlan, gru_sil_update, validate_gru_sil
from training.ppo_trainer import PPOTrainer
from training.main_ppo import main
from training.recovery_archive import RecoveryArchive, RecoverySegment
from training.reference_corridor import CorridorSegment, ReferenceCorridor
from training.reference_fingerprint import reference_fingerprint
from training.sil_pool_schedule import pool_for_slot


class TwoPoolSILTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.GRUSILTests()
        self.f.setUp()
        self.f.cfg.recovery_archive_enabled = True
        self.f.cfg.gru_sil_sampling_mode = "quality_dynamics_v3"
        self.f.cfg.gru_sil_max_updates = 16
        self.f.cfg.gru_sil_budget_ratio = 1.
        for g in self.f.trainer.optimizer.param_groups:
            g["lr"] = 1e-5
        # Recovery is after the elite high-water cut at action 8. Its original
        # reset prefix and failed tail stay available for hidden state/returns.
        self.segment = RecoverySegment(self.f.traj, dict(candidate_id="tail", first_outside_step=8, end_step=11))
        self.recovery = SimpleNamespace(segments=[self.segment], trajectories=[self.f.traj])

    def update(self, budget, iteration=51):
        f = self.f
        return gru_sil_update(f.trainer, f.model, f.archive, f.encoder, f.buffer,
                              budget_updates=budget, iteration=iteration, recovery_archive=self.recovery)

    def test_16_slots_keep_elite_weights_and_train_recovery_beyond_elite_cap(self):
        expected = GRUSILSamplingPlan(self.f.archive.trajectories, [0], 3, "quality_dynamics_v3")
        metrics = self.update(16)
        rows = self.f.trainer.last_gru_sil_sampling["chunks"]
        self.assertEqual((metrics["gru_sil_elite_candidates"], metrics["gru_sil_recovery_candidates"]), (12, 4))
        self.assertEqual((metrics["gru_sil_elite_updates"], metrics["gru_sil_recovery_updates"]), (12, 4))
        for r in rows:
            if r["pool"] == "recovery":
                self.assertEqual((r["start"], r["end"], r["cap_kind"]), (8, 11, "first_return"))
            else:
                self.assertLessEqual(r["end"], 8)
                key = (r["trajectory"], r["start"], r["end"], r["cap"])
                self.assertEqual(r["sampling_weight"]["probability"], expected.probabilities()[key])
        self.assertEqual(metrics["gru_sil_recovery_accepted_positive_frames"], 12)

    def test_small_budgets_resume_quota_and_zero_advantage_consumes_it(self):
        self.f.traj.step_rewards[:] = -100
        before = copy.deepcopy((self.f.model.state_dict(), self.f.trainer.optimizer.state_dict()))
        observed = []
        for i in range(16):
            m = self.update(1, i)
            observed.extend(r["pool"] for r in self.f.trainer.last_gru_sil_sampling["chunks"])
            self.assertEqual(m["gru_sil_updates"], 0)
            self.assertEqual(m["gru_sil_attempts"], 0)
            self.assertEqual(self.f.trainer.gru_sil_pool_cursor, i+1)
            state = copy.deepcopy(self.f.trainer.checkpoint_state())
            resumed = PPOTrainer(self.f.cfg)
            resumed.restore_checkpoint_state(state, self.f.model)
            resumed.optimizer = self.f.trainer.optimizer
            self.f.trainer = resumed
        self.assertEqual(observed, [pool_for_slot(i, .25) for i in range(16)])
        self.assertEqual(observed.count("recovery"), 4)
        self.f.assert_state_equal(before, (self.f.model.state_dict(), self.f.trainer.optimizer.state_dict()))

    def test_kl_reject_consumes_only_drawn_slot_and_next_round_reaches_other_pool(self):
        before = copy.deepcopy((self.f.model.state_dict(), self.f.trainer.optimizer.state_dict()))
        with patch("training.gru_sil.conditional_kl", return_value=torch.ones(11)):
            first = self.update(16)
            self.assertEqual(first["gru_sil_recovery_candidates"], 1)
            self.assertEqual(first["gru_sil_recovery_rejected_steps"], 1)
            self.assertEqual(first["gru_sil_elite_candidates"], 0)
            self.assertEqual(self.f.trainer.gru_sil_pool_cursor, 1)
            second = self.update(16, 52)
            self.assertEqual(second["gru_sil_elite_rejected_steps"], 1)
            self.assertEqual(second["gru_sil_recovery_candidates"], 0)
            self.assertEqual(self.f.trainer.gru_sil_pool_cursor, 2)
        self.f.assert_state_equal(before, (self.f.model.state_dict(), self.f.trainer.optimizer.state_dict()))

    def test_empty_pool_lends_slots_without_debt_and_zero_budget_does_not_advance(self):
        self.assertEqual(self.update(0)["gru_sil_budget"], 0)
        self.recovery.segments = []
        self.recovery.trajectories = []
        m = self.update(3)
        self.assertEqual((m["gru_sil_elite_candidates"], m["gru_sil_recovery_candidates"]), (3, 0))
        self.assertEqual(self.f.trainer.gru_sil_pool_cursor, 0)
        self.recovery.segments = [self.segment]
        self.recovery.trajectories = [self.f.traj]
        self.f.archive.trajectories = []
        m = self.update(3, 52)
        self.assertEqual((m["gru_sil_elite_candidates"], m["gru_sil_recovery_candidates"]), (0, 3))
        self.assertEqual(self.f.trainer.gru_sil_pool_cursor, 0)

    def test_recovery_segments_equal_mass_despite_length_and_only_own_intervals(self):
        self.recovery.segments.append(RecoverySegment(self.f.traj,
            dict(candidate_id="long", first_outside_step=1, end_step=7)))
        p = RecoverySamplingPlan(self.recovery, 3, self.f.cfg.feature_step_seconds).probabilities()
        self.assertAlmostEqual(sum(v for (i,*_),v in p.items() if i==0), .5)
        self.assertAlmostEqual(sum(v for (i,*_),v in p.items() if i==1), .5)
        self.assertEqual({(a,b) for (i,a,b,_),v in p.items() if i==1}, {(1,4), (4,7)})

    def test_configuration_and_checkpoint_compatibility_fail_closed(self):
        state = self.f.trainer.checkpoint_state()
        del state["gru_sil_pool_cursor"]
        self.f.trainer.gru_sil_pool_cursor = 9
        self.f.trainer.restore_checkpoint_state(state, self.f.model)
        self.assertEqual(self.f.trainer.gru_sil_pool_cursor, 0)
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                self.f.trainer.restore_checkpoint_state(dict(state, gru_sil_pool_cursor=bad), self.f.model)
        for bad in (0., 1., float("nan")):
            with self.assertRaises(ValueError):
                validate_gru_sil(replace(self.f.cfg, gru_sil_recovery_fraction=bad))
        with self.assertRaisesRegex(ValueError, "no independent recovery"):
            gru_sil_update(self.f.trainer, self.f.model, self.f.archive, self.f.encoder,
                           self.f.buffer, budget_updates=16, iteration=1)

    def test_cli_explicit_enable_and_saved_resume_preserve_two_pool_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder)/"source.pt"
            saved = asdict(self.f.cfg)
            saved.pop("recovery_archive_enabled")
            saved.pop("gru_sil_recovery_fraction")
            torch.save(dict(config=saved, model={}), checkpoint)
            base = ["main_ppo", "--dry-run", "--resume", str(checkpoint), "--run-dir", "runs/test"]
            def resolved(extra):
                output = io.StringIO()
                with patch.object(sys, "argv", base+extra), contextlib.redirect_stdout(output):
                    main()
                return json.loads(output.getvalue())
            self.assertFalse(resolved([])["recovery_archive_enabled"])
            enabled = resolved(["--recovery-archive", "--gru-sil-recovery-fraction", ".25"])
            self.assertTrue(enabled["recovery_archive_enabled"])
            self.assertEqual(enabled["gru_sil_recovery_fraction"], .25)
            torch.save(dict(config=asdict(self.f.cfg), model={}), checkpoint)
            self.assertTrue(resolved([])["recovery_archive_enabled"])
            self.assertFalse(resolved(["--no-recovery-archive"])["recovery_archive_enabled"])
            with self.assertRaisesRegex(ValueError, "requires --recovery-archive"):
                resolved(["--no-recovery-archive", "--recovery-archive-resume", "missing.npz"])


class RecoveryArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = replace(fixture.TrainConfig.ppo_mainline(), reference_trajectory="test",
                           reference_anchor_id="player_origin", reference_corridor_radius=1.5,
                           reference_max_step=10., reference_projection_window=10., recovery_archive_enabled=True)
        self.route = ReferenceCorridor([CorridorSegment(np.array([[0.,0.],[20.,0.]]), np.ones(1))], self.cfg)
        self.fp = reference_fingerprint(self.cfg, schema_id="anchor35", raw_dim=35)

    def archive(self, cfg=None):
        return RecoveryArchive(cfg or self.cfg, self.root, self.fp, self.route)

    def trajectory(self, outside_x=(1., 2.), upper=False):
        xy = np.array([(0.,-.5), *[(x,-2.) for x in outside_x], (3.,-.5), (4.,-.5)])
        if upper:
            xy[:,1] *= -1
        traj = episode(t=1, tail=len(xy)-2)
        traj.raw_states[:, :2] = xy
        traj.raw_states[:, 33:35] = xy
        # No new confirmed progress is required for a recovery candidate.
        traj.corridor_credit_events[:] = False
        traj.corridor_confirmed_progress[:] = 0
        return traj

    def test_cell_replaces_slow_with_fast_keeps_full_context_and_deduplicates(self):
        a = self.archive()
        slow = self.trajectory((.5, 1., 2.))
        fast = self.trajectory()
        self.assertEqual(a.ingest([slow], 1)["recovery_archive_added"], 1)
        m = a.ingest([fast], 2)
        self.assertEqual((m["recovery_archive_added"], m["recovery_archive_evicted"]), (1, 1))
        self.assertEqual(a.segments[0].descriptor["T_steps"], 3)
        self.assertEqual((a.segments[0].start, a.segments[0].end), (1, 3))
        self.assertEqual(a.ingest([fast, slow], 3)["recovery_archive_added"], 0)
        self.assertEqual(fast.iteration, 0)
        path = a.save(3)
        b = self.archive()
        self.assertEqual(b.restore(path), 1)
        np.testing.assert_array_equal(b.trajectories[0].raw_states, fast.raw_states)
        np.testing.assert_array_equal(b.trajectories[0].step_rewards, fast.step_rewards)
        self.assertEqual(len(b.trajectories[0].actions), 4)

    def test_upper_backward_return_internal_reset_and_missing_context_rejected(self):
        a = self.archive()
        reset = self.trajectory()
        reset.terminated[1] = True
        missing = self.trajectory()
        missing.contact = None
        m = a.ingest([self.trajectory(upper=True), self.trajectory((1., 4.)), reset, missing], 1)
        self.assertEqual(m["recovery_archive_incomplete"], 2)
        self.assertEqual(m["recovery_archive_size"], 0)
        self.assertEqual(self.archive().restore(a.save(1)), 0)

    def test_restore_rechecks_rules_and_geometric_descriptors(self):
        a = self.archive()
        a.ingest([self.trajectory()], 1)
        path = a.save(1)
        with self.assertRaisesRegex(ValueError, "selection rule"):
            self.archive(replace(self.cfg, recovery_min_forward_m=3.)).restore(path)
        with np.load(path, allow_pickle=False) as data:
            payload = dict(data)
        meta = json.loads(str(payload["archive_metadata_json"]))
        meta["segments"][0]["end_step"] += 1
        payload["archive_metadata_json"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **payload)
        with self.assertRaisesRegex(ValueError, "descriptor differs"):
            self.archive().restore(path)


if __name__ == "__main__":
    unittest.main()

"""Archive without SIL: preserve reset prefixes/tails, provenance and restart semantics."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.config import TrainConfig
from training.dataset import Trajectory
from training.elite_archive import EliteTrajectoryArchive
from training.main_ppo import main, parse_args
from training.reference_corridor import trajectory_endpoint
from training.reference_fingerprint import reference_fingerprint
from training.run_logging import load_elites, save_elites


def episode(t=4, distance=3.0, tail=6):
    n = t + tail
    states = np.zeros((n + 1, 35), np.float32)
    states[:, 0] = np.arange(n + 1) * .1
    states[:, 1] = 3
    states[t + 1:, 1] = 1  # Keep the post-progress fall in the archive.
    events = np.zeros(n, bool)
    events[t - 1] = True
    confirmed = np.zeros(n + 1, np.float64)
    confirmed[t:] = distance
    return Trajectory(
        states, np.arange(n * 2, dtype=np.float32).reshape(n, 2),
        contact=np.ones((n + 1, 5), np.float32),
        corridor_progress=np.ones(n, np.float32) * 99,
        guidance_index=np.zeros(n + 1, np.int8), corridor_credit_events=events,
        corridor_confirmed_progress=confirmed, step_rewards=np.arange(n, dtype=np.float64),
        terminated=np.zeros(n, bool), progress_mode="confirmed_endpoint_v2")


class EliteArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = TrainConfig.ppo_mainline()
        self.config.reference_trajectory = "test-route.json"
        self.fingerprint = reference_fingerprint(self.config, schema_id="anchor35", raw_dim=35)

    def archive(self, **changes):
        return EliteTrajectoryArchive(replace(self.config, **changes), self.root, self.fingerprint)

    def test_full_prefix_and_failed_tail_roundtrip_without_sil(self):
        original = episode()
        archive = self.archive()
        stats = archive.ingest([original], 51)
        self.assertFalse(self.config.sil_enabled)
        self.assertEqual(stats["elite_archive_max_distance_m"], 3.0)
        self.assertEqual((original.score, original.iteration, original.archive_role), (0., 0, None))
        self.assertEqual(archive.save(51), 1)
        path = self.root / "elites_iter_0051.npz"
        restored = self.archive()
        self.assertEqual(restored.restore(path), 1)
        kept = restored.trajectories[0]
        for name in ("raw_states", "actions", "contact", "guidance_index", "corridor_progress",
                     "corridor_credit_events", "corridor_confirmed_progress", "step_rewards", "terminated"):
            np.testing.assert_array_equal(getattr(kept, name), getattr(original, name))
        self.assertEqual(kept.iteration, 51)
        self.assertEqual(kept.archive_role, "pareto")
        self.assertGreater(len(kept.actions), trajectory_endpoint(kept).t_cap)
        self.assertFalse(path.with_suffix(".npz.tmp").exists())

    def test_admission_keeps_fast_and_far_with_capacity_and_rejects_invalid(self):
        archive = self.archive(elite_archive_capacity=2)
        bad = episode(distance=90)
        bad.trajectory_valid = False
        water = episode(distance=100)
        water.raw_states[-1, 1] = -20
        stats = archive.ingest([episode(2, 2), episode(8, 8), episode(6, 1), bad, water], 52)
        self.assertEqual(stats["elite_archive_size"], 2)
        self.assertEqual(sorted(trajectory_endpoint(t).distance for t in archive.trajectories), [2, 8])

    def test_rejects_missing_context_and_does_not_consume_numpy_rng(self):
        traj = episode()
        traj.contact = None
        np.random.seed(19)
        expected = np.random.random(4)
        np.random.seed(19)
        stats = self.archive().ingest([traj], 1)
        np.testing.assert_array_equal(np.random.random(4), expected)
        self.assertEqual(stats["elite_archive_rejected_incomplete"], 1)

    def test_empty_snapshot_is_restorable(self):
        archive = self.archive()
        self.assertEqual(archive.save(50), 0)
        self.assertEqual(self.archive().restore(self.root / "elites_iter_0050.npz"), 0)

    def test_rejects_legacy_pool_and_different_reference(self):
        path = self.root / "legacy.npz"
        save_elites(path, [episode()], self.fingerprint)
        with self.assertRaisesRegex(ValueError, "unpruned"):
            self.archive().restore(path)
        archive = self.archive()
        archive.ingest([episode()], 1)
        archive.save(1)
        archive.fingerprint = {"fingerprint": "other"}
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            archive.restore(self.root / "elites_iter_0001.npz")

    def test_saved_validity_is_not_silently_promoted(self):
        traj = episode()
        traj.trajectory_valid = False
        traj.termination_known = False
        path = self.root / "flags.npz"
        save_elites(path, [traj])
        row = load_elites(path)[0]
        self.assertFalse(row["trajectory_valid"])
        self.assertFalse(row["termination_known"])

    def test_missing_terminal_array_remains_unknown(self):
        traj = episode()
        traj.terminated = None
        path = self.root / "unknown.npz"
        save_elites(path, [traj])
        self.assertFalse(load_elites(path)[0]["termination_known"])

    def test_cli_enables_archive_with_sil_off_and_accepts_stop_file(self):
        args = ["main_ppo", "--dry-run", "--run-dir", str(self.root),
                "--reference-trajectory", "--elite-archive", "--recipe", "ppo-ou",
                "--stop-file", str(self.root / "stop")]
        output = io.StringIO()
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(output):
            main()
        parsed = json.loads(output.getvalue())
        self.assertTrue(parsed["elite_archive_enabled"])
        self.assertFalse(parsed["sil_enabled"])
        self.assertEqual(parse_args(args[1:]).stop_file, str(self.root / "stop"))


if __name__ == "__main__":
    unittest.main()

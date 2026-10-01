"""Confirmed-endpoint formal prep stays on init-from and the rescored pool."""
import contextlib
import io
import unittest

from src.training import main_ppo
from src.tests.analysis.campaign import prepare_v2_formal as prep
from training.reference_corridor import PROGRESS_MODE_V2


class V2FormalPrepTests(unittest.TestCase):
    def test_command_uses_init_from_and_rescored_pool(self):
        args = main_ppo.parse_args(prep.training_arguments())
        self.assertTrue(args.init_from.endswith("ppo_iter_0118.pt"))
        self.assertTrue(args.sil_init_pool.endswith("elites_v2.npz"))
        self.assertIsNone(args.resume)
        self.assertIsNone(args.sil_resume_pool)
        self.assertIsNone(args.sil_seed)
        self.assertEqual(args.sil_archive_mode, "td_pareto")
        self.assertTrue(args.corridor_sil)
        self.assertEqual(args.target_physical_samples, 96000)
        self.assertEqual(prep.launch_config().reference_progress_mode, PROGRESS_MODE_V2)

    def test_pool_flags_are_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main_ppo.parse_args(["--sil-init-pool", "pool.npz"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main_ppo.parse_args([
                "--resume", "ckpt.pt", "--init-from", "other.pt", "--sil-init-pool", "pool.npz",
            ])

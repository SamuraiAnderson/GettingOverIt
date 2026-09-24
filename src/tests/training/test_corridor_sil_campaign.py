"""Check actual CLI configuration against the historical formal experiment."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from dataclasses import asdict
import tempfile
import numpy as np

from src.training import main_ppo
from src.tests.analysis.run_corridor_sil import training_arguments, validate_configuration


class CaptureConfig(Exception):
    pass


class SILCampaignTests(unittest.TestCase):
    def test_elite_roundtrip_preserves_corridor_resume_state(self):
        from src.training.run_logging import save_elites, load_elites
        from src.training.dataset import Trajectory
        traj = Trajectory(raw_states=np.zeros((4, 33), np.float32),
                          actions=np.ones((3, 2), np.float32),
                          contact=np.zeros((4, 5), np.float32),
                          corridor_progress=np.array([0., 1., 2.], np.float32),
                          guidance_index=np.array([0, 0, 1, 1], np.int8),
                          score=3.5, iteration=8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pool.npz"
            save_elites(path, [traj])
            restored = Trajectory(**load_elites(path)[0])
        for field in ("raw_states", "actions", "contact", "corridor_progress", "guidance_index"):
            np.testing.assert_array_equal(getattr(traj, field), getattr(restored, field))
        self.assertEqual((restored.score, restored.iteration), (3.5, 8))

    def test_continuation_uses_remaining_samples_and_matching_pool(self):
        from src.tests.analysis.extend_corridor_sil import resume_arguments
        args = main_ppo.parse_args(resume_arguments(31, {"last_iteration": 13, "physical_samples": 97939}))
        self.assertEqual(args.additional_physical_samples, 190061)
        self.assertTrue(args.resume.endswith("ppo_iter_0013.pt"))
        self.assertTrue(args.sil_resume_pool.endswith("elites_iter_0013.npz"))
        self.assertIsNone(args.sil_seed)

    def test_actual_configuration_only_changes_sil(self):
        for seed in (31, 47):
            arguments = training_arguments(Path("unused_campaign"), seed, "samples")
            index = arguments.index("--run-dir")
            del arguments[index:index + 2]
            args = main_ppo.parse_args(arguments)
            captured = []

            def capture(config):
                captured.append(asdict(config))
                raise CaptureConfig

            with patch.object(main_ppo, "parse_args", return_value=args), \
                    patch.object(main_ppo, "_setup_logging"), \
                    patch.object(main_ppo, "create_actor_critic", side_effect=capture):
                with self.assertRaises(CaptureConfig):
                    main_ppo.main()
            validate_configuration(captured[0], seed)
            self.assertEqual(args.target_physical_samples, 96000)
            self.assertEqual(captured[0]["value_warmup_iters"], 2)
            self.assertEqual(captured[0]["sil_start_iter"], 5)

    def test_sample_endpoint_does_not_silently_reset_resume_count(self):
        for arguments in (["--target-physical-samples", "0"],
                          ["--target-physical-samples", "96000", "--resume", "unused.pt"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main_ppo.parse_args(arguments)

    def test_iteration_endpoint_remains_exactly_ten(self):
        args = main_ppo.parse_args(training_arguments(Path("unused_campaign"), 31, "iterations"))
        self.assertEqual(args.max_iterations, 10)
        self.assertIsNone(args.target_physical_samples)


if __name__ == "__main__":
    unittest.main()

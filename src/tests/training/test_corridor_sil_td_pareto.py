"""Seed-47 long SIL campaign locks td_pareto and inherits checkpoint σ."""
import contextlib
from dataclasses import asdict
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.training import main_ppo
from src.tests.analysis import run_corridor_sil_td_pareto as campaign


class CaptureConfig(Exception):
    pass


class SILParetoCampaignTests(unittest.TestCase):
    def test_budget_and_output_constraints(self):
        output = str(campaign.ROOT / "runs/unused_sil_td_pareto")
        parsed = campaign.parse_args(["--output", output])
        self.assertEqual(parsed.hours, 12)
        self.assertEqual(campaign.SEED, 47)
        self.assertEqual(campaign.SEEDS, (47,))
        self.assertEqual(campaign.ARM, "S")
        self.assertEqual(campaign.ARCHIVE_MODE, "td_pareto")
        parsed = campaign.parse_args(["--output", output, "--hours", "16"])
        self.assertEqual(parsed.hours, 16)
        for extra in (["--hours", "0"], ["--hours", "17"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parse_args(["--output", output, *extra])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            campaign.parse_args(["--output", str(campaign.ROOT / "unused")])

    def test_training_command_enables_td_pareto_and_inherits_sigma(self):
        arguments = campaign.training_arguments(Path("unused_campaign"), 47)
        self.assertIn("--corridor-sil", arguments)
        self.assertIn("--sparse-geometry", arguments)
        self.assertEqual(arguments[arguments.index("--sil-archive-mode") + 1], "td_pareto")
        self.assertNotIn("--phase-noise", arguments)
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
        config = captured[0]
        campaign.validate_configuration(config, 47)
        self.assertEqual(config["sil_archive_mode"], "td_pareto")
        self.assertTrue(config["sil_enabled"])
        self.assertTrue(config["sparse_geometry"])
        self.assertFalse(config["ppo_reset_log_std_on_init"])
        self.assertEqual(config["ppo_log_std_anneal_rate"], 0.0)
        self.assertEqual(config["ou_phi"], 0.85)
        self.assertEqual(config["gamma"], 0.99)
        self.assertEqual(config["gae_lambda"], 0.95)
        self.assertEqual(args.target_physical_samples, 96000)
        self.assertEqual(args.seed, 47)
        with self.assertRaises(ValueError):
            campaign.training_arguments(Path("unused_campaign"), 31)

    def test_resume_keeps_td_pareto_and_matching_elite_pool(self):
        current = {"last_iteration": 14, "physical_samples": 97939}
        args, count = campaign.resume_arguments(Path("unused_campaign"), 47, current, 192000)
        parsed = main_ppo.parse_args(args)
        self.assertEqual(count, 5)
        self.assertTrue(parsed.resume.endswith("ppo_iter_0014.pt"))
        self.assertTrue(parsed.sil_resume_pool.endswith("elites_iter_0014.npz"))
        self.assertEqual(parsed.sil_archive_mode, "td_pareto")
        self.assertEqual(parsed.additional_physical_samples, 192000 - 97939)
        self.assertEqual(parsed.max_iterations, 19)
        self.assertIsNone(parsed.sil_seed)
        self.assertTrue(parsed.corridor_sil)
        with self.assertRaises(ValueError):
            campaign.resume_arguments(Path("unused_campaign"), 31, current, 192000)

    def test_choose_next_target_uses_timing_not_scores(self):
        row = {"physical_samples": 96000, "recent_samples_per_iteration": 8000,
               "recent_iteration_seconds": 70}
        target, estimate = campaign.choose_next_target(row, 20000, 600)
        self.assertEqual(target, 192000)
        self.assertGreater(estimate, 0)
        self.assertLess(estimate, 20000)
        with self.assertRaises(campaign.BudgetStop):
            campaign.choose_next_target(row, 10, 600)
        self.assertEqual(campaign.next_milestone(96000), 192000)
        self.assertEqual(campaign.next_milestone(96001), 192000)
        self.assertEqual(campaign.next_milestone(191999), 192000)

    def test_runner_rejects_other_seeds_and_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            current = Path(folder)
            (current / "source_hashes.json").write_text("[]", encoding="utf-8")
            runner = campaign.SILParetoRunner(
                datetime.now().astimezone() + timedelta(hours=12), current)
        self.assertTrue(runner.run_name("S", 47).endswith("_S_s47"))
        with self.assertRaises(ValueError):
            runner.run_name("C", 47)
        with self.assertRaises(ValueError):
            runner.run_name("S", 31)
        with self.assertRaises(ValueError):
            runner.run_name("S", 61)

    def test_validate_rejects_map_elites_and_sigma_reset(self):
        path = campaign.configuration_reference(47)
        config = json.loads(path.read_text(encoding="utf-8"))["config"]
        config["sil_enabled"] = True
        config["sil_advantage_weighting_enabled"] = True
        config["sil_archive_mode"] = "td_pareto"
        config["ppo_reset_log_std_on_init"] = False
        config["ppo_log_std_anneal_rate"] = 0.0
        config["ou_phi"] = 0.85
        config["sparse_geometry"] = True
        config["reference_trajectory"] = (
            "src/Data/video_reference/37258923076-1-192/reference_trajectory_game_world.json")
        campaign.validate_configuration(config, 47)
        config["sil_archive_mode"] = "map_elites"
        with self.assertRaises(ValueError):
            campaign.validate_configuration(config, 47)
        config["sil_archive_mode"] = "td_pareto"
        config["ppo_reset_log_std_on_init"] = True
        with self.assertRaises(ValueError):
            campaign.validate_configuration(config, 47)
        config["ppo_reset_log_std_on_init"] = False
        config["lr"] = config["lr"] * 2
        with self.assertRaises(ValueError):
            campaign.validate_configuration(config, 47)


if __name__ == "__main__":
    unittest.main()

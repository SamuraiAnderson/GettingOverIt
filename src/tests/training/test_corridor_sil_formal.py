"""Formal SIL-only campaign keeps the C recipe and inherits checkpoint σ."""
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
from src.tests.analysis.campaign import run_corridor_sil_formal as campaign


class CaptureConfig(Exception):
    pass


class SILFormalCampaignTests(unittest.TestCase):
    def test_budget_and_output_constraints(self):
        output = str(campaign.ROOT / "runs/unused_sil_formal")
        parsed = campaign.parse_args(["--output", output])
        self.assertEqual(parsed.hours, 6)
        self.assertEqual(campaign.SEEDS, (31, 47, 61))
        self.assertEqual(campaign.ARM, "S")
        for extra in (["--hours", "0"], ["--hours", "7"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parse_args(["--output", output, *extra])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            campaign.parse_args(["--output", str(campaign.ROOT / "unused")])

    def test_training_command_is_sil_only_and_inherits_sigma(self):
        for seed in campaign.SEEDS:
            arguments = campaign.training_arguments(Path("unused_campaign"), seed)
            self.assertIn("--corridor-sil", arguments)
            self.assertIn(str(campaign.SIL_SEED), arguments)
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
            campaign.validate_configuration(config, seed)
            self.assertTrue(config["sil_enabled"])
            self.assertFalse(config["ppo_reset_log_std_on_init"])
            self.assertEqual(config["ppo_log_std_anneal_rate"], 0.0)
            self.assertEqual(config["ou_phi"], 0.85)
            self.assertEqual(config["gamma"], 0.99)
            self.assertEqual(config["gae_lambda"], 0.95)
            self.assertFalse(config["phase_noise_enabled"])
            self.assertEqual(args.target_physical_samples, 96000)
            self.assertEqual(args.seed, seed)

    def test_resume_restores_matching_elite_pool(self):
        current = {"last_iteration": 14, "physical_samples": 97939}
        args, count = campaign.resume_arguments(Path("unused_campaign"), 31, current, 192000)
        parsed = main_ppo.parse_args(args)
        self.assertEqual(count, 5)
        self.assertTrue(parsed.resume.endswith("ppo_iter_0014.pt"))
        self.assertTrue(parsed.sil_resume_pool.endswith("elites_iter_0014.npz"))
        self.assertEqual(parsed.additional_physical_samples, 192000 - 97939)
        self.assertEqual(parsed.max_iterations, 19)
        self.assertIsNone(parsed.sil_seed)
        self.assertTrue(parsed.corridor_sil)

    def test_choose_endpoint_uses_run_count_not_six_holdouts(self):
        rows = [{"physical_samples": 96000, "recent_samples_per_iteration": 8000,
                 "recent_iteration_seconds": 70}] * 3
        target, estimates = campaign.choose_endpoint(rows, 20000, 600)
        self.assertIn(target, campaign.ENDPOINT_CANDIDATES)
        self.assertGreaterEqual(target, 96000)
        self.assertEqual(len(estimates), campaign.ENDPOINT_CANDIDATES.index(target) + 1)
        with self.assertRaises(campaign.BudgetStop):
            campaign.choose_endpoint(rows, 10, 600)

    def test_runner_rejects_non_sil_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            current = Path(folder)
            (current / "source_hashes.json").write_text("[]", encoding="utf-8")
            runner = campaign.SILFormalRunner(
                datetime.now().astimezone() + timedelta(hours=6), current)
        self.assertTrue(runner.run_name("S", 61).endswith("_S_s61"))
        with self.assertRaises(ValueError):
            runner.run_name("C", 31)
        with self.assertRaises(ValueError):
            runner.run_name("T", 47)
        with self.assertRaises(ValueError):
            runner.run_name("S", 59)

    def test_validate_rejects_sigma_reset_and_hyperparameter_drift(self):
        path = campaign.configuration_reference(31)
        config = json.loads(path.read_text(encoding="utf-8"))["config"]
        config["sil_enabled"] = True
        config["sil_advantage_weighting_enabled"] = True
        config["ppo_reset_log_std_on_init"] = False
        config["ppo_log_std_anneal_rate"] = 0.0
        config["ou_phi"] = 0.85
        campaign.validate_configuration(config, 31)
        config["ppo_reset_log_std_on_init"] = True
        with self.assertRaises(ValueError):
            campaign.validate_configuration(config, 31)
        config["ppo_reset_log_std_on_init"] = False
        config["lr"] = config["lr"] * 2
        with self.assertRaises(ValueError):
            campaign.validate_configuration(config, 31)


if __name__ == "__main__":
    unittest.main()

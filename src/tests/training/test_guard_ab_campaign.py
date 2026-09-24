"""Additional independent seeds retain the previous A/B experiment contract."""
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
from src.tests.analysis import run_corridor_sil_guard_ab as campaign


class CaptureConfig(Exception):
    pass


class GuardABCampaignTests(unittest.TestCase):
    def test_seed_selection_and_scaled_deadline(self):
        output = str(campaign.ROOT / "runs/unused_guard_test")
        default = campaign.parse_args(["--output", output])
        self.assertEqual(tuple(default.seeds), (31, 47))
        self.assertEqual(default.hours, 3)
        additional = campaign.parse_args([
            "--output", output, "--seeds", "59", "71", "83", "97", "--hours", "6"])
        self.assertEqual(additional.seeds, [59, 71, 83, 97])
        for invalid in (["--seeds", "59", "59"], ["--seeds", "-1"],
                        ["--seeds", str(2**32)], ["--hours", "4"],
                        ["--seeds", "59", "71", "83", "97", "--hours", "7"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parse_args(["--output", output, *invalid])

    def test_new_seeds_use_the_same_actual_training_configuration(self):
        previous = campaign.ROOT / "runs/corridor_sil_guard_ab3_20260924_3h_090758"
        for seed in (59, 71, 83, 97):
            for arm in campaign.ARMS:
                with self.subTest(seed=seed, arm=arm):
                    arguments = campaign.training_arguments(Path("unused_campaign"), arm, seed)
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
                    campaign.validate_configuration(captured[0], seed, arm)
                    old = json.loads((previous.parent / f"{previous.name}_{arm}_s31" /
                                      "run_meta.json").read_text(encoding="utf-8"))["config"]
                    differences = {key for key in old.keys() | captured[0].keys()
                                   if old.get(key) != captured[0].get(key)}
                    # sil_phase_* / phase_noise_* 是 ab3 之后引入的 P3.1-SIL 接触相位
                    # 加权与空中相探索噪声字段（旧 run_meta 无这些键）。属于有意的配置
                    # 演进；新 campaign 必须在自身 manifest 里显式固定它们，保证
                    # campaign 内两臂一致。
                    self.assertLessEqual(differences, {"checkpoint_dir", "log_dir",
                                                       "sil_phase_weighting_enabled",
                                                       "sil_phase_window",
                                                       "sil_phase_airborne_weight",
                                                       "sil_phase_airborne_sample",
                                                       "phase_noise_enabled",
                                                       "phase_noise_airborne_log_bonus"})
                    self.assertEqual(args.seed, seed)
                    self.assertEqual(args.target_physical_samples, 96000)

    def test_missing_historical_seed_is_not_replaced_by_another_seed_score(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            current = root / "campaign"
            current.mkdir()
            (current / "source_hashes.json").write_text("[]", encoding="utf-8")
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "C_s31_midpoint_096k.json").write_text("{}", encoding="utf-8")
            runner = campaign.GuardABRunner(
                datetime.now().astimezone() + timedelta(hours=6), current, (59, 71, 83, 97))
            with patch.object(campaign, "BASELINE", baseline):
                campaign.comparison(runner, "running")
            result = json.loads((current / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual([row["seed"] for row in result["runs"]], [59, 71, 83, 97])
            self.assertTrue(all(row["baseline"] is None for row in result["runs"]))
            self.assertNotIn("baseline", result["pooled_episodes"])
            self.assertIn("4 个独立训练 seed", (current / "RESULT.md").read_text(encoding="utf-8"))
            self.assertTrue(runner.run_name("S", 97).endswith("_S_s97"))
            with self.assertRaises(ValueError):
                runner.run_name("C", 31)
            with self.assertRaises(ValueError):
                runner.run_name("T", 59)

    def test_configuration_template_still_rejects_hyperparameter_drift(self):
        path = campaign.configuration_reference(59)
        self.assertIn("_C_s31", str(path))
        config = json.loads(path.read_text(encoding="utf-8"))["config"]
        campaign.validate_configuration(config, 59, "C")
        config["lr"] *= 2
        with self.assertRaises(ValueError):
            campaign.validate_configuration(config, 59, "C")


if __name__ == "__main__":
    unittest.main()

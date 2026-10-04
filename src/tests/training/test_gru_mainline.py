"""Fresh PPO uses GRU; explicit and historical checkpoint contracts remain loadable."""
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.config import TrainConfig
from training.experimental_policy import create_actor_critic
from training.experiment_options import ppo_config, apply_experiment_arguments
from training.main_ppo import parse_args, main


class GRUMainlineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_fresh_default_constructs_recurrent_policy(self):
        cfg = ppo_config()
        apply_experiment_arguments(cfg, parse_args([]))
        model = create_actor_critic(cfg)
        self.assertTrue(hasattr(model, "gru"))
        self.assertFalse(hasattr(model, "transformer"))
        self.assertEqual(model.gru.hidden_size, 128)
        self.assertEqual(cfg.recurrent_seq_len, 32)
        self.assertTrue(cfg.normalize_action_history)
        self.assertFalse(cfg.random_deploy)
        self.assertFalse(cfg.sil_enabled)
        self.assertEqual((cfg.gru_sil_max_updates, cfg.gru_sil_budget_ratio), (16, 1.0))

    def test_legacy_checkpoint_missing_new_fields_loads_strictly(self):
        old = TrainConfig(d_model=16, nhead=2, num_layers=1, sparse_geometry=False)
        model = create_actor_critic(old)
        saved = asdict(old)
        for name in ("policy_backbone", "normalize_action_history", "recurrent_seq_len"):
            saved.pop(name)
        checkpoint = {"config": saved, "model": model.state_dict()}
        cfg = ppo_config(checkpoint)
        apply_experiment_arguments(cfg, parse_args([]), checkpoint)
        restored = create_actor_critic(cfg)
        restored.load_state_dict(checkpoint["model"], strict=True)
        self.assertEqual(cfg.policy_backbone, "transformer")
        self.assertFalse(cfg.normalize_action_history)
        self.assertEqual(cfg.recurrent_seq_len, 16)

    def test_saved_gru_contract_and_local_connection_are_preserved(self):
        saved = asdict(TrainConfig.ppo_mainline())
        saved.update(recurrent_seq_len=12, random_deploy=True, game_root="old-host", port=1)
        cfg = ppo_config({"config": saved, "model": {}})
        self.assertEqual(cfg.policy_backbone, "gru")
        self.assertEqual(cfg.recurrent_seq_len, 12)
        self.assertTrue(cfg.random_deploy)
        self.assertEqual(cfg.game_root, TrainConfig().game_root)
        self.assertEqual(cfg.port, TrainConfig().port)

    def test_explicit_transformer_and_legacy_bc_remain_available(self):
        cfg = ppo_config()
        apply_experiment_arguments(cfg, parse_args(["--policy-backbone", "transformer"]))
        self.assertTrue(hasattr(create_actor_critic(cfg), "transformer"))
        self.assertEqual(ppo_config(legacy_bc=True).policy_backbone, "transformer")

    def test_gru_rejects_unsupported_sil_before_dry_run_returns(self):
        for switches in (["--recipe", "ppo-ou-sil"], ["--corridor-sil"]):
            with self.subTest(switches=switches), patch.object(sys, "argv", ["main_ppo", "--dry-run", *switches]):
                with self.assertRaisesRegex(ValueError, "SIL"):
                    main()


if __name__ == "__main__":
    unittest.main()

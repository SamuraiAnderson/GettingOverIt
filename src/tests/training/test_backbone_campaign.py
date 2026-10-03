import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import torch

from src.tests.analysis.campaign.compare_backbones import (
    paired_models, physical_samples, config_for, executed_world_steps, validate_baseline, validate_exact_start, create_actor_critic,
)


class BackboneCampaignTests(unittest.TestCase):
    def test_shared_initialization_and_fresh_temporal_operators(self):
        mean = np.linspace(-1, 1, 39, dtype=np.float32)
        std = np.full(39, 2, dtype=np.float32)
        pair, keys = paired_models(7, mean, std)
        states = [pair[name].state_dict() for name in ("transformer", "gru")]
        self.assertTrue(all(torch.equal(states[0][key], states[1][key]) for key in keys))
        self.assertTrue(pair["gru"].dynamics_stats_ready)
        self.assertTrue(any(key.startswith("gru.") for key in states[1]))
        self.assertFalse(any(key.startswith("transformer.") for key in states[1]))
        other, _ = paired_models(17, mean, std)
        self.assertTrue(any(not torch.equal(states[1][key], other["gru"].state_dict()[key]) for key in keys))

    def test_config_changes_only_backbone(self):
        first, second = vars(config_for("transformer")), vars(config_for("gru"))
        changed = [key for key in first if first[key] != second[key]]
        self.assertEqual(changed, ["policy_backbone"])
        self.assertFalse(first["sil_enabled"])
        self.assertTrue(first["normalize_action_history"])

    def test_physical_budget_is_not_minibatch_size(self):
        self.assertEqual(physical_samples([{"rollout_environment_steps": "9600.0"}] * 3), 28800)
        with self.assertRaises(KeyError):
            physical_samples([{"batch_size": "9600"}])

    def test_terminated_agents_still_count_as_executed_worlds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "checkpoints/feature_diagnostics/iter_0001.json"
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps({"segments": [
                {"agent": 0, "start": 699, "duration": 1},
                {"agent": 1, "start": 1199, "duration": 1},
            ]}), encoding="utf-8")
            rows = [{"iteration": "1", "rollout_environment_steps": "8700"}]
            self.assertEqual(physical_samples(rows), 8700)
            self.assertEqual(executed_world_steps(root, rows), 9600)

    def test_baseline_consistency_rejects_divergence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for agent in (0, 1):
                (root / f"deterministic_000_agent{agent}.json").write_text(
                    json.dumps({"states": [[0, 1], [2, 3]]}), encoding="utf-8")
            validate_baseline(root)
            (root / "deterministic_000_agent1.json").write_text(
                json.dumps({"states": [[0, 1], [2, 3.01]]}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_baseline(root)

    def test_exact_start_checks_across_resets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for trial in (0, 1):
                for agent in (0, 1):
                    (root / f"deterministic_{trial:03d}_agent{agent}.json").write_text(
                        json.dumps({"states": [[0, 1], [2, 3]]}), encoding="utf-8")
            validate_exact_start(root)
            (root / "deterministic_001_agent0.json").write_text(
                json.dumps({"states": [[0, 1], [2, 3.01]]}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                validate_exact_start(root)

    def test_latest_known_action_reaches_both_backbones(self):
        for backbone in ("transformer", "gru"):
            with self.subTest(backbone=backbone):
                torch.manual_seed(7)
                cfg = config_for(backbone)
                cfg.d_model, cfg.nhead, cfg.num_layers, cfg.sparse_geometry = 16, 2, 1, False
                model = create_actor_critic(cfg).eval()
                dynamics = torch.zeros(1, 2, 39)
                patches = torch.zeros(1, 2, 4, 32, 32)
                history = torch.zeros(1, 2, 2)
                with torch.no_grad():
                    before = model._backbone(dynamics, patches, history)
                    history[:, -1] = torch.tensor([100., -100.])
                    after = model._backbone(dynamics, patches, history)
                self.assertGreater(float((before - after).abs().max()), 1e-6)


if __name__ == "__main__":
    unittest.main()

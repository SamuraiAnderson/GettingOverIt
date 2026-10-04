"""Policy lag is not input corruption; KL guards must precede optimizer steps."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))
from tests.training import test_ppo_contract as contract
from training.decision_buffer import evaluated_batches
from training.ppo_trainer import PPOTrainer, StalePolicyError
from training.rollout_pipeline import check_version, RolloutPipeline


class PipelineContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = contract.PPOContractTests()
        self.fixture.setUp()

    def _update_with_rejected_second_batch(self, adaptive):
        cfg, model, _, buffer, _, _ = self.fixture.collect()
        cfg.ppo_batch_size = 2
        cfg.ppo_target_kl_adaptive = adaptive
        trainer = PPOTrainer(cfg)
        def shifted(*args, **kwargs):
            for i, (batch, lp, ent, val) in enumerate(evaluated_batches(*args, **kwargs)):
                yield batch, lp+(1.0 if i else 0.0), ent, val
        # Keep the frozen audit real, while controlling only training batch 2.
        from training.policy_audit import audit_policy
        verified = audit_policy(model, buffer)
        with patch("training.policy_audit.audit_policy", return_value=verified), \
             patch("training.decision_buffer.evaluated_batches", side_effect=shifted):
            metrics = trainer.update(model, buffer)
        return cfg, trainer, metrics

    def test_kl_guard_rejects_second_batch_before_step(self):
        cfg, trainer, metrics = self._update_with_rejected_second_batch(adaptive=False)
        self.assertEqual(metrics["n_updates"], 1)
        self.assertEqual([x["applied"] for x in trainer.last_update_trace], [True, False])
        self.assertTrue(all(int(state["step"]) == 1 for state in trainer.optimizer.state.values()))
        self.assertEqual(metrics["target_kl"], cfg.target_kl)

    def test_adaptive_target_kl_shrinks_after_early_stop(self):
        cfg, _, metrics = self._update_with_rejected_second_batch(adaptive=True)
        self.assertLess(metrics["target_kl"], cfg.target_kl)

    def test_grad_norm_stays_fixed_unless_adaptive(self):
        from training.ppo_trainer import _GradNormAdapter
        fixed = _GradNormAdapter(0.5, 5.0, warmup_updates=3, enabled=False)
        adaptive = _GradNormAdapter(0.5, 5.0, warmup_updates=3)
        for norm in (9.0, 9.0, 9.0):
            fixed.record(norm)
            adaptive.record(norm)
        self.assertEqual(fixed.max_grad_norm, 0.5)
        self.assertEqual(adaptive.max_grad_norm, 5.0)

    def test_update_starts_without_stale_first_moment(self):
        cfg, model, _, buffer, _, _ = self.fixture.collect()
        behavior = copy.deepcopy(model)
        trainer = PPOTrainer(cfg)
        trainer.update(model, buffer)
        self.assertTrue(trainer.optimizer.state)
        for state in trainer.optimizer.state.values():
            state["exp_avg"].fill_(1e3)
        metrics = trainer.update(model, buffer, behavior_model=behavior, max_behavior_kl=1.0)
        self.assertGreater(metrics["n_updates"], 0)
        # Clipped gradients keep a freshly accumulated first moment far below the stale fill.
        for state in trainer.optimizer.state.values():
            self.assertLess(float(state["exp_avg"].abs().max()), 1.0)
        self.assertTrue(any(float(s["exp_avg_sq"].max()) > 0 for s in trainer.optimizer.state.values()))

    def test_stale_rollout_rejected_without_changing_weights(self):
        cfg, model, _, buffer, _, _ = self.fixture.collect()
        behavior = copy.deepcopy(model)
        with torch.no_grad():
            model.actor_mean.bias.add_(1)
        before = copy.deepcopy(model.state_dict())
        trainer = PPOTrainer(cfg)
        with self.assertRaises(StalePolicyError):
            trainer.update(model, buffer, behavior_model=behavior, max_behavior_kl=.001)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor.cpu(), before[key].cpu(), rtol=0, atol=0)
        self.assertFalse(trainer.optimizer.state)

    def test_bad_behavior_probabilities_still_fail(self):
        cfg, model, _, buffer, _, _ = self.fixture.collect()
        wrong = copy.deepcopy(model)
        with torch.no_grad():
            wrong.actor_mean.bias.add_(.1)
        with self.assertRaisesRegex(ValueError, "Frozen behavior-policy replay failed"):
            PPOTrainer(cfg).update(model, buffer, behavior_model=wrong)

    def test_small_lag_can_update_without_weakening_strict_default(self):
        cfg, model, _, buffer, _, _ = self.fixture.collect()
        behavior = copy.deepcopy(model)
        with torch.no_grad():
            # Define a small lag relative to the configured exploration std.
            model.actor_mean.bias.add_(.02 * model.actor_log_std.exp())
        with self.assertRaisesRegex(ValueError, "Frozen behavior-policy replay failed"):
            PPOTrainer(cfg).update(model, buffer)
        metrics = PPOTrainer(cfg).update(model, buffer, behavior_model=behavior)
        self.assertGreater(metrics["n_updates"], 0)
        self.assertGreater(metrics["behavior_kl"], 0)
        self.assertLess(metrics["preupdate_ratio_max_error"], .001)

    def test_queue_and_policy_versions_have_hard_bounds(self):
        self.assertEqual(check_version(3, 4), 1)
        for a, b in ((3, 2), (1, 3)):
            with self.assertRaises(ValueError):
                check_version(a, b)
        pipeline = object.__new__(RolloutPipeline)
        pipeline.pending = {"id":1}
        with self.assertRaisesRegex(RuntimeError, "one collection"):
            pipeline.submit({"id":2})
        pipeline.pending = None
        with self.assertRaisesRegex(RuntimeError, "No collection"):
            pipeline.receive()


if __name__ == "__main__":
    unittest.main()

"""σ 调度（M2 修复）的离线测试：init-from 重设 + 每轮退火。

背景：σ≈0.52 从源 checkpoint 经 --init-from 原样继承且全程不退化（entropy 钉死
1.51 nats），技能住在噪声里。修复 = 重设到 ppo_init_log_std + 每轮线性退火。
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.actor_critic import ActorCritic
from training.config import TrainConfig


def _model():
    cfg = TrainConfig(d_model=16, nhead=2, num_layers=1, context_len=3,
                      num_agents=2, random_deploy=False, deploy_exclusion_check=False,
                      contact_heads_enabled=False)
    torch.manual_seed(3)
    return ActorCritic(cfg), cfg


class SigmaScheduleTests(unittest.TestCase):
    def test_reset_log_std(self):
        """重设把源 checkpoint 继承的大 σ 拉回 ppo_init_log_std。"""
        model, cfg = _model()
        model.actor_log_std.data.fill_(-0.66)  # 模拟源 checkpoint 的 σ≈0.52
        model.reset_log_std(cfg.ppo_init_log_std)
        self.assertTrue(torch.allclose(model.actor_log_std.data,
                                       torch.full_like(model.actor_log_std, cfg.ppo_init_log_std)))
        # forward clamp 内生效：std ≈ 0.20
        self.assertGreaterEqual(cfg.ppo_init_log_std, cfg.ppo_log_std_min)
        self.assertLessEqual(cfg.ppo_init_log_std, cfg.ppo_log_std_max)

    def test_anneal_log_std_linear_and_floor(self):
        """每轮线性递减，触及下限后钳住。"""
        model, cfg = _model()
        model.reset_log_std(cfg.ppo_init_log_std)
        model.anneal_log_std(0.02, cfg.ppo_log_std_min)
        self.assertTrue(torch.allclose(
            model.actor_log_std.data,
            torch.full_like(model.actor_log_std, cfg.ppo_init_log_std - 0.02)))
        for _ in range(200):
            model.anneal_log_std(0.02, cfg.ppo_log_std_min)
        self.assertTrue(torch.allclose(
            model.actor_log_std.data,
            torch.full_like(model.actor_log_std, cfg.ppo_log_std_min)))

    def test_anneal_zero_rate_is_noop(self):
        model, cfg = _model()
        before = model.actor_log_std.data.clone()
        model.anneal_log_std(0.0, cfg.ppo_log_std_min)
        self.assertTrue(torch.equal(model.actor_log_std.data, before))


if __name__ == "__main__":
    unittest.main()

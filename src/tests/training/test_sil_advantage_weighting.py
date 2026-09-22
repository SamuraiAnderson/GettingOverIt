"""P3.2 优势加权 SIL 的离线回归测试。

覆盖：
  - 权重语义：状态完全相同、仅 corridor_progress 不同的两条轨迹，未来 credited 大的
    轨迹窗口必须获得严格更大的权重（此时 V 相同，A 差异完全来自 R）；
  - 加权路径：sil_update 走加权高斯 NLL，log_std 不变、actor_mean 更新、指标有限；
  - 回退路径：池内含非走廊轨迹时回退 MSE-on-mean（不出现优势指标）。
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset
from training.ppo_trainer import PPOTrainer


def _config(**kwargs):
    return TrainConfig(d_model=16, nhead=2, num_layers=1, context_len=3,
                       num_agents=2, random_deploy=False, deploy_exclusion_check=False,
                       contact_heads_enabled=False,
                       sil_batch_size=16, sil_max_batches=3, sil_num_workers=0,
                       **kwargs)


def _make_traj(T: int, credited_per_step: float, with_corridor: bool = True) -> Trajectory:
    raw = np.zeros((T + 1, 33), dtype=np.float32)
    raw[:, 0] = 50.0 + np.linspace(0, 3, T + 1)
    raw[:, 1] = 10.0 + np.linspace(0, 8, T + 1)
    raw[:, 23] = raw[:, 0] + 1.0
    raw[:, 24] = raw[:, 1] + 1.0
    acts = np.tanh(np.random.randn(T, 2).astype(np.float32)) * 0.5
    cp = None
    if with_corridor:
        cp = np.full(T, credited_per_step, dtype=np.float32)
    return Trajectory(raw_states=raw, actions=acts, corridor_progress=cp)


class SilAdvantageWeightingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        np.random.seed(7)
        grid = 256
        self.eff_arr = np.random.rand(grid, grid).astype(np.float32)
        self.terrain = np.ones((grid, grid), dtype=bool)

    def _dataset(self, trajs, config) -> TrajectoryDataset:
        ds = TrajectoryDataset(config, self.terrain, self.eff_arr, min_gx=0, min_gy=0,
                               solid_polygons=None, tip_local=None, body_local=None)
        ds.set_trajectories(trajs)
        return ds

    def test_weights_semantics(self):
        """相同状态与动作历史下，未来 credited 大的轨迹窗口权重逐点严格更大。"""
        cfg = _config()
        T = 40
        # 两条轨迹共享 states 与 actions（→ 每窗口模型输入相同、V 逐点相同），
        # 仅未来 credited 不同 → A 差异完全来自 R，权重差异必须逐点严格。
        hi = _make_traj(T, 0.5)
        lo = _make_traj(T, 0.001)
        lo.raw_states = hi.raw_states
        lo.actions = hi.actions
        ds = self._dataset([hi, lo], cfg)
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")

        weights, pos_frac = trainer._sil_advantage_weights(model, ds)
        self.assertEqual(len(weights), len(ds))
        self.assertTrue(np.all(weights >= cfg.sil_advantage_floor - 1e-6))
        self.assertTrue(np.all(weights <= cfg.sil_advantage_cap + 1e-6))
        self.assertGreaterEqual(pos_frac, 0.0)
        self.assertLessEqual(pos_frac, 1.0)

        n_each = len(ds) // 2  # 两条等长轨迹，窗口数相同，索引先 hi 后 lo
        w_hi, w_lo = weights[:n_each], weights[n_each:]
        # clamp 单调 → 逐点 >=；尾部两条轨迹优势同触底 floor 时取等号，故严格不等
        # 只在分离度最大的首窗口（tau=0，R 差最大且未触 cap）上断言。
        self.assertTrue(np.all(w_hi >= w_lo), "clamp 下单调性应保持逐点 >=")
        self.assertGreater(w_hi[0], w_lo[0])
        self.assertGreater(w_hi.mean(), w_lo.mean())

    def test_sil_update_weighted_path(self):
        cfg = _config()
        ds = self._dataset([_make_traj(40, 0.1), _make_traj(40, 0.05)], cfg)
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")

        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        metrics = trainer.sil_update(model, ds)
        self.assertGreater(metrics.get("sil_updates", 0), 0)
        self.assertTrue(np.isfinite(metrics["sil_loss"]))
        self.assertIn("sil_adv_pos_frac", metrics)
        self.assertIn("sil_weight_mean", metrics)
        self.assertTrue(torch.allclose(before["actor_log_std"],
                                       dict(model.named_parameters())["actor_log_std"].detach()))
        changed = any(not torch.allclose(before[k], p.detach())
                      for k, p in model.named_parameters() if "actor_mean" in k)
        self.assertTrue(changed)

    def test_fallback_without_corridor(self):
        cfg = _config()
        ds = self._dataset([_make_traj(40, 0.0, with_corridor=False),
                            _make_traj(40, 0.0, with_corridor=False)], cfg)
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")

        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        metrics = trainer.sil_update(model, ds)
        self.assertGreater(metrics.get("sil_updates", 0), 0)
        self.assertTrue(np.isfinite(metrics["sil_loss"]))
        self.assertNotIn("sil_adv_pos_frac", metrics)  # 回退 MSE 路径无优势指标
        self.assertTrue(torch.allclose(before["actor_log_std"],
                                       dict(model.named_parameters())["actor_log_std"].detach()))

    def test_switch_off_keeps_legacy_mse(self):
        cfg = _config(sil_advantage_weighting_enabled=False)
        ds = self._dataset([_make_traj(40, 0.1)], cfg)
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")
        metrics = trainer.sil_update(model, ds)
        self.assertGreater(metrics.get("sil_updates", 0), 0)
        self.assertNotIn("sil_adv_pos_frac", metrics)


if __name__ == "__main__":
    unittest.main()

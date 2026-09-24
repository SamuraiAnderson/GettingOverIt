"""P3.2 优势加权 SIL 的离线回归测试。

覆盖：
  - 权重语义：状态完全相同、仅 corridor_progress 不同的两条轨迹，未来 credited 大的
    轨迹窗口必须获得严格更大的权重（此时 V 相同，A 差异完全来自 R）；
  - 加权路径：sil_update 走加权高斯 NLL，log_std 不变、actor_mean 更新、指标有限；
  - 回退路径：池内含非走廊轨迹时回退 MSE-on-mean（不出现优势指标）；
  - 三道防线（20260923 C+SIL det 塌陷后引入）：行为门控（V-free，池 median
    credited vs 当前轮行为水平）、PPO 预算约束、饱和目标限幅。
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


def _make_traj(T: int, credited_per_step: float, with_corridor: bool = True,
               contact_states: np.ndarray | None = None) -> Trajectory:
    raw = np.zeros((T + 1, 33), dtype=np.float32)
    raw[:, 0] = 50.0 + np.linspace(0, 3, T + 1)
    raw[:, 1] = 10.0 + np.linspace(0, 8, T + 1)
    raw[:, 23] = raw[:, 0] + 1.0
    raw[:, 24] = raw[:, 1] + 1.0
    acts = np.tanh(np.random.randn(T, 2).astype(np.float32)) * 0.5
    cp = None
    if with_corridor:
        cp = np.full(T, credited_per_step, dtype=np.float32)
    contact = None
    if contact_states is not None:
        # contact 列序：tip_contact / tip_grip / body_contact / pot_contact / fall_dist
        contact = np.zeros((T + 1, 5), dtype=np.float32)
        contact[contact_states, 0] = 1.0  # tip_contact
    return Trajectory(raw_states=raw, actions=acts, corridor_progress=cp,
                      contact=contact)


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

        weights, pos_frac, ret_full = trainer._sil_advantage_weights(model, ds)
        self.assertEqual(len(ret_full), len(ds))
        self.assertEqual(len(weights), len(ds))
        # (R−V)₊ 语义：权重下界为 0（非正优势零权重），上界 cap
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.all(weights <= cfg.sil_advantage_cap + 1e-6))
        self.assertGreaterEqual(pos_frac, 0.0)
        self.assertLessEqual(pos_frac, 1.0)

        n_each = len(ds) // 2  # 两条等长轨迹，窗口数相同，索引先 hi 后 lo
        w_hi, w_lo = weights[:n_each], weights[n_each:]
        # 门控+正部归一化保持单调 → 逐点 >=；尾部两轨迹优势同被门控置零时取等号，
        # 故严格不等只在分离度最大的首窗口（tau=0，R 差最大）上断言。
        self.assertTrue(np.all(w_hi >= w_lo), "门控下单调性应保持逐点 >=")
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

    # ── P3.1-SIL：接触相位加权 + 空中帧稀疏采样 ──

    def test_phase_weights_semantics(self):
        """接触块前后 K 帧权重 1.0，远离接触的空中帧权重 = airborne_weight。"""
        cfg = _config(sil_phase_airborne_sample=1.0)  # 关稀疏，只看权重
        T, K = 40, cfg.sil_phase_window
        contact_states = np.arange(18, 23)  # 状态 18-22 接触
        traj = _make_traj(T, 0.1, contact_states=contact_states)
        ds = self._dataset([traj], cfg)
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")

        w = trainer._sil_phase_weights(ds)
        self.assertIsNotNone(w)
        self.assertEqual(len(w), len(ds))
        # 接触块 [18,22] ± K=7 → 近接触窗口 tau ∈ [11, 29]
        near = np.arange(11, 30)
        far = np.array([t for t in range(40) if t < 11 or t > 29])
        self.assertTrue(np.all(w[near] == 1.0))
        self.assertTrue(np.allclose(w[far], cfg.sil_phase_airborne_weight))

    def test_phase_sparse_sampling_drops_airborne(self):
        """sample=0 时空中帧全丢，仅保留接触附近窗口；log_std 不变。"""
        cfg = _config(sil_advantage_weighting_enabled=False,  # 隔离相位分量
                      sil_phase_airborne_sample=0.0)
        T = 40
        contact_states = np.arange(18, 23)
        traj = _make_traj(T, 0.1, contact_states=contact_states)
        ds = self._dataset([traj], cfg)
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")

        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        metrics = trainer.sil_update(model, ds)
        self.assertGreater(metrics.get("sil_updates", 0), 0)
        # 近接触窗口 [11,29] 共 19 个；空中帧（sample=0）全丢。窗口总数 = 动作数 40。
        self.assertEqual(metrics["sil_windows_kept"], 19.0)
        self.assertAlmostEqual(metrics["sil_airborne_frac"], 1.0 - 19.0 / 40.0, places=3)
        self.assertTrue(torch.allclose(
            before["actor_log_std"],
            dict(model.named_parameters())["actor_log_std"].detach()))

    def test_phase_disabled_without_contact(self):
        """轨迹缺接触信号时相位分量缺失，不影响优势加权路径。"""
        cfg = _config()  # 相位加权默认开，但轨迹无 contact
        ds = self._dataset([_make_traj(40, 0.1)], cfg)
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")
        self.assertIsNone(trainer._sil_phase_weights(ds))
        metrics = trainer.sil_update(model, ds)
        self.assertGreater(metrics.get("sil_updates", 0), 0)
        self.assertNotIn("sil_airborne_frac", metrics)

    def test_positive_advantage_gating(self):
        """(R−V)₊：critic 系统性高估（V≫R）时全池零权重；无价值校准时自门控跳过。"""
        cfg = _config(sil_value_coef=0.0)  # 关价值校准，测纯门控跳过
        ds = self._dataset([_make_traj(40, 0.1), _make_traj(40, 0.05)], cfg)
        model = ActorCritic(cfg).eval()
        with torch.no_grad():
            model.critic_head[2].bias.fill_(100.0)  # V 系统性远超 R
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")

        weights, pos_frac, _ = trainer._sil_advantage_weights(model, ds)
        self.assertTrue(np.all(weights == 0.0))
        self.assertEqual(pos_frac, 0.0)

        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        metrics = trainer.sil_update(model, ds)
        self.assertEqual(metrics.get("sil_updates", 1), 0)   # 自门控跳过
        self.assertEqual(metrics.get("sil_skipped"), 1.0)
        for k, p in model.named_parameters():
            self.assertTrue(torch.allclose(before[k], p.detach()), k)

    def test_gated_off_still_calibrates_value(self):
        """门控期间（无 A>0 窗口）价值校准（1.3）继续运行——修复高估才能解除门控。"""
        cfg = _config(sil_value_coef=0.1)  # 价值校准开
        ds = self._dataset([_make_traj(40, 0.1), _make_traj(40, 0.05)], cfg)
        model = ActorCritic(cfg).eval()
        with torch.no_grad():
            model.critic_head[2].bias.fill_(100.0)
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")

        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        metrics = trainer.sil_update(model, ds)
        # 不跳过：价值校准继续；actor_mean 不动（权重全零），critic 动
        self.assertGreater(metrics.get("sil_updates", 0), 0)
        self.assertIn("sil_value_loss", metrics)
        self.assertTrue(torch.allclose(
            before["actor_mean.weight"],
            dict(model.named_parameters())["actor_mean.weight"].detach()))
        # V 从 100 向 R_full（~0-3）校准 → bias 严格下降（幅度小，不用 allclose 判）
        after_bias = dict(model.named_parameters())["critic_head.2.bias"].detach()
        self.assertLess(after_bias.item(), before["critic_head.2.bias"].item())

    def test_gate_skips_when_pool_not_better(self):
        """池 median 总 credited ≤ 行为水平 ×(1+margin) → 整轮跳过，模型参数不变。"""
        cfg = _config()  # sil_gate_margin 默认 0.0
        ds = self._dataset([_make_traj(40, 0.1)], cfg)  # 池总量 = 0.1×40 = 4.0m
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")

        before = {k: v.detach().clone() for k, v in model.named_parameters()}
        metrics = trainer.sil_update(model, ds, behavior_progress=10.0)  # 行为 10m > 池 4m
        self.assertEqual(metrics.get("sil_updates", 1), 0.0)
        self.assertEqual(metrics.get("sil_skipped", 0), 1.0)
        self.assertAlmostEqual(metrics.get("sil_pool_level"), 4.0, places=5)
        self.assertEqual(metrics.get("sil_behavior_level"), 10.0)
        self.assertNotIn("sil_loss", metrics)  # 跳过时不产生损失
        self.assertNotIn("sil_adv_pos_frac", metrics)  # 门控在权重前向前，不计算优势
        for k, p in model.named_parameters():
            self.assertTrue(torch.allclose(before[k], p.detach()), f"{k} 被改动")

    def test_gate_opens_when_pool_better(self):
        """池优于当前行为水平 → 门控打开，正常加权更新并记录两侧水平。"""
        cfg = _config()
        ds = self._dataset([_make_traj(40, 0.1)], cfg)  # 4.0m
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")
        metrics = trainer.sil_update(model, ds, behavior_progress=1.0)  # 行为 1m < 池 4m
        self.assertGreater(metrics["sil_updates"], 0)
        self.assertAlmostEqual(metrics.get("sil_pool_level"), 4.0, places=5)
        self.assertEqual(metrics.get("sil_behavior_level"), 1.0)
        self.assertIn("sil_adv_pos_frac", metrics)

    def test_budget_constraint_caps_batches(self):
        """SIL minibatch 预算 = ratio × 当轮 PPO 更新数；预算 0 → 整轮跳过。"""
        cfg = _config()  # sil_batch_size=16, sil_max_batches=3, ratio 默认 1.0
        ds = self._dataset([_make_traj(40, 0.1), _make_traj(40, 0.05)], cfg)
        model = ActorCritic(cfg).eval()
        trainer = PPOTrainer(cfg)
        trainer.device = torch.device("cpu")

        metrics = trainer.sil_update(model, ds, budget_updates=1)
        self.assertEqual(metrics["sil_updates"], 1.0)
        self.assertEqual(metrics.get("sil_budget_cap"), 1.0)

        metrics = trainer.sil_update(model, ds, budget_updates=2)
        self.assertEqual(metrics["sil_updates"], 2.0)
        self.assertEqual(metrics.get("sil_budget_cap"), 2.0)

        # 预算 0 → 跳过（在计算权重之前返回，不带优势指标）
        metrics = trainer.sil_update(model, ds, budget_updates=0)
        self.assertEqual(metrics["sil_updates"], 0.0)
        self.assertEqual(metrics.get("sil_skipped"), 1.0)
        self.assertEqual(metrics.get("sil_budget_cap"), 0.0)

        # 不传预算 → 旧行为：仅受 sil_max_batches 约束
        metrics = trainer.sil_update(model, ds)
        self.assertEqual(metrics["sil_updates"], 3.0)
        self.assertNotIn("sil_budget_cap", metrics)

    def test_sil_value_calibration(self):
        """1.3 critic 精英回报：coef>0 时 critic_head 更新且 V 向 R_full 靠拢；coef=0 不动。"""
        def _run(coef: float):
            torch.manual_seed(13)
            np.random.seed(13)
            cfg = _config(sil_value_coef=coef)
            ds = self._dataset([_make_traj(40, 0.1), _make_traj(40, 0.05)], cfg)
            model = ActorCritic(cfg).eval()
            # 伪造恒定高 V：所有窗口的 R_full 目标都低于 V → V 应被拉低
            last = model.critic_head[-1]
            last.weight.data.zero_()
            last.bias.data.fill_(100.0)
            trainer = PPOTrainer(cfg)
            trainer.device = torch.device("cpu")
            before = {k: v.detach().clone() for k, v in model.named_parameters()}
            metrics = trainer.sil_update(model, ds)
            v_after = model.critic_head[-1].bias.item()
            return metrics, before, model, v_after

        # coef=0：critic_head 不被 SIL 改动（actor NLL 不经过 critic_head）
        m0, b0, model0, _ = _run(0.0)
        self.assertNotIn("sil_value_loss", m0)
        for k, p in model0.named_parameters():
            if "critic_head" in k:
                self.assertTrue(torch.allclose(b0[k], p.detach()), f"{k} 被改动")

        # coef>0：critic_head 更新，且 V 从恒定 100 向 R_full 目标下降
        m1, _, _, v1 = _run(0.1)
        self.assertIn("sil_value_loss", m1)
        self.assertTrue(np.isfinite(m1["sil_value_loss"]))
        self.assertLess(v1, 100.0)

    def test_target_clamp_bounds_pull(self):
        """满饱和精英动作（atanh 后 ±7.6）：限幅使 NLL 损失严格更小（目标被约束）。"""
        def _loss(clamp: float) -> float:
            torch.manual_seed(11)
            np.random.seed(11)
            cfg = _config(sil_target_clamp=clamp)
            traj = _make_traj(40, 0.1)
            traj.actions = np.full((40, 2), 100.0, dtype=np.float32)  # 满饱和
            ds = self._dataset([traj], cfg)
            model = ActorCritic(cfg).eval()
            trainer = PPOTrainer(cfg)
            trainer.device = torch.device("cpu")
            metrics = trainer.sil_update(model, ds)
            self.assertGreater(metrics["sil_updates"], 0)
            return metrics["sil_loss"]

        self.assertLess(_loss(3.0), _loss(0.0))  # 0 = 关闭限幅


if __name__ == "__main__":
    unittest.main()

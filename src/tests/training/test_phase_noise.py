"""相位探索噪声（空中相放大）的离线回归测试。

验证：
  - 空中帧（tip/body/pot 全空）的 std 相对接触帧恰好放大 exp(bonus) 倍；
  - 调制是观测的确定性函数：同一输入两次前向 std 一致（PPO 比率自洽的前提）；
  - 开关关闭时空中/接触 std 相同；
  - clamp 上限仍然生效（bonus 不会把 σ 推出 ppo_log_std_max）。
"""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.dataset import BASE_DYNAMICS_DIM, DYNAMICS_DIM


def _config(**kw):
    return TrainConfig(d_model=16, nhead=2, num_layers=1, context_len=3,
                       num_agents=2, random_deploy=False, deploy_exclusion_check=False,
                       contact_heads_enabled=False, **kw)


def _inputs(tip: float, body: float = 0.0, pot: float = 0.0):
    """(B=2, T=3, DYNAMICS_DIM) 输入；末帧接触列按参数设置。"""
    torch.manual_seed(0)
    dyn = torch.randn(2, 3, DYNAMICS_DIM) * 0.1
    dyn[:, -1, BASE_DYNAMICS_DIM:] = 0.0
    dyn[:, -1, BASE_DYNAMICS_DIM + 0] = tip    # tip_contact
    dyn[:, -1, BASE_DYNAMICS_DIM + 2] = body   # body_contact
    dyn[:, -1, BASE_DYNAMICS_DIM + 3] = pot    # pot_contact
    pat = torch.randn(2, 3, 4, 32, 32)
    act = torch.randn(2, 3, 2) * 0.1
    mask = torch.ones(2, 3)
    return dyn, pat, act, mask


class PhaseNoiseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def _model(self, cfg):
        model = ActorCritic(cfg).eval()
        with torch.no_grad():
            # 基线 log_std 放到 clamp 区间中段，避免顶到上限掩盖差异
            model.actor_log_std.data.fill_(-1.0)
        return model

    def test_airborne_std_boosted(self):
        cfg = _config()
        model = self._model(cfg)
        with torch.no_grad():
            dist_air, _ = model.forward(*_inputs(tip=0.0))
            dist_con, _ = model.forward(*_inputs(tip=1.0))
        ratio = (dist_air.stddev / dist_con.stddev).mean().item()
        self.assertAlmostEqual(ratio, float(torch.exp(torch.tensor(
            cfg.phase_noise_airborne_log_bonus)).item()), places=4)

    def test_body_or_pot_contact_counts_as_contact(self):
        cfg = _config()
        model = self._model(cfg)
        with torch.no_grad():
            std_body = model.forward(*_inputs(tip=0.0, body=1.0))[0].stddev.mean().item()
            std_pot = model.forward(*_inputs(tip=0.0, pot=1.0))[0].stddev.mean().item()
            std_air = model.forward(*_inputs(tip=0.0))[0].stddev.mean().item()
        self.assertAlmostEqual(std_body, std_pot, places=6)
        self.assertLess(std_body, std_air)

    def test_deterministic_across_forwards(self):
        cfg = _config()
        model = self._model(cfg)
        with torch.no_grad():
            s1 = model.forward(*_inputs(tip=0.0))[0].stddev
            s2 = model.forward(*_inputs(tip=0.0))[0].stddev
        self.assertTrue(torch.equal(s1, s2))

    def test_disabled_when_switch_off(self):
        cfg = _config(phase_noise_enabled=False)
        model = self._model(cfg)
        with torch.no_grad():
            std_air = model.forward(*_inputs(tip=0.0))[0].stddev.mean().item()
            std_con = model.forward(*_inputs(tip=1.0))[0].stddev.mean().item()
        self.assertAlmostEqual(std_air, std_con, places=6)

    def test_clamp_cap_respected(self):
        cfg = _config(phase_noise_airborne_log_bonus=5.0)
        model = self._model(cfg)  # log_std=-1.0 + 5.0 → 远超上限
        with torch.no_grad():
            std_air = model.forward(*_inputs(tip=0.0))[0].stddev.mean().item()
        self.assertAlmostEqual(std_air, float(torch.exp(torch.tensor(
            cfg.ppo_log_std_max)).item()), places=5)


if __name__ == "__main__":
    unittest.main()

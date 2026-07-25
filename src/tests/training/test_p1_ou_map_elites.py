"""离线单测：P1.1 OU 条件高斯 + P1.2 MAP-Elites SIL 准入。无需游戏。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset


def _make_traj(T: int, x0: float, y0: float, dy: float = 2.0) -> Trajectory:
    """合成轨迹：y 缓慢上升，secured peak 落在终点附近。"""
    raw = np.zeros((T + 1, 33), dtype=np.float32)
    raw[:, 0] = x0
    raw[:, 1] = y0 + np.linspace(0, dy, T + 1)
    raw[:, 23] = raw[:, 0] + 1.0
    raw[:, 24] = raw[:, 1] + 1.0
    acts = np.zeros((T, 2), dtype=np.float32)
    return Trajectory(raw_states=raw, actions=acts)


def test_ou_conditional_and_log_prob_consistency() -> None:
    """采集 log_prob 与 evaluate_actions 在同一 noise_prev 下应一致。"""
    torch.manual_seed(0)
    cfg = TrainConfig()
    cfg.ou_enabled = True
    cfg.ou_phi = 0.85
    model = ActorCritic(cfg).eval()

    from training.dataset import DYNAMICS_DIM
    B, T, D = 4, cfg.context_len, DYNAMICS_DIM  # 现为 39 (34 base + 5 contact)
    dynamics = torch.randn(B, T, D)
    patches = torch.randn(B, T, 4, 32, 32)
    act_hist = torch.randn(B, T, 2)
    valid = torch.ones(B, T)
    z_prev = torch.randn(B, 2)

    av = model.get_action_and_value(
        dynamics, patches, act_hist, valid_mask=valid,
        noise_prev=z_prev, noise_phi=cfg.ou_phi,
    )
    lp2, ent, _ = model.evaluate_actions(
        dynamics, patches, act_hist, av.action.detach(),
        valid_mask=valid, noise_prev=z_prev, noise_phi=cfg.ou_phi,
    )
    assert torch.allclose(av.log_prob, lp2, atol=1e-4), (
        f"OU log_prob 不一致: sample={av.log_prob} eval={lp2}"
    )
    assert av.noise is not None and av.noise.shape == (B, 2)
    assert torch.isfinite(ent).all()

    # φ=0 应退化为 iid：两次独立采样的条件均值应回到 policy mean
    mean, std = model._ou_conditional(
        torch.zeros(2), torch.ones(2), z_prev[0], 0.0,
    )
    assert torch.allclose(mean, torch.zeros(2)) and torch.allclose(std, torch.ones(2))


def test_ou_ar1_lag1_near_phi() -> None:
    """合成 AR(1) 序列的 lag-1 自相关应接近 φ。"""
    phi = 0.85
    n = 20000
    z = np.zeros(n, dtype=np.float64)
    z[0] = np.random.randn()
    scale = (1.0 - phi * phi) ** 0.5
    for t in range(1, n):
        z[t] = phi * z[t - 1] + scale * np.random.randn()
    lag1 = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    assert abs(lag1 - phi) < 0.03, f"lag-1={lag1:.3f} 偏离 φ={phi}"
    assert abs(z.var() - 1.0) < 0.08, f"marginal var={z.var():.3f} 应≈1"


def test_map_elites_diversity_and_cold_start() -> None:
    """同 cell 只留最优；冷启动期同 cell 落选仍可填到 cold_start_min。"""
    np.random.seed(0)
    cfg = TrainConfig()
    cfg.dwell_steps = 1  # 缩短驻留，合成单调上升即可确认 peak
    grid = 64
    ds = TrajectoryDataset(
        cfg, np.ones((grid, grid), dtype=bool),
        np.zeros((grid, grid), dtype=np.float32),
        min_gx=0, min_gy=0,
    )

    # 全部挤在同一 5m cell（x∈[0,5), y∈[10,15)），分数递增
    trajs = [_make_traj(20, x0=1.0, y0=10.0 + 0.1 * i) for i in range(6)]
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    stats = ds.add_trajectories_map_elites(
        trajs, scores, iteration=0, cell_size=5.0, cold_start_min=5,
    )
    assert stats["n_cells"] == 1, f"应只有 1 个 cell, got {stats['n_cells']}"
    assert len(ds.trajectories) == 5, (
        f"冷启动应填到 5 条（1 elite + extras）, got {len(ds.trajectories)}"
    )
    assert max(t.score for t in ds.trajectories) == 6.0

    # 新开一个远处 cell：应扩多样性；cell 数≥cold_start_min 后 extras 清空
    far = [
        _make_traj(20, x0=20.0 + 6.0 * i, y0=10.0) for i in range(5)
    ]
    far_scores = [10.0, 11.0, 12.0, 13.0, 14.0]
    stats2 = ds.add_trajectories_map_elites(
        far, far_scores, iteration=1, cell_size=5.0, cold_start_min=5,
    )
    assert stats2["n_cells"] >= 5, f"cells 应≥5, got {stats2['n_cells']}"
    # cell 达标后不应再保留同 cell extras → 池大小 == cell 数
    assert len(ds.trajectories) == stats2["n_cells"], (
        f"稳态池应等于 cell 数: trajs={len(ds.trajectories)} cells={stats2['n_cells']}"
    )

    # 落水轨迹硬拒
    water = _make_traj(10, x0=0.0, y0=-20.0, dy=0.0)
    before = len(ds.trajectories)
    ds.add_trajectories_map_elites(
        [water], [99.0], iteration=2, cell_size=5.0, cold_start_min=5,
    )
    assert len(ds.trajectories) == before, "落水轨迹不应入池"


def main() -> None:
    test_ou_conditional_and_log_prob_consistency()
    print("OU log_prob consistency OK")
    test_ou_ar1_lag1_near_phi()
    print("OU lag-1 ≈ φ OK")
    test_map_elites_diversity_and_cold_start()
    print("MAP-Elites cold-start / diversity OK")
    print("TEST_P1_OK")


if __name__ == "__main__":
    main()

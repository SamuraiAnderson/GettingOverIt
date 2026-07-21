"""离线冒烟：验证 PPO Self-Imitation (sil_update) 代码路径无需游戏即可跑通。

合成最小 TrajectoryDataset + ActorCritic，走完
dataset 构建 → patch 重建 → DataLoader → forward → MSE-on-mean → backward/step。
"""
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
from training.ppo_trainer import PPOTrainer


def _make_traj(T: int, y0: float) -> Trajectory:
    raw = np.zeros((T + 1, 33), dtype=np.float32)
    raw[:, 0] = 50.0 + np.linspace(0, 3, T + 1)      # x 缓慢右移
    raw[:, 1] = y0 + np.linspace(0, 8, T + 1)        # y 单调上升 → secured peak 覆盖全程
    raw[:, 23] = raw[:, 0] + 1.0                      # tip x
    raw[:, 24] = raw[:, 1] + 1.0                      # tip y
    acts = np.tanh(np.random.randn(T, 2).astype(np.float32)) * 0.5
    return Trajectory(raw_states=raw, actions=acts)


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    config = TrainConfig()
    print(f"patch_mode={config.patch_mode}, sil_enabled={config.sil_enabled}, "
          f"sil_batch_size={config.sil_batch_size}, sil_loss_coef={config.sil_loss_coef}")

    # 合成效率图 + 全通行地形（dualscale 只用 efficiency_arr[0]，其余通道无几何→零）
    grid = 256
    eff_arr = np.random.rand(grid, grid).astype(np.float32)
    terrain = np.ones((grid, grid), dtype=bool)

    dataset = TrajectoryDataset(
        config, terrain, eff_arr,
        min_gx=0, min_gy=0,
        solid_polygons=None, tip_local=None, body_local=None,
    )

    trajs = [_make_traj(60, y0=10.0), _make_traj(60, y0=12.0), _make_traj(60, y0=8.0)]
    scores = [5.0, 9.0, 3.0]
    dataset.add_trajectories(trajs, scores, keep_ratio=0.3, iteration=0)
    top = sorted(dataset.trajectories, key=lambda t: t.score, reverse=True)[:config.sil_traj_cap]
    dataset.set_trajectories(top)
    print(f"精英池: {len(dataset.trajectories)} 轨迹, {len(dataset)} 训练窗口")
    assert len(dataset) > 0, "精英池窗口为空"

    # __getitem__ 形状检查
    dyn, pat, act_hist, vm, target = dataset[0]
    print(f"样本形状: dyn={tuple(dyn.shape)} pat={tuple(pat.shape)} "
          f"act={tuple(act_hist.shape)} vm={tuple(vm.shape)} target={tuple(target.shape)}")

    model = ActorCritic(config)
    trainer = PPOTrainer(config)
    trainer.device = torch.device("cpu")

    # 空数据集应安全返回 {}
    empty = TrajectoryDataset(config, terrain, eff_arr, 0, 0)
    assert trainer.sil_update(model, empty) == {}, "空数据集未返回 {}"

    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    metrics = trainer.sil_update(model, dataset)
    print("sil_update metrics:", {k: round(v, 4) for k, v in metrics.items()})

    assert metrics.get("sil_updates", 0) > 0, "无 SIL 更新步"
    assert np.isfinite(metrics["sil_loss"]), "sil_loss 非有限"
    # log_std 不应被 MSE-on-mean 修改（只作用 mean）
    ls_before = before["actor_log_std"]
    ls_after = dict(model.named_parameters())["actor_log_std"].detach()
    assert torch.allclose(ls_before, ls_after), "actor_log_std 被 SIL 改动（应不变）"
    # backbone/actor_mean 应有更新
    changed = any(
        not torch.allclose(before[k], p.detach())
        for k, p in model.named_parameters() if "actor_mean" in k
    )
    assert changed, "actor_mean 权重未更新"

    print("SMOKE_SIL_OK")


if __name__ == "__main__":
    main()

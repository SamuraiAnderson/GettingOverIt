"""
离线 sanity：验证 dualscale / legacy 两种 patch_mode 的 build_patch 形状与通道装配，
以及 ActionPredictor / ActorCritic 在两种模式下前向不报错。无需游戏 / 训练。

运行：python src/tests/training/sanity_dualscale_patch.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.dataset import DYNAMICS_DIM, build_patch
from training.model import ActionPredictor


def _square(cx, cy, w, h):
    return np.array([[cx - w, cy - h], [cx + w, cy - h],
                     [cx + w, cy + h], [cx - w, cy + h]], dtype=np.float64)


def _make_state():
    """构造一个 33D 状态：player 在 (10, 10)，hub 在其上方，tip 在右侧。"""
    s = np.zeros(33, dtype=np.float32)
    s[0], s[1] = 10.0, 10.0        # player
    s[5], s[6] = 10.1, 10.9        # hub（上方）
    s[10], s[11] = 10.0, 10.4      # slider
    s[15], s[16] = 11.0, 10.5      # handle
    s[19], s[20] = 11.6, 10.6      # pole
    s[23], s[24] = 12.2, 10.7      # tip（右侧）
    s[27] = 15.0                   # hammerAngle
    return s


def _geometry():
    # 全局 1m 栅格的地形/效率图（100x100，覆盖原点周边）
    terrain = np.zeros((100, 100), dtype=np.float32)
    terrain[:5, :] = 1.0           # 底部一片实体
    eff = np.random.default_rng(0).random((100, 100)).astype(np.float32)
    # 接触判定用的实心多边形（世界坐标）：player 下方一块地面
    polygons = [_square(11.0, 9.0, 3.0, 1.0), _square(12.2, 10.0, 0.4, 0.4)]
    tip_local = _square(0.0, 0.0, 0.07, 0.22)          # tip 自身系
    body_local = [_square(0.0, 0.5, 0.3, 0.5),          # body（上）
                  _square(0.0, -0.8, 0.35, 0.14)]       # pot（下）
    return terrain, eff, polygons, tip_local, body_local


def main():
    torch.manual_seed(0)
    base = TrainConfig()
    state = _make_state()
    terrain, eff, polygons, tip_local, body_local = _geometry()
    min_gx = min_gy = 0

    # ── build_patch: dualscale ──
    cfg_dual = replace(base, patch_mode="dualscale")
    p = build_patch(state, terrain, eff, cfg_dual, min_gx, min_gy,
                    solid_polygons=polygons, tip_local=tip_local, body_local=body_local)
    assert p.shape == (cfg_dual.patch_channels, cfg_dual.patch_size, cfg_dual.patch_size), p.shape
    ch_sum = [float(p[i].sum()) for i in range(p.shape[0])]
    print(f"[dualscale] patch shape={p.shape} channel_sums={['%.1f' % c for c in ch_sum]}")
    assert ch_sum[0] > 0, "wide eff 空"
    assert ch_sum[1] > 0, "contact 地形空"
    assert ch_sum[2] > 0, "contact 锅/body 轮廓空"
    assert ch_sum[3] > 0, "contact 锤头轮廓空"
    assert np.isfinite(p).all()

    # ── build_patch: legacy ──
    cfg_leg = replace(base, patch_mode="legacy")
    pl = build_patch(state, terrain, eff, cfg_leg, min_gx, min_gy)
    assert pl.shape == (cfg_leg.patch_channels, cfg_leg.patch_size, cfg_leg.patch_size), pl.shape
    print(f"[legacy]    patch shape={pl.shape} "
          f"channel_sums={['%.1f' % float(pl[i].sum()) for i in range(pl.shape[0])]}")
    assert np.isfinite(pl).all()

    # ── 模型前向（两种模式）──
    B, T = 2, base.context_len
    for mode in ("dualscale", "legacy"):
        cfg = replace(base, patch_mode=mode)
        dyn = torch.randn(B, T, DYNAMICS_DIM)
        pat = torch.randn(B, T, cfg.patch_channels, cfg.patch_size, cfg.patch_size)
        act = torch.randn(B, T, cfg.action_dim)
        vmask = torch.ones(B, T)

        ap = ActionPredictor(cfg).eval()
        with torch.no_grad():
            out = ap(dyn, pat, act, vmask)
        assert out.shape == (B, cfg.action_dim), out.shape

        ac = ActorCritic(cfg).eval()
        with torch.no_grad():
            dist, value = ac.forward(dyn, pat, act, vmask)
        assert value.shape[0] == B
        print(f"[{mode}] ActionPredictor out={tuple(out.shape)} "
              f"ActorCritic value={tuple(value.shape)}  OK")

    print("\nALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()

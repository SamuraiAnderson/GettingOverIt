"""离线定位 NaN 来源：扫描 warmup_state.pkl 轨迹的 raw_states / dynamics / patch 各通道。

用法：
  python -m src.tests.training.diagnose_nan
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig
from training.dataset import (
    Trajectory,
    build_dynamics,
    build_patch,
)
from training.deploy_sampling import extract_body_local, extract_tip_local
from training.reward import ClimbingEfficiencyMap

sys.path.insert(0, str(_REPO_ROOT / "src" / "tests" / "control_interaction"))
from test_l7_surface_airdrop import extract_polygons, load_colliders


def _bad(arr: np.ndarray) -> tuple[int, int]:
    a = np.asarray(arr, dtype=np.float64)
    return int(np.isnan(a).sum()), int(np.isinf(a).sum())


def main() -> None:
    config = TrainConfig()

    # ── 加载轨迹 ──
    warmup_path = Path(config.checkpoint_dir) / "warmup_state.pkl"
    with open(warmup_path, "rb") as f:
        state = pickle.load(f)
    trajs = [
        Trajectory(
            raw_states=d["raw_states"],
            actions=d["actions"],
            score=d["score"],
            iteration=d["iteration"],
        )
        for d in state["trajectories"]
    ]
    print(f"[load] {len(trajs)} trajectories")

    # ── 1) 原始状态 / 动作 NaN 扫描 ──
    print("\n=== 1) raw_states / actions ===")
    total_raw_nan = total_raw_inf = 0
    for i, t in enumerate(trajs):
        rn, ri = _bad(t.raw_states)
        an, ai = _bad(t.actions)
        total_raw_nan += rn + an
        total_raw_inf += ri + ai
        if rn or ri or an or ai:
            print(f"  traj[{i}] raw_states nan={rn} inf={ri} | actions nan={an} inf={ai} "
                  f"| shape={t.raw_states.shape}")
            # 定位到具体行/列
            a = np.asarray(t.raw_states, dtype=np.float64)
            rows = np.where(~np.isfinite(a).all(axis=1))[0]
            for r in rows[:5]:
                cols = np.where(~np.isfinite(a[r]))[0]
                print(f"      row {r}: bad cols {cols.tolist()} vals={a[r][cols].tolist()}")
    print(f"  TOTAL raw nan={total_raw_nan} inf={total_raw_inf}")

    # 也报告极端值（可能溢出）
    all_raw = np.concatenate([np.asarray(t.raw_states, np.float64) for t in trajs], axis=0)
    finite = all_raw[np.isfinite(all_raw)]
    print(f"  raw finite range: min={finite.min():.3g} max={finite.max():.3g} "
          f"absmax={np.abs(finite).max():.3g}")

    # ── 2) dynamics NaN 扫描 + 各维极值 ──
    print("\n=== 2) build_dynamics ===")
    dyn_bad = 0
    all_dyn = []
    for i, t in enumerate(trajs):
        d = build_dynamics(t.raw_states)
        all_dyn.append(d)
        n, inf = _bad(d)
        dyn_bad += n + inf
        if n or inf:
            print(f"  traj[{i}] dynamics nan={n} inf={inf}")
    print(f"  TOTAL dynamics nan/inf={dyn_bad}")
    stacked = np.concatenate(all_dyn, axis=0)
    absmax = np.abs(stacked).max(axis=0)
    worst = np.argsort(absmax)[::-1][:6]
    print(f"  dynamics per-dim absmax top6: "
          + ", ".join(f"dim{d}={absmax[d]:.3g}" for d in worst))

    # ── 3) patch 各通道 NaN 扫描（需要几何 + 效率图）──
    print("\n=== 3) build_patch (per channel) ===")
    game_root = Path(config.game_root)
    env_data, player_data = load_colliders(game_root)
    polygons = extract_polygons(env_data)
    tip_local = extract_tip_local(player_data)
    body_local = extract_body_local(player_data)

    cache_dir = Path(config.checkpoint_dir) / "cache"
    mask_cache = str(cache_dir / "traversable_mask.npz")
    eff_map = ClimbingEfficiencyMap(
        polygons,
        grid_resolution=config.grid_resolution,
        diffusion_iterations=config.diffusion_iterations,
        diffusion_alpha=config.diffusion_alpha,
        cache_path=mask_cache,
        height_prior_weight=config.height_prior_weight,
    )
    # 复刻训练 Phase0：用轨迹 update 效率图后再取 arr
    eff_map.update(trajs, config, prev_eff_map=None)
    eff_arr = eff_map.get_arr()
    terrain_mask = getattr(eff_map, "traversable", None)
    min_gx, min_gy = eff_map._min_gx, eff_map._min_gy

    if eff_arr is None:
        print("  efficiency_arr is None after update!")
    else:
        en, ei = _bad(eff_arr)
        print(f"  efficiency_arr nan={en} inf={ei} shape={eff_arr.shape} "
              f"range=[{np.nanmin(eff_arr):.3g},{np.nanmax(eff_arr):.3g}]")

    ch_bad = np.zeros(config.patch_channels, dtype=int)
    examples: list[str] = []
    n_steps = 0
    for i, t in enumerate(trajs):
        for s in t.raw_states:
            n_steps += 1
            patch = build_patch(
                s, terrain_mask, eff_arr, config, min_gx, min_gy,
                solid_polygons=polygons, tip_local=tip_local, body_local=body_local,
            )
            for c in range(config.patch_channels):
                n, inf = _bad(patch[c])
                if n or inf:
                    ch_bad[c] += 1
                    if len(examples) < 10:
                        examples.append(
                            f"  traj[{i}] step player=({s[0]:.2f},{s[1]:.2f}) "
                            f"ch{c} nan={n} inf={inf}")
    print(f"  scanned {n_steps} steps")
    print(f"  per-channel bad-patch counts: {ch_bad.tolist()}")
    for e in examples:
        print(e)

    # ── 4) 模型 forward NaN 扫描（eval 模式，复刻 val）──
    print("\n=== 4) model forward ===")
    import torch

    from training.dataset import TrajectoryDataset, compute_dynamics_stats
    from training.model import ActionPredictor

    dataset = TrajectoryDataset(
        config, terrain_mask, eff_arr,
        min_gx=min_gx, min_gy=min_gy,
        solid_polygons=polygons, tip_local=tip_local, body_local=body_local,
    )
    dataset.set_trajectories(trajs)
    model = ActionPredictor(config)
    mean, std = compute_dynamics_stats(trajs)
    model.set_dynamics_stats(mean, std)
    model.eval()
    print(f"  dataset windows={len(dataset)}; dyn_std min={std.min():.3g} "
          f"(dim of min={int(np.argmin(std))})")

    nan_samples = 0
    worst_examples: list[str] = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            dyn, pat, act, vm, tgt = dataset[idx]
            out = model(dyn.unsqueeze(0), pat.unsqueeze(0),
                        act.unsqueeze(0), valid_mask=vm.unsqueeze(0))
            if not torch.isfinite(out).all():
                nan_samples += 1
                if len(worst_examples) < 8:
                    dn = model._normalize_dynamics(dyn)
                    worst_examples.append(
                        f"  idx={idx} out={out.flatten().tolist()} "
                        f"raw_dyn_absmax={dyn.abs().max():.3g} "
                        f"norm_dyn_absmax={dn.abs().max():.3g}")
    print(f"  NaN/inf forward outputs: {nan_samples}/{len(dataset)}")
    for e in worst_examples:
        print(e)

    print("\n=== SUMMARY ===")
    print(f"  raw nan/inf={total_raw_nan + total_raw_inf}, "
          f"dynamics nan/inf={dyn_bad}, patch bad per ch={ch_bad.tolist()}, "
          f"model-forward NaN={nan_samples}")


if __name__ == "__main__":
    main()

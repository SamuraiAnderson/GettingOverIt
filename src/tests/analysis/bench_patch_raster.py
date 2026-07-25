r"""基准：面积覆盖率栅格化对 build_patch 热路径的开销。

`build_patch` 在 rollout 每 agent 每帧都调用，ch2/ch3 改用面积覆盖率后需确认
未拖慢采集。对比 `subsamples=1`（旧二值语义）与默认 `PART_RASTER_SUBSAMPLES`。

关键设计：子采样只在多边形 bbox 覆盖的格子上做，部件轮廓 bbox 仅几格，
故开销应与二值版同量级、且不随 patch_size 放大。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig
from training.dataset import (
    PART_RASTER_SUBSAMPLES,
    build_patch,
    rasterize_polygon_fill,
)

_TIP_LOCAL = np.array([
    [-0.0709, -0.2791], [0.0768, -0.2791], [0.0768, 0.1687], [-0.0709, 0.1687],
], dtype=np.float64)
_BODY_LOCAL = np.array(
    [[-0.2919, 0.0828], [0.3460, 0.0828], [0.3460, 1.0401], [-0.2919, 1.0401]],
    dtype=np.float64,
)
_POT_LOCAL = np.array(
    [[-0.3239, -0.9767], [0.3824, -0.9767], [0.3824, -0.7000], [-0.3239, -0.7000]],
    dtype=np.float64,
)


def _make_state(px: float, py: float, tip_dx: float, tip_dy: float) -> np.ndarray:
    s = np.zeros(33, dtype=np.float32)
    s[0], s[1] = px, py
    s[5], s[6] = px, py + 0.3546
    s[23], s[24] = px + tip_dx, py + tip_dy
    s[19], s[20] = px + tip_dx * 0.7, py + tip_dy * 0.7
    return s


def _fake_terrain(n_poly: int = 40) -> list[np.ndarray]:
    """造一片地形多边形，模拟 rasterize_solid_local 的实际负载。"""
    rng = np.random.default_rng(0)
    polys = []
    for _ in range(n_poly):
        cx, cy = rng.uniform(-20, 20), rng.uniform(-20, 20)
        w, h = rng.uniform(1.0, 6.0), rng.uniform(1.0, 6.0)
        polys.append(np.array([
            [cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
            [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2],
        ], dtype=np.float64))
    return polys


def bench_primitive(reps: int = 2000) -> None:
    print("── 图元级：rasterize_polygon_fill（单个 tip 轮廓）──")
    res = TrainConfig().contact_patch_resolution
    world = _TIP_LOCAL + np.array([1.5, 0.8])
    for n in (1, PART_RASTER_SUBSAMPLES, 8):
        t0 = time.perf_counter()
        for _ in range(reps):
            rasterize_polygon_fill(world, 0.0, 0.0, 32, res, subsamples=n)
        dt = (time.perf_counter() - t0) / reps * 1e6
        tag = " (旧二值语义)" if n == 1 else (
            " ← 当前配置" if n == PART_RASTER_SUBSAMPLES else "")
        print(f"  subsamples={n}: {dt:8.1f} μs/次{tag}")


def bench_build_patch(reps: int = 500) -> None:
    print("\n── 整体：build_patch（含 ch1 地形栅格化 + ch2/ch3 轮廓）──")
    cfg = TrainConfig()
    terrain = _fake_terrain()
    eff = np.random.rand(200, 200).astype(np.float32)
    states = [
        _make_state(0.0, 0.0, 3.5 * np.cos(t), 3.5 * np.sin(t))
        for t in np.linspace(0, 2 * np.pi, 16)
    ]

    t0 = time.perf_counter()
    for i in range(reps):
        build_patch(
            states[i % len(states)], None, eff, cfg, 0, 0,
            solid_polygons=terrain,
            tip_local=_TIP_LOCAL,
            body_local=[_BODY_LOCAL, _POT_LOCAL],
        )
    dt = (time.perf_counter() - t0) / reps * 1e3
    print(f"  build_patch: {dt:.3f} ms/次")

    n_agents = cfg.num_agents
    print(f"  → {n_agents} agents 每帧 patch 构建 ≈ {dt * n_agents:.2f} ms")
    print(f"  → 若每轮 rollout 400 步: ≈ {dt * n_agents * 400 / 1000:.1f} s")


def main() -> None:
    print(f"contact_patch_resolution = {TrainConfig().contact_patch_resolution} m/px")
    print(f"PART_RASTER_SUBSAMPLES   = {PART_RASTER_SUBSAMPLES}\n")
    bench_primitive()
    bench_build_patch()
    print("\n判读：ch2/ch3 的子采样只在轮廓 bbox（几格）内做，故图元开销应与二值同量级；")
    print("      build_patch 的主要成本在 ch1 地形栅格化（1024 格 × 多边形预筛），未改动。")


if __name__ == "__main__":
    main()

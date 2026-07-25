r"""诊断：tip 轮廓在 contact patch 上的栅格化命中率。

起因：给 `test_patch_tip_coverage.py` 写回归测试时发现，锤子全伸到几何上界后
ch3（锤头轮廓通道）在 0°/90°/180°/270° 等方向上**完全为空**。排查方向指向
栅格化走样而非出窗：

- 真实 tip 轮廓 bbox = 0.148 (宽) × 0.448 (高) m
- `rasterize_polygon_fill` 用「格中心点是否在多边形内」判定（MplPath.contains_points）
- 0.25 m/px 下轮廓宽仅 0.59 px、0.2 m/px 下 0.74 px —— **都不足 1 像素**
  → 多边形可以整体落在相邻格中心之间，一个格心都不含 → 通道全零

本脚本用真实 `player_contour.json` 轮廓，扫描 tip 的朝向 × 亚像素位置偏移，
统计各分辨率下 ch3 的空通道率与平均命中像素数，用于判断是否需要改栅格化方式
（而非只调窗口大小）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.dataset import rasterize_polygon_fill
from training.deploy_sampling import reconstruct_polygon_world

_PROJECT_CONFIG = _REPO_ROOT / "src" / "config" / "project.json"

_TIP_FALLBACK = np.array([
    [-0.0709, -0.2791], [0.0768, -0.2791], [0.0768, 0.1687], [-0.0709, 0.1687],
], dtype=np.float64)


def _load_tip_local() -> tuple[np.ndarray, str]:
    try:
        with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
            game_root = Path(json.load(f)["game"]["executable_path"]).parent
        path = game_root / "GoiData" / "Colliders" / "player_contour.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        paths = data["parts"]["tip"]["paths"]
        return np.asarray(paths[0], dtype=np.float64), "player_contour.json (真实)"
    except Exception:
        return _TIP_FALLBACK, "兜底矩形 (近似)"


def scan(tip_local: np.ndarray, res: float, size: int = 32) -> dict:
    """扫描朝向 × 亚像素偏移，返回命中率统计。"""
    empty = 0
    total = 0
    hits: list[int] = []
    # 亚像素偏移：在一个格子内均匀取 5×5 个 tip 中心位置
    sub = np.linspace(0.0, res, 5, endpoint=False)
    for deg in range(0, 360, 15):
        for ox in sub:
            for oy in sub:
                # tip 放在 patch 中心附近 + 亚像素偏移；patch 中心 = (0,0)
                world = reconstruct_polygon_world(
                    (float(ox), float(oy)), float(deg), tip_local, 1.0,
                )
                ch = rasterize_polygon_fill(world, 0.0, 0.0, size, res)
                n = int((ch > 0).sum())
                hits.append(n)
                total += 1
                if n == 0:
                    empty += 1
    arr = np.asarray(hits)
    return {
        "res": res,
        "empty_rate": empty / total,
        "mean_px": float(arr.mean()),
        "median_px": float(np.median(arr)),
        "max_px": int(arr.max()),
        "total": total,
    }


def main() -> None:
    tip_local, src = _load_tip_local()
    w = tip_local[:, 0].max() - tip_local[:, 0].min()
    h = tip_local[:, 1].max() - tip_local[:, 1].min()
    print("=" * 72)
    print("tip 轮廓栅格化命中率诊断")
    print("=" * 72)
    print(f"轮廓来源: {src}, 顶点数 {len(tip_local)}")
    print(f"轮廓 bbox: {w:.4f} (宽) × {h:.4f} (高) m")
    print(f"R_max = {np.hypot(tip_local[:, 0], tip_local[:, 1]).max():.4f} m\n")

    print(f"{'res(m/px)':>10} {'宽(px)':>8} {'空通道率':>10} {'平均命中px':>12} "
          f"{'中位数':>8} {'最大':>6}")
    print("-" * 72)
    for res in (0.5, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05):
        st = scan(tip_local, res)
        print(f"{res:>10.2f} {w / res:>8.2f} {st['empty_rate'] * 100:>9.1f}% "
              f"{st['mean_px']:>12.2f} {st['median_px']:>8.1f} {st['max_px']:>6d}")

    print("\n判读：空通道率 = 锤头在 ch3 里完全看不见的姿态/位置比例。")
    print("      宽(px) < 1 时，「格心在多边形内」判定必然大量漏采。")


if __name__ == "__main__":
    main()

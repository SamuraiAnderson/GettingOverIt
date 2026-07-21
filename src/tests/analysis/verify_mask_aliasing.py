"""
效率图 traversable mask 走样核验（离线，只用 environment.json，无需 torch）。

动机：效率图网格默认 1.0m，而碰撞边中位仅 0.69m、最近邻间距 0.53~2.53m（见
effectiveness_report.md B）。`ClimbingEfficiencyMap._build_traversable_mask` 只在格子
**中心**做一次点在多边形判定，薄墙(<1m)会被漏判（中心落空 → 误判可通行 → 扩散穿墙），
薄台阶(<1m)会被抹掉。本工具直观核验 1m 中心采样是否走样，以及降到 0.5m / 加 3x3 子格
采样能否恢复这些薄特征，据此决定「维持 1m / 加子格采样 / 降 0.5m」。

用法:
  python -m src.tests.analysis.verify_mask_aliasing
  python -m src.tests.analysis.verify_mask_aliasing --zoom 10 70 120 240
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tests.analysis.env_geometry import (
    default_env_path,
    extract_polygons,
    load_environment,
)

logger = logging.getLogger(__name__)


def build_mask(
    polygons: list[np.ndarray], res: float, subsample: int = 1
) -> tuple[np.ndarray, int, int]:
    """构建 traversable(可通行) 掩码。

    subsample=1: 仅格子中心采样（与生产 _build_traversable_mask 同口径）。
    subsample=k: 每格 k×k 子采样，任一子点在多边形内即判固体（对薄墙保守，恢复漏判）。
    返回 (traversable[h,w] bool, min_gx, min_gy)。
    """
    all_pts = np.concatenate(polygons)
    x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
    y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())

    min_gx = int(np.floor(x_min / res)) - 1
    min_gy = int(np.floor(y_min / res)) - 1
    max_gx = int(np.ceil(x_max / res)) + 1
    max_gy = int(np.ceil(y_max / res)) + 1
    w = max_gx - min_gx + 1
    h = max_gy - min_gy + 1

    paths = [MplPath(p) for p in polygons]
    gx = np.arange(w)
    gy = np.arange(h)
    GX, GY = np.meshgrid(gx, gy)  # (h, w)

    if subsample <= 1:
        offsets = [(0.5, 0.5)]
    else:
        offsets = [
            ((i + 0.5) / subsample, (j + 0.5) / subsample)
            for i in range(subsample)
            for j in range(subsample)
        ]

    solid = np.zeros((h, w), dtype=bool)
    for ox, oy in offsets:
        cx = ((min_gx + GX) + ox) * res
        cy = ((min_gy + GY) + oy) * res
        pts = np.column_stack([cx.ravel(), cy.ravel()])
        inside = np.zeros(len(pts), dtype=bool)
        for p in paths:
            inside |= p.contains_points(pts)
        solid |= inside.reshape(h, w)

    return ~solid, min_gx, min_gy


def _extent(mask: np.ndarray, min_gx: int, min_gy: int, res: float):
    h, w = mask.shape
    return [
        min_gx * res, (min_gx + w) * res,
        min_gy * res, (min_gy + h) * res,
    ]


def _draw_polys(ax, polygons, lw=0.3):
    for p in polygons:
        poly = np.vstack([p, p[0]])
        ax.plot(poly[:, 0], poly[:, 1], color="k", lw=lw, alpha=0.6)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default=None)
    ap.add_argument(
        "--zoom", type=float, nargs=4, default=[10.0, 70.0, 120.0, 240.0],
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        help="放大区域，看薄特征细节",
    )
    ap.add_argument(
        "--out", type=str,
        default=str(_REPO_ROOT / "src" / "Data" / "analysis" / "F_mask_aliasing.png"),
    )
    args = ap.parse_args()

    env_path = Path(args.env) if args.env else default_env_path()
    env = load_environment(env_path)
    polys = extract_polygons(env)
    logger.info("多边形数: %d", len(polys))

    # 三种口径
    m1c, gx1, gy1 = build_mask(polys, 1.0, subsample=1)     # 1m 中心（生产口径）
    m1s, _, _ = build_mask(polys, 1.0, subsample=3)         # 1m 3x3 子采样
    m05c, gx05, gy05 = build_mask(polys, 0.5, subsample=1)  # 0.5m 中心

    # 走样量化：1m 中心判「可通行」但 1m 子采样判「固体」的格 = 中心采样漏判的薄固体
    # （潜在假通道：扩散会从这些格穿墙）
    false_free = m1c & (~m1s)
    n_false = int(false_free.sum())
    n_trav = int(m1c.sum())
    logger.info(
        "1m 中心可通行格: %d; 其中被 3x3 子采样判为固体(疑似漏判薄墙): %d (%.2f%%)",
        n_trav, n_false, 100.0 * n_false / max(n_trav, 1),
    )
    logger.info(
        "可通行占比 — 1m中心 %.3f, 1m子采样 %.3f, 0.5m中心 %.3f",
        m1c.mean(), m1s.mean(), m05c.mean(),
    )

    xmin, xmax, ymin, ymax = args.zoom
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))

    # 标签用英文，避免默认字体缺 CJK 字形渲染成方块（不影响数据）。
    panels = [
        (m1c, gx1, gy1, 1.0, "1m center (production)"),
        (m1s, gx1, gy1, 1.0, "1m 3x3 subsample (recovers thin walls)"),
        (m05c, gx05, gy05, 0.5, "0.5m center"),
    ]
    for col, (mask, mgx, mgy, res, title) in enumerate(panels):
        ext = _extent(mask, mgx, mgy, res)
        ax = axes[0, col]
        ax.imshow(
            mask, origin="lower", extent=ext, cmap="Greys_r",
            interpolation="nearest", aspect="equal",
        )
        _draw_polys(ax, polys, lw=0.2)
        ax.set_title(f"Full map | {title}")
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")

        axz = axes[1, col]
        axz.imshow(
            mask, origin="lower", extent=ext, cmap="Greys_r",
            interpolation="nearest", aspect="equal",
        )
        _draw_polys(axz, polys, lw=0.6)
        axz.set_xlim(xmin, xmax)
        axz.set_ylim(ymin, ymax)
        axz.set_title(f"Zoom [{xmin:.0f},{xmax:.0f}]x[{ymin:.0f},{ymax:.0f}] | {title}")
        axz.set_xlabel("world x (m)")
        axz.set_ylabel("world y (m)")

    fig.suptitle(
        "traversable mask aliasing check: white=traversable, black=solid, "
        "thin lines=collider polygons\n"
        f"thin-wall cells missed by 1m center-sampling: "
        f"{n_false}/{n_trav} ({100.0*n_false/max(n_trav,1):.2f}%)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    logger.info("已保存: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

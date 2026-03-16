"""
可视化 main_train.py 缓存的投放点。

读取 checkpoints/cache/landable_segments.npz，在地形多边形上绘制
所有可着陆线段及全部可能的投放点位置。

用法：
  python src/tests/control_interaction/visualize_cached_drops.py
  python src/tests/control_interaction/visualize_cached_drops.py --drop-height 3.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_l7_surface_airdrop import (
    compute_landable_surfaces,
    compute_player_height,
    extract_polygons,
    load_colliders,
)


def _precompute_segment_arcs(
    segments: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    arc_lengths = []
    cum_lengths = []
    for seg in segments:
        d = np.diff(seg, axis=0)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
        arc_lengths.append(cum[-1])
        cum_lengths.append(cum)
    return np.array(arc_lengths), cum_lengths


def load_env_data() -> dict:
    env_path = _REPO_ROOT / "checkpoints" / "environment.json"
    with open(env_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="可视化 main_train 缓存的投放点")
    parser.add_argument("--drop-height", type=float, default=None,
                        help="投放高度 (默认取 config.drop_height)")
    parser.add_argument("--save", type=str, default="logs/cached_drops.png",
                        help="保存图片路径")
    parser.add_argument("--no-show", action="store_true",
                        help="不弹出交互窗口")
    args = parser.parse_args()

    config = TrainConfig()
    drop_height = args.drop_height or config.drop_height

    game_root = Path(config.game_root)
    cache_dir = Path(config.checkpoint_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "landable_segments.npz"

    if cache_path.exists():
        seg_data = np.load(cache_path, allow_pickle=True)
        segments = list(seg_data["segments"])
        min_clearance = float(seg_data["min_clearance"])
        print(f"已加载 {len(segments)} 条可着陆线段 (缓存, min_clearance={min_clearance:.2f})")
    else:
        print("缓存不存在，从碰撞体数据重新计算...")
        env_data_raw, player_data = load_colliders(game_root)
        polygons = extract_polygons(env_data_raw)
        player_height = compute_player_height(player_data)
        min_clearance = player_height + config.surface_padding
        segments = compute_landable_surfaces(
            polygons, min_clearance,
            resolution=0.1,
            y_max_cutoff=config.y_max_cutoff,
        )
        np.savez_compressed(
            cache_path,
            segments=np.array(segments, dtype=object),
            min_clearance=np.array(min_clearance),
        )
        print(f"已计算 {len(segments)} 条可着陆线段 (min_clearance={min_clearance:.2f}, 已缓存)")

    arc_lengths, cum_lengths = _precompute_segment_arcs(segments)
    total_arc = float(arc_lengths.sum())
    print(f"总弧长: {total_arc:.1f}")

    # 收集所有线段顶点作为全部可能投放位置
    all_surface_pts = np.concatenate(segments, axis=0)  # (N, 2)
    all_drop_pts = all_surface_pts.copy()
    all_drop_pts[:, 1] += drop_height
    print(f"全部可能投放点: {len(all_drop_pts)} 个 (drop_height={drop_height:.1f})")

    env_data = load_env_data()

    import matplotlib
    matplotlib.use("TkAgg" if not args.no_show else "Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(22, 14))

    excluded = {"Snake"}
    env_patches = []
    for collider in env_data.get("colliders", []):
        if collider.get("name") in excluded:
            continue
        for path in collider.get("paths", []):
            pts = np.array(path, dtype=np.float64)
            if len(pts) >= 3:
                env_patches.append(plt.Polygon(pts, closed=True))
    if env_patches:
        pc = PatchCollection(
            env_patches, facecolor="#e0e0e0", edgecolor="#666666",
            linewidth=0.3, alpha=0.7, zorder=1,
        )
        ax.add_collection(pc)

    all_y = np.concatenate([s[:, 1] for s in segments])
    y_lo, y_hi = all_y.min(), all_y.max()
    y_range = y_hi - y_lo if y_hi > y_lo else 1.0
    cmap = matplotlib.colormaps["viridis"]

    for seg in segments:
        avg_y = seg[:, 1].mean()
        color = cmap((avg_y - y_lo) / y_range)
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=1.2,
                alpha=0.9, zorder=3)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=y_lo, vmax=y_hi),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("表面高度 Y (world)", fontsize=10)

    ax.scatter(
        all_drop_pts[:, 0], all_drop_pts[:, 1],
        c="red", s=4, marker=".", alpha=0.5,
        zorder=5, label=f"全部可能投放点 ({len(all_drop_pts)} 个, h={drop_height:.1f})",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("X (world)", fontsize=11)
    ax.set_ylabel("Y (world)", fontsize=11)
    ax.set_title(
        f"main_train.py 缓存的可着陆线段 & 全部可能投放点\n"
        f"{len(segments)} 条线段, {len(all_drop_pts)} 个投放点, "
        f"总弧长={total_arc:.0f}, min_clearance={min_clearance:.2f}, "
        f"y_max_cutoff={config.y_max_cutoff:.0f}",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150)
        print(f"图片已保存: {out}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

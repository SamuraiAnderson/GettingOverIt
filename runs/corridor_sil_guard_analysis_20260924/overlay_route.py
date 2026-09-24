# 决定性可视化：地形 + 参考路线（含 ±1.5m 走廊带）+ 评估回合 agent 轨迹叠加。
# 若参考线有坐标系标定偏移，agent 轨迹会系统性地贴在路线某一侧（或路线穿进岩石）。
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.reference_corridor import ReferenceCorridor
from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons

# 每个 seed 取 credited 最高的 2 条 + 中位的 1 条（deterministic 优先）
EVAL_DIRS = [f"corridor_formal_20260922_C_s{s}/checkpoints/eval_holdout_192k" for s in (31, 47, 61)]


def main() -> None:
    cfg = TrainConfig()
    env_data, _player = load_colliders(Path(cfg.game_root))
    polygons = extract_polygons(env_data)
    route = ReferenceCorridor.from_file(
        ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json", cfg)

    episodes = []
    for d in EVAL_DIRS:
        eps = []
        for ep in sorted((ROOT / "runs" / d).glob("deterministic_*_agent*.json")):
            j = json.loads(ep.read_text(encoding="utf-8"))
            if j.get("states"):
                eps.append(j)
        eps.sort(key=lambda j: -j["metrics"]["credited_progress_m"])
        pick = eps[:2] + eps[len(eps) // 2: len(eps) // 2 + 1]
        episodes.extend(pick)

    fig, ax = plt.subplots(figsize=(9, 10))
    for poly in polygons:
        ax.add_patch(MplPolygon(poly, closed=True, facecolor="0.85", edgecolor="0.6", lw=0.4))
    # 走廊带：沿路线法向 ±radius 的粗线近似
    for seg in route.segments:
        ax.plot(seg.points[:, 0], seg.points[:, 1], color="tab:red", lw=1.8,
                label="reference" if seg is route.segments[0] else None, zorder=4)
        ax.plot(seg.points[:, 0], seg.points[:, 1], color="tab:red", lw=2 * cfg.reference_corridor_radius * 72,
                alpha=0.12, solid_capstyle="round", zorder=3)
    for j in episodes:
        xy = np.array([[s[0], s[1]] for s in j["states"]])
        ax.plot(xy[:, 0], xy[:, 1], lw=0.9, alpha=0.75, zorder=5,
                label=None)
    # 起点与坐标参考
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_title("Reference route (red, band≈±1.5m) vs eval trajectories (thin) + terrain")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "route_overlay.png", dpi=150)
    plt.close(fig)
    print(f"episodes plotted: {len(episodes)}")
    print("saved: route_overlay.png")


if __name__ == "__main__":
    main()

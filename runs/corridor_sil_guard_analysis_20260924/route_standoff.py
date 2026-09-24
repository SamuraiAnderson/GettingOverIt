# 参考线站立距离（standoff）检验：参考线每个点到最近地形表面的距离剖面。
# 判别两种解释：
#   (a) 坐标标定错误（pot 视觉中心 vs Player 原点）→ 整条线常数悬空，
#       连必然接触地形的"爬行段"最低点都远离表面（min standoff ≫ 锅半径）；
#   (b) TAS 合法线路差异（TAS 飞、策略爬）→ 接触点处 standoff ≈ 锅半径（~0.3-0.5m），
#       只有飞跃段远离。
# 纯现有数据：reference_trajectory.json + 地形多边形，不需要任何运行。
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.reference_corridor import ReferenceCorridor
from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons


def dist_point_to_polygons(p: np.ndarray, polys: list[np.ndarray]) -> float:
    """点到多边形集合表面的距离（在多边形内部为 0）。"""
    from matplotlib.path import Path as MplPath
    best = np.inf
    for poly in polys:
        if MplPath(poly).contains_point(p):
            return 0.0
        # 到边的距离
        d = np.linalg.norm(poly - p, axis=1)
        best = min(best, float(d.min()))
    return best


def main() -> None:
    cfg = TrainConfig()
    env_data, _player = load_colliders(Path(cfg.game_root))
    polygons = extract_polygons(env_data)
    route = ReferenceCorridor.from_file(
        ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json", cfg)

    # 顶点级距离会高估（折线段的边比顶点更密），对每条段按弧长加密采样
    records = []  # (seg_idx, arc_len, x, y, standoff)
    for si, seg in enumerate(route.segments):
        pts = seg.points
        cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        dense_t = np.arange(0, cum[-1], 0.1)
        if len(dense_t) == 0:
            continue
        dx = np.interp(dense_t, cum, pts[:, 0])
        dy = np.interp(dense_t, cum, pts[:, 1])
        for t, x, y in zip(dense_t, dx, dy):
            records.append((si, float(t), float(x), float(y),
                            dist_point_to_polygons(np.array([x, y]), polygons)))

    standoff = np.array([r[4] for r in records])
    arc = np.array([r[1] for r in records])
    seg_ids = np.array([r[0] for r in records])
    report = {
        "n_samples": int(len(standoff)),
        "standoff_min": float(standoff.min()),
        "standoff_p05": float(np.percentile(standoff, 5)),
        "standoff_p25": float(np.percentile(standoff, 25)),
        "standoff_median": float(np.median(standoff)),
        "standoff_p90": float(np.percentile(standoff, 90)),
        "frac_within_pot_radius_0.4": float(np.mean(standoff <= 0.4)),
        "frac_within_1m": float(np.mean(standoff <= 1.0)),
        "per_segment_min": {int(s): float(standoff[seg_ids == s].min())
                            for s in np.unique(seg_ids)},
    }
    (OUT / "route_standoff.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for s in np.unique(seg_ids):
        m = seg_ids == s
        ax.plot(arc[m], standoff[m], lw=1.2, label=f"segment {s}")
    ax.axhline(0.4, color="tab:red", ls="--", lw=1, label="pot radius ≈0.4m")
    ax.set_xlabel("arc length within segment (m)")
    ax.set_ylabel("standoff to terrain (m)")
    ax.set_title("Reference route standoff profile — contact sections must approach pot radius")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "route_standoff.png", dpi=140)
    plt.close(fig)
    print("saved: route_standoff.json / route_standoff.png")


if __name__ == "__main__":
    main()

# 系统性偏移检验：agent 位置相对参考路线最近点的【有符号】偏移向量。
# 若参考线存在坐标系标定误差（pot 视觉中心 vs Player Transform 原点），
# agent 会系统性地偏在路线某一侧 → 均值向量显著非零且方向一致。
# 对照：若偏离是双向随机游走，均值向量应接近 0（纯横向摆动会抵消）。
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

EVAL_DIRS = [f"corridor_formal_20260922_C_s{s}/checkpoints/eval_holdout_192k" for s in (31, 47, 61)]


def nearest_point(polyline: np.ndarray, p: np.ndarray) -> np.ndarray:
    d2 = ((polyline - p) ** 2).sum(axis=1)
    return polyline[int(d2.argmin())]


def main() -> None:
    cfg = TrainConfig()
    route = ReferenceCorridor.from_file(
        ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json", cfg)
    pts = np.concatenate([s.points for s in route.segments], axis=0)

    offsets = []   # agent - nearest_route_point，逐帧
    agent_y, route_y = [], []
    for d in EVAL_DIRS:
        for ep in sorted((ROOT / "runs" / d).glob("*_agent*.json")):
            j = json.loads(ep.read_text(encoding="utf-8"))
            states = j.get("states")
            if not states:
                continue
            xy = np.array([[s[0], s[1]] for s in states], dtype=np.float64)
            for st in range(0, len(xy), 8):  # 每 8 帧采样，足够统计
                p = xy[st]
                q = nearest_point(pts, p)
                offsets.append(p - q)
                agent_y.append(p[1])
                route_y.append(q[1])

    off = np.array(offsets)
    agent_y = np.array(agent_y)
    route_y = np.array(route_y)
    dy = agent_y - route_y
    report = {
        "n_samples": int(len(off)),
        "mean_offset_x": float(off[:, 0].mean()),
        "mean_offset_y": float(off[:, 1].mean()),
        "median_offset_x": float(np.median(off[:, 0])),
        "median_offset_y": float(np.median(off[:, 1])),
        "std_offset_x": float(off[:, 0].std()),
        "std_offset_y": float(off[:, 1].std()),
        "frac_above_line": float(np.mean(dy > 0)),   # agent 在路线点上方比例
        "mean_dy": float(dy.mean()),
    }
    (OUT / "systematic_offset.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(off[:, 0], off[:, 1], s=2, alpha=0.15)
    ax.axvline(0, color="k", lw=0.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.scatter([off[:, 0].mean()], [off[:, 1].mean()], color="tab:red", s=80,
               marker="x", lw=3, label=f"mean ({off[:,0].mean():.2f}, {off[:,1].mean():.2f})")
    ax.set_xlabel("offset x (m)")
    ax.set_ylabel("offset y (m)")
    ax.set_title("Signed offset: agent vs nearest route point")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "systematic_offset.png", dpi=140)
    plt.close(fig)
    print("saved: systematic_offset.json / systematic_offset.png")


if __name__ == "__main__":
    main()

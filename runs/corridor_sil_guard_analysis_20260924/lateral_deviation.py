# 横向偏离分析：评估回合的 agent 位置相对参考路线的横向距离分布。
# 回答：走廊外移动是"贴着边界"（管径略紧/惩罚太弱）还是"另走一条线"（物理上必须）。
# 方法：用 ReferenceCorridor 的折线，对每条评估 trace 的逐帧 (x,y) 重算到全路线的最近距离。
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent

import sys
sys.path.insert(0, str(ROOT / "src"))
from training.config import TrainConfig
from training.reference_corridor import ReferenceCorridor

EVAL_DIRS = [f"corridor_formal_20260922_C_s{s}/checkpoints/eval_holdout_192k" for s in (31, 47, 61)]


def main() -> None:
    cfg = TrainConfig()
    route = ReferenceCorridor.from_file(
        ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json", cfg)
    # 全路线点云（段内折线展开）
    pts = np.concatenate([s.points for s in route.segments], axis=0)

    dists, projs, insides = [], [], []
    for d in EVAL_DIRS:
        for ep in sorted((ROOT / "runs" / d).glob("*_agent*.json")):
            j = json.loads(ep.read_text(encoding="utf-8"))
            trace = j.get("trace")
            states = j.get("states")
            if not trace or not states:
                continue
            xy = np.array([[s[0], s[1]] for s in states], dtype=np.float64)
            # 逐帧到全路线最近点距离（向量化分块）
            for st in range(0, len(xy), 256):
                chunk = xy[st:st + 256]
                dd = np.linalg.norm(chunk[:, None, :] - pts[None, :, :], axis=-1).min(axis=1)
                dists.extend(dd.tolist())
            insides.extend([bool(t["inside"]) for t in trace])

    dists = np.array(dists)
    insides = np.array(insides)
    radius = cfg.reference_corridor_radius
    report = {
        "n_frames": int(len(dists)),
        "radius_m": radius,
        "dist_median": float(np.median(dists)),
        "dist_p75": float(np.percentile(dists, 75)),
        "dist_p90": float(np.percentile(dists, 90)),
        "dist_p99": float(np.percentile(dists, 99)),
        "frac_within_radius": float(np.mean(dists <= radius)),
        "frac_within_2x_radius": float(np.mean(dists <= 2 * radius)),
        "frac_beyond_3m": float(np.mean(dists > 3.0)),
        "frac_beyond_5m": float(np.mean(dists > 5.0)),
        # 出走廊帧的偏离条件分布：贴边还是远行
        "outside_dist_median": float(np.median(dists[dists > radius])) if np.any(dists > radius) else None,
        "outside_dist_p90": float(np.percentile(dists[dists > radius], 90)) if np.any(dists > radius) else None,
        # 在线 tracker inside 与离线最近距离口径的一致性（应近似）
        "online_inside_frac": float(np.mean(insides)),
    }
    (OUT / "lateral_deviation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dists, bins=np.arange(0, min(dists.max(), 12) + 0.25, 0.25), color="tab:blue")
    ax.axvline(radius, color="tab:red", lw=1.5, label=f"corridor radius {radius}m")
    ax.set_xlabel("lateral distance to reference route (m)")
    ax.set_ylabel("frames")
    ax.set_title("Lateral deviation distribution (formal C holdout, all agents)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "lateral_deviation.png", dpi=140)
    plt.close(fig)
    print("saved: lateral_deviation.json / lateral_deviation.png")


if __name__ == "__main__":
    main()

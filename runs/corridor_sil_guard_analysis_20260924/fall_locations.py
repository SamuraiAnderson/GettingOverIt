# 进度死亡位置分析：评估回合的投影峰值/末端沿走廊弧长的分布。
# 回答："短程没到最优"具体卡在哪一段——是所有位置均匀死，还是卡在某个特定动作难点。
# 数据源：formal C 臂（σ=0.52）与 ab3 C 臂（σ 退火）的评估归档 trace。
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent

SETS = {
    "formal_C_holdout": [f"corridor_formal_20260922_C_s{s}/checkpoints/eval_holdout_192k" for s in (31, 47, 61)],
    "formal_C_midpoint": [f"corridor_formal_20260922_C_s{s}/checkpoints/eval_midpoint_096k" for s in (31, 47, 61)],
    "ab3_C_final": [f"corridor_sil_guard_ab3_20260924_3h_090758_C_s{s}/checkpoints" for s in (31, 47)]
                   + [f"corridor_sil_guard_ab3_extra4_20260924_C_s{s}/checkpoints" for s in (59, 71, 83, 97)],
}


def collect_episodes(dirs: list[str]) -> list[dict]:
    out = []
    for d in dirs:
        base = ROOT / "runs" / d
        if not base.is_dir():
            # ab3 的评估目录可能在 checkpoints/eval_* 下
            cands = list(base.glob("eval_*"))
            eps_dirs = cands if cands else [base]
        else:
            eps_dirs = [base]
        for ed in eps_dirs:
            for ep in sorted(ed.glob("*_agent*.json")):
                try:
                    j = json.loads(ep.read_text(encoding="utf-8"))
                except Exception:
                    continue
                trace = j.get("trace")
                if not trace:
                    continue
                proj = np.array([t["progress_m"] for t in trace], dtype=np.float64)
                credited = float(j["metrics"]["credited_progress_m"])
                out.append({
                    "peak_projection": float(proj.max()),
                    "end_projection": float(proj[-1]),
                    "credited": credited,
                    "fell": bool(j["metrics"]["fell"]),
                    "mode": j["metrics"]["mode"],
                })
    return out


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    report = {}
    for ax, (name, dirs) in zip(axes, SETS.items()):
        eps = collect_episodes(dirs)
        if not eps:
            ax.set_title(f"{name} (no data)")
            continue
        peaks = np.array([e["peak_projection"] for e in eps])
        ends = np.array([e["end_projection"] for e in eps])
        drops = peaks - ends
        report[name] = {
            "n": len(eps),
            "peak_median": float(np.median(peaks)),
            "peak_p90": float(np.percentile(peaks, 90)),
            "end_median": float(np.median(ends)),
            "drop_median": float(np.median(drops)),
            "drop_p75": float(np.percentile(drops, 75)),
            "fell_rate": float(np.mean([e["fell"] for e in eps])),
        }
        ax.hist(peaks, bins=np.arange(0, peaks.max() + 0.5, 0.5), alpha=0.65,
                label="peak projection", color="tab:blue")
        ax.hist(ends, bins=np.arange(0, peaks.max() + 0.5, 0.5), alpha=0.65,
                label="end projection", color="tab:red")
        ax.set_title(f"{name} (n={len(eps)})")
        ax.set_xlabel("arc-length along corridor (m)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("episodes")
    fig.suptitle("Where progress dies along the corridor")
    fig.tight_layout()
    fig.savefig(OUT / "fall_locations.png", dpi=140)
    plt.close(fig)

    (OUT / "fall_locations.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

# 0.1 重裁 corridor_formal_20260922：保持口径废弃后的新指标体系。
#
# 主指标：credited 进度（m）与速度（m/step，cap = 首次封顶步）。
# 护卫指标：末端投影进度（trace[-1].progress_m）——监控棘轮记账与物理位置的发散。
# 次级：outside_fraction / fell / steps。
#
# 数据源：六组 eval 归档（initial / midpoint_096k / holdout_192k × deterministic / exploration）。
# 输出：rejudge_summary.json + credited_distributions.png + ratchet_guard.png + trends.png
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "runs" / "corridor_formal_20260922"
GROUPS = [f"{arm}_s{seed}" for arm in ("C", "T") for seed in (31, 47, 61)]
PHASES = ["eval_initial", "eval_midpoint_096k", "eval_holdout_192k"]
MODES = ["deterministic", "exploration"]
OUT = Path(__file__).parent


def episode_record(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    m = d["metrics"]
    trace = d["trace"]
    # 评估归档 trace 的 credited_progress_m 是【累计值】口径
    # （replay_corridor_fields 文档：cumsum(逐步增量)[t] = trace 的 credited_progress_m）。
    cum = np.array([t["credited_progress_m"] for t in trace], dtype=np.float64)
    total = float(cum[-1]) if len(cum) else 0.0
    # 与 metrics.credited_progress_m 交叉核验（同一口径的两种来源）
    assert abs(total - float(m["credited_progress_m"])) < 1e-3, (path.name, total, m["credited_progress_m"])
    # cap = 累计首次封顶的动作索引（与 corridor_cap_index 同口径）
    cap = int(np.searchsorted(cum, total * (1 - 1e-6), side="left")) if total > 0 else 0
    speed = total / (cap + 1) if total > 0 else 0.0  # m/step（物理步 120Hz）
    end_projection = float(trace[-1]["progress_m"]) if trace else 0.0
    return {
        "file": path.name,
        "mode": m["mode"],
        "credited_m": float(m["credited_progress_m"]),
        "cap_step": cap + 1,
        "speed_m_per_step": speed,
        "speed_m_per_s": speed * 120.0,  # 物理步 120Hz → 游戏秒
        "end_projection_m": end_projection,
        "ratchet_gap_m": total - end_projection,
        "outside_fraction": float(m["outside_fraction"]),
        "fell": bool(m["fell"]),
        "steps": int(m["steps"]),
        "saw_gap": any(t["gap"] for t in trace),
    }


def collect() -> dict:
    rows = []
    for g in GROUPS:
        gdir = CAMPAIGN.parent / f"corridor_formal_20260922_{g}" / "checkpoints"
        for phase in PHASES:
            edir = gdir / phase
            if not edir.is_dir():
                continue
            for mode in MODES:
                for ep in sorted(edir.glob(f"{mode}_*_agent*.json")):
                    r = episode_record(ep)
                    r.update(group=g, arm=g[0], seed=int(g.split("s")[1]),
                             phase=phase.replace("eval_", ""), mode=mode)
                    rows.append(r)
    return rows


def agg(records: list[dict]) -> dict:
    if not records:
        return {}
    credited = np.array([r["credited_m"] for r in records])
    speed = np.array([r["speed_m_per_step"] for r in records])
    endp = np.array([r["end_projection_m"] for r in records])
    gap = np.array([r["ratchet_gap_m"] for r in records])
    return {
        "n": len(records),
        "credited_median": float(np.median(credited)),
        "credited_mean": float(credited.mean()),
        "credited_max": float(credited.max()),
        "credited_q25": float(np.percentile(credited, 25)),
        "credited_q75": float(np.percentile(credited, 75)),
        "speed_median": float(np.median(speed)),
        "speed_max": float(speed.max()),
        "speed_m_per_s_median": float(np.median(speed) * 120.0),
        "end_projection_median": float(np.median(endp)),
        "ratchet_gap_median": float(np.median(gap)),
        "ratchet_gap_max": float(gap.max()),
        "outside_fraction_mean": float(np.mean([r["outside_fraction"] for r in records])),
        "fell_rate": float(np.mean([r["fell"] for r in records])),
        "zero_progress_rate": float(np.mean(credited <= 1e-9)),
    }


def main() -> None:
    rows = collect()
    print(f"episodes collected: {len(rows)}")

    summary: dict = {}
    for g in GROUPS:
        for phase in [p.replace("eval_", "") for p in PHASES]:
            for mode in MODES:
                rec = [r for r in rows if r["group"] == g and r["phase"] == phase and r["mode"] == mode]
                a = agg(rec)
                if a:
                    summary[f"{g}/{phase}/{mode}"] = a

    # ── C vs T 裁决（holdout，分 seed + 汇总）──
    verdict = {}
    for mode in MODES:
        for seed in (31, 47, 61):
            for arm in ("C", "T"):
                key = f"{arm}_s{seed}/holdout_192k/{mode}"
                if key in summary:
                    verdict.setdefault(mode, {}).setdefault(f"s{seed}", {})[arm] = {
                        "median": summary[key]["credited_median"],
                        "mean": summary[key]["credited_mean"],
                        "max": summary[key]["credited_max"],
                        "end_proj_median": summary[key]["end_projection_median"],
                    }
        pooled = {}
        for arm in ("C", "T"):
            rec = [r for r in rows
                   if r["arm"] == arm and r["phase"] == "holdout_192k" and r["mode"] == mode]
            a = agg(rec)
            pooled[arm] = {k: a[k] for k in ("n", "credited_median", "credited_mean",
                                             "credited_max", "end_projection_median",
                                             "ratchet_gap_median", "fell_rate")} if a else {}
        verdict[mode]["pooled"] = pooled

    (OUT / "rejudge_summary.json").write_text(
        json.dumps({"episodes": rows, "summary": summary, "verdict": verdict},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 图 1：credited 分布（holdout，分模式）──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, mode in zip(axes, MODES):
        data, labels = [], []
        for g in GROUPS:
            rec = [r["credited_m"] for r in rows
                   if r["group"] == g and r["phase"] == "holdout_192k" and r["mode"] == mode]
            data.append(rec)
            labels.append(g)
        bp = ax.boxplot(data, labels=labels, showmeans=True)
        for i, rec in enumerate(data):
            ax.scatter(np.full(len(rec), i + 1) + np.random.uniform(-0.12, 0.12, len(rec)),
                       rec, s=14, alpha=0.6, zorder=3)
        ax.set_title(f"holdout_192k credited progress — {mode}")
        ax.set_ylabel("credited progress (m)" if mode == MODES[0] else "")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT / "credited_distributions.png", dpi=140)
    plt.close(fig)

    # ── 图 2：棘轮护卫（credited vs 末端投影，全阶段全模式）──
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for arm, marker in (("C", "o"), ("T", "^")):
        for phase, color in (("initial", "gray"), ("midpoint_096k", "tab:blue"),
                             ("holdout_192k", "tab:red")):
            rec = [r for r in rows if r["arm"] == arm and r["phase"] == phase]
            if not rec:
                continue
            ax.scatter([r["credited_m"] for r in rec], [r["end_projection_m"] for r in rec],
                       s=16, alpha=0.55, marker=marker, color=color,
                       label=f"{arm}/{phase}")
    lim = max(r["credited_m"] for r in rows) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="ratchet == end position")
    ax.set_xlabel("credited progress (m)")
    ax.set_ylabel("end projection (m)")
    ax.set_title("Ratchet guard: credited vs final physical projection")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "ratchet_guard.png", dpi=140)
    plt.close(fig)

    # ── 图 3：逐阶段趋势（deterministic 中位数 + 均值带）──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    phase_order = ["initial", "midpoint_096k", "holdout_192k"]
    for ax, mode in zip(axes, MODES):
        for g in GROUPS:
            meds, means = [], []
            for phase in phase_order:
                key = f"{g}/{phase}/{mode}"
                meds.append(summary.get(key, {}).get("credited_median", np.nan))
                means.append(summary.get(key, {}).get("credited_mean", np.nan))
            style = "-" if g.startswith("C") else "--"
            ax.plot(phase_order, meds, style, marker="o", label=g, lw=1.6)
        ax.set_title(f"credited median by phase — {mode}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylabel("credited progress median (m)" if mode == MODES[0] else "")
    fig.tight_layout()
    fig.savefig(OUT / "trends.png", dpi=140)
    plt.close(fig)

    # ── 控制台简报 ──
    print("\n== holdout_192k credited median (det / exp) ==")
    for g in GROUPS:
        d = summary.get(f"{g}/holdout_192k/deterministic", {})
        e = summary.get(f"{g}/holdout_192k/exploration", {})
        print(f"  {g}: det={d.get('credited_median', float('nan')):.2f}m "
              f"(max {d.get('credited_max', 0):.2f})  "
              f"exp={e.get('credited_median', float('nan')):.2f}m "
              f"(max {e.get('credited_max', 0):.2f})  "
              f"end_proj_det={d.get('end_projection_median', float('nan')):.2f}m")
    print("\n== C vs T pooled (holdout) ==")
    for mode in MODES:
        p = verdict[mode]["pooled"]
        print(f"  {mode}: C median={p['C']['credited_median']:.2f} mean={p['C']['credited_mean']:.2f} "
              f"max={p['C']['credited_max']:.2f} | T median={p['T']['credited_median']:.2f} "
              f"mean={p['T']['credited_mean']:.2f} max={p['T']['credited_max']:.2f}")
    print("\nsaved: rejudge_summary.json / credited_distributions.png / "
          "ratchet_guard.png / trends.png")


if __name__ == "__main__":
    main()

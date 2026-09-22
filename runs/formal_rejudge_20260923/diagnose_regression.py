# 0.2 回退机制诊断：4/6 组 midpoint→holdout 回退的机制定位。
#
# 前置事实（run_meta.json 核实）：正式训练 sil_enabled=False、explore_boost_enabled=False，
# 即纯 PPO+OU。因此"池停滞→boost 提方差→冲坏技能"的恶性循环假设对本批运行**结构上不成立**，
# 诊断对象改为纯 PPO 动力学候选机制：
#   H2 entropy 塌缩/爆炸（模式塌缩到更差行为 / 噪声主导）
#   H3 KL/早停动力学（破坏性大更新 / 更新饥饿）
#   H4 value_loss 发散（优势估计崩坏）
#   H5 动作饱和（tanh 打满 → 梯度消失）
#   H6 训练 rollout 的 corridor_progress_m 何时见顶回落（回退的时间定位）
#
# 输入：六组 metrics.csv（22 轮/组）。输出：diagnosis.json + 三张图 + 控制台简报。
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GROUPS = [f"{arm}_s{seed}" for arm in ("C", "T") for seed in (31, 47, 61)]
OUT = Path(__file__).parent

# 评估锚点：midpoint≈96k 样本、holdout≈192k 样本；每轮约 9.6k 样本
MID_WINDOW = (8, 11)    # 中期评估对应的训练轮次邻域
LATE_WINDOW = (19, 22)  # 保留集评估对应的训练轮次邻域


def load_metrics(group: str) -> list[dict]:
    path = ROOT / "runs" / f"corridor_formal_20260922_{group}" / "metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = []
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if v not in ("", None) else float("nan"))
                         for k, v in r.items()})
        return rows


def window_mean(rows: list[dict], key: str, lo: int, hi: int) -> float:
    vals = [r[key] for r in rows if lo <= r["iteration"] <= hi and np.isfinite(r.get(key, float("nan")))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    data = {g: load_metrics(g) for g in GROUPS}
    for g, rows in data.items():
        for r in rows:
            r["progress_per_agent"] = r["corridor_progress_m"] / 8.0  # 8 agents

    # ── 逐组定量摘要 ──
    diagnosis = {}
    for g, rows in data.items():
        prog_mid = window_mean(rows, "progress_per_agent", *MID_WINDOW)
        prog_late = window_mean(rows, "progress_per_agent", *LATE_WINDOW)
        ent_mid = window_mean(rows, "entropy", *MID_WINDOW)
        ent_late = window_mean(rows, "entropy", *LATE_WINDOW)
        ent_all = [r["entropy"] for r in rows if not r["value_only"]]
        prog_all = [r["progress_per_agent"] for r in rows]
        peak_iter = int(rows[int(np.nanargmax(prog_all))]["iteration"])
        diagnosis[g] = {
            "progress_mid_mean_per_agent": prog_mid,
            "progress_late_mean_per_agent": prog_late,
            "progress_delta": prog_late - prog_mid,
            "progress_peak_iter": peak_iter,
            "entropy_mid": ent_mid,
            "entropy_late": ent_late,
            "entropy_delta": ent_late - ent_mid,
            "entropy_final_min": float(np.nanmin(ent_all)) if ent_all else float("nan"),
            "entropy_final_max": float(np.nanmax(ent_all)) if ent_all else float("nan"),
            "approx_kl_late_mean": window_mean(rows, "approx_kl", *LATE_WINDOW),
            "approx_kl_max": float(np.nanmax([r["approx_kl"] for r in rows])),
            "n_updates_late_mean": window_mean(rows, "n_updates", *LATE_WINDOW),
            "early_stopped_count": int(sum(1 for r in rows if r.get("early_stopped") == 1.0)),
            "value_loss_late_mean": window_mean(rows, "value_loss", *LATE_WINDOW),
            "value_loss_trend": window_mean(rows, "value_loss", *LATE_WINDOW)
                               - window_mean(rows, "value_loss", *MID_WINDOW),
            "action_saturation_late": window_mean(rows, "executed_action_saturation_fraction", *LATE_WINDOW),
            "inside_fraction_mid": window_mean(rows, "corridor_inside_fraction", *MID_WINDOW),
            "inside_fraction_late": window_mean(rows, "corridor_inside_fraction", *LATE_WINDOW),
            "mean_reward_mid": window_mean(rows, "mean_reward", *MID_WINDOW),
            "mean_reward_late": window_mean(rows, "mean_reward", *LATE_WINDOW),
        }

    (OUT / "diagnosis.json").write_text(
        json.dumps(diagnosis, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 图 1：逐组 训练进度/entropy 双轴 ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, g in zip(axes.flat, GROUPS):
        rows = data[g]
        it = [r["iteration"] for r in rows]
        ax.plot(it, [r["progress_per_agent"] for r in rows], "tab:blue", lw=1.6,
                label="credited/agent (train)")
        ax.axvspan(*MID_WINDOW, color="gray", alpha=0.12)
        ax.axvspan(*LATE_WINDOW, color="tab:red", alpha=0.10)
        ax.set_title(g)
        ax.set_ylabel("credited m/agent", color="tab:blue")
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(it, [r["entropy"] for r in rows], "tab:orange", lw=1.2, alpha=0.85,
                 label="entropy")
        ax2.set_ylabel("entropy", color="tab:orange")
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Training-rollout credited progress vs entropy (red window = holdout anchor)")
    fig.tight_layout()
    fig.savefig(OUT / "diag_progress_entropy.png", dpi=140)
    plt.close(fig)

    # ── 图 2：KL / 更新数 / 早停 ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, g in zip(axes.flat, GROUPS):
        rows = data[g]
        it = [r["iteration"] for r in rows]
        ax.plot(it, [r["approx_kl"] for r in rows], "tab:red", lw=1.5, label="approx_kl")
        ax.plot(it, [r["preupdate_kl"] for r in rows], "tab:pink", lw=1.0, alpha=0.7,
                label="preupdate_kl")
        ax.axhline(0.03, color="k", ls="--", lw=0.8)
        ax.set_title(g)
        ax.grid(alpha=0.3)
        ax.set_ylabel("KL")
        ax2 = ax.twinx()
        ax2.step(it, [r["n_updates"] for r in rows], "tab:blue", lw=1.2, label="n_updates")
        ax2.set_ylabel("n_updates", color="tab:blue")
        for r in rows:
            if r.get("early_stopped") == 1.0:
                ax.axvline(r["iteration"], color="k", alpha=0.25, lw=0.8)
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("KL dynamics & update counts (vertical lines = early-stopped iters)")
    fig.tight_layout()
    fig.savefig(OUT / "diag_kl_updates.png", dpi=140)
    plt.close(fig)

    # ── 图 3：饱和 / value_loss / 走廊内比例 ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for ax, g in zip(axes.flat, GROUPS):
        rows = data[g]
        it = [r["iteration"] for r in rows]
        ax.plot(it, [r["executed_action_saturation_fraction"] for r in rows],
                "tab:purple", lw=1.5, label="action saturation")
        ax.plot(it, [r["corridor_inside_fraction"] for r in rows],
                "tab:green", lw=1.5, label="inside fraction")
        ax.set_title(g)
        ax.grid(alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(it, [r["value_loss"] for r in rows], "tab:brown", lw=1.1, alpha=0.8,
                 label="value_loss")
        ax2.set_ylabel("value_loss", color="tab:brown")
    axes.flat[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Action saturation / corridor-inside fraction / value loss")
    fig.tight_layout()
    fig.savefig(OUT / "diag_saturation_value.png", dpi=140)
    plt.close(fig)

    # ── 控制台简报 ──
    print(f"{'group':7} {'prog_mid':>8} {'prog_late':>9} {'Δprog':>7} "
          f"{'ent_mid':>7} {'ent_late':>8} {'kl_late':>7} {'kl_max':>6} "
          f"{'nupd_late':>9} {'estop':>5} {'sat_late':>8} {'vloss_Δ':>8}")
    for g in GROUPS:
        d = diagnosis[g]
        print(f"{g:7} {d['progress_mid_mean_per_agent']:8.2f} "
              f"{d['progress_late_mean_per_agent']:9.2f} {d['progress_delta']:7.2f} "
              f"{d['entropy_mid']:7.3f} {d['entropy_late']:8.3f} "
              f"{d['approx_kl_late_mean']:7.4f} {d['approx_kl_max']:6.3f} "
              f"{d['n_updates_late_mean']:9.1f} {d['early_stopped_count']:5d} "
              f"{d['action_saturation_late']:8.3f} {d['value_loss_trend']:8.3f}")
    print("\nsaved: diagnosis.json / diag_progress_entropy.png / "
          "diag_kl_updates.png / diag_saturation_value.png")


if __name__ == "__main__":
    main()

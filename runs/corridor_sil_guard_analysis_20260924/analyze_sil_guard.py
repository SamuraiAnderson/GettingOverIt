# SIL 对照实验（corridor_sil_guard_ab3）负收益机制分析
#
# 输入：ab3 两批共 6 个 seed 的 C（纯 PPO）/ S（走廊 SIL）双臂 metrics.csv。
# 输出：控制台简报 + sil_internal.png（SIL 内部指标）+ arm_comparison.png（臂间对照）
#       + analysis.json（逐 seed 汇总）
#
# 关注假设：
#   H1 优势权重失准（pos_frac 钉 0/1、weight 分布极端）
#   H2 SIL 损失干扰 PPO 稳定性（KL/entropy/n_updates 臂间差异）
#   H3 精英池质量/多样性塌缩（cells、trajs、池峰停滞）
#   H4 训练采集进度本身受损（corridor_progress_m 趋势臂间对比）
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent

ARMS = {
    "batch1": {s: (f"corridor_sil_guard_ab3_20260924_3h_090758_C_s{s}",
                   f"corridor_sil_guard_ab3_20260924_3h_090758_S_s{s}")
               for s in (31, 47)},
    "batch2": {s: (f"corridor_sil_guard_ab3_extra4_20260924_C_s{s}",
                   f"corridor_sil_guard_ab3_extra4_20260924_S_s{s}")
               for s in (59, 71, 83, 97)},
}


def load(run_dir: str) -> list[dict]:
    path = ROOT / "runs" / run_dir / "metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [{k: (float(v) if v not in ("", None) else float("nan"))
                 for k, v in r.items()} for r in csv.DictReader(fh)]


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r.get(key, float("nan")) for r in rows], dtype=np.float64)


def main() -> None:
    data: dict[str, dict[str, list[dict]]] = {}
    for batch, seeds in ARMS.items():
        for seed, (c_dir, s_dir) in seeds.items():
            data[f"C_{seed}"] = {"rows": load(c_dir), "batch": batch}
            data[f"S_{seed}"] = {"rows": load(s_dir), "batch": batch}

    # ── 逐 seed 汇总 ──
    summary = {}
    print("== 训练采集进度（corridor_progress_m 每 agent，后半段均值）==")
    for seed in (31, 47, 59, 71, 83, 97):
        line = {}
        for arm in ("C", "S"):
            rows = data[f"{arm}_{seed}"]["rows"]
            its = col(rows, "iteration")
            prog = col(rows, "corridor_progress_m") / 8.0
            late = prog[its >= its.max() - 4]
            ent = col(rows, "entropy")
            kl = col(rows, "approx_kl")
            vl = col(rows, "value_loss")
            line[arm] = {
                "prog_late_mean": float(np.nanmean(late)),
                "prog_max": float(np.nanmax(prog)),
                "entropy_late": float(np.nanmean(ent[its >= its.max() - 4])),
                "kl_mean": float(np.nanmean(kl[2:])),
                "value_loss_late": float(np.nanmean(vl[its >= its.max() - 4])),
            }
            if arm == "S":
                pos = col(rows, "sil_adv_pos_frac")
                wm = col(rows, "sil_weight_mean")
                sl = col(rows, "sil_loss")
                tr = col(rows, "sil_trajs")
                line["S_sil"] = {
                    "pos_frac_series": [round(float(x), 4) for x in pos if np.isfinite(x)],
                    "weight_mean_series": [round(float(x), 4) for x in wm if np.isfinite(x)],
                    "sil_loss_series": [round(float(x), 4) for x in sl if np.isfinite(x)],
                    "sil_trajs_last": float(tr[-1]) if len(tr) else None,
                }
        summary[str(seed)] = line
        print(f"  s{seed}: C prog_late={line['C']['prog_late_mean']:.2f} "
              f"(max {line['C']['prog_max']:.2f}) | S prog_late={line['S']['prog_late_mean']:.2f} "
              f"(max {line['S']['prog_max']:.2f}) | S pos_frac={line['S_sil']['pos_frac_series']}")

    (OUT / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 图 1：SIL 内部指标（S 臂 6 seeds）──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for seed in (31, 47, 59, 71, 83, 97):
        rows = data[f"S_{seed}"]["rows"]
        its = col(rows, "iteration")
        for ax, key, title in (
                (axes[0], "sil_adv_pos_frac", "adv pos frac (raw A>0)"),
                (axes[1], "sil_weight_mean", "weight mean"),
                (axes[2], "sil_loss", "sil loss")):
            v = col(rows, key)
            mask = np.isfinite(v)
            ax.plot(its[mask], v[mask], marker="o", ms=3, lw=1.2, label=f"s{seed}")
    axes[0].axhline(0.0, color="k", lw=0.6)
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.6)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlabel("iteration")
    for ax, title in zip(axes, ("adv pos frac", "weight mean", "sil loss")):
        ax.set_title(title)
    fig.suptitle("SIL internals — S arms")
    fig.tight_layout()
    fig.savefig(OUT / "sil_internal.png", dpi=140)
    plt.close(fig)

    # ── 图 2：臂间对照（训练进度 / KL / entropy / value_loss），每指标一横排 ──
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True)
    for ax, (key, title) in zip(axes, (
            ("corridor_progress_m", "train credited per agent (m)"),
            ("approx_kl", "approx_kl"),
            ("entropy", "entropy"),
            ("value_loss", "value_loss"))):
        for seed in (31, 47, 59, 71, 83, 97):
            for arm, color in (("C", "tab:blue"), ("S", "tab:red")):
                rows = data[f"{arm}_{seed}"]["rows"]
                its = col(rows, "iteration")
                v = col(rows, key)
                if key == "corridor_progress_m":
                    v = v / 8.0
                ax.plot(its, v, lw=1.1, alpha=0.7, color=color)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[Line2D([], [], color="tab:blue", label="C (pure PPO)"),
                            Line2D([], [], color="tab:red", label="S (corridor SIL)")],
                   fontsize=9)
    axes[-1].set_xlabel("iteration")
    fig.suptitle("C (blue, 6 seeds) vs S (red, 6 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "arm_comparison.png", dpi=140)
    plt.close(fig)

    print("\nsaved: analysis.json / sil_internal.png / arm_comparison.png")


if __name__ == "__main__":
    main()

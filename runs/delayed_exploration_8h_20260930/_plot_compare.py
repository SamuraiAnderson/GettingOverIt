"""One-off start/end envelope overlay. Deleted after the figure is written."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from src.tests.analysis.plot_sil_td_envelope import FONT, catalog_pool, load_trajectories

START = ROOT / "runs/log_std_clamp_4h_20260929/elites/elites_iter_0163.npz"
END = ROOT / "runs/delayed_exploration_8h_20260930/elites/elites_iter_0251.npz"
OUT = ROOT / "runs/delayed_exploration_8h_20260930/td_envelope"


def front_of(path: Path):
    catalog = catalog_pool(load_trajectories(path), source=str(path))
    return catalog["envelope"]


def main():
    before, after = front_of(START), front_of(END)
    before_keys = {(row["t"], round(row["D"], 3)) for row in before}
    fig, ax = plt.subplots(figsize=(8.6, 5.8), facecolor="#faf9f6")
    ax.set_facecolor("#faf9f6")
    ax.plot([row["t"] for row in before], [row["D"] for row in before],
            color="#7a868c", linewidth=1.5, marker="o", markersize=4.5,
            label="第 163 轮（训练前）")
    ax.plot([row["t"] for row in after], [row["D"] for row in after],
            color="#c0392b", linewidth=1.8, marker="o", markersize=5,
            label="第 251 轮（训练后）")
    for row in after:
        if (row["t"], round(row["D"], 3)) in before_keys:
            continue
        ax.annotate(f"{row['D']:.2f} m", (row["t"], row["D"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontproperties=FONT, fontsize=8, color="#c0392b")
    farthest = max(after, key=lambda row: row["D"])
    ax.annotate(f"最远仍是 t={farthest['t']}\nD={farthest['D']:.2f} m",
                (farthest["t"], farthest["D"]), textcoords="offset points",
                xytext=(-78, -28), fontproperties=FONT, fontsize=9, color="#7a4a3a")
    ax.set_xlabel("封顶步数 t（越小越快）", fontproperties=FONT, fontsize=11)
    ax.set_ylabel("累计有效进度 D（米，越大越远）", fontproperties=FONT, fontsize=11)
    ax.set_title("延迟探索前后的 (t, D) 左上包络", fontproperties=FONT, fontsize=14)
    ax.grid(True, color="#d8d2c8", linewidth=0.6)
    legend = ax.legend(prop=FONT, loc="lower right")
    legend.get_frame().set_facecolor("#faf9f6")
    fig.text(0.5, 0.02, "红字是这次留下的新前沿点。20 米和 24 米两段没有外推。",
             ha="center", fontproperties=FONT, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "envelope_before_after.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()

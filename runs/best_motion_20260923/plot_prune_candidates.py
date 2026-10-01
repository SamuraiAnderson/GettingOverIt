# 绘制最优 episode 的高度/进度曲线，标注候选剪枝点（一次性分析脚本）
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.config import TrainConfig
from training.reward import progress_metric

EP = r"C:\Users\Symbol\aCodes\GettingOverIt\runs\corridor_formal_20260922_T_s31\checkpoints\eval_midpoint_096k\exploration_001_agent0.json"
MANIFEST = r"C:\Users\Symbol\aCodes\GettingOverIt\runs\corridor_formal_20260922_T_s31\checkpoints\eval_midpoint_096k\manifest.json"
OUT = Path(__file__).parent / "prune_candidates.png"

d = json.load(open(EP, encoding="utf-8"))
manifest = json.load(open(MANIFEST, encoding="utf-8"))
cfg = TrainConfig(**manifest["config"])

states = np.asarray(d["states"], dtype=np.float32)
tr = d["trace"]
y = states[:, 1].astype(np.float64)
steps = np.arange(len(y))  # states 索引 = 物理步（0..1200）

prog = np.array([0.0] + [t["progress_m"] for t in tr])
cred = np.array([0.0] + [t["credited_progress_m"] for t in tr])
inside = np.array([True] + [bool(t["inside"]) for t in tr])

_, peak_idx, _, _ = progress_metric(states, cfg)
cred_max = max(t["credited_progress_m"] for t in tr)
cred_cap = next(t["step"] for t in tr if t["credited_progress_m"] >= cred_max - 1e-9)
prog_peak = int(prog.argmax())

font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
fig, ax = plt.subplots(figsize=(12, 6.4), facecolor="#faf9f6")
ax.set_facecolor("#faf9f6")

# 走廊外区域着色
out = ~inside
ax.fill_between(steps, y.min() - 0.5, y.max() + 0.5, where=out,
                color="#e8b4b4", alpha=0.25, label="走廊外", zorder=0)

ax.plot(steps, y - y[0], color="#32789d", lw=1.6, label="相对高度 y (m)")
ax.plot(steps, prog, color="#689573", lw=1.4, label="走廊投影 progress (m)")
ax.plot(steps, cred, color="#946640", lw=1.4, ls="--", label="累计有效进度 credited (m)")

for x, name, c in [(peak_idx, f"secured peak={peak_idx}\n(BC 截断口径)", "#c0392b"),
                   (cred_cap, f"credited 封顶={cred_cap}", "#8e44ad"),
                   (prog_peak, f"progress 峰值={prog_peak}", "#16a085")]:
    ax.axvline(x, color=c, lw=1.4, ls=":", alpha=0.9)
    ax.text(x + 6, y.max() - y[0] - 0.4, name, color=c, fontproperties=font,
            fontsize=10, va="top")

ax.set_xlabel("物理步 (120 步 = 1 秒)", fontproperties=font, fontsize=11)
ax.set_ylabel("米", fontproperties=font, fontsize=11)
ax.set_title("最优 episode 曲线与候选剪枝点（红=训练BC截断口径，紫=有效进度封顶，青=投影峰值）",
             fontproperties=font, fontsize=13)
ax.legend(prop=font, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT, dpi=140)
print("saved:", OUT)
print(f"peak_idx={peak_idx}, cred_cap={cred_cap}, prog_peak={prog_peak}")

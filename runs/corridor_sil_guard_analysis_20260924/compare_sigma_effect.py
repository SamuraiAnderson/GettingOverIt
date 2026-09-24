# σ 调度对纯 PPO 的影响：ab3 C 臂（σ 重置+退火）vs formal C 臂（σ≈0.52 钉死）
# 两侧均为纯 PPO、同配方（γ=0.99/λ=0.95）、同 init-from 源；唯一差异是 σ 调度。
# 注意样本量不同：ab3 目标 96k（~10 轮），formal 跑到 192k（22 轮）——比较同轮次段。
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

ARMS = {
    "ab3_C_s31": "corridor_sil_guard_ab3_20260924_3h_090758_C_s31",
    "ab3_C_s47": "corridor_sil_guard_ab3_20260924_3h_090758_C_s47",
    "ab3_C_s59": "corridor_sil_guard_ab3_extra4_20260924_C_s59",
    "ab3_C_s71": "corridor_sil_guard_ab3_extra4_20260924_C_s71",
    "ab3_C_s83": "corridor_sil_guard_ab3_extra4_20260924_C_s83",
    "ab3_C_s97": "corridor_sil_guard_ab3_extra4_20260924_C_s97",
    "formal_C_s31": "corridor_formal_20260922_C_s31",
    "formal_C_s47": "corridor_formal_20260922_C_s47",
    "formal_C_s61": "corridor_formal_20260922_C_s61",
}


def series(run_dir: str):
    rows = list(csv.DictReader((ROOT / "runs" / run_dir / "metrics.csv").open(encoding="utf-8")))
    its = np.array([int(float(r["iteration"])) for r in rows])
    prog = np.array([float(r["corridor_progress_m"]) for r in rows]) / 8.0
    ent = np.array([float(r["entropy"]) for r in rows])
    return its, prog, ent


print(f"{'arm':15} {'iters':>5} {'prog@5-8':>9} {'prog@9-11':>10} {'prog_last3':>10} "
      f"{'ent@3':>7} {'ent_last':>8}")
for name, d in ARMS.items():
    its, prog, ent = series(d)
    n = len(its)
    seg_a = prog[(its >= 5) & (its <= 8)]
    seg_b = prog[(its >= 9) & (its <= 11)]
    print(f"{name:15} {n:5d} {np.mean(seg_a) if len(seg_a) else float('nan'):9.2f} "
          f"{np.mean(seg_b) if len(seg_b) else float('nan'):10.2f} "
          f"{np.mean(prog[-3:]):10.2f} {ent[2] if n > 2 else float('nan'):7.3f} "
          f"{ent[-1]:8.3f}")

# 同轮次窗口（9-11 轮 ≈ 96k 样本点）的分组均值
ab3, formal = [], []
for name, d in ARMS.items():
    its, prog, _ = series(d)
    seg = prog[(its >= 9) & (its <= 11)]
    if not len(seg):
        continue
    (ab3 if name.startswith("ab3") else formal).append(float(np.mean(seg)))
print(f"\n96k 窗口训练进度均值: ab3(σ调度)={np.mean(ab3):.2f}m  formal(σ=0.52)={np.mean(formal):.2f}m")

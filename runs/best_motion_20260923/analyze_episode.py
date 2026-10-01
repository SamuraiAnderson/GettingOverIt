# 分析最优 episode 的进度/高度曲线，寻找跌落与震荡的起始点（一次性分析脚本）
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.config import TrainConfig
from training.reward import progress_metric

EP = r"C:\Users\Symbol\aCodes\GettingOverIt\runs\corridor_formal_20260922_T_s31\checkpoints\eval_midpoint_096k\exploration_001_agent0.json"
MANIFEST = r"C:\Users\Symbol\aCodes\GettingOverIt\runs\corridor_formal_20260922_T_s31\checkpoints\eval_midpoint_096k\manifest.json"

d = json.load(open(EP, encoding="utf-8"))
tr = d["trace"]
states = d["states"]

print("metrics:", json.dumps({k: v for k, v in d["metrics"].items() if k != "native_comparison"}, ensure_ascii=False))
print()

# states 是 33D，先看每个状态里哪些维度像位置（身体/锤头）
# 打印前 3 帧确认语义未知，先用 trace 的 progress 为主
print("step  progress  credited  inside  action_dx action_dy")
for t in tr[::40]:
    a = t["action"]
    print(f"{t['step']:5d} {t['progress_m']:8.3f} {t['credited_progress_m']:8.3f} {str(t['inside']):5s} {a[0]:9.2f} {a[1]:9.2f}")

best = max(tr, key=lambda t: t["credited_progress_m"])
print()
print("max credited step:", best["step"], "credited:", round(best["credited_progress_m"], 3), "progress:", round(best["progress_m"], 3))

first_out = next((t["step"] for t in tr if not t["inside"]), None)
print("first outside corridor step:", first_out)

# arrival_steps=519 之后的 progress 走势
arr = d["metrics"]["arrival_steps"]
print()
print(f"--- after arrival_steps={arr}, every 20 steps ---")
for t in tr[arr - 1::20]:
    print(f"{t['step']:5d} {t['progress_m']:8.3f} {t['credited_progress_m']:8.3f} inside={t['inside']}")

# 找 progress 峰值及之后的最低点（判断跌落幅度）
peak_prog = max(t["progress_m"] for t in tr)
peak_step = next(t["step"] for t in tr if t["progress_m"] == peak_prog)
print()
print("peak progress:", round(peak_prog, 3), "at step", peak_step)
after = [t["progress_m"] for t in tr if t["step"] > peak_step]
print("min progress after peak:", round(min(after), 3), "(drop of", round(peak_prog - min(after), 3), "m)")

# ── 用训练同款 progress_metric 计算 secured peak（BC 截断点口径）──
print()
print("=== 训练同款 secured peak（progress_metric, BC 截断点）===")
manifest = json.load(open(MANIFEST, encoding="utf-8"))
cfg = TrainConfig(**manifest["config"])
print("dwell_steps:", cfg.dwell_steps, "dwell_drop_tol:", cfg.dwell_drop_tol)
states_arr = np.asarray(d["states"], dtype=np.float32)
progress, peak_idx, secured_dy, reach = progress_metric(states_arr, cfg)
y = states_arr[:, 1].astype(np.float64)
print(f"peak_idx={peak_idx}  (即 BC 监督只覆盖动作 0..{min(peak_idx, len(tr)-1)})")
print(f"secured_dy={secured_dy:.3f} m, reach={reach:.3f} m, progress={progress:.3f}")
print(f"y[start]={y[0]:.3f}  y[peak]={y[peak_idx]:.3f}  y[max]={y.max():.3f} @ step {int(y.argmax())}")
print(f"y[end]={y[-1]:.3f}")
# peak 之后的高度走势
tail = y[peak_idx:]
print(f"peak 之后: min y={tail.min():.3f} (跌落 {y[peak_idx]-tail.min():.3f} m), end y={tail[-1]:.3f}")
# 逐段统计 peak 前后的震荡
for lo, hi, name in [(0, peak_idx, "peak 之前"), (peak_idx, len(y) - 1, "peak 之后")]:
    seg = y[lo:hi + 1]
    print(f"{name}: steps {lo}..{hi}, y ∈ [{seg.min():.2f}, {seg.max():.2f}]")

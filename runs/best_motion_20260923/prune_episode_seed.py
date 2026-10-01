# 把最优 episode 在 credited 封顶步（590）处剪枝，导出：
#   1) pruned_590_episode.json  — 剪枝版评估归档（展示/核对用）
#   2) sil_seed_590.npz         — main_ppo --sil-seed 冷启动种子（raw_states/actions/contact）
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
OUT_DIR = Path(__file__).parent
PRUNE_STEP = 590  # credited_progress_m 首次封顶 8.099 的物理步（trace step 口径）

d = json.load(open(EP, encoding="utf-8"))
cfg = TrainConfig(**json.load(open(MANIFEST, encoding="utf-8"))["config"])

states = np.asarray(d["states"], dtype=np.float32)      # (1201, 33)
contacts = np.asarray(d["contacts"], dtype=np.float32)  # (1201, 5)
trace = d["trace"]                                       # 1200 条，trace[i].step = i+1
assert states.shape[0] == len(trace) + 1 and contacts.shape[0] == states.shape[0]

# trace[i] 的 action 把 states[i] 推进到 states[i+1]；保留动作 0..PRUNE_STEP-1
cut = PRUNE_STEP
pr_states = states[: cut + 1]                            # (591, 33)
pr_contacts = contacts[: cut + 1]                        # (591, 5)
pr_trace = trace[:cut]                                   # 590 条
actions = np.asarray([t["action"] for t in pr_trace], dtype=np.float32)  # (590, 2)
assert pr_states.shape == (cut + 1, 33) and actions.shape == (cut, 2)

# ── 剪枝版归档：metrics 如实重算 ──
m = dict(d["metrics"])
m["steps"] = cut
m["fell"] = False
m["left_corridor"] = any(not t["inside"] for t in pr_trace)
m["outside_fraction"] = float(np.mean([not t["inside"] for t in pr_trace]))
m["arrival_steps"] = min(m["arrival_steps"], cut)
m["pruned_from"] = {"source_steps": 1200, "prune_step": cut,
                    "reason": "credited_progress 封顶，后续为走廊外震荡与跌落（负面段）"}
pruned = {"metrics": m, "states": pr_states.tolist(),
          "contacts": pr_contacts.tolist(), "trace": pr_trace}
(OUT_DIR / "pruned_590_episode.json").write_text(
    json.dumps(pruned, ensure_ascii=False), encoding="utf-8")

# ── 种子 npz（main_ppo --sil-seed 加载；score 在训练侧用 eff_map 现算）──
meta = {"source_episode": EP, "prune_step": cut,
        "credited_progress_m": m["credited_progress_m"],
        "note": "pruned at credited-progress cap; score computed at load time"}
np.savez_compressed(OUT_DIR / "sil_seed_590.npz",
                    raw_states=pr_states, actions=actions, contact=pr_contacts,
                    meta=json.dumps(meta, ensure_ascii=False))

# ── 验证输出 ──
prog, peak_idx, dy, reach = progress_metric(pr_states, cfg)
y = pr_states[:, 1].astype(np.float64)
print(f"剪枝步: {cut}  raw_states={pr_states.shape}  actions={actions.shape}  contact={pr_contacts.shape}")
print(f"剪枝后: secured_peak_idx={peak_idx}  secured_dy={dy:.3f}m  reach={reach:.3f}m")
print(f"高度: start={y[0]:.3f}  max={y.max():.3f}@{int(y.argmax())}  end={y[-1]:.3f}  min={y.min():.3f}")
print(f"water_y_threshold={cfg.water_y_threshold} → 触水={bool(y.min() < cfg.water_y_threshold)}")
print(f"metrics: steps={m['steps']} fell={m['fell']} left_corridor={m['left_corridor']} "
      f"outside_fraction={m['outside_fraction']:.4f} credited={m['credited_progress_m']:.3f}")
print("saved:", OUT_DIR / "pruned_590_episode.json")
print("saved:", OUT_DIR / "sil_seed_590.npz")

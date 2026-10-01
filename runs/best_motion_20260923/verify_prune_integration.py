# 用真实最优 episode 验证 prune_trajectory 在训练口径下的行为（离线，不启动游戏）
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset, prune_trajectory
from training.reward import progress_metric, score_trajectory

EP = r"C:\Users\Symbol\aCodes\GettingOverIt\runs\corridor_formal_20260922_T_s31\checkpoints\eval_midpoint_096k\exploration_001_agent0.json"
MANIFEST = r"C:\Users\Symbol\aCodes\GettingOverIt\runs\corridor_formal_20260922_T_s31\checkpoints\eval_midpoint_096k\manifest.json"

cfg = TrainConfig(**json.load(open(MANIFEST, encoding="utf-8"))["config"])
d = json.load(open(EP, encoding="utf-8"))
traj = Trajectory(
    raw_states=np.asarray(d["states"], dtype=np.float32),
    actions=np.asarray([t["action"] for t in d["trace"]], dtype=np.float32),
    contact=np.asarray(d["contacts"], dtype=np.float32),
)

pruned = prune_trajectory(traj, cfg)
for name, t in [("完整", traj), ("剪枝", pruned)]:
    prog, pk, dy, reach = progress_metric(t.raw_states, cfg)
    print(f"{name}: states={t.raw_states.shape} actions={t.actions.shape} "
          f"peak_idx={pk} secured_dy={dy:.3f} score={score_trajectory(t.raw_states, None, cfg):.4f}")

# BC 窗口对比
ds = TrajectoryDataset(cfg, np.ones((4, 4), dtype=np.float32), None)
ds.set_trajectories([traj])
n_full = len(ds)
ds.set_trajectories([pruned])
n_pruned = len(ds)
print(f"BC 窗口数: 完整={n_full}  剪枝={n_pruned}  一致={n_full == n_pruned}")
print(f"存储缩减: actions {len(traj.actions)} → {len(pruned.actions)} "
      f"({100 * (1 - len(pruned.actions) / len(traj.actions)):.1f}% 负面段被物理移除)")

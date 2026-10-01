# 验证 sil_seed_590.npz 能按 main_ppo 的种子加载路径消费（不启动游戏、不建效率图）
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset
from training.reward import score_trajectory

MANIFEST = r"C:\Users\Symbol\aCodes\GettingOverIt\runs\corridor_formal_20260922_T_s31\checkpoints\eval_midpoint_096k\manifest.json"
SEED = Path(__file__).parent / "sil_seed_590.npz"

cfg = TrainConfig(**json.load(open(MANIFEST, encoding="utf-8"))["config"])

# 与 main_ppo 种子加载分支完全相同的代码路径（eff_map=None：bc_waypoint_weight=0 时不参与打分）
z = np.load(SEED)
traj = Trajectory(
    raw_states=np.asarray(z["raw_states"], dtype=np.float32),
    actions=np.asarray(z["actions"], dtype=np.float32),
    contact=np.asarray(z["contact"], dtype=np.float32) if "contact" in z.files else None,
    iteration=-1,
)
traj.score = score_trajectory(traj.raw_states, None, cfg)
print(f"种子: T={len(traj.actions)}  score={traj.score:.3f}  meta={z['meta'].item()}")

# 精英池入池 + BC 窗口索引（需要地形 mask，这里用全 1 假 mask 只验证索引逻辑）
fake_mask = np.ones((4, 4), dtype=np.float32)
ds = TrajectoryDataset(cfg, fake_mask, None)
ds.set_trajectories([traj])
print(f"入池后: 轨迹数={len(ds.trajectories)}  BC 窗口数={len(ds)}")
taus = [tau for _, tau in ds._index]
print(f"BC 监督 tau ∈ [0, {max(taus)}]（secured peak 截断；590 步中仅前 {max(taus)+1} 步进 BC）")

# 取一个样本确认 __getitem__ 不炸（patch 用假 mask 会全 1，仅验证形状）
dyn, patches, hist, mask, target = ds[len(ds) - 1]
print(f"样本: dynamics={tuple(dyn.shape)} patches={tuple(patches.shape)} "
      f"hist={tuple(hist.shape)} target={tuple(target.shape)}")
print("OK")

# 端到端验证：真实最优 episode 走走廊口径 SIL 管线（重放字段 → 剪枝 → 打分 → bd → BC 窗口 → patch 重建）
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset, prune_trajectory
from training.reference_corridor import ReferenceCorridor, corridor_cap_index, replay_corridor_fields
from training.reward import corridor_score, progress_metric

EP = ROOT / "runs/corridor_formal_20260922_T_s31/checkpoints/eval_midpoint_096k/exploration_001_agent0.json"
MANIFEST = ROOT / "runs/corridor_formal_20260922_T_s31/checkpoints/eval_midpoint_096k/manifest.json"
REF = ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json"

cfg = TrainConfig(**json.loads(MANIFEST.read_text(encoding="utf-8"))["config"])
route = ReferenceCorridor.from_file(str(REF), cfg)
d = json.loads(EP.read_text(encoding="utf-8"))

# 1) 重放走廊字段（rollout 在线记录的离线等价物）
states = np.asarray(d["states"], dtype=np.float32)
progress, guidance = replay_corridor_fields(route, states)
traj = Trajectory(
    raw_states=states,
    actions=np.asarray([t["action"] for t in d["trace"]], dtype=np.float32),
    contact=np.asarray(d["contacts"], dtype=np.float32),
    corridor_progress=progress,
    guidance_index=guidance,
)
print(f"重放: credited_total={progress.sum():.4f} (评估口径 8.0991)  "
      f"cap={corridor_cap_index(progress)} (期望 589)  guidance 段={sorted(set(guidance.tolist()))}")

# 2) 走廊口径剪枝
pruned = prune_trajectory(traj, cfg)
print(f"剪枝: {len(traj.actions)} → {len(pruned.actions)} 动作 "
      f"(states {len(traj.raw_states)} → {len(pruned.raw_states)})")
print(f"  对比高度口径会剪到 545；走廊口径保留到 credited 封顶（590）+ dwell")

# 3) 排序分
print(f"corridor_score={corridor_score(pruned.corridor_progress, cfg):.4f}")

# 4) bd
fake_mask = np.ones((4, 4), dtype=np.float32)
ds = TrajectoryDataset(cfg, fake_mask, None)
cid = ds._behavior_cell_id(pruned.raw_states, 5.0, pruned.corridor_progress)
cap_state = pruned.raw_states[corridor_cap_index(pruned.corridor_progress) + 1]
print(f"bd cell={cid}  封顶位置=({cap_state[0]:.2f}, {cap_state[1]:.2f})")

# 5) BC 窗口（对比高度口径的 531）
ds.set_trajectories([pruned])
taus = sorted(t for _, t in ds._index)
print(f"BC 窗口: τ ∈ [0, {max(taus)}]，共 {len(taus)} 个（高度口径仅 531 个，530 后有效推进不再被截掉）")

# 6) patch 重建：guidance_index 选图与 rollout 在线路径一致
ref_maps = route.guidance_maps(fake_mask.shape, 0, 0, cfg.grid_resolution)
ds2 = TrajectoryDataset(cfg, fake_mask, None, reference_maps=ref_maps)
ds2.set_trajectories([pruned])
sample = ds2[len(ds2) - 1]
print(f"patch 样本: {tuple(sample[1].shape)}，末窗口 τ={max(taus)} 取样 OK")

# 7) 与在线 _ppo_patches 的 wide 通道一致性数值抽查（同 state 同 guidance → 同图）
from training.dataset import build_patch
k = 550  # 走廊外段（step 550 guidance 值）
gid = int(pruned.guidance_index[k])
online_eff = ref_maps[gid] if gid >= 0 else None
p_replay = ds2._build_patch(pruned.raw_states[k], gid)
p_online = build_patch(pruned.raw_states[k], fake_mask, online_eff, cfg, 0, 0)
print(f"patch 一致性（step {k}, guidance={gid}）: wide 通道最大差 "
      f"{np.abs(p_replay[0] - p_online[0]).max():.2e}（应为 0）")
print("OK")

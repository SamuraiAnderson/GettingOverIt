# 离线隔离：corridor_progress 在 剪枝→入池→快照 链路中的保留情况
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset, prune_trajectory
from training.run_logging import save_elites

cfg = TrainConfig(d_model=16, nhead=2, num_layers=1, context_len=3, num_agents=2,
                  random_deploy=False, deploy_exclusion_check=False,
                  contact_heads_enabled=False)

T = 60
raw = np.zeros((T + 1, 33), dtype=np.float32)
raw[:, 0] = 50 + np.linspace(0, 3, T + 1)
raw[:, 1] = 10 + np.linspace(0, 8, T + 1)
raw[:, 23] = raw[:, 0] + 1
raw[:, 24] = raw[:, 1] + 1
acts = (np.tanh(np.random.randn(T, 2)) * 0.5).astype(np.float32)
cp = np.full(T, 0.1, dtype=np.float32)
gid = np.zeros(T + 1, dtype=np.int8)

t = Trajectory(raw_states=raw, actions=acts, corridor_progress=cp, guidance_index=gid)
print("orig   cp is None?", t.corridor_progress is None)

p = prune_trajectory(t, cfg)
print("pruned cp is None?", p.corridor_progress is None,
      "len:", None if p.corridor_progress is None else len(p.corridor_progress))

grid = 256
eff = np.random.rand(grid, grid).astype(np.float32)
terr = np.ones((grid, grid), dtype=bool)
ds = TrajectoryDataset(cfg, terr, eff, 0, 0)
ds.add_trajectories_map_elites([p], [5.0], iteration=0)
inpool = ds.trajectories[0]
print("pool   cp is None?", inpool.corridor_progress is None)

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "e.npz"
    save_elites(path, ds.trajectories)
    z = np.load(path)
    print("npz keys:", sorted(z.files))

"""Fine-tune the demonstrator mean to match the two stored elite action traces."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset
from training.experimental_policy import create_actor_critic
from training.reference_corridor import ReferenceCorridor
from training.reward import ClimbingEfficiencyMap
from training.run_logging import load_elites
from tests.control_interaction.test_l7_surface_airdrop import extract_polygons, load_colliders
from training.deploy_sampling import extract_body_local, extract_tip_local

INIT = ROOT / "runs/sparse_geometry_20260926_S_s47/checkpoints/ppo_iter_0109.pt"
ELITES = ROOT / "runs/sparse_geometry_20260926_S_s47/elites/elites_iter_0118.npz"
OUT = Path(__file__).resolve().parent / "imitation"
PAIRS = ((1, 17), (2, 16))
MORE_EPOCHS = 40


def load_windows(cfg):
    cache = OUT / "windows.pt"
    if cache.exists():
        keys = torch.load(cache, map_location="cpu", weights_only=True)
        print(f"cached windows {keys[0].shape[0]}", flush=True)
        return keys
    env, player = load_colliders(Path(cfg.game_root))
    polygons = extract_polygons(env)
    eff = ClimbingEfficiencyMap(
        polygons, grid_resolution=cfg.grid_resolution,
        diffusion_iterations=cfg.diffusion_iterations, diffusion_alpha=cfg.diffusion_alpha,
        height_prior_weight=cfg.height_prior_weight,
    )
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    maps = route.guidance_maps(eff.traversable.shape, eff._min_gx, eff._min_gy, cfg.grid_resolution)
    dataset = TrajectoryDataset(
        cfg, eff.traversable, eff.get_arr(), eff._min_gx, eff._min_gy,
        solid_polygons=polygons, tip_local=extract_tip_local(player),
        body_local=extract_body_local(player), reference_maps=maps,
    )
    rows = load_elites(ELITES)
    windows = []
    for rank, index in PAIRS:
        row = rows[index]
        traj = Trajectory(
            raw_states=np.asarray(row["raw_states"], np.float32),
            actions=np.asarray(row["actions"], np.float32),
            contact=None if row.get("contact") is None else np.asarray(row["contact"], np.float32),
            guidance_index=None if row.get("guidance_index") is None else np.asarray(row["guidance_index"]),
            corridor_progress=np.asarray(row["corridor_progress"], np.float32),
        )
        print(f"encode rank {rank} steps {len(traj.actions)}", flush=True)
        for tau in range(len(traj.actions)):
            windows.append(dataset.encode_window(traj, tau))
    keys = [torch.stack([w[i] for w in windows]) for i in range(9)]
    torch.save(keys, cache)
    print(f"windows {keys[0].shape[0]}", flush=True)
    return keys


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    resume = OUT / "imitation.pt"
    source = resume if resume.exists() else INIT
    saved = torch.load(source, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in saved["config"].items() if k in TrainConfig.__dataclass_fields__})
    model = create_actor_critic(cfg)
    model.load_state_dict(saved["model"])
    model.eval()
    keys = load_windows(cfg)
    n = keys[0].shape[0]
    history_path = OUT / "train.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {
        "init": str(INIT), "windows": n, "losses": [],
    }
    losses = list(history["losses"])
    start = len(losses)
    params = [p for name, p in model.named_parameters() if not name.startswith("critic_head") and p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
    print(f"resume epoch {start} from {source}", flush=True)
    for epoch in range(start, start + MORE_EPOCHS):
        perm = torch.randperm(n)
        total = 0.0
        seen = 0
        for begin in range(0, n, 64):
            idx = perm[begin:begin + 64]
            dist, _ = model.forward(
                keys[0][idx], keys[1][idx], keys[2][idx], keys[3][idx],
                keys[5][idx], keys[6][idx], keys[7][idx], keys[8][idx],
            )
            pred = torch.tanh(dist.mean) * model._action_scale
            loss = torch.mean((pred - keys[4][idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        mean = total / seen
        losses.append(mean)
        print(f"epoch {epoch} mse {mean:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "config": saved["config"],
                    "reference_corridor": saved.get("reference_corridor"),
                    "policy_context": saved.get("policy_context", {})}, resume)
        history.update(windows=n, losses=losses, checkpoint=str(resume))
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if mean < 1.0:
            break
    print(resume, flush=True)


if __name__ == "__main__":
    main()

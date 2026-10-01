"""Deterministic 16-player playback of imitation.pt on the current desktop."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.experimental_policy import create_actor_critic
from training.model import expand_state_dict_for_dynamics_dim
from training.reference_corridor import ReferenceCorridor
from training.reward import ClimbingEfficiencyMap
from training.rollout import RolloutWorker
from start.game_launcher import GameLauncher
from tests.control_interaction.test_l7_surface_airdrop import extract_polygons, load_colliders

CKPT = Path(__file__).resolve().parent / "imitation.pt"
STEPS = 2400

_launch = GameLauncher.launch


def launch_on_current_desktop(self, wait=False, *, desktop="GoiRuntime"):
    return _launch(self, wait=wait, desktop=None)


def main():
    GameLauncher.launch = launch_on_current_desktop
    saved = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in saved["config"].items() if k in TrainConfig.__dataclass_fields__})
    cfg.num_agents = 16
    cfg.random_deploy = False
    cfg.ou_enabled = False
    cfg.sil_enabled = False
    cfg.explore_boost_enabled = False
    game_root = Path(cfg.game_root)
    runtime = game_root / "GoiData" / "runtime_config.json"
    original = runtime.read_bytes() if runtime.exists() else None
    model = create_actor_critic(cfg)
    model.load_state_dict(expand_state_dict_for_dynamics_dim(saved["model"], model.state_dict()))
    model.eval()
    worker = RolloutWorker(cfg)
    env_data, player_data = load_colliders(game_root)
    polygons = extract_polygons(env_data)
    eff = ClimbingEfficiencyMap(
        polygons, grid_resolution=cfg.grid_resolution,
        diffusion_iterations=cfg.diffusion_iterations, diffusion_alpha=cfg.diffusion_alpha,
        height_prior_weight=cfg.height_prior_weight,
    )
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    worker.set_terrain_info(eff.traversable, eff._min_gx, eff._min_gy)
    worker._reference_maps = route.guidance_maps(
        eff.traversable.shape, eff._min_gx, eff._min_gy, cfg.grid_resolution)
    worker.set_deploy_obstacle_geometry(polygons, player_data)
    print("当前桌面，16 players，确定性策略测试", flush=True)
    try:
        worker.launch_game()
        worker.env.set_camera_free(False)
        worker.collect_ppo(model, STEPS, eff, deterministic=True, initial_offsets=np.zeros((16, 2), np.float32))
        print("closed-loop finished", flush=True)
    finally:
        worker.teardown()
        if original is None:
            if runtime.exists():
                runtime.unlink()
        else:
            runtime.write_bytes(original)


if __name__ == "__main__":
    main()

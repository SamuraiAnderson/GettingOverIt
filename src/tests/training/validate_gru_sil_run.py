"""Offline acceptance of an actual --gru-sil --save-rollout-inputs run.

Rebuild every archived elite's raw inputs and compare with the actual online
frames. Audit every recorded behavior action with its saved frozen weights.
Does not start a game or perform any optimizer updates.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from training.main_ppo import load_colliders, extract_polygons
from training.dataset import TrajectoryDataset
from training.decision_buffer import DecisionRolloutBuffer
from training.deploy_sampling import extract_body_local, extract_tip_local
from training.elite_archive import EliteTrajectoryArchive
from training.experimental_policy import create_actor_critic
from training.experiment_options import ppo_config
from training.policy_audit import audit_policy
from training.reference_corridor import ReferenceCorridor
from training.reward import ClimbingEfficiencyMap


def validate(root):
    root = Path(root)
    torch.set_num_threads(4)
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "completed", meta["status"]
    rows = list(csv.DictReader((root / "metrics.csv").open(encoding="utf-8")))
    result = {"run": str(root.resolve()), "iterations": [], "elite_inputs": []}
    for row in rows:
        iteration = int(row["iteration"])
        saved = torch.load(root / "checkpoints" / f"ppo_iter_{iteration:04d}.pt", map_location="cpu", weights_only=False)
        assert saved["optimizer"] and saved["config"]["gru_sil_enabled"]
        cfg = ppo_config(saved)
        assert float(row["gru_sil_updates"]) <= min(cfg.gru_sil_max_updates,
                                                    int(float(row["n_updates"]) * cfg.gru_sil_budget_ratio))
        assert float(row["gru_sil_kl"]) <= cfg.gru_sil_target_kl
        assert float(row["gru_sil_frame_kl_max"]) <= cfg.gru_sil_max_frame_kl
        rollout_dir = root / "checkpoints" / "rollouts"
        behavior = torch.load(rollout_dir / f"behavior_{iteration:04d}.pt", map_location="cpu", weights_only=False)
        model = create_actor_critic(ppo_config(behavior)).eval()
        model.load_state_dict(behavior["model"])
        buffer = DecisionRolloutBuffer.load(rollout_dir / f"iter_{iteration:04d}.npz")
        audit = audit_policy(model, buffer, max_samples=None)
        result["iterations"].append({"iteration": iteration, **audit,
                                     **{k: float(v) for k, v in row.items() if k.startswith("gru_sil_")}})
    cfg = ppo_config(saved)
    archive = EliteTrajectoryArchive(cfg, root / "elites" / "full_episodes", meta["reference_fingerprint"])
    assert archive.restore(root / "elites" / "full_episodes" / f"elites_iter_{iteration:04d}.npz") > 0
    env_data, player_data = load_colliders(Path(cfg.game_root))
    polygons = extract_polygons(env_data)
    eff = ClimbingEfficiencyMap(polygons, grid_resolution=cfg.grid_resolution,
        diffusion_iterations=cfg.diffusion_iterations, diffusion_alpha=cfg.diffusion_alpha,
        cache_path=str(root / "checkpoints" / "cache" / "traversable_mask.npz"),
        height_prior_weight=cfg.height_prior_weight)
    maps = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg).guidance_maps(
        eff.traversable.shape, eff._min_gx, eff._min_gy, cfg.grid_resolution)
    encoder = TrajectoryDataset(cfg, eff.traversable, eff.get_arr(), min_gx=eff._min_gx, min_gy=eff._min_gy,
        solid_polygons=polygons, tip_local=extract_tip_local(player_data), body_local=extract_body_local(player_data),
        reference_maps=maps)
    for traj in archive.trajectories:
        source = DecisionRolloutBuffer.load(rollout_dir / f"iter_{traj.iteration:04d}.npz")
        match = [ep for ep in source.episodes if np.array_equal(ep["actions"], traj.actions)]
        assert len(match) == 1, "Cannot uniquely identify the elite's online episode"
        rebuilt = encoder.encode_sequence(traj)
        errors = {name: float(np.max(np.abs(data - match[0][name]))) for name, data in rebuilt.items()}
        assert all(v == 0 for v in errors.values()), errors
        result["elite_inputs"].append({"source_iteration": traj.iteration,
                                       "frames": len(traj.raw_states), "max_errors": errors})
    assert sum(r["gru_sil_updates"] for r in result["iterations"]) > 0, "No accepted SIL update exercised"
    result["passed"] = True
    (root / "gru_sil_acceptance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    validate(parser.parse_args().run_dir)

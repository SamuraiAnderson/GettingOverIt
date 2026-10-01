"""Prepare a confirmed-endpoint formal run. Does not launch the game.

    conda run --no-capture-output -n getting-over-it-analysis python -m src.tests.analysis.prepare_v2_formal
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from training.config import TrainConfig
from training.experimental_policy import create_actor_critic
from training.reference_corridor import (
    PROGRESS_MODE_V1, PROGRESS_MODE_V2, ReferenceCorridor, progress_mode_of, trajectory_endpoint,
)
from training.reference_fingerprint import reference_fingerprint, write_fingerprint_sidecar
from training.run_logging import load_elites

SOURCE = ROOT / "runs/sparse_geometry_20260926_S_s47/checkpoints/ppo_iter_0118.pt"
POOL = ROOT / "runs/sparse_geometry_20260926/td_v2_rerank/elites_v2.npz"
RANKING = ROOT / "runs/sparse_geometry_20260926/td_v2_rerank/ranking.json"
ROUTE = ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory_game_world.json"
CAMPAIGN = "confirmed_endpoint_v2_formal_20260928"
PREP = ROOT / "runs" / CAMPAIGN


def training_arguments():
    return list(map(str, [
        "--init-from", SOURCE,
        "--sil-init-pool", POOL,
        "--reference-trajectory", ROUTE,
        "--reference-corridor-radius", 1.5,
        "--reference-deviation-weight", 0.01,
        "--geometry-profile", "native-pose-v1",
        "--no-random-deploy",
        "--recipe", "ppo-ou",
        "--seed", 47,
        "--num-agents", 8,
        "--steps-per-rollout", 1200,
        "--max-iterations", 20,
        "--value-warmup-iters", 2,
        "--lr", 0.00003,
        "--ent-coef", 0.001,
        "--ppo-batch-size", 128,
        "--save-interval", 1,
        "--torch-threads", 4,
        "--features", "none",
        "--policy-backbone", "transformer",
        "--normalize-action-history",
        "--event-source", "native",
        "--no-event-branch",
        "--action-segment-steps", 1,
        "--sparse-geometry",
        "--run-dir", f"{CAMPAIGN}_S_s47",
        "--persist-game",
        "--save-rollout-inputs",
        "--gamma", 0.99,
        "--gae-lambda", 0.95,
        "--corridor-sil",
        "--sil-archive-mode", "td_pareto",
        "--target-physical-samples", 96000,
    ]))


def launch_config() -> TrainConfig:
    config = TrainConfig()
    config.reference_trajectory = str(ROUTE.resolve())
    config.reference_corridor_radius = 1.5
    config.reference_deviation_weight = 0.01
    return config


def expected_fingerprint(config: TrainConfig | None = None) -> dict:
    config = launch_config() if config is None else config
    route = ReferenceCorridor.from_file(config.reference_trajectory, config)
    return reference_fingerprint(
        config, route_meta=route.metadata, schema_id="anchor35", raw_dim=35,
    )


def pool_endpoints() -> list[dict]:
    rows = []
    for index, row in enumerate(load_elites(POOL)):
        if row["progress_mode"] != PROGRESS_MODE_V2 or row["corridor_confirmed_progress"] is None:
            raise ValueError(f"elite {index} is missing confirmed_endpoint_v2 fields")
        from training.dataset import Trajectory
        endpoint = trajectory_endpoint(Trajectory(**row))
        rows.append({
            "source_index": index,
            "t": endpoint.t_cap,
            "D": endpoint.distance,
            "valid": endpoint.valid,
            "cap": endpoint.cap,
        })
    return rows


def check_ranking(endpoints: list[dict]) -> None:
    ranking = json.loads(RANKING.read_text(encoding="utf-8"))
    recorded = sorted((row["t"], round(row["D"], 6), row["valid"]) for row in ranking["by_D"])
    stored = sorted((row["t"], round(row["D"], 6), row["valid"]) for row in endpoints)
    if recorded != stored:
        raise ValueError("elites_v2.npz (t, D) does not match ranking.json")


def check_checkpoint() -> dict:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    saved = torch.load(SOURCE, map_location="cpu", weights_only=False)
    mode = progress_mode_of(saved.get("config", {}))
    if mode != PROGRESS_MODE_V1:
        raise ValueError(f"expected a legacy checkpoint, found {mode}")
    if saved.get("config", {}).get("ppo_reset_log_std_on_init"):
        raise ValueError("source checkpoint would reset sigma")
    config = launch_config()
    model = create_actor_critic(config)
    model.load_state_dict(saved["model"])
    before = [tensor.detach().clone() for tensor in model.critic_head.parameters()]
    actor = model.actor_mean.weight.detach().clone()
    model.reset_critic_head()
    if all(torch.equal(old, new) for old, new in zip(before, model.critic_head.parameters())):
        raise ValueError("critic head did not change")
    if not torch.equal(actor, model.actor_mean.weight.detach()):
        raise ValueError("actor weights changed while resetting the critic")
    return {
        "iteration": int(saved.get("iteration", -1)),
        "source_mode": mode,
        "target_mode": PROGRESS_MODE_V2,
        "critic_reinitialized": True,
        "actor_preserved": True,
    }


def main() -> None:
    fingerprint = expected_fingerprint()
    write_fingerprint_sidecar(POOL, fingerprint)
    endpoints = pool_endpoints()
    check_ranking(endpoints)
    checkpoint = check_checkpoint()
    PREP.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign": CAMPAIGN,
        "init_from": str(SOURCE),
        "sil_init_pool": str(POOL),
        "fingerprint": fingerprint["fingerprint"],
        "checkpoint": checkpoint,
        "pool_n": len(endpoints),
        "command": ["python", "-m", "src.training.main_ppo", *training_arguments()],
        "notes": [
            "Do not --resume the sparse-geometry checkpoint; that locks legacy_credit_v1.",
            "Critic is reinitialized because the progress mode changes. Actor and log_std are kept.",
            "The original elites_iter_0118.npz is not modified.",
            "First segment stops at 96000 physical samples, including critic warmup.",
        ],
    }
    (PREP / "prep.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"prepared {PREP / 'prep.json'}", flush=True)


if __name__ == "__main__":
    main()

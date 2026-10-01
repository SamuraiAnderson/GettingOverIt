"""Open-loop and closed-loop controls for rank1 and rank2.

Five game launches, two resets each (rank1 then rank2). Sixteen agents:
four per condition, group-to-slot rotated every reset.
"""
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from tests.control_interaction.test_l7_surface_airdrop import extract_polygons, load_colliders
from training.config import TrainConfig
from training.contact_features import ContactPrevState
from training.dataset import build_dynamics, left_pad_sequence
from training.experimental_policy import create_actor_critic
from training.reference_anchor import reference_position, scoring_context
from training.reward import ClimbingEfficiencyMap, is_water
from training.rollout import RolloutWorker
from training.run_logging import load_elites

ELITES = ROOT / "runs/sparse_geometry_20260926_S_s47/elites/elites_iter_0118.npz"
PARENT = Path(__file__).resolve().parent
FROZEN = PARENT / "lr_control_20260927" / "frozen_epoch69.pt"
WINDOWS = PARENT / "windows.pt"
LR_OUT = PARENT / "lr_control_20260927"
OUT = LR_OUT / "game"
GROUPS = ("original", "predicted", "closed_69", "closed_best")
RANKS = (
    {"name": "rank1", "elite": 17, "steps": 815, "window_slice": slice(0, 815)},
    {"name": "rank2", "elite": 16, "steps": 491, "window_slice": slice(815, 1306)},
)
LAUNCHES = 5
SPARSE = ("contour_tokens", "contour_mask", "relation_tokens", "relation_mask")


def choose_best():
    rows = []
    for group in ("A", "B"):
        for seed in (47, 48, 49):
            metrics = json.loads((LR_OUT / f"{group}_s{seed}" / "best_metrics.json").read_text(encoding="utf-8"))
            rows.append((
                metrics["overall"]["mse"], metrics["overall"]["p95"], seed, group, metrics["epoch"],
            ))
    rows.sort()
    mse, p95, seed, group, epoch = rows[0]
    return {
        "group": group, "seed": seed, "epoch": epoch, "eval_mse": mse, "p95": p95,
        "path": str(LR_OUT / f"{group}_s{seed}" / "best.pt"),
        "ranking": [
            {"eval_mse": a, "p95": b, "seed": c, "group": d, "epoch": e}
            for a, b, c, d, e in rows
        ],
    }


def load_policy(path, dev):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in saved["config"].items() if k in TrainConfig.__dataclass_fields__})
    model = create_actor_critic(cfg).to(dev)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, saved


def predict_rows(model, keys, dev):
    from run_lr_control import forward_all
    return forward_all(model, keys, dev).numpy().astype(np.float32)


def assignment(launch, reset):
    shift = (launch * 2 + reset) % len(GROUPS)
    order = GROUPS[shift:] + GROUPS[:shift]
    mapping = {}
    for block, name in enumerate(order):
        for slot in range(4):
            mapping[block * 4 + slot] = name
    return mapping


def window_batch(traces, ids, cfg, dev):
    arrays, masks = [], []
    for agent in ids:
        frames = traces[agent]
        packed = [
            left_pad_sequence(np.asarray(frames[name][-cfg.context_len:]), cfg.context_len)
            for name in ("dynamics", "patches", "act_history")
        ]
        arrays.append([item for item, _mask in packed])
        masks.append(packed[0][1])
    tensors = [torch.as_tensor(np.stack([row[k] for row in arrays]), dtype=torch.float32, device=dev) for k in range(3)]
    mask = torch.as_tensor(np.stack(masks), dtype=torch.float32, device=dev)
    sparse = []
    for name in SPARSE:
        stacked = np.stack([
            left_pad_sequence(np.asarray(traces[agent][name][-cfg.context_len:]), cfg.context_len)[0]
            for agent in ids
        ])
        sparse.append(torch.as_tensor(stacked, dtype=torch.float32, device=dev))
    return tensors, mask, sparse


def closed_actions(models, traces, ids_by_model, cfg, dev):
    actions = {}
    for model, ids in ids_by_model.items():
        (dynamics, patches, acts), mask, sparse = window_batch(traces, ids, cfg, dev)
        noise = torch.zeros(len(ids), model.parameter_dim, device=dev)
        with torch.no_grad():
            value = model.get_action_and_value(
                dynamics, patches, acts, mask, noise, 0.0, True,
                sparse[0], sparse[1], sparse[2], sparse[3],
            )
        taken = value.action.detach().cpu().numpy().astype(np.float32)
        for row, agent in enumerate(ids):
            actions[agent] = taken[row]
    return actions


def play(worker, route, cfg, eff, obs, rank, actions, models, dev, schema, meta):
    steps = rank["steps"]
    agents = list(range(cfg.num_agents))
    trackers = [route.tracker_from_state(obs[i], schema, meta) for i in agents]
    contact_prev = {i: ContactPrevState() for i in agents}
    contacts, contact_prev = worker._compute_contact_batch(obs, contact_prev)
    traces = {}
    ids_by_model = {}
    for agent, name in actions["group"].items():
        if name not in ("closed_69", "closed_best"):
            continue
        model = models[name]
        traces[agent] = {key: [] for key in ("dynamics", "patches", "act_history", *SPARSE)}
        ids_by_model.setdefault(model, []).append(agent)
    water = np.zeros(cfg.num_agents, dtype=bool)
    peak = np.zeros(cfg.num_agents)
    retreat = np.zeros(cfg.num_agents)
    for tracker, i in zip(trackers, agents):
        peak[i] = tracker.route_progress()
    for step in range(steps):
        patches = worker._ppo_patches(obs, eff, {i: trackers[i] for i in agents})
        sparse = worker._ppo_sparse(obs)
        for agent in traces:
            prev = actions["played"][agent][-1] if actions["played"][agent] else np.zeros(2, np.float32)
            row = traces[agent]
            row["dynamics"].append(build_dynamics(obs[agent], contacts[agent]))
            row["patches"].append(np.asarray(patches[agent], np.float32).copy())
            row["act_history"].append(np.asarray(prev, np.float32).copy())
            for name, plane in zip(SPARSE, sparse):
                row[name].append(np.asarray(plane[agent], np.float32).copy())
        live = closed_actions(models, traces, ids_by_model, cfg, dev) if traces else {}
        batch = np.zeros((cfg.num_agents, 2), np.float32)
        for agent, name in actions["group"].items():
            if agent in live:
                batch[agent] = live[agent]
            else:
                batch[agent] = actions["script"][name][step]
            actions["played"][agent].append(batch[agent].copy())
        obs, _dones = worker.env.step(batch)
        contacts, contact_prev = worker._compute_contact_batch(obs, contact_prev)
        for i, tracker in enumerate(trackers):
            route.step_state(tracker, obs[i], schema, meta)
            progress = tracker.route_progress()
            peak[i] = max(peak[i], progress)
            retreat[i] = max(retreat[i], peak[i] - progress)
            if is_water(float(obs[i, 1]), cfg):
                water[i] = True
    rows = []
    for i, tracker in enumerate(trackers):
        rows.append({
            "agent": i,
            "group": actions["group"][i],
            "D": float(tracker.confirmed_d),
            "endpoint_progress": float(tracker.route_progress()),
            "water": bool(water[i]),
            "max_retreat": float(retreat[i]),
            "valid": not tracker.invalid,
            "cap": int(tracker.last_credit_action),
        })
    return rows


def prepare_rank(rank, rows, pred, demo_schema, demo_meta, route):
    elite = rows[rank["elite"]]
    states = np.asarray(elite["raw_states"], np.float64)
    demo = reference_position(states[0], demo_schema, demo_meta)
    return {
        "states0": states[0],
        "demo_pot": demo,
        "original": np.asarray(elite["actions"], np.float32),
        "predicted": pred[rank["window_slice"]],
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    choice = choose_best()
    model69, saved = load_policy(FROZEN, dev)
    model_best, _ = load_policy(choice["path"], dev)
    keys = torch.load(WINDOWS, map_location="cpu", weights_only=True)
    predicted = predict_rows(model69, keys, dev)
    cfg = TrainConfig(**{k: v for k, v in saved["config"].items() if k in TrainConfig.__dataclass_fields__})
    cfg.num_agents = 16
    cfg.random_deploy = False
    if not cfg.reference_trajectory:
        raise RuntimeError("frozen checkpoint has no reference trajectory")
    env_colliders, player = load_colliders(Path(cfg.game_root))
    polygons = extract_polygons(env_colliders)
    eff = ClimbingEfficiencyMap(
        polygons, grid_resolution=cfg.grid_resolution,
        diffusion_iterations=cfg.diffusion_iterations, diffusion_alpha=cfg.diffusion_alpha,
        height_prior_weight=cfg.height_prior_weight,
    )
    worker = RolloutWorker(cfg)
    worker.set_terrain_info(eff.traversable, eff._min_gx, eff._min_gy)
    worker.set_deploy_obstacle_geometry(polygons, player)
    route = worker.reference_corridor
    rows = load_elites(ELITES)
    demo_schema, demo_meta = scoring_context(np.asarray(rows[17]["raw_states"][0]), cfg, None, route)
    rank_data = {
        spec["name"]: prepare_rank(spec, rows, predicted, demo_schema, demo_meta, route)
        for spec in RANKS
    }
    config = {
        "launches": LAUNCHES,
        "resets_per_launch": 2,
        "agents": 16,
        "agents_per_group": 4,
        "groups": list(GROUPS),
        "rotation": "block shift (launch * 2 + reset) mod 4",
        "ranks": [{"name": spec["name"], "steps": spec["steps"], "elite": spec["elite"]} for spec in RANKS],
        "closed_69": str(FROZEN),
        "closed_best": choice,
        "predicted_source": "epoch 69 deterministic mean on the stored demonstration windows",
        "random_deploy": False,
        "exploration": False,
        "snapshot": "one canonical snapshot per launch, shared by both resets",
        "D": "confirmed_endpoint_v2 confirmed_d",
        "endpoint_progress": "route arc at the final frame",
        "max_retreat": "max drop of route arc from its running peak",
        "device": str(dev),
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"launches": []}
    runtime = Path(cfg.game_root) / "GoiData" / "runtime_config.json"
    original = runtime.read_bytes() if runtime.exists() else None
    models = {"closed_69": model69, "closed_best": model_best}
    try:
        for launch in range(LAUNCHES):
            worker.launch_game()
            launch_row = {"launch": launch, "resets": []}
            report["launches"].append(launch_row)
            try:
                for reset, spec in enumerate(RANKS):
                    obs = worker.env.reset()
                    schema, meta = scoring_context(obs[0], cfg, worker.env, route)
                    data = rank_data[spec["name"]]
                    pots = reference_position(obs, schema, meta)
                    pot_err = np.linalg.norm(pots - data["demo_pot"], axis=1)
                    player_err = np.linalg.norm(obs[:, :2] - data["states0"][:2], axis=1)
                    groups = assignment(launch, reset)
                    bundle = {
                        "group": groups,
                        "script": {"original": data["original"], "predicted": data["predicted"]},
                        "played": {i: [] for i in range(16)},
                    }
                    agents = play(worker, route, cfg, eff, obs, spec, bundle, models, dev, schema, meta)
                    launch_row["resets"].append({
                        "reset": reset,
                        "rank": spec["name"],
                        "steps": spec["steps"],
                        "assignment": {str(i): groups[i] for i in range(16)},
                        "init_pot_err_vs_demo_m": [float(v) for v in pot_err],
                        "init_player_err_vs_demo_m": [float(v) for v in player_err],
                        "init_pot_spread_m": float(np.max(np.linalg.norm(pots[:, None, :] - pots[None, :, :], axis=2))),
                        "agents": agents,
                    })
                    med = {
                        name: float(np.median([row["D"] for row in agents if row["group"] == name]))
                        for name in GROUPS
                    }
                    print(
                        f"launch {launch} reset {reset} {spec['name']} "
                        + " ".join(f"{name} {med[name]:.2f}" for name in GROUPS),
                        flush=True,
                    )
                    (OUT / "episodes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            finally:
                worker.close_game()
        print("game control finished", flush=True)
    finally:
        if original is None:
            if runtime.exists():
                runtime.unlink()
        else:
            runtime.write_bytes(original)


if __name__ == "__main__":
    main()

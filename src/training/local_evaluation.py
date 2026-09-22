"""Fixed corridor evaluation, using the same collect_ppo observation pipeline."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.geometry_profile import configure_geometry
from training.experimental_policy import create_actor_critic
from training.experiment_options import add_experiment_arguments, apply_experiment_arguments, restore_initialization_shape
from training.model import expand_state_dict_for_dynamics_dim
from training.policy_audit import audit_policy
from training.ppo_trainer import PPOTrainer
from training.reward import ClimbingEfficiencyMap, is_water
from training.rollout import RolloutWorker, _warmup_and_snapshot
from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons


def score_corridor_episode(trajectory, route, goal_progress=5.0, dwell=10, fall_drop=3.0):
    """Local success = credited progress + current forward position, held in corridor.

    The target is relative to the fixed reset location on its continuous segment.
    Gaps are never counted as success. Drop is peak-to-current player height.
    """
    if goal_progress <= 0 or dwell < 1 or fall_drop <= 0:
        raise ValueError("Positive goal, dwell, and fall threshold required")
    states = trajectory.raw_states
    tracker = route.tracker(states[0, :2])
    start_segment, start_progress = tracker.segment_index, tracker.progress
    if start_progress + goal_progress > route.segments[start_segment].length - route.config.reference_corridor_radius:
        raise ValueError("Local goal reaches a reference gap; choose a shorter goal")
    held = 0
    arrival = None
    peak_y = float(states[0, 1])
    fell = False
    outside = 0
    trace = []
    previous_contacts = None
    previous_action = np.zeros(2)
    for step, (state, action) in enumerate(zip(states[1:], trajectory.actions), 1):
        result = tracker.step(state[:2])
        peak_y = max(peak_y, float(state[1]))
        fell |= is_water(state[1], route.config) or peak_y - float(state[1]) >= fall_drop
        outside += int(not result.inside and not result.gap)
        qualifies = (result.inside and not result.gap and tracker.segment_index == start_segment
                     and tracker.progress >= start_progress + goal_progress
                     and tracker.total_progress >= goal_progress)
        held = held + 1 if qualifies else 0
        if held >= dwell and arrival is None:
            arrival = step
        contacts = trajectory.contact[step - 1, [0, 2, 3]] if trajectory.contact is not None else None
        enter = np.zeros(3) if previous_contacts is None or contacts is None else (contacts > previous_contacts)
        leave = np.zeros(3) if previous_contacts is None or contacts is None else (contacts < previous_contacts)
        trace.append({"step": step, "progress_m": tracker.progress - start_progress,
                      "credited_progress_m": tracker.total_progress, "inside": result.inside,
                      "gap": result.gap, "contact_before_action": None if contacts is None else contacts.tolist(),
                      "contact_enter": enter.astype(int).tolist(), "contact_exit": leave.astype(int).tolist(),
                      "action": action.tolist(), "action_delta": float(np.linalg.norm(action - previous_action))})
        previous_contacts, previous_action = contacts, action
    return {
        "success": arrival is not None, "arrival_steps": arrival, "fell": bool(fell),
        "left_corridor": outside > 0, "outside_fraction": outside / max(1, len(trace)),
        "credited_progress_m": tracker.total_progress, "start_segment": start_segment,
        "start_progress_m": start_progress, "steps": len(trace),
    }, trace


def summarize(records):
    result = {}
    for mode in sorted({r["mode"] for r in records}):
        rows = [r for r in records if r["mode"] == mode]
        n = len(rows)
        successes = sum(r["success"] for r in rows)
        p, z = successes / n, 1.96
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        arrivals = [r["arrival_steps"] for r in rows if r["success"]]
        result[mode] = {"episodes": n, "success_rate": p, "success_wilson95": [center-half, center+half],
                        "interval_note": "Descriptive only: agents share trial seeds; episodes are not independent.",
                        "fall_rate": float(np.mean([r["fell"] for r in rows])),
                        "leave_corridor_rate": float(np.mean([r["left_corridor"] for r in rows])),
                        "mean_credited_progress_m": float(np.mean([r["credited_progress_m"] for r in rows])),
                        "mean_arrival_steps": float(np.mean(arrivals)) if arrivals else None}
    return result


def compare_native_contacts(contacts, packets, agent):
    aligned = all(p.get("aligned", False) for p in packets[:len(contacts)])
    native = np.array([p["agents"][agent]["contact"] for p in packets[:len(contacts)]])
    known = np.array([p["agents"][agent]["contactKnown"] for p in packets[:len(contacts)]], dtype=bool)
    proxy = np.column_stack([contacts[:, 0], contacts[:, 1], np.maximum(contacts[:, 2], contacts[:, 3])])
    names = ["tip_contact", "tip_grip", "player_contact_union"]
    if packets[0].get("version", 1) >= 2:
        physics = np.array([p["agents"][agent]["physicsContact"] for p in packets[:len(contacts)]])
        valid = np.array([p["agents"][agent]["physicsKnown"] for p in packets[:len(contacts)]], dtype=bool)
        # Direct engine contacts are the reference for geometry; retain callback grip separately.
        native = np.column_stack([physics[:, 0], native[:, 1], physics[:, 1:].max(axis=1),
                                  physics[:, 1], physics[:, 2]])
        known = np.column_stack([valid[:, 0], known[:, 1], valid[:, 1:].all(axis=1), valid[:, 1], valid[:, 2]])
        proxy = np.column_stack([proxy, contacts[:, 2], contacts[:, 3]])
        names += ["body_contact", "pot_contact"]
    result = {}
    for k, name in enumerate(names):
        mask = known[:, k]
        a, b = proxy[mask, k] > .5, native[mask, k] > .5
        result[name] = {"aligned": aligned, "known_frames": int(mask.sum()), "total_frames": len(mask),
                        "native_positive": int(b.sum()), "proxy_positive": int(a.sum()),
                        "false_positive": int((a & ~b).sum()), "false_negative": int((~a & b).sum()),
                        "mismatch_fraction": float(np.mean(a != b)) if len(a) else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-trajectory", type=Path, default=None,
                        help="Override the saved route; defaults to checkpoint route or the project's video reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--jitter", type=float, default=.01)
    parser.add_argument("--goal-progress", type=float, default=5.)
    parser.add_argument("--dwell", type=int, default=10)
    parser.add_argument("--noise-phi", type=float, default=.85)
    parser.add_argument("--inference-batch-size", type=int, default=0)
    parser.add_argument("--geometry-profile", choices=("native-pose-v1", "legacy"), default=None)
    parser.add_argument("--native-diagnostics", action="store_true")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--train-smoke", type=int, default=0, help="Fresh rollouts and updates after frozen evaluation")
    parser.add_argument("--replay", type=Path, help="Offline audit of archived NPZ; requires behavior checkpoint")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if (args.steps < 1 or args.rounds < 1 or args.num_agents < 1 or args.jitter < 0
            or args.train_smoke < 0 or args.goal_progress <= 0 or args.dwell < 1
            or args.inference_batch_size < 0 or not 0 <= args.noise_phi < 1):
        raise ValueError("Invalid evaluation lengths, jitter, goal, batch size or noise_phi (must be in [0, 1))")
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = TrainConfig(contact_heads_enabled=False, phase_weighting_enabled=False,
                      sil_advantage_weighting_enabled=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else {}
    if args.init_from and args.checkpoint:
        raise ValueError("--init-from and --checkpoint are mutually exclusive")
    init_state = torch.load(args.init_from, map_location="cpu", weights_only=False) if args.init_from else {}
    restore_initialization_shape(cfg, init_state)
    for f in fields(cfg):
        if f.name in checkpoint.get("config", {}) and f.name not in ("game_root", "port"):
            setattr(cfg, f.name, checkpoint["config"][f.name])
    if checkpoint:
        cfg.contact_heads_enabled = "actor_mean_contact.weight" in checkpoint["model"]
    configure_geometry(cfg, checkpoint or init_state, args.geometry_profile)
    apply_experiment_arguments(cfg, args, checkpoint)
    cfg.num_agents = args.num_agents
    cfg.random_deploy = cfg.sil_enabled = cfg.explore_boost_enabled = False
    reference_path = args.reference_trajectory or cfg.reference_trajectory or (
        ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json")
    cfg.reference_trajectory = str(Path(reference_path).resolve())
    cfg.ou_enabled, cfg.ou_phi = args.noise_phi > 0, args.noise_phi
    cfg.ppo_inference_batch_size = args.inference_batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_actor_critic(cfg).to(device).eval()
    if checkpoint:
        model.load_state_dict(expand_state_dict_for_dynamics_dim(checkpoint["model"], model.state_dict()))
    elif init_state:
        if hasattr(model, "initialize_from"):
            model.initialize_from(init_state["model"])
        else:
            model.load_state_dict(init_state["model"])
    model.exploration_boost = checkpoint.get("policy_context", {}).get("exploration_boost", 0.)
    if args.replay:
        if not args.checkpoint:
            raise ValueError("Replay requires the archived behavior checkpoint")
        from training.ppo_buffer import PPORolloutBuffer
        metrics = audit_policy(model, PPORolloutBuffer.load(args.replay))
        (args.output / "replay.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics))
        return
    worker = RolloutWorker(cfg)
    saved_hash = (checkpoint.get("reference_corridor") or {}).get("sha256")
    if saved_hash and args.reference_trajectory is None and saved_hash != worker.reference_corridor.metadata["sha256"]:
        raise ValueError("Checkpoint reference file changed; select --reference-trajectory explicitly to switch")
    env_data, player_data = load_colliders(Path(cfg.game_root))
    polygons = extract_polygons(env_data)
    eff_map = ClimbingEfficiencyMap(polygons, grid_resolution=cfg.grid_resolution,
        diffusion_iterations=cfg.diffusion_iterations, diffusion_alpha=cfg.diffusion_alpha,
        cache_path=str(args.output / "terrain.npz"), height_prior_weight=cfg.height_prior_weight)
    worker.set_terrain_info(eff_map.traversable, eff_map._min_gx, eff_map._min_gy)
    worker.set_deploy_obstacle_geometry(polygons, player_data)
    rng = np.random.default_rng(args.seed)
    offsets = rng.uniform(-args.jitter, args.jitter, (args.rounds, args.num_agents, 2)).astype(np.float32)
    offsets[0] = 0  # Always include unperturbed repeated reset.
    manifest = {"version": 1, "seed": args.seed, "config": asdict(cfg),
        "checkpoint": str(args.checkpoint) if args.checkpoint else (
            f"warm start from {args.init_from}" if args.init_from else "random initialization (engineering smoke only)"),
        "init_from": args.init_from,
        "reference": worker.reference_corridor.metadata,
        "criterion": {"credited_progress_m": args.goal_progress, "dwell_steps": args.dwell,
                      "fall_peak_drop_m": 3., "arrival_unit": "environment steps"},
        "offsets": offsets.tolist(), "contact_source": "geometry_proxy",
        "native_diagnostics_requested": args.native_diagnostics,
        "initialization": "warmup then canonical snapshot; teleport jitter also clears rigid-body velocities",
        "geometry_sha256": hashlib.sha256(json.dumps([env_data, player_data], sort_keys=True).encode()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "reference_corridor": worker.reference_corridor.metadata,
                "policy_context": {"exploration_boost": model.exploration_boost}}, args.output / "behavior.pt")
    records, runs, updates = [], [], []
    failure = None
    try:
        if args.no_launch:
            from env import GoiEnv
            worker.env = GoiEnv(port=cfg.port, num_agents=cfg.num_agents)
            worker.env.connect()
            worker.env.reset()
            _warmup_and_snapshot(worker.env, cfg.num_agents, cfg.warmup_steps)
        else:
            worker.launch_game()
        for mode in ("deterministic", "exploration"):
            for trial in range(args.rounds):
                # Paired initial perturbations and explicitly recorded independent policy seeds.
                np.random.seed(args.seed + trial)
                torch.manual_seed(args.seed + trial)
                buffer, trajectories = worker.collect_ppo(model, args.steps, eff_map,
                    deterministic=mode == "deterministic", initial_offsets=offsets[trial],
                    native_diagnostics=args.native_diagnostics)
                audit = audit_policy(model, buffer)
                stem = f"{mode}_{trial:03d}"
                buffer.save(args.output / f"{stem}.npz")
                if hasattr(worker, "last_feature_scales"):
                    (args.output / f"{stem}_features.json").write_text(json.dumps({
                        "scales": worker.last_feature_scales, "segments": worker.last_segment_trace}, indent=2), encoding="utf-8")
                if worker.last_native_diagnostics:
                    (args.output / f"{stem}_native.json").write_text(
                        json.dumps(worker.last_native_diagnostics), encoding="utf-8")
                for agent, trajectory in enumerate(trajectories):
                    metrics, trace = score_corridor_episode(trajectory, worker.reference_corridor,
                                                           args.goal_progress, args.dwell)
                    records.append({"mode": mode, "trial": trial, "agent": agent, **metrics})
                    if worker.last_native_diagnostics:
                        records[-1]["native_comparison"] = compare_native_contacts(
                            trajectory.contact, worker.last_native_diagnostics, agent)
                    (args.output / f"{stem}_agent{agent}.json").write_text(json.dumps({
                        "metrics": records[-1], "states": trajectory.raw_states.tolist(),
                        "contacts": trajectory.contact.tolist(), "trace": trace}), encoding="utf-8")
                runs.append({"mode": mode, "trial": trial, **audit, **worker.last_profile})
                logging.info("%s trial=%d audit=%s profile=%s", mode, trial, audit, worker.last_profile)
        trainer = PPOTrainer(cfg)
        for iteration in range(args.train_smoke):
            buffer, _ = worker.collect_ppo(model, args.steps, eff_map)
            before = torch.cat([p.detach().flatten() for p in model.parameters()]).clone()
            start = time.perf_counter()
            metrics = trainer.update(model, buffer)
            delta = torch.cat([p.detach().flatten() for p in model.parameters()]) - before
            updates.append({"iteration": iteration + 1, **metrics, **worker.last_profile,
                            "parameter_delta_l2": delta.norm().item(), "update_seconds": time.perf_counter()-start})
            logging.info("Smoke update: %s", updates[-1])
        print(json.dumps(summarize(records), indent=2))
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result = {"summary": summarize(records), "episodes": records, "rollouts": runs,
                  "updates": updates, "contact_native_validation": "measured" if any(
                      "native_comparison" in record for record in records) else "pending",
                  "failure": failure}
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        worker.teardown()


if __name__ == "__main__":
    main()

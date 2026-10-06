"""Fixed corridor evaluation, using the same collect_ppo observation pipeline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from training.experiment_options import add_experiment_arguments, apply_experiment_arguments, restore_initialization_shape, ppo_config
from training.model import expand_state_dict_for_dynamics_dim
from training.policy_audit import audit_policy
from training.exploration_windows import add_arguments as add_window_arguments, apply_arguments as apply_window_arguments
from training.ppo_trainer import PPOTrainer
from training.reward import ClimbingEfficiencyMap, is_water
from training.rollout import RolloutWorker, _warmup_and_snapshot
from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons


def projection_from_start(route, tracker, start_segment, start_progress):
    """Current projection along verified segments, measured from the reset location.

    Later continuous segments count after the agent enters them. A gap
    (`waiting`) reports the end of the last confirmed segment.
    """
    if tracker.segment_index < start_segment:
        return 0.0
    first = route.segments[start_segment].length - start_progress
    if tracker.segment_index == start_segment:
        return first if tracker.waiting else max(0.0, tracker.progress - start_progress)
    middle = sum(route.segments[i].length for i in range(start_segment + 1, tracker.segment_index))
    tail = route.segments[tracker.segment_index].length if tracker.waiting else tracker.progress
    return first + middle + tail


def _remaining_reference_m(route, start_segment, start_progress):
    return (route.segments[start_segment].length - start_progress
            + sum(segment.length for segment in route.segments[start_segment + 1:]))


HOLD_STEPS = 120
HOLD_PROGRESS_SLACK_M = 0.3
HOLD_INSIDE_FRACTION = 0.95
STEPS_PER_SECOND = 120.0


def _tolerant_hold(frames, arrival, goal_progress, hold_steps, hold_progress_floor,
                   hold_inside_fraction, config):
    """1 s after first arrival. Gaps are unconfirmed; a short window is unobserved.

    Floor slack and inside fraction are unverified parameters.
    """
    window = [frame for frame in frames if arrival <= frame["step"] < arrival + hold_steps]
    if len(window) < hold_steps:
        return False, True, False
    if any(frame["gap"] for frame in window):
        return False, False, True
    watered = any(is_water(frame["y"], config) for frame in window)
    held = (
        all(frame["projection_m"] >= hold_progress_floor for frame in window)
        and float(np.mean([frame["inside"] for frame in window])) >= hold_inside_fraction
        and window[-1]["projection_m"] >= goal_progress
        and not watered
    )
    return held, False, False


def _max_retreat_m(frames, arrival):
    after = [frame for frame in frames if frame["step"] >= arrival]
    if not after:
        return 0.0
    peak = after[0]["projection_m"]
    retreat = 0.0
    for frame in after:
        peak = max(peak, frame["projection_m"])
        retreat = max(retreat, peak - frame["projection_m"])
    return float(retreat)


def score_corridor_episode(trajectory, route, goal_progress=5.0, dwell=10, fall_drop=3.0,
                           hold_steps=HOLD_STEPS, hold_progress_floor=None,
                           hold_inside_fraction=HOLD_INSIDE_FRACTION):
    """Corridor episode score: first arrival, tolerant 1 s hold, and retention.

    Arrival is the first step whose projection and credited progress are both
    at least `goal_progress`. Subsequent verified segments count. A reference
    gap is `reference_unconfirmed`; a hold cut short by the episode end is
    `hold_incomplete`. Neither is a control failure.

    Hold floor slack and inside fraction are unverified parameters.

    `success` remains the original strict dwell on the start segment.
    """
    if hold_progress_floor is None:
        hold_progress_floor = goal_progress - HOLD_PROGRESS_SLACK_M
    if (goal_progress <= 0 or dwell < 1 or fall_drop <= 0 or hold_steps < 1
            or hold_progress_floor < 0 or not 0 <= hold_inside_fraction <= 1):
        raise ValueError("Positive goal, dwell, hold window, and valid hold fraction required")
    states = trajectory.raw_states
    from training.reference_anchor import scoring_context
    schema, meta = scoring_context(states[0], route.config, None, route)
    tracker = route.tracker_from_state(states[0], schema, meta)
    start_segment, start_progress = tracker.segment_index, tracker.progress
    if _remaining_reference_m(route, start_segment, start_progress) < goal_progress:
        raise ValueError("Local goal exceeds remaining reference length")
    held = 0
    strict_arrival = None
    arrival = None
    peak_y = float(states[0, 1])
    fell = False
    watered = is_water(peak_y, route.config)
    outside = 0
    frames = []
    trace = []
    previous_contacts = None
    previous_action = np.zeros(2)
    for step, (state, action) in enumerate(zip(states[1:], trajectory.actions), 1):
        result = route.step_state(tracker, state, schema, meta)
        from training.reference_corridor import PROGRESS_MODE_V2, progress_mode_of
        distance_now = (tracker.confirmed_d if progress_mode_of(route.config) == PROGRESS_MODE_V2
                        else tracker.total_progress)
        projection = projection_from_start(route, tracker, start_segment, start_progress)
        peak_y = max(peak_y, float(state[1]))
        watered |= is_water(state[1], route.config)
        fell |= watered or peak_y - float(state[1]) >= fall_drop
        outside += int(not result.inside and not result.gap)
        frames.append({
            "step": step, "projection_m": projection, "credited_m": tracker.total_progress,
            "confirmed_progress_m": tracker.confirmed_d, "delta_d": result.delta_d,
            "credit_event": result.credit_event, "reward": result.reward,
            "inside": result.inside, "gap": result.gap, "y": float(state[1]),
            "segment": tracker.segment_index,
        })
        if (arrival is None and projection >= goal_progress and distance_now >= goal_progress):
            arrival = step
        qualifies = (result.inside and not result.gap and tracker.segment_index == start_segment
                     and tracker.progress >= start_progress + goal_progress
                     and distance_now >= goal_progress)
        held = held + 1 if qualifies else 0
        if held >= dwell and strict_arrival is None:
            strict_arrival = step
        contacts = trajectory.contact[step - 1, [0, 2, 3]] if trajectory.contact is not None else None
        enter = np.zeros(3) if previous_contacts is None or contacts is None else (contacts > previous_contacts)
        leave = np.zeros(3) if previous_contacts is None or contacts is None else (contacts < previous_contacts)
        trace.append({"step": step, "progress_m": tracker.progress - start_progress,
                      "projection_m": projection,
                      "credited_progress_m": tracker.total_progress,
                      "confirmed_progress_m": tracker.confirmed_d,
                      "delta_d": result.delta_d,
                      "credit_event": bool(result.credit_event),
                      "reward": result.reward,
                      "inside": result.inside,
                      "gap": result.gap, "contact_before_action": None if contacts is None else contacts.tolist(),
                      "contact_enter": enter.astype(int).tolist(), "contact_exit": leave.astype(int).tolist(),
                      "action": action.tolist(), "action_delta": float(np.linalg.norm(action - previous_action))})
        previous_contacts, previous_action = contacts, action
    hold_success, hold_incomplete, hold_unconfirmed = (
        _tolerant_hold(frames, arrival, goal_progress, hold_steps, hold_progress_floor,
                       hold_inside_fraction, route.config)
        if arrival is not None else (False, False, False)
    )
    terminal_gap = bool(frames and frames[-1]["gap"])
    same_segment = tracker.segment_index == start_segment and not tracker.waiting
    return {
        "success": strict_arrival is not None, "arrival_steps": arrival, "fell": bool(fell),
        "watered": bool(watered),
        "left_corridor": outside > 0, "outside_fraction": outside / max(1, len(trace)),
        "credited_progress_m": tracker.total_progress,
        "cap_progress_m": tracker.confirmed_d,
        "cap_action": tracker.last_credit_action,
        "t_cap": None if tracker.last_credit_action < 0 else tracker.last_credit_action + 1,
        "start_segment": start_segment,
        "start_progress_m": start_progress, "steps": len(trace),
        "hold_success": hold_success, "hold_incomplete": hold_incomplete,
        "reference_unconfirmed": hold_unconfirmed or terminal_gap,
        "terminal_progress_m": frames[-1]["projection_m"] if frames else 0.0,
        "max_retreat_after_arrival_m": (
            _max_retreat_m(frames, arrival) if arrival is not None else None),
        "terminal_same_segment_progress_m": (
            float(tracker.progress - start_progress) if same_segment else None),
        "continued_on_later_segment": any(
            frame["segment"] > start_segment and not frame["gap"] for frame in frames),
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
        arrivals = [r["arrival_steps"] for r in rows if r.get("arrival_steps") is not None]
        arrived = len(arrivals)
        held = sum(bool(r.get("hold_success")) for r in rows)
        credited = [r["credited_progress_m"] for r in rows]
        capped = [r.get("cap_progress_m", r["credited_progress_m"]) for r in rows]
        retreats = [r["max_retreat_after_arrival_m"] for r in rows
                    if r.get("max_retreat_after_arrival_m") is not None]
        result[mode] = {"episodes": n, "success_rate": p, "success_wilson95": [center-half, center+half],
                        "interval_note": "Descriptive only: agents share trial seeds; episodes are not independent.",
                        "fall_rate": float(np.mean([r["fell"] for r in rows])),
                        "water_count": sum(bool(r.get("watered")) for r in rows),
                        "water_rate": float(np.mean([bool(r.get("watered")) for r in rows])),
                        "leave_corridor_rate": float(np.mean([r["left_corridor"] for r in rows])),
                        "mean_credited_progress_m": float(np.mean(credited)),
                        "credited_progress_median_m": float(np.median(credited)),
                        "credited_progress_q25_m": float(np.quantile(credited, 0.25)),
                        "mean_cap_progress_m": float(np.mean(capped)),
                        "cap_progress_median_m": float(np.median(capped)),
                        "cap_progress_q25_m": float(np.quantile(capped, 0.25)),
                        "cap_progress_max_m": float(np.max(capped)),
                        "mean_arrival_steps": float(np.mean(arrivals)) if arrivals else None,
                        "mean_arrival_game_seconds": (
                            float(np.mean(arrivals)) / STEPS_PER_SECOND if arrivals else None),
                        "arrival_count": arrived, "arrival_rate": arrived / n,
                        "hold_success_count": held,
                        "hold_rate_after_arrival": held / arrived if arrived else 0.0,
                        "hold_incomplete_count": sum(bool(r.get("hold_incomplete")) for r in rows),
                        "terminal_progress_median_m": float(np.median(
                            [r.get("terminal_progress_m", 0.0) for r in rows])),
                        "max_retreat_after_arrival_median_m": (
                            float(np.median(retreats)) if retreats else None),
                        "reference_unconfirmed_count": sum(
                            bool(r.get("reference_unconfirmed")) for r in rows)}
    return result


def format_eval_line(values):
    """Current-stage line: credited median/Q25 first; arrival and hold separate."""
    n = values["episodes"]
    if "credited_progress_median_m" not in values:
        return (f"严格 1 秒 {values['success_hold_120_count']}/{n}；"
                f"10 步成功 {values.get('successes', 0)}/{n}")
    arrived = int(values.get("arrival_count") or 0)
    time_s = values.get("mean_arrival_game_seconds")
    time_bit = f"，用时 {time_s:.2f}s" if time_s is not None else ""
    retreat = values.get("max_retreat_after_arrival_median_m")
    retreat_bit = f"；到达后最大回退中位 {retreat:.3f}m" if retreat is not None else ""
    if "cap_progress_median_m" in values:
        head = (f"D 中位 {values['cap_progress_median_m']:.3f}m"
                f"（Q25 {values['cap_progress_q25_m']:.3f}m）；"
                f"legacy credited 中位 {values['credited_progress_median_m']:.3f}m；")
    else:
        head = (f"credited 中位 {values['credited_progress_median_m']:.3f}m"
                f"（Q25 {values['credited_progress_q25_m']:.3f}m）；")
    return (
        f"{head}"
        f"到达 {arrived}/{n}{time_bit}；"
        f"到达后保持 {int(values.get('hold_success_count') or 0)}/{arrived}；"
        f"未观察完 {int(values.get('hold_incomplete_count') or 0)}；"
        f"参考无法确认 {int(values.get('reference_unconfirmed_count') or 0)}；"
        f"期末投影中位 {values.get('terminal_progress_median_m', 0.0):.3f}m"
        f"{retreat_bit}；"
        f"落水 {int(values.get('water_count') or 0)}/{n}"
    )


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
    add_window_arguments(parser)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference-trajectory", type=Path, default=None,
                        help="Override the saved route; defaults to checkpoint route or the project's video reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=2)
    parser.add_argument("--game-root", type=Path, help="Game installation or isolated test copy")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--jitter", type=float, default=0., help="Spawn position jitter in meters; default exact spawn")
    parser.add_argument("--goal-progress", type=float, default=5.)
    parser.add_argument("--dwell", type=int, default=10)
    parser.add_argument("--noise-phi", type=float, default=None,
                        help="Override exploration OU coefficient; default inherits checkpoint (0 for iid)")
    parser.add_argument("--inference-batch-size", type=int, default=0)
    parser.add_argument("--geometry-profile", choices=("native-pose-v1", "legacy"), default=None)
    parser.add_argument("--native-diagnostics", action="store_true")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--current-desktop", action="store_true",
                        help="Show the policy-controlled game on the current desktop")
    parser.add_argument("--collider-visual", action="store_true",
                        help="Display training collision outlines instead of scene textures")
    parser.add_argument("--train-smoke", type=int, default=0, help="Fresh rollouts and updates after frozen evaluation")
    parser.add_argument("--replay", type=Path, help="Offline audit of archived NPZ; requires behavior checkpoint")
    parser.add_argument("--modes", default="deterministic,exploration",
                        help="Comma-separated rollout modes: deterministic and/or exploration")
    parser.add_argument("--prefix-archive", type=Path, help="Full elite archive for frozen demonstration-prefix handoff")
    parser.add_argument("--prefix-source-iteration", type=int, help="Unique source iteration in the selected elite archive")
    parser.add_argument("--prefix-steps", help="Comma-separated prefix lengths, one per agent; action k is first policy action")
    parser.add_argument("--noise-plan", type=Path,
                        help="Fixed per-agent noise windows JSON; frozen exact-spawn exploration only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    prefix_args = (args.prefix_archive is not None, args.prefix_source_iteration is not None, args.prefix_steps is not None)
    if args.noise_plan and (any(prefix_args) or args.train_smoke or args.replay or args.jitter
                           or not args.checkpoint or args.modes != ["exploration"]):
        raise ValueError("Noise-plan evaluation requires a checkpoint, exploration-only, exact spawn and no prefix/training/replay")
    if any(prefix_args) and (not all(prefix_args) or args.train_smoke or args.replay
                            or args.modes != ["deterministic"] or args.jitter or not args.checkpoint):
        raise ValueError("Prefix handoff requires all three prefix options, checkpoint, deterministic-only, exact spawn and no training/replay")
    if (args.steps < 1 or args.rounds < 1 or args.num_agents < 1 or args.jitter < 0
            or args.train_smoke < 0 or args.goal_progress <= 0 or args.dwell < 1
            or args.inference_batch_size < 0 or (args.noise_phi is not None and not 0 <= args.noise_phi < 1)
            or not args.modes or any(mode not in ("deterministic", "exploration") for mode in args.modes)):
        raise ValueError("Invalid evaluation lengths, jitter, goal, batch size, noise_phi or modes")
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else {}
    cfg = ppo_config(checkpoint)
    for name in ("contact_heads_enabled", "phase_weighting_enabled", "sil_advantage_weighting_enabled"):
        if name not in checkpoint.get("config", {}):
            setattr(cfg, name, False)
    if args.init_from and args.checkpoint:
        raise ValueError("--init-from and --checkpoint are mutually exclusive")
    init_state = torch.load(args.init_from, map_location="cpu", weights_only=False) if args.init_from else {}
    restore_initialization_shape(cfg, init_state)
    if checkpoint:
        cfg.contact_heads_enabled = "actor_mean_contact.weight" in checkpoint["model"]
    configure_geometry(cfg, checkpoint or init_state, args.geometry_profile)
    apply_experiment_arguments(cfg, args, checkpoint)
    if not args.replay:
        cfg.num_agents = args.num_agents
        cfg.steps_per_rollout = args.steps
        cfg.training_seed = args.seed
    apply_window_arguments(cfg, args)
    if args.game_root is not None:
        cfg.game_root = str(args.game_root.resolve())
    cfg.random_deploy = cfg.sil_enabled = cfg.gru_sil_enabled = cfg.explore_boost_enabled = False
    noise_plan = None
    if args.noise_plan:
        from training.evaluation_noise_plan import fixed_plan
        noise_plan = json.loads(args.noise_plan.read_text(encoding="utf-8"))
        fixed_plan(noise_plan, cfg, args.steps)
    reference_path = args.reference_trajectory or cfg.reference_trajectory or (
        ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory_game_world.json")
    cfg.reference_trajectory = str(Path(reference_path).resolve())
    if args.noise_phi is None:
        args.noise_phi = cfg.ou_phi if cfg.ou_enabled else 0.0
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
    prefix = None
    if args.prefix_archive is not None:
        from training.elite_archive import EliteTrajectoryArchive
        from training.prefix_handoff import PrefixHandoff
        from training.reference_fingerprint import reference_fingerprint
        fingerprint = reference_fingerprint(cfg, route_meta=worker.reference_corridor.metadata,
            schema_id="anchor35" if cfg.tcp_protocol == "anchor35" else "legacy33",
            raw_dim=35 if cfg.tcp_protocol == "anchor35" else 33)
        archive = EliteTrajectoryArchive(cfg, args.output / "unused", fingerprint)
        archive.restore(args.prefix_archive)
        matches = [t for t in archive.trajectories if t.iteration == args.prefix_source_iteration]
        if len(matches) != 1:
            raise ValueError(f"Expected unique source iteration; found {len(matches)}")
        prefix = PrefixHandoff(matches[0], [int(s) for s in args.prefix_steps.split(",")], {
            "archive": str(args.prefix_archive.resolve()), "iteration": args.prefix_source_iteration,
            "archive_sha256": hashlib.sha256(args.prefix_archive.read_bytes()).hexdigest()})
        prefix.validate(cfg, args.steps, True)
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
        "presentation": {"current_desktop": args.current_desktop, "collider_visual": args.collider_visual},
        "torch_precision": {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
        "checkpoint": str(args.checkpoint) if args.checkpoint else (
            f"warm start from {args.init_from}" if args.init_from else "random initialization (engineering smoke only)"),
        "init_from": args.init_from,
        "reference": worker.reference_corridor.metadata,
        "criterion": {"credited_progress_m": args.goal_progress, "dwell_steps": args.dwell,
                      "fall_peak_drop_m": 3., "arrival_unit": "environment steps",
                      "hold_steps": HOLD_STEPS,
                      "hold_progress_floor_m": args.goal_progress - HOLD_PROGRESS_SLACK_M,
                      "hold_inside_fraction": HOLD_INSIDE_FRACTION,
                      "hold_parameters_unverified": True},
        "offsets": offsets.tolist(), "contact_source": "geometry_proxy",
        "native_diagnostics_requested": args.native_diagnostics,
        "initialization": "warmup then canonical snapshot; teleport jitter also clears rigid-body velocities",
        "geometry_sha256": hashlib.sha256(json.dumps([env_data, player_data], sort_keys=True).encode()).hexdigest()}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if prefix is not None:
        manifest["prefix_handoff"] = prefix.metadata()
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if noise_plan is not None:
        manifest["evaluation_noise_plan"] = noise_plan
        manifest["noise_plan_sha256"] = hashlib.sha256(args.noise_plan.read_bytes()).hexdigest()
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "reference_corridor": worker.reference_corridor.metadata,
                "policy_context": {"exploration_boost": model.exploration_boost}}, args.output / "behavior.pt")
    records, runs, updates = [], [], []
    frozen_weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    failure = None
    try:
        if args.no_launch:
            from env import make_goi_env
            worker.env = make_goi_env(cfg)
            worker.env.connect()
            worker.env.reset()
            _warmup_and_snapshot(worker.env, cfg.num_agents, cfg.warmup_steps)
            if args.collider_visual:
                worker.env.set_camera_free(True)
                worker.env.toggle_collider_visual(True)
        else:
            worker.launch_game(current_desktop=args.current_desktop, collider_visual=args.collider_visual)
        for mode in args.modes:
            for trial in range(args.rounds):
                # Paired initial perturbations and explicitly recorded independent policy seeds.
                np.random.seed(args.seed + trial)
                torch.manual_seed(args.seed + trial)
                buffer, trajectories = worker.collect_ppo(model, args.steps, eff_map,
                    deterministic=mode == "deterministic", initial_offsets=offsets[trial],
                    native_diagnostics=args.native_diagnostics, evaluation_prefix=prefix,
                    evaluation_noise_plan=noise_plan)
                stem = f"{mode}_{trial:03d}"
                # Keep the evidence even when the frozen-probability audit rejects it.
                buffer.save(args.output / (f"{stem}_policy_probes.npz" if prefix else f"{stem}.npz"))
                if hasattr(worker, "last_feature_scales"):
                    (args.output / f"{stem}_features.json").write_text(json.dumps({
                        "scales": worker.last_feature_scales, "segments": worker.last_segment_trace,
                        "collection": buffer.collection_metadata}, indent=2), encoding="utf-8")
                if worker.last_native_diagnostics:
                    (args.output / f"{stem}_native.json").write_text(
                        json.dumps(worker.last_native_diagnostics), encoding="utf-8")
                for episode_index, trajectory in enumerate(trajectories):
                    agent = getattr(buffer, "collection_metadata", {}).get(
                        "episode_agent_ids", list(range(len(trajectories))))[episode_index]
                    metrics, trace = score_corridor_episode(trajectory, worker.reference_corridor,
                                                           args.goal_progress, args.dwell)
                    records.append({"mode": mode, "trial": trial, "agent": agent, **metrics})
                    if noise_plan is not None:
                        records[-1]["noise_intervention"] = noise_plan["agents"][agent]
                    if prefix is not None:
                        records[-1]["prefix_handoff"] = prefix.compare(
                            trajectory, agent, buffer.episodes[episode_index]["actions"])
                    if worker.last_native_diagnostics:
                        records[-1]["native_comparison"] = compare_native_contacts(
                            trajectory.contact, worker.last_native_diagnostics, agent)
                    (args.output / f"{stem}_agent{agent}.json").write_text(json.dumps({
                        "metrics": records[-1], "states": trajectory.raw_states.tolist(),
                        "contacts": trajectory.contact.tolist(), "trace": trace}), encoding="utf-8")
                audit = audit_policy(model, buffer)
                if prefix is not None:
                    audit = {"policy_probe_replay_audit": audit,
                             "audit_scope": "Frozen policy predictions on actual histories; does not attribute demonstration actions to policy"}
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
        if not args.train_smoke and not all(torch.equal(frozen_weights[k], v.detach().cpu())
                                            for k, v in model.state_dict().items()):
            raise RuntimeError("Frozen evaluation changed model weights")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result = {"summary": summarize(records), "episodes": records, "rollouts": runs,
                  "updates": updates, "contact_native_validation": "measured" if any(
                      "native_comparison" in record for record in records) else "pending",
                  "failure": failure}
        result["model_weights_unchanged"] = all(torch.equal(frozen_weights[k], v.detach().cpu())
                                               for k, v in model.state_dict().items())
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        worker.teardown()


if __name__ == "__main__":
    main()

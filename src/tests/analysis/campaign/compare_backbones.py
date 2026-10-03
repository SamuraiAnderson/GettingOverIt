"""Paired fresh Transformer/GRU campaign on the installed isolated-world runtime.

Run with the getting-over-it-analysis interpreter. Stages are restartable;
training/evaluation use the production entry points, never a second PPO loop.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from training.config import TrainConfig
from training.dataset import compute_dynamics_stats
from training.experimental_policy import create_actor_critic
from training.geometry_profile import configure_geometry
from training.rollout import RolloutWorker
from training.local_evaluation import summarize
from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons

SEEDS = (7, 17, 47)
BACKBONES = ("transformer", "gru")
ROUTE = ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory_game_world.json"
EXPECTED_DLL = "632067533df42c1406d4ebfc1bc5ce4207f944d99d187d67261d59c181080bfe"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def config_for(backbone):
    config = TrainConfig(
        num_agents=8, steps_per_rollout=1200, random_deploy=False,
        policy_backbone=backbone, normalize_action_history=True, recurrent_seq_len=32,
        contact_heads_enabled=False, sil_enabled=False, explore_boost_enabled=False,
        phase_weighting_enabled=False, sil_advantage_weighting_enabled=False,
        reference_trajectory=str(ROUTE), value_warmup_iters=2,
        ppo_batch_size=32, lr=1e-4, gamma=.99, gae_lambda=.95,
    )
    configure_geometry(config, profile="native-pose-v1")
    return config


def paired_models(seed, mean, std):
    """Identical common tensors, independently initialized temporal operators."""
    models = {}
    for backbone in BACKBONES:
        torch.manual_seed(seed)
        model = create_actor_critic(config_for(backbone))
        model.set_dynamics_stats(mean, std)
        model.reset_log_std(model.config.ppo_init_log_std)
        models[backbone] = model
    source, target = (models[name].state_dict() for name in BACKBONES)
    common = sorted(source.keys() & target.keys())
    for key in common:
        if source[key].shape != target[key].shape:
            raise ValueError(f"Shared tensor shape mismatch: {key}")
        target[key].copy_(source[key])
    models["gru"].load_state_dict(target)
    assert all(torch.equal(source[key], target[key]) for key in common)
    return models, common


def runtime_identity():
    dll = Path(config_for("transformer").game_root) / "BepInEx/plugins/GameRuntime_v2.dll"
    digest = hashlib.sha256(dll.read_bytes()).hexdigest()
    if digest != EXPECTED_DLL:
        raise RuntimeError(f"Runtime differs from accepted binary: {digest}")
    return {"dll": str(dll), "sha256": digest}


def prepare(output):
    runtime = runtime_identity()
    stats_path = output / "calibration/stats.npz"
    cfg = config_for("transformer")
    worker = RolloutWorker(cfg)
    if not stats_path.exists():
        random.seed(20261002)
        np.random.seed(20261002)
        env_data, player_data = load_colliders(Path(cfg.game_root))
        worker.set_deploy_obstacle_geometry(extract_polygons(env_data), player_data)
        try:
            worker.launch_game()
            hello = worker.env.hello
            if hello.get("physics_backend") != "isolated_worlds":
                raise RuntimeError(f"Unexpected physics backend: {hello}")
            if hello.get("step_frames") != 1 or not np.isclose(hello.get("fixed_delta_time", 0), 1/120):
                raise RuntimeError(f"Unexpected step duration: {hello}")
            trajectories = worker.collect_random(cfg.steps_per_rollout)
            if not trajectories:
                raise RuntimeError("Calibration discarded every trajectory")
            if any(t.raw_states.shape[1] != 35 or not np.isfinite(t.raw_states).all() for t in trajectories):
                raise RuntimeError("Invalid calibration states")
            mean, std = compute_dynamics_stats(trajectories)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            for i, trajectory in enumerate(trajectories):
                np.savez_compressed(stats_path.parent / f"agent{i}.npz", states=trajectory.raw_states,
                                    actions=trajectory.actions, contact=trajectory.contact)
            write_json(stats_path.parent / "manifest.json", {
                "seed": 20261002, "physical_steps": 9600,
                "retained_trajectories": len(trajectories), "hello": hello, "runtime": runtime,
                "excluded_from_training_budget": True,
                "first_reference_segment_length_m": worker.reference_corridor.segments[0].length,
                "reference": worker.reference_corridor.metadata,
            })
            np.savez(stats_path, mean=mean, std=std)
        finally:
            worker.close_game()
    with np.load(stats_path) as stats:
        mean, std = stats["mean"], stats["std"]
    pairs = []
    for seed in SEEDS:
        models, common = paired_models(seed, mean, std)
        for name, model in models.items():
            path = output / f"initial/{name}_s{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            state = {"model": model.state_dict(), "config": asdict(model.config), "iteration": 0,
                     "value_warmup_end": 2, "reference_corridor": worker.reference_corridor.metadata}
            if path.exists():
                old = torch.load(path, weights_only=False, map_location="cpu")
                if old["config"] != state["config"] or any(
                        not torch.equal(old["model"][key], value) for key, value in state["model"].items()):
                    raise RuntimeError(f"Existing initialization differs: {path}")
            else:
                torch.save(state, path)
            pairs.append({"seed": seed, "backbone": name,
                          "parameters": sum(p.numel() for p in model.parameters()),
                          "common_tensor_count": len(common), "checkpoint": str(path)})
    write_json(output / "protocol.json", {
        "version": 2, "runtime": runtime, "seeds": SEEDS, "pairs": pairs,
        "source_sha256": {name: hashlib.sha256((ROOT / "src/training" / name).read_bytes()).hexdigest()
                          for name in ("config.py", "experimental_policy.py", "experimental_rollout.py",
                                       "decision_buffer.py", "ppo_trainer.py", "main_ppo.py", "local_evaluation.py")},
        "precheck_iterations": 3, "screen_executed_world_steps_per_model_seed": 96000,
        "budget_definition": "8 worlds * 1200 vector steps * 10 rollouts. Terminal agents receive zero actions; active training transitions are reported separately.",
        "calibration_sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest(),
        "evaluation": {"seed": 20261002, "rounds": 2, "agents": 2, "steps": 1200,
                       "goal_m": 5, "hold_steps": 120, "jitter_m": 0,
                       "scope": "Exact canonical spawn only, as requested; four repeats check determinism, one outcome per policy.",
                       "hold_parameters_unverified": True},
        "scope": "Existing architectures under equal physical sample budget; parameter counts differ.",
        "resumption": "Production resume restores optimizer; each invocation reseeds RNG with the paired seed.",
    })


def training_rows(directory):
    path = directory / "metrics.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def physical_samples(rows):
    # Production's historical field counts active transitions, not inactive worlds.
    return sum(int(float(row["rollout_environment_steps"])) for row in rows)


def executed_world_steps(directory, rows):
    total = 0
    for row in rows:
        iteration = int(float(row["iteration"]))
        path = directory / f"checkpoints/feature_diagnostics/iter_{iteration:04d}.json"
        segments = json.loads(path.read_text(encoding="utf-8"))["segments"]
        # An action segment ends on the last STEP consumed by an active agent.
        # All worlds still advance until the last active agent terminates.
        vector_steps = max(segment["start"] + segment["duration"] for segment in segments)
        total += 8 * vector_steps
    return total


def run_command(output, name, arguments):
    directory = output / "commands"
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, *map(str, arguments)]
    record_path = directory / f"{name}.json"
    record = {"command": command, "started": time.time(), "status": "running"}
    write_json(record_path, record)
    logging.info("START %s", name)
    with (directory / f"{name}.log").open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    record.update(status="passed" if result.returncode == 0 else "failed", returncode=result.returncode,
                  seconds=time.time()-record["started"])
    write_json(record_path, record)
    logging.info("END %s: %s in %.1fs", name, record["status"], record["seconds"])
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {directory / (name + '.log')}")


def train(output, backbone, seed, iterations):
    directory = output / f"{backbone}_s{seed}"
    rows = training_rows(directory)
    completed = max((int(float(row["iteration"])) for row in rows), default=0)
    if completed >= iterations:
        if executed_world_steps(directory, rows) != len(rows) * 9600:
            raise RuntimeError(f"Existing run has incomplete world-step budget: {directory}")
        return
    checkpoint = (directory / f"checkpoints/ppo_iter_{completed:04d}.pt" if completed
                  else output / f"initial/{backbone}_s{seed}.pt")
    if not checkpoint.exists():
        raise RuntimeError(f"Missing checkpoint for last recorded iteration: {checkpoint}")
    run_command(output, f"{backbone}_s{seed}_to{iterations}", [
        "-m", "src.training.main_ppo", "--resume", checkpoint,
        "--run-dir", directory, "--seed", seed, "--recipe", "ppo-ou",
        "--num-agents", 8, "--steps-per-rollout", 1200, "--max-iterations", iterations,
        "--persist-game", "--save-interval", 1, "--torch-threads", 4,
        "--no-random-deploy", "--save-rollout-inputs",
    ])
    rows = training_rows(directory)
    if len(rows) != iterations or executed_world_steps(directory, rows) != iterations * 9600:
        raise RuntimeError(f"Unequal/incomplete physical budget for {directory}")


def evaluate(output, backbone, seed, iteration):
    directory = output / f"eval_exact/{backbone}_s{seed}_i{iteration}"
    result = directory / "results.json"
    if result.exists():
        previous = json.loads(result.read_text(encoding="utf-8"))
        if previous["failure"] is None and len(previous["episodes"]) == 4:
            validate_exact_start(directory)
            return
        raise RuntimeError(f"Incomplete evaluation needs inspection: {directory}")
    checkpoint = (output / f"{backbone}_s{seed}/checkpoints/ppo_iter_{iteration:04d}.pt" if iteration
                  else output / f"initial/{backbone}_s{seed}.pt")
    run_command(output, f"eval_exact_{backbone}_s{seed}_i{iteration}", [
        "-m", "src.training.local_evaluation", "--checkpoint", checkpoint, "--output", directory,
        "--steps", 1200, "--rounds", 2, "--num-agents", 2, "--seed", 20261002,
        "--jitter", 0, "--goal-progress", 5, "--modes", "deterministic",
    ])
    data = json.loads(result.read_text(encoding="utf-8"))
    if data["failure"] is not None or len(data["episodes"]) != 4:
        raise RuntimeError(f"Evaluation did not cover all paired cases: {directory}")
    validate_exact_start(directory)


def validate_exact_start(directory):
    paths = [directory / f"deterministic_{trial:03d}_agent{agent}.json"
             for trial in (0, 1) for agent in (0, 1)]
    states = [np.asarray(json.loads(path.read_text(encoding="utf-8"))["states"]) for path in paths]
    shapes_equal = all(state.shape == states[0].shape for state in states)
    error = max(float(np.max(np.abs(state - states[0]))) for state in states) if shapes_equal else None
    check = {"repeats": 4, "same_shape": shapes_equal, "state_max_abs_error": error,
             "bitwise_equal": shapes_equal and all(np.array_equal(states[0], state) for state in states)}
    write_json(directory / "baseline_consistency.json", check)
    if not shapes_equal or not np.isfinite(error) or error > 1e-6:
        raise RuntimeError(f"Exact-spawn repeats diverged: {directory}: {check}")


def validate_baseline(directory):
    paths = [directory / f"deterministic_000_agent{i}.json" for i in (0, 1)]
    states = [np.asarray(json.loads(path.read_text(encoding="utf-8"))["states"]) for path in paths]
    shapes_equal = states[0].shape == states[1].shape
    error = float(np.max(np.abs(states[0] - states[1]))) if shapes_equal else None
    check = {"same_shape": shapes_equal, "state_max_abs_error": error,
             "bitwise_equal": shapes_equal and bool(np.array_equal(*states))}
    write_json(directory / "baseline_consistency.json", check)
    if not shapes_equal or error > 1e-6:
        raise RuntimeError(f"Duplicate baseline diverged: {directory}: {check}")


def audit_precheck(output):
    checks = []
    for backbone in BACKBONES:
        archive = output / f"{backbone}_s7/checkpoints/rollouts"
        destination = output / f"audit/{backbone}_s7_i3"
        initial = torch.load(output / f"initial/{backbone}_s7.pt", weights_only=False, map_location="cpu")["model"]
        behavior = torch.load(archive / "behavior_0003.pt", weights_only=False, map_location="cpu")["model"]
        trained = torch.load(archive.parent / "ppo_iter_0003.pt", weights_only=False, map_location="cpu")["model"]
        policy_keys = [key for key in initial if not key.startswith("critic_head.")]
        if any(not torch.equal(initial[key], behavior[key]) for key in policy_keys):
            raise RuntimeError(f"Critic warmup changed frozen policy: {backbone}")
        if not any(not torch.equal(behavior[key], trained[key]) for key in policy_keys):
            raise RuntimeError(f"PPO did not change policy weights: {backbone}")
        rows = training_rows(output / f"{backbone}_s7")[:3]
        if len(rows) != 3 or any(float(row["n_updates"]) <= 0 for row in rows):
            raise RuntimeError(f"Incomplete critic/PPO optimizer updates: {backbone}")
        for row in rows:
            for key, value in row.items():
                if value and not np.isfinite(float(value)):
                    raise RuntimeError(f"Nonfinite metric {backbone} {key}: {value}")
        with np.load(archive / "iter_0003.npz", allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            for episode in range(metadata["episodes"]):
                history = data[f"episode_{episode}_act_history"]
                actions = data[f"episode_{episode}_actions"]
                if not np.array_equal(history[0], np.zeros(2)) or not np.array_equal(history[1:], actions):
                    raise RuntimeError(f"Previous-action alignment failed: {backbone}, episode {episode}")
        if not (destination / "replay.json").exists():
            run_command(output, f"audit_{backbone}_s7_i3", [
                "-m", "src.training.local_evaluation", "--checkpoint", archive / "behavior_0003.pt",
                "--replay", archive / "iter_0003.npz", "--output", destination,
            ])
        metrics = json.loads((destination / "replay.json").read_text(encoding="utf-8"))
        expected = next(int(float(row["buffer_size"])) for row in rows if int(float(row["iteration"])) == 3)
        if (metrics["audit_samples"] != expected or not np.isfinite(list(metrics.values())).all()
                or metrics["preupdate_ratio_max_error"] > 1e-3):
            raise RuntimeError(f"Incomplete frozen-policy audit: {metrics}")
        checks.append({"backbone": backbone, "critic_warmup_preserved_policy": True,
                       "ppo_changed_policy": True, "finite_metrics": True,
                       "previous_action_alignment": True, **metrics})
    write_json(output / "precheck.json", {"status": "passed", "checks": checks})


def split_evaluation(episodes):
    return {"baseline_duplicates": [r for r in episodes if r["trial"] == 0],
            "robustness": summarize([r for r in episodes if r["trial"] != 0])}


def report(output):
    result = {"training": {}, "evaluations": {}, "inference":
              "Exact spawn only. Four deterministic repeats validate reset consistency; one outcome per policy. Compare three paired training seeds."}
    if (output / "precheck.json").exists():
        result["engineering"] = json.loads((output / "precheck.json").read_text(encoding="utf-8"))
    for seed in SEEDS:
        for backbone in BACKBONES:
            key = f"{backbone}_s{seed}"
            rows = training_rows(output / key)
            result["training"][key] = {"iterations": len(rows), "active_transitions": physical_samples(rows),
                                       "executed_world_steps": executed_world_steps(output / key, rows),
                                       "collection_seconds": sum(float(r["rollout_total_seconds"]) for r in rows),
                                       "inference_seconds": sum(float(r["rollout_inference_seconds"]) for r in rows),
                                       "geometry_patch_seconds": sum(float(r["rollout_patch_seconds"]) for r in rows),
                                       "update_seconds": sum(float(r["update_and_sil_seconds"]) for r in rows),
                                       "ppo_updates": sum(float(r["n_updates"]) for r in rows if float(r.get("value_only") or 0) == 0),
                                       "last_metrics": rows[-1] if rows else None}
    for path in sorted((output / "eval_exact").glob("*/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result["evaluations"][path.parent.name] = {"failure": data["failure"],
                                                   "repeat_count": len(data["episodes"]),
                                                   "primary_episode": data["episodes"][0] if data["episodes"] else None,
                                                   "exact_start": summarize(data["episodes"][:1])}
        state_path = path.parent / "deterministic_000_agent0.json"
        if state_path.exists():
            states = np.asarray(json.loads(state_path.read_text(encoding="utf-8"))["states"])
            result["evaluations"][path.parent.name]["player_origin_height"] = {
                "max_gain_m": float(np.max(states[:, 1]) - states[0, 1]),
                "final_gain_m": float(states[-1, 1] - states[0, 1]),
                "note": "Raw Player origin y; distinct from confirmed route distance D.",
            }
        consistency = path.parent / "baseline_consistency.json"
        if consistency.exists():
            result["evaluations"][path.parent.name]["baseline_consistency"] = json.loads(
                consistency.read_text(encoding="utf-8"))
    paired = []
    for seed in SEEDS:
        names = [f"{backbone}_s{seed}_i{iteration}" for backbone in BACKBONES for iteration in (0, 10)]
        if not all(name in result["evaluations"] for name in names):
            continue
        if any(result["evaluations"][name]["failure"] is not None or
               result["evaluations"][name]["repeat_count"] != 4
               for name in names):
            continue
        row = {"seed": seed}
        for backbone in BACKBONES:
            initial, final = (result["evaluations"][f"{backbone}_s{seed}_i{i}"]["exact_start"]["deterministic"]
                              for i in (0, 10))
            row[backbone] = {
                "initial_cap_median_m": initial["cap_progress_median_m"],
                "final_cap_median_m": final["cap_progress_median_m"],
                "cap_median_change_m": final["cap_progress_median_m"] - initial["cap_progress_median_m"],
                "final_arrivals": final["arrival_count"], "final_holds": final["hold_success_count"],
                "final_water_count": final["water_count"], "cases": final["episodes"],
            }
        row["gru_minus_transformer_final_cap_m"] = row["gru"]["final_cap_median_m"] - row["transformer"]["final_cap_median_m"]
        row["gru_minus_transformer_learning_change_m"] = row["gru"]["cap_median_change_m"] - row["transformer"]["cap_median_change_m"]
        paired.append(row)
    result["paired_seeds"] = paired
    write_json(output / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/backbone_compare_20261002")
    parser.add_argument("--stage", choices=("prepare", "precheck", "audit", "screen", "evaluate", "report"), required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(4)
    if args.stage == "report":
        report(args.output)
        return
    prepare(args.output)
    if args.stage == "precheck":
        for backbone in BACKBONES:
            train(args.output, backbone, SEEDS[0], 3)
        audit_precheck(args.output)
    elif args.stage == "audit":
        audit_precheck(args.output)
    elif args.stage == "screen":
        audit_precheck(args.output)
        # Keep identical process boundaries (3 + 7 iterations) for every paired seed.
        for seed in SEEDS:
            for backbone in BACKBONES:
                train(args.output, backbone, seed, 3)
        for seed in SEEDS:
            for backbone in BACKBONES:
                train(args.output, backbone, seed, 10)
    elif args.stage == "evaluate":
        for seed in SEEDS:
            for backbone in BACKBONES:
                for iteration in (0, 10):
                    evaluate(args.output, backbone, seed, iteration)
    report(args.output)


if __name__ == "__main__":
    main()

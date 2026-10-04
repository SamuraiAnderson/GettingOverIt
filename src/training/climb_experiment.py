"""Strict, hash-locked warm-start recipes; loading never launches the game."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from .config import TrainConfig
from .climb_diagnostics import PhaseRules

ARMS = {
    "control": {},
    "guidance_wide": {"reference_guidance_sigma": 3.0},
    "sil_mixed": {"gru_sil_sampling_mode": "quality_dynamics_v3"},
    "long_discount": {"gamma": .9975},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}


def resolve_config(saved, arm, seed, iterations):
    if arm not in ARMS or type(seed) is not int or not 0 <= seed < 2**32 or type(iterations) is not int or iterations < 3:
        raise ValueError("Invalid arm, seed or budget (minimum three rounds includes critic warmup)")
    cfg = TrainConfig(**saved)
    if (cfg.policy_backbone != "gru" or not cfg.gru_sil_enabled or not cfg.elite_archive_enabled or cfg.sil_enabled
            or cfg.random_deploy or cfg.action_segment_steps != 1 or cfg.steps_per_rollout != 1200
            or cfg.num_agents != 8 or cfg.reference_progress_mode != "confirmed_endpoint_v2"):
        raise ValueError("Baseline must be exact-spawn, 8x1200, GRU SIL with v2 rewards")
    # The control remains the archived two-level sampler even when fresh PPO defaults change.
    cfg.gru_sil_sampling_mode = "trajectory_uniform"
    cfg = replace(cfg, training_seed=seed, gru_sil_seed=seed, max_iterations=iterations,
                  value_warmup_iters=2, save_interval=1, ppo_reset_log_std_on_init=False,
                  sil_start_iter=2,
                  gru_sil_phase_rules=asdict(PhaseRules()), **ARMS[arm])
    cfg.reference_trajectory = str(Path(cfg.reference_trajectory).resolve())
    return cfg


def load_recipe(path):
    recipe = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema", "arm", "seed", "iterations", "baseline_config", "resolved_config", "inputs", "run_dir",
                "initialization", "budget", "phase_rules", "source_files"}
    if set(recipe) != required or recipe["schema"] != "tree_climb_experiment_v1":
        raise ValueError("Invalid experiment recipe schema")
    if set(recipe["inputs"]) != {"checkpoint", "elite_pool", "pool_fingerprint", "route", "runtime_dll"}:
        raise ValueError("Incomplete frozen inputs")
    for item in [*recipe["inputs"].values(), *recipe["source_files"]]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input/source changed: {item['path']}")
    cfg = resolve_config(recipe["baseline_config"], recipe["arm"], recipe["seed"], recipe["iterations"])
    if asdict(cfg) != recipe["resolved_config"] or recipe["phase_rules"] != asdict(PhaseRules()):
        raise ValueError("Recipe contains unregistered config changes")
    if recipe["initialization"] != "same_actor_and_std_fresh_optimizer_reset_critic_two_warmup_rounds":
        raise ValueError("Unpaired initialization")
    expected_budget = {"iterations": cfg.max_iterations, "maximum_executed_world_steps": cfg.max_iterations * 8 * 1200,
                       "warmup_included": True, "unit": "executed_world_steps_not_active_transitions"}
    if recipe["budget"] != expected_budget:
        raise ValueError("Budget does not match resolved configuration")
    source_run = Path(recipe["inputs"]["checkpoint"]["path"]).resolve().parent.parent
    run = Path(recipe["run_dir"]).resolve()
    if run == source_run or source_run in run.parents or run in source_run.parents:
        raise ValueError("Experiment output must be separate from the source run")
    return recipe, cfg


def configure_arguments(args, defaults, recipe):
    # One source of configuration: reject a CLI overlay, including --resume.
    allowed = {"experiment_config", "dry_run", "torch_threads"}
    changed = [k for k, v in vars(args).items() if k not in allowed and v != getattr(defaults, k)]
    if changed:
        raise ValueError(f"Experiment recipes reject CLI overrides: {changed}")
    args.init_from = recipe["inputs"]["checkpoint"]["path"]
    args.elite_archive_resume = recipe["inputs"]["elite_pool"]["path"]
    args.seed = recipe["seed"]
    args.run_dir = recipe["run_dir"]
    args.save_rollout_inputs = True
    args.persist_game = True
    if not args.dry_run and Path(args.run_dir).exists():
        raise ValueError("Experiment run directory already exists; use an explicit checkpoint resume separately")


def reset_experiment_critic(model):
    """Same reset protocol in every arm; actor/backbone/std remain inherited."""
    model.reset_critic_head()

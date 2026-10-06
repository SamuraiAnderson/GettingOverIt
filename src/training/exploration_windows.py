"""Exogenous temporal exploration schedules; no task-stage or observation labels."""
import argparse
import copy

import numpy as np


FIELDS = ("floor", "peak", "min_steps", "max_steps", "suffix_min_steps", "transition_steps")
TRAINING_FIELDS = ("anchor_scale", "anchor_coef")


def zero_noise_enabled(cfg):
    return cfg.exploration_windows_enabled and cfg.exploration_window_floor == 0


def settings(cfg):
    return {"version": 2 if zero_noise_enabled(cfg) else 1,
            "allocation": [1, 1, 4, 2], "cuda_precision": "fp32",
            **{name: getattr(cfg, "exploration_window_" + name) for name in FIELDS}}


def validate(cfg, horizon=None):
    if not cfg.exploration_windows_enabled:
        return
    horizon = cfg.steps_per_rollout if horizon is None else horizon
    if cfg.policy_backbone != "gru" or cfg.action_segment_steps != 1 or cfg.num_agents != 8:
        raise ValueError("Exploration windows currently require GRU, single-step actions and 8 agents")
    if cfg.delayed_exploration_enabled or cfg.phase_noise_enabled:
        raise ValueError("Exploration windows cannot combine with delayed_exploration or phase_noise")
    s = settings(cfg)
    if not (np.isfinite([s["floor"], s["peak"]]).all() and 0 <= s["floor"] <= s["peak"] and s["peak"] > 0):
        raise ValueError("Exploration window scales must be finite and 0 <= floor <= peak, peak > 0")
    if any(not np.isfinite(value) or value <= 0
           for value in (cfg.deterministic_anchor_scale, cfg.exploration_window_anchor_coef)):
        raise ValueError("Deterministic anchor scale and coefficient must be finite and positive")
    if not (1 <= s["min_steps"] <= s["max_steps"] <= horizon
            and 1 <= s["suffix_min_steps"] <= horizon):
        raise ValueError("Exploration window lengths must fit the rollout horizon")
    if s["transition_steps"] != 0:
        raise ValueError("Only rectangular exploration windows (transition_steps=0) are validated")


def make_rng(cfg):
    # Independent of deployment, OU innovations, PPO shuffling and SIL sampling.
    return np.random.default_rng(np.random.SeedSequence([cfg.training_seed, 0x57494E44]))


def configure_precision(cfg):
    """Process-wide policy math, shared by collection, PPO, SIL and frozen replay.

    Low-noise likelihoods amplify batch-shape-dependent TF32 rounding. Opting in
    to windows therefore opts the process into FP32; disabled runs keep defaults.
    """
    if cfg.exploration_windows_enabled:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def sample_plan(cfg, horizon, rng, *, deterministic=False):
    validate(cfg, horizon)
    if not cfg.exploration_windows_enabled or deterministic:
        return None
    s = settings(cfg)
    before = copy.deepcopy(rng.bit_generator.state)
    rows = [{"kind": "deterministic" if zero_noise_enabled(cfg) else "low", "start": 0, "stop": 0},
            {"kind": "full", "start": 0, "stop": int(horizon)}]
    # One duration per log stratum, independently shuffled across onset strata.
    u = (np.arange(4) + rng.random(4)) / 4
    lengths = np.rint(np.exp(np.log(s["min_steps"]) + u *
                            np.log(s["max_steps"] / s["min_steps"]))).astype(int)
    rng.shuffle(lengths)
    for j, length in enumerate(lengths):
        start = int((j + rng.random()) / 4 * (horizon - int(length) + 1))
        rows.append({"kind": "window", "start": start, "stop": start + int(length)})
    for j in range(2):
        start = int((j + rng.random()) / 2 * (horizon - s["suffix_min_steps"] + 1))
        rows.append({"kind": "suffix", "start": start, "stop": int(horizon)})
    rng.shuffle(rows)
    for agent, row in enumerate(rows):
        row["agent"] = agent
    return {"horizon": int(horizon), "settings": s, "agents": rows,
            "rng_before": before, "rng_after": copy.deepcopy(rng.bit_generator.state)}


def scales_for_plan(plan, agent_ids, step):
    if not 0 <= step < plan["horizon"]:
        raise ValueError("Exploration window step outside rollout")
    s = plan["settings"]
    return np.asarray([s["peak"] if plan["agents"][i]["start"] <= step < plan["agents"][i]["stop"]
                       else s["floor"] for i in agent_ids], dtype=np.float32)


def add_arguments(parser):
    parser.add_argument("--exploration-windows", action=argparse.BooleanOptionalAction, default=None,
                        help="Opt-in temporal noise windows for 8-agent GRU collection")
    for name in FIELDS + TRAINING_FIELDS:
        parser.add_argument("--exploration-window-" + name.replace("_", "-"),
                            type=float if name in ("floor", "peak") + TRAINING_FIELDS else int, default=None)


def apply_arguments(cfg, args):
    if args.exploration_windows is not None:
        cfg.exploration_windows_enabled = args.exploration_windows
    for name in FIELDS + TRAINING_FIELDS:
        field = "exploration_window_" + name
        if getattr(args, field) is not None:
            setattr(cfg, field, getattr(args, field))
    validate(cfg)

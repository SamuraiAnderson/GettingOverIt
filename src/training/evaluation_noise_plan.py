"""Fixed, exogenous noise windows for frozen comparisons only."""
import copy

import numpy as np

from .exploration_windows import settings, validate, zero_noise_enabled


def fixed_plan(spec, cfg, horizon, *, deterministic=False, initial_offsets=None):
    validate(cfg, horizon)
    if (not zero_noise_enabled(cfg) or deterministic or cfg.random_deploy
            or (initial_offsets is not None and (
                np.shape(initial_offsets) != (cfg.num_agents, 2) or np.any(initial_offsets)))):
        raise ValueError("Fixed evaluation noise requires zero-floor windows and exact-spawn exploration")
    if (not isinstance(spec, dict) or type(spec.get("version")) is not int or spec["version"] != 1
            or type(spec.get("horizon")) is not int or spec["horizon"] != horizon):
        raise ValueError("Invalid fixed evaluation noise version/horizon")
    rows = spec.get("agents")
    if not isinstance(rows, list) or len(rows) != cfg.num_agents:
        raise ValueError("Fixed evaluation noise needs one ordered row per agent")
    for i, row in enumerate(rows):
        if (not isinstance(row, dict) or type(row.get("agent")) is not int or row["agent"] != i
                or any(type(row.get(k)) is not int for k in ("start", "stop"))
                or not 0 <= row["start"] <= row["stop"] <= horizon
                or not isinstance(row.get("condition"), str) or not row["condition"].strip()):
            raise ValueError("Invalid fixed evaluation noise agent/window/condition")
    return {"horizon": horizon, "settings": settings(cfg), "agents": copy.deepcopy(rows),
            "evaluation_only": True, "schedule_source": "fixed_evaluation_plan",
            "scheduler_rng_consumed": False}

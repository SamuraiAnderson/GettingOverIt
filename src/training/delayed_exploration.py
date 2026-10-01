"""Per-player exploration delay. The schedule is a first-round probe, not a tuned optimum."""
from __future__ import annotations

import numpy as np

# Two players each: full noise from step 0, or a quiet prefix until 200/400/600.
OPEN_STEPS = (0, 0, 200, 200, 400, 400, 600, 600)


def noise_scale(step: int, open_step: int, floor: float, ramp_steps: int) -> float:
    """Quiet prefix keeps `floor`; after the open step, scale rises linearly to 1.

    An open step of 0 has no prefix, so the scale is 1 immediately. The ramp
    avoids a jump when a delayed player starts exploring. At the open step the
    scale is still `floor`; it reaches 1 exactly `ramp_steps` later.
    """
    step, open_step, ramp_steps = int(step), int(open_step), int(ramp_steps)
    floor = float(floor)
    if step < 0 or open_step < 0 or ramp_steps < 1 or not 0.0 < floor <= 1.0:
        raise ValueError("delayed exploration scale inputs are out of range")
    if open_step == 0:
        return 1.0
    if step < open_step:
        return floor
    progress = min(1.0, (step - open_step) / ramp_steps)
    return floor + (1.0 - floor) * progress


def scales_for_agents(config, agent_ids, step: int, deterministic: bool = False) -> np.ndarray:
    """One scale per agent at this physical step. Disabled or deterministic runs stay at 1."""
    ids = list(agent_ids)
    if deterministic or not getattr(config, "delayed_exploration_enabled", False):
        return np.ones(len(ids), dtype=np.float32)
    if int(config.num_agents) != len(OPEN_STEPS):
        raise ValueError("延迟探索的四档分配按 8 个 player 定义")
    if int(config.steps_per_rollout) <= OPEN_STEPS[-1]:
        raise ValueError("每轮步数必须大于 600，否则最晚一档不会打开噪声")
    floor = float(config.delayed_exploration_floor)
    ramp = int(config.delayed_exploration_ramp_steps)
    return np.asarray([noise_scale(step, OPEN_STEPS[int(i)], floor, ramp) for i in ids], dtype=np.float32)

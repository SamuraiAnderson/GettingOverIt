"""Switchable SIL archive: MAP-Elites (default) or endpoint (t, D) Pareto.

This module owns the (t, D) point, dominance, ingest, capacity, and the
helpers used by sampling / gate. BC admission is not routed through here.
Prefix-milestone archives are not implemented yet.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .reference_corridor import corridor_cap_index
from .reward import is_water_trajectory

if TYPE_CHECKING:
    from .config import TrainConfig
    from .dataset import Trajectory

ARCHIVE_MAP_ELITES = "map_elites"
ARCHIVE_TD_PARETO = "td_pareto"
ROLE_PARETO = "pareto"
ROLE_BEHAVIOR = "behavior"


def td_point(traj: "Trajectory") -> tuple[int, float] | None:
    """t = cap+1, D = float64 credited sum. None if the trajectory is not comparable."""
    progress = getattr(traj, "corridor_progress", None)
    states = getattr(traj, "raw_states", None)
    actions = getattr(traj, "actions", None)
    if progress is None or states is None or actions is None:
        return None
    progress = np.asarray(progress, dtype=np.float64)
    if (progress.ndim != 1 or len(progress) != len(actions)
            or len(states) != len(actions) + 1
            or not np.isfinite(progress).all()):
        return None
    total = float(progress.sum(dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        return None
    cap = corridor_cap_index(progress)
    t = int(cap) + 1
    if t < 1:
        return None
    return t, total


def dominates(a: tuple[int, float], b: tuple[int, float]) -> bool:
    """A eliminates B when it is at least as fast and far, and better in one."""
    t_a, d_a = a
    t_b, d_b = b
    return t_a <= t_b and d_a >= d_b and (t_a < t_b or d_a > d_b)


def _dedup_key(point: tuple[int, float]) -> tuple[int, int]:
    t, d = point
    return t, int(np.round(d * 1e12))


def _dedup_better(new: "Trajectory", old: "Trajectory") -> bool:
    if int(new.iteration) != int(old.iteration):
        return int(new.iteration) > int(old.iteration)
    if float(new.score) != float(old.score):
        return float(new.score) > float(old.score)
    return len(new.actions) > len(old.actions)


def _valid_candidates(trajs: list["Trajectory"], config: "TrainConfig") -> list["Trajectory"]:
    kept = []
    for traj in trajs:
        if is_water_trajectory(traj.raw_states, config):
            continue
        if td_point(traj) is None:
            continue
        kept.append(traj)
    return kept


def dedup_td(trajs: list["Trajectory"]) -> list["Trajectory"]:
    best: dict[tuple[int, int], "Trajectory"] = {}
    for traj in trajs:
        point = td_point(traj)
        if point is None:
            continue
        key = _dedup_key(point)
        previous = best.get(key)
        if previous is None or _dedup_better(traj, previous):
            best[key] = traj
    return [best[key] for key in sorted(best)]


def pareto_front(trajs: list["Trajectory"]) -> list["Trajectory"]:
    points = [(traj, td_point(traj)) for traj in trajs]
    points = [(traj, point) for traj, point in points if point is not None]
    kept = []
    for traj, point in points:
        if any(dominates(other, point) for other_traj, other in points
               if other_traj is not traj):
            continue
        kept.append(traj)
    return kept


def _farthest(trajs: list["Trajectory"]) -> "Trajectory":
    return min(trajs, key=lambda traj: (-td_point(traj)[1], td_point(traj)[0], -traj.iteration))


def _fastest(trajs: list["Trajectory"]) -> "Trajectory":
    return min(trajs, key=lambda traj: (td_point(traj)[0], -td_point(traj)[1], -traj.iteration))


def bucket_compress(trajs: list["Trajectory"], bucket_m: float) -> list["Trajectory"]:
    """Approximate front: one min-t representative per D bucket, plus farthest."""
    if not trajs or bucket_m <= 0:
        return list(trajs)
    groups: dict[int, list] = {}
    for traj in trajs:
        _t, distance = td_point(traj)
        groups.setdefault(int(np.floor(distance / bucket_m)), []).append(traj)
    reps = []
    for bucket in sorted(groups):
        reps.append(min(groups[bucket],
                        key=lambda traj: (td_point(traj)[0], -td_point(traj)[1], -traj.iteration)))
    farthest = _farthest(trajs)
    if all(traj is not farthest for traj in reps):
        reps.append(farthest)
    return dedup_td(reps)


def _norm_xy(trajs: list["Trajectory"]) -> np.ndarray:
    pts = np.array([td_point(traj) for traj in trajs], dtype=np.float64)
    span = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1e-12)
    return (pts - pts.min(axis=0)) / span


def sparse_cap(trajs: list["Trajectory"], cap: int) -> list["Trajectory"]:
    """Keep fastest and farthest, then greedily sparse points. Approximate if over cap."""
    if len(trajs) <= cap:
        return list(trajs)
    if cap < 2:
        return [_farthest(trajs)]
    ordered = []
    for required in (_fastest(trajs), _farthest(trajs)):
        if all(traj is not required for traj in ordered):
            ordered.append(required)
    remaining = [traj for traj in trajs if all(traj is not kept for kept in ordered)]
    remaining.sort(key=lambda traj: (td_point(traj)[0], -td_point(traj)[1], -traj.iteration))
    while len(ordered) < cap and remaining:
        kept_xy = _norm_xy(ordered)
        rest_xy = _norm_xy(ordered + remaining)[len(ordered):]
        dist = np.linalg.norm(rest_xy[:, None, :] - kept_xy[None, :, :], axis=2).min(axis=1)
        pick = int(np.argmax(dist))
        # Ties: earliest t, then larger D (remaining is already sorted; scan equals).
        best = dist[pick]
        for i, value in enumerate(dist):
            if value + 1e-15 >= best:
                pick = i
                break
        ordered.append(remaining.pop(pick))
    return ordered


def _cell_id(traj: "Trajectory", cell_size: float) -> tuple[int, int] | None:
    point = td_point(traj)
    if point is None:
        return None
    idx = point[0]
    if idx < 0 or idx >= len(traj.raw_states):
        return None
    x, y = float(traj.raw_states[idx, 0]), float(traj.raw_states[idx, 1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return int(np.floor(x / cell_size)), int(np.floor(y / cell_size))


def behavior_extras(
    leftover: list["Trajectory"],
    kept: list["Trajectory"],
    cell_size: float,
    slots: int,
) -> list["Trajectory"]:
    """Keep a few spatially distinct leftovers the endpoint front dropped."""
    if slots <= 0 or not leftover:
        return []
    occupied = {cid for traj in kept if (cid := _cell_id(traj, cell_size)) is not None}
    best: dict[tuple[int, int], "Trajectory"] = {}
    for traj in leftover:
        cid = _cell_id(traj, cell_size)
        if cid is None or cid in occupied:
            continue
        previous = best.get(cid)
        if previous is None or float(traj.score) > float(previous.score):
            best[cid] = traj
    ranked = sorted(best.values(), key=lambda traj: (-float(traj.score), -traj.iteration))
    return ranked[:slots]


def credited_distance(traj: "Trajectory") -> float:
    point = td_point(traj)
    return 0.0 if point is None else point[1]


def gate_eligible_indices(
    trajs: list["Trajectory"], behavior_progress: float, margin: float,
) -> list[int]:
    """Per-trajectory D gate. A short-fast track cannot close SIL for a farther one."""
    threshold = float(behavior_progress) * (1.0 + float(margin))
    return [i for i, traj in enumerate(trajs) if credited_distance(traj) > threshold]


def equal_traj_window_index(
    trajs: list["Trajectory"], windows_per_traj: int = 0,
) -> list[tuple[int, int]]:
    """Pick trajectory first, then equally spaced windows. Does not drop stored steps."""
    caps = []
    for traj in trajs:
        if traj.corridor_progress is None:
            caps.append(-1)
            continue
        cap = min(corridor_cap_index(np.asarray(traj.corridor_progress, dtype=np.float64)),
                  len(traj.actions) - 1)
        caps.append(int(cap))
    usable = [c for c in caps if c >= 0]
    if not usable:
        return []
    budget = min(usable) + 1
    if windows_per_traj > 0:
        budget = min(budget, int(windows_per_traj))
    budget = max(1, budget)
    index = []
    for ti, cap in enumerate(caps):
        if cap < 0:
            continue
        if cap + 1 <= budget:
            taus = range(cap + 1)
        else:
            taus = np.linspace(0, cap, budget)
            taus = np.unique(np.rint(taus).astype(int))
        for tau in taus:
            index.append((ti, int(tau)))
    return index


def ingest_td_pareto(
    existing: list["Trajectory"],
    incoming: list["Trajectory"],
    scores: list[float],
    iteration: int,
    config: "TrainConfig",
) -> tuple[list["Trajectory"], dict[str, float]]:
    """Merge → reject → dedup → Pareto → optional D-buckets → cap → behavior extras."""
    new = []
    for traj, score in zip(incoming, scores):
        traj.score = float(score)
        traj.iteration = iteration
        new.append(traj)
    merged = _valid_candidates([*existing, *new], config)
    unique = dedup_td(merged)
    front = pareto_front(unique)
    bucketed = False
    enable_at = int(getattr(config, "sil_td_bucket_enable_at", 16))
    bucket_m = float(getattr(config, "sil_td_bucket_m", 0.10))
    if len(front) >= enable_at:
        front = bucket_compress(front, bucket_m)
        bucketed = True
    cap = int(getattr(config, "sil_traj_cap", 20))
    capped = False
    if len(front) > cap:
        front = sparse_cap(front, cap)
        capped = True
    leftover = [traj for traj in unique if all(traj is not kept for kept in front)]
    extras = behavior_extras(
        leftover, front, float(getattr(config, "sil_cell_size", 5.0)),
        int(getattr(config, "sil_td_behavior_slots", 3)),
    )
    room = max(0, cap - len(front))
    extras = extras[:room]
    for traj in front:
        traj.archive_role = ROLE_PARETO
    for traj in extras:
        traj.archive_role = ROLE_BEHAVIOR
    kept = front + extras
    return kept, {
        "sil_archive_pareto": float(len(front)),
        "sil_archive_behavior": float(len(extras)),
        "sil_archive_bucketed": float(bucketed),
        "sil_archive_capped": float(capped),
        "sil_archive_size": float(len(kept)),
        "sil_windows_per_traj": float(getattr(config, "sil_td_windows_per_traj", 0)),
    }

"""Switchable SIL archive: MAP-Elites (default) or endpoint (t, D) Pareto.

This module owns the (t, D) point, dominance, ingest, capacity, and the
helpers used by sampling / gate. BC admission is not routed through here.
Prefix-milestone archives are not implemented yet.
"""
from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Sequence

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
SAMPLING_LEGACY = "legacy_min_length"
SAMPLING_BALANCED = "balanced_stratified"
SAMPLING_COVERAGE = "coverage_share"
SAMPLING_MODES = (SAMPLING_LEGACY, SAMPLING_BALANCED, SAMPLING_COVERAGE)
SAMPLING_RULE_VERSION = "coverage_v1"
NEW_SAMPLING_MODES = (SAMPLING_BALANCED, SAMPLING_COVERAGE)


def recover_v2_termination(traj: "Trajectory") -> bool:
    """Fill a missing v2 termination array with all-false steps.

    Saved confirmed-endpoint elites sometimes omit ``terminated`` even though
    every sibling row and the collector's ``done`` flag are false. Leaving
    ``termination_known`` false blocks advantage weighting for the whole pool.
    An explicit array is never rewritten. Non-v2 rows stay unknown.
    """
    from .reference_corridor import PROGRESS_MODE_V2
    if getattr(traj, "termination_known", True):
        return False
    if getattr(traj, "progress_mode", None) != PROGRESS_MODE_V2:
        return False
    if getattr(traj, "terminated", None) is not None:
        return False
    actions = getattr(traj, "actions", None)
    if actions is None:
        return False
    traj.terminated = np.zeros(len(actions), dtype=bool)
    traj.termination_known = True
    return True


def td_point(traj: "Trajectory") -> tuple[int, float] | None:
    """(t_cap, distance). v2 distance is confirmed D; v1 remains the credit sum."""
    from .reference_corridor import trajectory_endpoint
    states = getattr(traj, "raw_states", None)
    actions = getattr(traj, "actions", None)
    if states is None or actions is None:
        return None
    if len(states) != len(actions) + 1:
        return None
    endpoint = trajectory_endpoint(traj)
    if (not endpoint.valid or endpoint.cap < 0 or endpoint.t_cap is None
            or not np.isfinite(endpoint.distance) or endpoint.distance <= 0.0):
        return None
    return int(endpoint.t_cap), float(endpoint.distance)


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
        from .reference_corridor import trajectory_endpoint
        endpoint = trajectory_endpoint(traj)
        if not endpoint.valid or endpoint.cap < 0:
            caps.append(-1)
            continue
        caps.append(min(int(endpoint.cap), len(traj.actions) - 1))
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


def sampling_generator(seed: int, iteration: int, epoch: int, category: str) -> np.random.Generator:
    """Stable per-purpose RNG. Python's salted hash() cannot reproduce a resumed plan."""
    payload = f"{int(seed)}|{int(iteration)}|{int(epoch)}|{category}|{SAMPLING_RULE_VERSION}".encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(value)


def _comparable_v2_point(traj: "Trajectory") -> tuple[int, float, int] | None:
    """(t, D, cap) for a v2 endpoint that can be supervised on actions 0..cap."""
    from .reference_corridor import PROGRESS_MODE_V2, trajectory_endpoint
    point = td_point(traj)
    if point is None:
        return None
    endpoint = trajectory_endpoint(traj)
    if endpoint.mode != PROGRESS_MODE_V2 or not endpoint.valid or endpoint.cap < 0:
        return None
    actions = getattr(traj, "actions", None)
    if actions is None or len(actions) == 0:
        return None
    cap = min(int(endpoint.cap), len(actions) - 1)
    if cap < 0:
        return None
    return int(point[0]), float(point[1]), cap


def full_candidate_index(
    trajs: list["Trajectory"], horizon: int | None = None,
) -> list[tuple[int, int]]:
    """Every supervised action 0..cap. Does not let the shortest trajectory shrink the rest."""
    index: list[tuple[int, int]] = []
    limit = None if horizon is None or int(horizon) <= 0 else int(horizon)
    for ti, traj in enumerate(trajs):
        parsed = _comparable_v2_point(traj)
        if parsed is None:
            continue
        t_cap, _distance, cap = parsed
        if limit is not None and t_cap > limit:
            continue
        for tau in range(cap + 1):
            index.append((ti, int(tau)))
    return index


def coverage_area(
    points: Sequence[tuple[int, float] | None], horizon: int,
) -> tuple[np.ndarray, float]:
    """Exact shared area of the (t, D) coverage envelope.

    Time cuts are 1, every finite t_i, and H+1. Distance cuts are 0 and every D_i.
    A cell [a, b) x [c, d) is covered by i when t_i <= a and D_i >= d. Its area is
    split equally. The returned total is the independent cell sum, which equals
    the sum of shares up to floating-point order.
    """
    parsed = list(points)
    weights = np.zeros(len(parsed), dtype=np.float64)
    horizon = int(horizon)
    if horizon < 1 or not parsed:
        return weights, 0.0
    usable = [(idx, int(point[0]), float(point[1]))
              for idx, point in enumerate(parsed)
              if point is not None and np.isfinite(point[0]) and np.isfinite(point[1])
              and int(point[0]) <= horizon and float(point[1]) > 0.0]
    if not usable:
        return weights, 0.0
    time_cuts = sorted({1, horizon + 1, *(t for _i, t, _d in usable)})
    distance_cuts = sorted({0.0, *(d for _i, _t, d in usable)})
    envelope = 0.0
    for left, right in zip(time_cuts[:-1], time_cuts[1:]):
        if left > horizon:
            break
        right = min(int(right), horizon + 1)
        width = right - int(left)
        if width <= 0:
            continue
        for low, high in zip(distance_cuts[:-1], distance_cuts[1:]):
            height = float(high) - float(low)
            if height <= 0.0:
                continue
            covered = [idx for idx, t_cap, distance in usable
                       if t_cap <= int(left) and distance >= float(high)]
            if not covered:
                continue
            cell = width * height
            share = cell / len(covered)
            for idx in covered:
                weights[idx] += share
            envelope += cell
    if not np.isfinite(weights).all() or np.any(weights < -1e-12):
        raise ValueError("coverage area is not finite and nonnegative")
    scale = max(1.0, abs(envelope))
    if abs(float(weights.sum()) - envelope) > 1e-8 * scale:
        raise ValueError("coverage shares do not conserve the envelope area")
    return weights, float(envelope)


def uniform_stratified_weights(
    points: Sequence[tuple[int, float] | None], horizon: int,
) -> np.ndarray:
    """Equal weight for every comparable point with t <= H and D > 0."""
    weights = np.zeros(len(points), dtype=np.float64)
    eligible = [idx for idx, point in enumerate(points)
                if point is not None and int(point[0]) <= int(horizon) and float(point[1]) > 0.0]
    if eligible:
        weights[eligible] = 1.0
    return weights


def allocate_quotas(
    probs: np.ndarray, budget: int, rng: np.random.Generator,
) -> np.ndarray:
    """Largest-remainder quotas. Sum is `budget`; each entry is floor or ceil.

    The `budget - sum(floor)` extras are placed by systematic sampling on the
    fractional parts: one uniform u in [0, 1) and the points u, u+1, ..., u+R-1.
    There is no extra "at least one" grant.
    """
    probs = np.asarray(probs, dtype=np.float64)
    quotas = np.zeros(len(probs), dtype=np.int64)
    budget = int(budget)
    if budget < 0:
        raise ValueError("quota budget must be nonnegative")
    if budget == 0 or len(probs) == 0:
        return quotas
    if not np.isfinite(probs).all() or np.any(probs < 0.0):
        raise ValueError("quota probabilities must be finite and nonnegative")
    total = float(probs.sum())
    if total <= 0.0:
        return quotas
    targets = budget * (probs / total)
    quotas = np.floor(targets).astype(np.int64)
    residual = targets - quotas
    extra = int(budget - int(quotas.sum()))
    if extra < 0:
        raise ValueError("quota floors exceed the budget")
    if extra == 0:
        return quotas
    edges = np.cumsum(residual)
    if float(edges[-1]) <= 0.0:
        order = np.argsort(-targets, kind="stable")
        for step in range(extra):
            quotas[int(order[step % len(quotas)])] += 1
        return quotas
    start = float(rng.random())
    for step in range(extra):
        point = start + step
        if point >= float(edges[-1]):
            point = float(np.nextafter(edges[-1], 0.0))
        choice = int(np.searchsorted(edges, point, side="left"))
        quotas[min(choice, len(quotas) - 1)] += 1
    ceilings = np.ceil(targets - 1e-9).astype(np.int64)
    overflow = np.flatnonzero(quotas > ceilings)
    for idx in overflow:
        surplus = int(quotas[idx] - ceilings[idx])
        quotas[idx] = int(ceilings[idx])
        room = np.flatnonzero(quotas < ceilings)
        if len(room) == 0 or surplus > len(room):
            raise ValueError("cannot keep quotas at floor/ceil while matching the budget")
        order = room[np.argsort(-residual[room], kind="stable")]
        for step in range(surplus):
            quotas[int(order[step])] += 1
    if int(quotas.sum()) != budget:
        raise ValueError("quota allocation does not sum to the budget")
    if np.any(quotas < np.floor(targets).astype(np.int64)) or np.any(quotas > ceilings):
        raise ValueError("quota left the floor/ceil band")
    return quotas


def stratified_taus(
    candidates: Sequence[int], quota: int, rng: np.random.Generator,
) -> list[int]:
    """Sample `quota` action indices from a time-ordered candidate list.

    q=0 draws nothing. q=1 draws one index from the whole span. 1<q<=L draws one
    index from each nearly equal contiguous bin. q>L repeats the full list and
    stratifies the remainder. Repeats are allowed only in that last case.
    """
    ordered = [int(tau) for tau in sorted(candidates)]
    quota = int(quota)
    count = len(ordered)
    if quota <= 0 or count == 0:
        return []
    if quota == 1:
        return [ordered[int(rng.integers(0, count))]]
    repeats, remainder = divmod(quota, count) if quota > count else (0, quota)
    chosen: list[int] = ordered * repeats
    if remainder == 0:
        return chosen
    if remainder == 1:
        chosen.append(ordered[int(rng.integers(0, count))])
        return chosen
    base, leftover = divmod(count, remainder)
    start = 0
    for bin_index in range(remainder):
        width = base + (1 if bin_index < leftover else 0)
        stop = start + width
        chosen.append(ordered[int(rng.integers(start, stop))])
        start = stop
    return chosen


def _phase_keep(
    traj: "Trajectory", taus: list[int], *, phase_window: int,
    airborne_sample: float, rng: np.random.Generator,
) -> tuple[list[int], str | None]:
    """Keep contact windows and a random subset of airborne windows."""
    contact = getattr(traj, "contact", None)
    if contact is None or not (0.0 <= float(airborne_sample) < 1.0):
        return list(taus), None
    kernel = np.ones(2 * max(1, int(phase_window)) + 1, dtype=np.float64)
    touched = ((contact[:, 0] > 0.5) | (contact[:, 2] > 0.5)
               | (contact[:, 3] > 0.5)).astype(np.float64)
    near = np.convolve(touched, kernel, mode="same") > 0
    kept: list[int] = []
    for tau in taus:
        rel = near[min(int(tau), len(near) - 1)]
        if rel or float(rng.random()) < float(airborne_sample):
            kept.append(int(tau))
    if not kept:
        return [], "empty_after_phase"
    return kept, None


def _third_counts(taus: Sequence[int], cap: int) -> tuple[int, int, int]:
    span = max(int(cap) + 1, 1)
    first = middle = last = 0
    for tau in taus:
        frac = float(tau) / span
        if frac < 1.0 / 3.0:
            first += 1
        elif frac < 2.0 / 3.0:
            middle += 1
        else:
            last += 1
    return first, middle, last


def build_td_sampling_plan(
    trajs: list["Trajectory"],
    *,
    mode: str,
    horizon: int,
    k: int,
    window_budget: int | None,
    seed: int,
    iteration: int,
    epoch: int = 0,
    phase_enabled: bool = False,
    phase_window: int = 7,
    airborne_sample: float = 1.0,
    behavior_progress: float | None = None,
) -> dict:
    """Turn a retained pool into one shuffled SIL window plan.

    Phase filtering happens before the area or uniform weights, so a trajectory
    with no surviving window does not receive budget. Invalid, non-positive, and
    t>H trajectories are not sampled. An empty positive-area set skips SIL
    instead of falling back to the first action.
    """
    if mode not in NEW_SAMPLING_MODES:
        raise ValueError(f"build_td_sampling_plan does not serve mode {mode}")
    horizon = int(horizon)
    average_k = 32 if int(k) <= 0 else int(k)
    phase_rng = sampling_generator(seed, iteration, epoch, "phase")
    quota_rng = sampling_generator(seed, iteration, epoch, "quota")
    within_rng = sampling_generator(seed, iteration, epoch, "within")
    shuffle_rng = sampling_generator(seed, iteration, epoch, "shuffle")

    parsed: list[tuple[int, float, int] | None] = []
    reasons: dict[int, str] = {}
    kept_taus: dict[int, list[int]] = {}
    for ti, traj in enumerate(trajs):
        point = _comparable_v2_point(traj)
        if point is None:
            parsed.append(None)
            reasons[ti] = "incomparable"
            continue
        t_cap, distance, cap = point
        parsed.append((t_cap, distance, cap))
        if t_cap > horizon or horizon < 1:
            reasons[ti] = "over_horizon" if horizon >= 1 else "invalid_horizon"
            continue
        candidates = list(range(cap + 1))
        if phase_enabled:
            candidates, empty_reason = _phase_keep(
                traj, candidates, phase_window=phase_window,
                airborne_sample=airborne_sample, rng=phase_rng,
            )
            if empty_reason:
                reasons[ti] = empty_reason
                continue
        kept_taus[ti] = candidates

    points: list[tuple[int, float] | None] = []
    for ti, point in enumerate(parsed):
        if ti in kept_taus and point is not None:
            points.append((point[0], point[1]))
        else:
            points.append(None)
    areas, envelope = coverage_area(points, horizon)
    if mode == SAMPLING_COVERAGE:
        raw_weights = areas
    else:
        raw_weights = uniform_stratified_weights(points, horizon)
    positive = raw_weights > 0.0
    probability = np.zeros(len(trajs), dtype=np.float64)
    if positive.any():
        probability[positive] = raw_weights[positive] / float(raw_weights[positive].sum())

    def _record(index: list[tuple[int, int]], quotas: np.ndarray, skip: bool, reason: str) -> dict:
        sampled: dict[int, list[int]] = {}
        for ti, tau in index:
            sampled.setdefault(ti, []).append(int(tau))
        frontier_ids = set()
        comparable = [trajs[ti] for ti, point in enumerate(parsed) if point is not None]
        for traj in pareto_front(comparable):
            frontier_ids.add(id(traj))
        rows = []
        frontier_windows = 0
        for ti, point in enumerate(parsed):
            chosen = sampled.get(ti, [])
            is_frontier = point is not None and id(trajs[ti]) in frontier_ids
            if is_frontier:
                frontier_windows += len(chosen)
            thirds = _third_counts(chosen, point[2] if point is not None else 0)
            rows.append({
                "idx": ti,
                "t": None if point is None else int(point[0]),
                "D": None if point is None else float(point[1]),
                "cap": None if point is None else int(point[2]),
                "is_frontier": bool(is_frontier),
                "area": float(areas[ti]),
                "prob": float(probability[ti]),
                "target_quota": int(quotas[ti]),
                "actual_quota": len(chosen),
                "unique_windows": len(set(chosen)),
                "min_tau": None if not chosen else int(min(chosen)),
                "max_tau": None if not chosen else int(max(chosen)),
                "n_first_third": thirds[0],
                "n_middle_third": thirds[1],
                "n_last_third": thirds[2],
                "reason": reasons.get(ti),
            })
        actual = len(index)
        unique = len(set(index))
        top_prob = float(probability.max()) if len(probability) else 0.0
        frontier_share = float(frontier_windows / actual) if actual else 0.0
        pool_key = hashlib.sha256(repr([
            None if point is None else (int(point[0]), round(float(point[1]), 9))
            for point in parsed
        ]).encode()).hexdigest()[:16]
        metrics = {
            "sil_sampling_mode": mode,
            "sil_sampling_H": float(horizon),
            "sil_sampling_K": float(average_k),
            "sil_sampling_n_eligible": float(int(positive.sum())),
            "sil_sampling_n_empty": float(sum(1 for reason in reasons.values()
                                              if reason == "empty_after_phase")),
            "sil_sampling_n_over_H": float(sum(1 for reason in reasons.values()
                                               if reason == "over_horizon")),
            "sil_sampling_total_area": float(envelope),
            "sil_sampling_planned_windows": float(int(quotas.sum())),
            "sil_sampling_actual_windows": float(actual),
            "sil_sampling_unique_windows": float(unique),
            "sil_sampling_frontier_share": frontier_share,
            "sil_sampling_top_prob": top_prob,
        }
        if behavior_progress is not None:
            distances = [point[1] for point in parsed if point is not None]
            metrics["sil_behavior_level"] = float(behavior_progress)
            metrics["sil_pool_level"] = float(max(distances)) if distances else 0.0
        record = {
            "iteration": int(iteration),
            "epoch": int(epoch),
            "mode": mode,
            "H": int(horizon),
            "K": int(average_k),
            "seed": int(seed),
            "rule_version": SAMPLING_RULE_VERSION,
            "pool_id": pool_key,
            "total_area": float(envelope),
            "skip": bool(skip),
            "skip_reason": reason,
            "sampled_index": [[int(ti), int(tau)] for ti, tau in index],
            "trajectories": rows,
        }
        return {"skip": skip, "skip_reason": reason, "index": index,
                "metrics": metrics, "record": record}

    if horizon < 1 or not positive.any():
        empty = np.zeros(len(trajs), dtype=np.int64)
        reason = "invalid_horizon" if horizon < 1 else "no_positive_area"
        return _record([], empty, True, reason)
    if window_budget is not None and int(window_budget) <= 0:
        empty = np.zeros(len(trajs), dtype=np.int64)
        return _record([], empty, True, "zero_update_budget")
    eligible_count = int(positive.sum())
    requested = eligible_count * average_k
    sample_budget = requested if window_budget is None else min(requested, int(window_budget))
    if sample_budget <= 0:
        empty = np.zeros(len(trajs), dtype=np.int64)
        return _record([], empty, True, "zero_sample_budget")
    quotas = allocate_quotas(probability, sample_budget, quota_rng)
    index: list[tuple[int, int]] = []
    for ti, quota in enumerate(quotas):
        if int(quota) <= 0:
            continue
        index.extend((ti, tau) for tau in stratified_taus(kept_taus[ti], int(quota), within_rng))
    shuffle_rng.shuffle(index)
    return _record(index, quotas, False, "")


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

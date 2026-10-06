"""Shared departure/return geometry and explicit cell selection; no model state."""
import math
import numpy as np
from .reference_anchor import scoring_context


def closed_excursions(inside, progress, segments, gaps):
    """Last inside state -> first return; never sum positive increments or merge visits."""
    rows, start, unfinished = [], None, 0
    for i in range(1, len(inside)):
        if inside[i - 1] and not inside[i]:
            start = i - 1
        if start is not None and inside[i]:
            if (segments[start] == segments[i]
                    and not np.any(gaps[start:i + 1])):
                rows.append(dict(start_step=start, first_outside_step=start + 1,
                                 end_step=i, T_steps=i - start,
                                 outside_states=i - start - 1,
                                 segment_index=int(segments[start]),
                                 s_start_m=float(progress[start]), s_end_m=float(progress[i]),
                                 delta_s_m=float(progress[i] - progress[start])))
            start = None
    if start is not None:
        unfinished = 1
    return rows, unfinished


def boundary_metrics(row, states, progress, route, step_seconds):
    """Upper means +world-y relative to the ordered projection, not upward velocity."""
    segment = route.segments[row["segment_index"]]
    offset = sum(s.length for s in route.segments[:row["segment_index"]])

    def projection_at(i):
        s = progress[i] - offset
        k = min(len(segment.lengths)-1, max(0, int(np.searchsorted(segment.starts, s, side="right")-1)))
        t = np.clip((s-segment.starts[k])/segment.lengths[k], 0., 1.)
        return segment.points[k] + t*segment.vectors[k]

    a, b = row["start_step"], row["end_step"]
    if step_seconds <= 0 or b < 1:
        raise ValueError("Return speed needs a positive time step and the preceding state")
    pa, pb = projection_at(a), projection_at(b)
    return dict(departure_projection_xy_m=pa.tolist(), return_projection_xy_m=pb.tolist(),
                departure_relative_y_m=float(states[a, 34]-pa[1]),
                return_relative_y_m=float(states[b, 34]-pb[1]),
                first_outside_relative_y_m=float(states[a+1, 34]-projection_at(a+1)[1]),
                last_outside_relative_y_m=float(states[b-1, 34]-projection_at(b-1)[1]),
                return_projection_speed_mps=float((progress[b]-progress[b-1])/step_seconds))


def admission_rejections(row, minimum, reject_upper=False, reject_negative_return=False):
    reasons = []
    if row["delta_s_m"] < minimum:
        reasons.append("below_min_delta")
    if reject_upper:
        if row["departure_relative_y_m"] > 0.:
            reasons.append("upper_departure")
        if row["return_relative_y_m"] > 0.:
            reasons.append("upper_return")
    if reject_negative_return and row["return_projection_speed_mps"] < 0.:
        reasons.append("negative_return_projection_speed")
    return reasons


def select_cells(rows, minimum, start_bin, delta_bin, *, reject_upper=False, reject_negative_return=False):
    """Identity=(segment, floor(s_start/ws), floor(delta/wd)); lower T wins."""
    if minimum <= 0 or start_bin <= 0 or delta_bin <= 0:
        raise ValueError("Forward threshold and bin widths must be positive")
    cells = {}
    decisions = {}
    for row in rows:
        cid = row["candidate_id"]
        reasons = admission_rejections(row, minimum, reject_upper, reject_negative_return)
        if reasons:
            decisions[cid] = {"reason": reasons[0], "rejection_reasons": reasons}
            continue
        key = (row["segment_index"], math.floor(row["s_start_m"] / start_bin),
               math.floor(row["delta_s_m"] / delta_bin))
        cells.setdefault(key, []).append(row)
    winners = []
    for key, members in sorted(cells.items()):
        ranked = sorted(members, key=lambda r: (r["T_seconds"], -r["delta_s_m"],
                                                r["s_start_m"], r["candidate_id"]))
        winner = ranked[0]
        winners.append(winner)
        for row in ranked:
            decisions[row["candidate_id"]] = {
                "reason": "selected" if row is winner else "same_cell_replaced",
                "cell": list(key), "winner_id": winner["candidate_id"],
                "cell_candidates": len(members),
            }
    return winners, decisions


def replay(states, cfg, route):
    schema, meta = scoring_context(states, config=cfg, route=route)
    tracker = route.tracker_from_state(states[0], schema, meta)
    offsets = np.r_[0., np.cumsum([s.length for s in route.segments])]
    inside = [tracker.previous_inside]
    progress = [float(offsets[tracker.segment_index] + tracker.progress)]
    segments, gaps, invalid = [tracker.segment_index], [False], []
    for i, state in enumerate(states[1:], 1):
        result = route.step_state(tracker, state, schema, meta)
        inside.append(result.inside)
        progress.append(float(offsets[tracker.segment_index] + tracker.progress))
        segments.append(tracker.segment_index)
        gaps.append(result.gap)
        if result.invalid:
            invalid.append(i)
    return np.asarray(inside), np.asarray(progress), np.asarray(segments), np.asarray(gaps), invalid



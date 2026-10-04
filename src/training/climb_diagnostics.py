"""Versioned, offline motion labels. Contacts are geometric proxies, not branch truth.

Labels at index t describe the observation before action t. Neither D nor reward
is used to label motion. The tree box is specific to the current first obstacle.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class PhaseRules:
    version: str = "tree_motion_proxy_v1"
    tree_x_min: float = -33.0
    tree_x_max: float = -27.0
    elevated_tip_y: float = 0.0
    upward_speed: float = 0.5
    minimum_flight_steps: int = 8
    minimum_flight_rise_m: float = 0.25
    preparation_steps: int = 12
    hold_steps: int = 120

    def __post_init__(self):
        if self.version != "tree_motion_proxy_v1":
            raise ValueError("Unknown motion rule version")
        numbers = [v for v in asdict(self).values() if not isinstance(v, str)]
        if not np.isfinite(numbers).all() or self.tree_x_min >= self.tree_x_max:
            raise ValueError("Invalid motion rules")
        if min(self.minimum_flight_steps, self.preparation_steps, self.hold_steps) < 1:
            raise ValueError("Phase step counts must be positive")
        if any(type(v) is not int for v in (self.minimum_flight_steps, self.preparation_steps, self.hold_steps)):
            raise ValueError("Phase step counts must be integers")
        if min(self.upward_speed, self.minimum_flight_rise_m) <= 0:
            raise ValueError("Motion thresholds must be positive")


def runs(mask):
    """Inclusive start, exclusive end; includes runs at either boundary."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def phase_counts(labels):
    names, counts = np.unique(labels, return_counts=True)
    return dict(zip(names.tolist(), counts.astype(int).tolist()))


def classify_motion(states, contacts, rules=None):
    rules = rules or PhaseRules()
    states, contacts = np.asarray(states), np.asarray(contacts)
    if (states.ndim != 2 or states.shape[1] != 35 or len(states) < 2
            or contacts.shape != (len(states), 5)
            or not np.isfinite(states).all() or not np.isfinite(contacts).all()):
        raise ValueError("Motion labels require finite 35D states and aligned N+1 by 5 contacts")
    n = len(states) - 1
    xy, vy = states[:, 33:35], states[:, 3]
    tree = (xy[:, 0] >= rules.tree_x_min) & (xy[:, 0] <= rules.tree_x_max)
    touching = np.any(contacts[:, [0, 2, 3]] > .5, axis=1)
    labels = np.full(len(states), "approach_support", dtype="<U40")
    labels[tree] = "tree_support"
    flights = []
    for a, b in runs(~touching):
        if b - a < rules.minimum_flight_steps:
            continue
        peak = a + int(xy[a:b, 1].argmax())
        rise = float(xy[peak, 1] - xy[a, 1])
        released = a > 0 and bool(touching[a - 1])
        qualifies = released and rise >= rules.minimum_flight_rise_m and float(vy[a:b].max()) > rules.upward_speed
        for t in range(a, b):
            labels[t] = ("tree_" if tree[t] else "approach_") + ("air_ascent" if vy[t] > rules.upward_speed else "air_other")
        if qualifies:
            first = max(0, a - rules.preparation_steps)
            for t in range(first, a):
                if touching[t]:
                    labels[t] = "takeoff_preparation"
        flights.append({"start_state": a, "end_state_exclusive": b, "peak_state": peak,
                        "observed_release": released, "takeoff_candidate": qualifies,
                        "rise_from_release_m": rise, "tree_at_release": bool(tree[a])})
    # This is NOT proof that a particular branch was gripped. No surface ID is archived.
    elevated = (contacts[:, 0] > .5) & (states[:, 24] >= rules.elevated_tip_y)
    elevated &= (states[:, 23] >= rules.tree_x_min) & (states[:, 23] <= rules.tree_x_max)
    labels[elevated] = "elevated_tip_contact_candidate"
    return labels[:n], {"flights": flights, "elevated_contact_runs": runs(elevated[:n]),
                        "counts": phase_counts(labels[:n]), "contact_semantics": "geometry_proxy"}


def launch_event_windows(states, contacts, cap, rules=None):
    """Offline launch proxies, spanning supported push through the airborne peak.

    Unlike v1's exclusive labels, an event retains its whole causal neighborhood.
    Trace the last uninterrupted supported rise back at most 64 states; include
    up to preparation_steps of preceding contact and the action entering it.
    Future archived states identify events, never become online policy inputs.
    All returned action windows stop at the confirmed cap (exclusive).
    """
    rules = rules or PhaseRules()
    _, motion = classify_motion(states, contacts, rules)
    if type(cap) is not int or not 0 < cap <= len(states) - 1:
        raise ValueError("Launch windows require a valid exclusive action cap")
    touching = np.any(np.asarray(contacts)[:, [0, 2, 3]] > .5, axis=1)
    vy = np.asarray(states)[:, 3]
    events = []
    for flight in motion["flights"]:
        release = flight["start_state"]
        if not flight["takeoff_candidate"] or release > cap:
            continue
        contact_start = release
        while contact_start > max(0, release - 64) and touching[contact_start - 1]:
            contact_start -= 1
        rise_start = release
        while rise_start > contact_start and vy[rise_start - 1] > rules.upward_speed:
            rise_start -= 1
        context_start = max(contact_start, rise_start - rules.preparation_steps)
        start, end = max(0, context_start - 1), min(cap, flight["peak_state"])
        if end > start:
            x = np.asarray(states)[start:end+1, 33]
            tree_connected = bool(((x >= rules.tree_x_min) & (x <= rules.tree_x_max)).any())
            events.append({"start": start, "end": end, "release_state": release,
                           "supported_rise_start": rise_start, "contact_start": contact_start,
                           "peak_state": flight["peak_state"],
                           "event_group": "tree_connected" if tree_connected else "approach_only",
                           "rise_from_release_m": flight["rise_from_release_m"]})
    return events


DYNAMICS_SPEC = {
    "version": "com_contact_dynamics_v1",
    "body_names": ["Player", "Hub", "Slider", "Handle", "PoleMiddle", "Tip"],
    "velocity_indices": [2, 7, 12, 17, 21, 25],
    "masses": [52., 10., 7., 3., 3., 4.],
    "linear_drag": [.2, .2, .2, 1., 0., 0.],
    "gravity_xy": [0., -30.],
    "minimum_acceleration_m_s2": 5., "saturation_acceleration_m_s2": 150.,
    "passive_impact_speed_retention": .5,
    "preparation_steps": 12, "response_steps": 6,
    "provenance": "native_dynamics_validation_20261002 bodyConfig; gravity checked against archived COM free flight",
}


def contact_dynamics_importance(states, contacts, cap, step_seconds=1/120):
    """A retrospective control-opportunity proxy, not causal action attribution.

    Use all six dynamic bodies' COM velocity so internal hammer motion cancels.
    Free flight predicts v_next = v + g*dt - weighted_drag(v_next)*dt.
    Split the residual into positive speed gain and direction change; passive
    braking alone is not a positive speed gain. Gate static gravity support by
    observed COM acceleration and reject dissipative pot/body-only impacts.
    The core requires geometric contact, with short tapered pre/post context.
    """
    s, c = np.asarray(states, np.float64), np.asarray(contacts, np.float64)
    if (s.ndim != 2 or s.shape[1] != 35 or c.shape != (len(s), 5)
            or not np.isfinite(s).all() or not np.isfinite(c).all()
            or type(cap) is not int or not 0 < cap < len(s)
            or not np.isfinite(step_seconds) or step_seconds <= 0):
        raise ValueError("Dynamics weights need finite aligned 35D states, contacts, cap and timestep")
    spec = DYNAMICS_SPEC
    mass = np.asarray(spec["masses"]); mass /= mass.sum()
    bodies = np.stack([s[:, j:j+2] for j in spec["velocity_indices"]], axis=1)
    velocity = (bodies * mass[None, :, None]).sum(axis=1)
    drag = (bodies * (mass*np.asarray(spec["linear_drag"]))[None, :, None]).sum(axis=1)
    delta = np.diff(velocity, axis=0)
    free = velocity[:-1] + step_seconds*(np.asarray(spec["gravity_xy"])-drag[1:])
    residual = velocity[1:] - free
    speed_next, speed_free = np.linalg.norm(velocity[1:], axis=1), np.linalg.norm(free, axis=1)
    speed_delta = speed_next - speed_free
    speed_gain = np.maximum(speed_delta, 0) / step_seconds
    # ||u-v||^2 = (||u||-||v||)^2 + 2||u||||v||(1-cos(theta)).
    turning = np.sqrt(np.maximum((residual**2).sum(axis=1)-speed_delta**2, 0)) / step_seconds
    observed_acceleration = np.linalg.norm(delta, axis=1) / step_seconds
    strength = np.minimum(np.hypot(speed_gain, turning), observed_acceleration)
    contact_state = np.any(c[:, [0, 2, 3]] > .5, axis=1)
    contact = contact_state[:-1] | contact_state[1:]
    tip = (c[:-1, 0] > .5) | (c[1:, 0] > .5)
    work_proxy = (residual * (velocity[1:]+velocity[:-1])*.5).sum(axis=1)
    passive_impact = (contact & ~tip & (work_proxy < 0)
                      & (speed_next < spec["passive_impact_speed_retention"]*speed_free))
    core = contact & ~passive_impact & (strength >= spec["minimum_acceleration_m_s2"])
    core[cap:] = False
    # Saturation contains isolated solver/contact spikes. Max combines overlapping
    # neighborhoods without counting an event repeatedly through multiple labels.
    priority = np.zeros(len(s)-1, np.float64)
    core_strength = (np.minimum(strength, spec["saturation_acceleration_m_s2"])
                     / spec["saturation_acceleration_m_s2"]) * core
    for t in np.flatnonzero(core):
        for offset in range(-spec["preparation_steps"], spec["response_steps"]+1):
            at = t + offset
            if not 0 <= at < cap: continue
            factor = (1. if offset == 0 else
                      .75 * (1-abs(offset)/(spec["preparation_steps"]+1)) if offset < 0 else
                      .5 * (1-offset/(spec["response_steps"]+1)))
            priority[at] = max(priority[at], float(core_strength[t]*factor))
    events = [{"start": a, "end": b, "context_start": max(0, a-spec["preparation_steps"]),
               "context_end": min(cap, b+spec["response_steps"]),
               "kind": "speed_gain" if speed_gain[a:b].sum() >= turning[a:b].sum() else "turning",
               "speed_gain_impulse_m_s": float(speed_gain[a:b].sum()*step_seconds),
               "turning_impulse_m_s": float(turning[a:b].sum()*step_seconds)} for a,b in runs(core[:cap])]
    air = ~contact_state
    interior_air = np.convolve(air.astype(int), np.ones(13, int), mode="same") == 13 if len(s) >= 13 else np.zeros(len(s), bool)
    audit = (interior_air[:-1] & interior_air[1:])[:cap]
    corrected_acceleration = delta / step_seconds + drag[1:]
    return {"spec": spec, "step_seconds": step_seconds, "events": events,
            "com_velocity": velocity[:cap+1], "non_gravity_acceleration": residual[:cap]/step_seconds,
            "speed_gain_acceleration": speed_gain[:cap], "turning_acceleration": turning[:cap],
            "contact": contact[:cap], "passive_impact": passive_impact[:cap], "core": core[:cap],
            "priority": priority[:cap], "free_flight": (~contact[:cap]) & (priority[:cap] == 0),
            "free_flight_audit_frames": int(audit.sum()),
            "free_flight_corrected_acceleration_quantiles": (
                np.percentile(corrected_acceleration[:cap][audit], [5,50,95], axis=0).tolist() if audit.any() else None)}


def discount_summary(rewards, gamma, step_seconds, starts):
    from .gru_sil import discounted_returns
    rewards = np.asarray(rewards, np.float64)
    if rewards.ndim != 1 or not np.isfinite(rewards).all() or not 0 < gamma <= 1 or step_seconds <= 0:
        raise ValueError("Invalid finite-horizon return inputs")
    # Explicitly no critic bootstrap: comparisons across gamma must not reuse the old value function.
    returns = discounted_returns(rewards, np.zeros(len(rewards), bool), 0., gamma)
    rows = []
    for start in sorted(set(int(s) for s in starts if 0 <= s < len(rewards))):
        hits = np.flatnonzero(rewards[start:] > 0)
        rows.append({"action": start, "finite_tail_return": float(returns[start]),
                     "next_positive_delay_s": None if not len(hits) else float(hits[0] * step_seconds),
                     "next_positive_discount": None if not len(hits) else float(gamma ** int(hits[0]))})
    return {"gamma": gamma, "bootstrap": 0., "truncation": "finite_observed_tail_not_value_target",
            "discount_at_seconds": {str(t): float(gamma ** (t / step_seconds)) for t in (1, 2, 3, 4)},
            "starts": rows}


def entry_state(states, rules=None):
    rules = rules or PhaseRules()
    states = np.asarray(states)
    hits = np.flatnonzero((states[:, 33] >= rules.tree_x_min) & (states[:, 33] <= rules.tree_x_max))
    if not len(hits):
        return None
    t = int(hits[0]); state = states[t]
    return {"state_index": t, "pot_xy": state[33:35].tolist(), "player_velocity_xy": state[2:4].tolist(),
            "hammer_angle": float(state[27]), "tip_relative_xy": (state[23:25]-state[:2]).tolist(),
            "cursor_relative_xy": (state[29:31]-state[:2]).tolist(), "cursor_velocity_xy": state[31:33].tolist(),
            "note": "First entry into fixed x-band; not a matched full physical state"}

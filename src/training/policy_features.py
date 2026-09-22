"""Versioned, shared online/offline PPO features. Geometry is an explicit proxy."""
from collections import deque
import numpy as np
from matplotlib.path import Path

RELATION_NAMES = (
    "cursor_tip_dx", "cursor_tip_dy", "error_along", "error_across",
    "relative_vx", "relative_vy", "velocity_along", "velocity_across",
    "angle_error_sin", "angle_error_cos", "hammer_omega", "rod_length",
    "gravity_along", "gravity_across", "rod_known", "angle_known",
)
EVENT_FIELDS = ("state", "enter", "exit", "since_enter", "since_exit", "known", "enter_seen", "exit_seen")
EVENT_NAMES = tuple(f"{part}_{field}" for part in ("tip", "body", "pot") for field in EVENT_FIELDS)
SURFACE_FIELDS = ("signed_distance", "normal_x", "normal_y", "normal_velocity", "tangent_velocity", "known")
GEOMETRY_NAMES = tuple(f"{part}_{field}" for part in ("tip", "player") for field in SURFACE_FIELDS) + (
    "cursor_margin", "extension_margin", "margins_known")
TREND_FIELDS = ("player_dx", "player_dy", "tip_relative_dx", "tip_relative_dy",
                "rod_turn_sin", "rod_turn_cos", "velocity_dx", "velocity_dy", "rod_turn_known", "known")
TREND_NAMES = tuple(f"lag{lag}_{name}" for lag in (4, 16) for name in TREND_FIELDS)
GROUP_NAMES = {"relations": RELATION_NAMES, "events": EVENT_NAMES,
               "geometry": GEOMETRY_NAMES, "trends": TREND_NAMES}


def experimental_enabled(config):
    return (config.policy_backbone != "transformer" or config.action_segment_steps > 1
            or config.normalize_action_history or any(getattr(config, name) for name in (
                "relation_features", "event_features", "event_branch", "geometry_features", "trend_features")))


def feature_layout(config):
    enabled = (config.relation_features, config.event_features or config.event_branch,
               config.geometry_features, config.trend_features)
    result, start = {}, 0
    for group, active in zip(GROUP_NAMES, enabled):
        if active:
            result[group] = slice(start, start + len(GROUP_NAMES[group]))
            start += len(GROUP_NAMES[group])
    return result, start


def validate_experiments(config):
    if config.policy_backbone not in ("transformer", "gru"):
        raise ValueError("policy_backbone must be transformer or gru")
    if config.event_source not in ("native", "geometry"):
        raise ValueError("event_source must be native or geometry")
    if config.action_segment_steps < 1 or config.recurrent_seq_len < 1:
        raise ValueError("Segment and sequence lengths must be positive")
    if config.ppo_inference_batch_size < 0:
        raise ValueError("ppo_inference_batch_size must be nonnegative")
    scales = np.asarray([config.feature_position_scale, config.feature_velocity_scale, config.feature_angular_scale,
                         config.feature_step_seconds, config.feature_clip, config.geometry_query_radius])
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("Feature scales, time step and query radius must be finite and positive")


def unit(vector):
    length = float(np.linalg.norm(vector))
    return (vector / length if length > 1e-5 else np.zeros(2)), length > 1e-5, length


def relations(state, config):
    s = np.asarray(state, dtype=np.float64)
    rod, rod_known, length = unit(s[23:25] - s[19:21])
    across = np.array([-rod[1], rod[0]])
    error, velocity = s[29:31] - s[23:25], s[31:33] - s[25:27]
    arm, arm_known, _ = unit(s[23:25] - s[5:7])
    target, target_known, _ = unit(s[29:31] - s[5:7])
    angle_known = arm_known and target_known
    sine = arm[0] * target[1] - arm[1] * target[0] if angle_known else 0.
    cosine = arm @ target if angle_known else 0.
    omega = ((s[25:27] - s[21:23]) @ across) / length if rod_known else 0.
    p, v = config.feature_position_scale, config.feature_velocity_scale
    values = [*(error / p), error @ rod / p, error @ across / p,
              *(velocity / v), velocity @ rod / v, velocity @ across / v,
              sine, cosine, omega / config.feature_angular_scale, length / p,
              -rod[1], -across[1], float(rod_known), float(angle_known)]
    return np.clip(values, -config.feature_clip, config.feature_clip).astype(np.float32)


class SurfaceIndex:
    """Cached polygon edges. Signed distance is from a part ORIGIN, not its collider."""
    def __init__(self, polygons):
        self.polygons = [np.array(p, dtype=np.float64, copy=True) for p in polygons if len(p) >= 3]
        self.paths = [Path(p) for p in self.polygons]
        self.boxes = np.array([[*p.min(axis=0), *p.max(axis=0)] for p in self.polygons]).reshape(-1, 4)
        self.edges = []
        for p in self.polygons:
            end = np.roll(p, -1, axis=0)
            direction = end - p
            area = np.sum(p[:, 0] * end[:, 1] - end[:, 0] * p[:, 1])
            normals = np.column_stack([direction[:, 1], -direction[:, 0]]) * (1 if area >= 0 else -1)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            self.edges.append((p, direction, np.sum(direction**2, axis=1), normals))

    def query(self, point, velocity, radius, velocity_scale):
        point = np.asarray(point)
        candidates = np.flatnonzero(np.all(self.boxes[:, :2] <= point + radius, axis=1)
                                    & np.all(self.boxes[:, 2:] >= point - radius, axis=1))
        best, normal, inside = float("inf"), np.zeros(2), False
        for i in candidates:
            p, direction, length2, normals = self.edges[i]
            t = np.clip(np.sum((point-p)*direction, axis=1) / np.maximum(length2, 1e-12), 0, 1)
            distances = np.linalg.norm(point - (p + t[:, None] * direction), axis=1)
            distances[length2 < 1e-12] = np.inf
            edge = int(np.argmin(distances))
            if distances[edge] < best:
                best, normal = float(distances[edge]), normals[edge]
            inside |= self.paths[i].contains_point(point)
        if best > radius:
            return np.zeros(6, dtype=np.float32)
        tangent = np.array([-normal[1], normal[0]])
        return np.array([(-best if inside else best) / radius, *normal,
                         velocity @ normal / velocity_scale, velocity @ tangent / velocity_scale, 1.], np.float32)


class FeatureBuilder:
    """Call once per physical observation, including reset; reuse returned row at bootstrap."""
    def __init__(self, config, surfaces=None):
        validate_experiments(config)
        self.config, self.surfaces = config, surfaces
        self.layout, self.dim = feature_layout(config)
        self.reset()

    def reset(self):
        self.history = deque(maxlen=17)
        self.previous = np.zeros(3, dtype=bool)
        self.previous_known = np.zeros(3, dtype=bool)
        self.age = np.zeros((3, 2), dtype=np.float64)
        self.seen = np.zeros((3, 2), dtype=bool)

    def step(self, state, contact_state=None, contact_known=None):
        cfg = self.config
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (33,) or not np.isfinite(state).all():
            raise ValueError("FeatureBuilder requires a finite 33D state")
        contact = np.asarray(contact_state if contact_state is not None else np.zeros(3), dtype=bool)
        known = np.asarray(contact_known if contact_known is not None else np.zeros(3), dtype=bool)
        changed = known & self.previous_known & (contact != self.previous)
        enter, leave = changed & contact, changed & ~contact
        self.age += cfg.feature_step_seconds
        for k, pulse in enumerate((enter, leave)):
            self.age[pulse, k] = 0.
            self.seen[pulse, k] = True
        self.age[~known] = 0.
        self.seen[~known] = False
        events = np.column_stack([contact & known, enter, leave,
            np.where(self.seen[:, 0], np.minimum(self.age[:, 0] / 2., 1.), 0),
            np.where(self.seen[:, 1], np.minimum(self.age[:, 1] / 2., 1.), 0),
            known, self.seen[:, 0], self.seen[:, 1]]).astype(np.float32).ravel()
        self.previous, self.previous_known = contact.copy(), known.copy()
        groups = {}
        if "relations" in self.layout:
            groups["relations"] = relations(state, cfg)
        if "events" in self.layout:
            groups["events"] = events
        if "geometry" in self.layout:
            surface = []
            for point, velocity in ((state[23:25], state[25:27]), (state[:2], state[2:4])):
                surface.extend(self.surfaces.query(point, velocity, cfg.geometry_query_radius,
                                                  cfg.feature_velocity_scale) if self.surfaces else np.zeros(6))
            surface.extend([1 - np.linalg.norm(state[29:31] - state[:2]) / 3.5,
                            1 - np.linalg.norm(state[23:25] - state[5:7]) / 2.955, 1.])
            groups["geometry"] = np.clip(surface, -cfg.feature_clip, cfg.feature_clip)
        if "trends" in self.layout:
            trends = []
            for lag in (4, 16):
                if len(self.history) < lag:
                    trends.extend(np.zeros(len(TREND_FIELDS)))
                    continue
                old = self.history[-lag]
                displacement_scale = cfg.feature_velocity_scale * cfg.feature_step_seconds * lag
                a, ka, _ = unit(state[23:25] - state[19:21])
                b, kb, _ = unit(old[23:25] - old[19:21])
                turn = (b[0]*a[1] - b[1]*a[0], b @ a) if ka and kb else (0., 0.)
                trends.extend([*((state[:2]-old[:2])/displacement_scale),
                    *((state[23:25]-state[:2]-old[23:25]+old[:2])/displacement_scale),
                    *turn, *((state[2:4]-old[2:4])/cfg.feature_velocity_scale), float(ka and kb), 1.])
            groups["trends"] = np.clip(trends, -cfg.feature_clip, cfg.feature_clip)
        self.history.append(state.copy())
        features = np.concatenate([groups[group] for group in self.layout]).astype(np.float32) if self.dim else np.zeros(0, np.float32)
        return features, bool(changed.any())

    def sequence(self, states, contacts=None, known=None):
        self.reset()
        return np.stack([self.step(s, None if contacts is None else contacts[i],
                                  None if known is None else known[i])[0] for i, s in enumerate(states)])


def feature_names(config):
    layout, _ = feature_layout(config)
    return [f"{group}.{name}" for group in layout for name in GROUP_NAMES[group]]


def scale_report(features, config):
    names = feature_names(config)
    features = np.asarray(features)
    return {name: {"rms": float(np.sqrt(np.mean(features[:, i] ** 2))),
                   "abs_p99": float(np.percentile(np.abs(features[:, i]), 99)),
                   "at_clip_fraction": float(np.mean(np.abs(features[:, i]) >= config.feature_clip))}
            for i, name in enumerate(names)}

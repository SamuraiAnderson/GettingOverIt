"""Player-centered sparse contour tokens and multi-candidate surface relations.

The coarse dualscale CNN keeps the low-resolution map. Fine interaction uses
continuous collision edges inside an 8.64 m window (256-cell precision budget):
air generates no tokens. Static terrain edges / normals / spatial index and
player/tip local shapes are cached. Terrain contours and normals are filled
only for polygons that have entered the current window; unreached terrain is
not precomputed. Each step only rebuilds part poses, nearby queries, and
relative features. Trainable encoder outputs are never cached.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .contact_features import TerrainContactIndex
from .deploy_sampling import (
    player_hub_angle_deg,
    reconstruct_polygon_world,
    rod_angle_deg,
)

if TYPE_CHECKING:
    from .config import TrainConfig

CLASS_TERRAIN = 0
CLASS_PLAYER = 1
CLASS_TIP = 2
N_CONTOUR_CLASS = 3

PART_TIP = 0
PART_POT = 1
PART_BODY = 2
N_RELATION_PART = 3

# class one-hot 3 + mid_xy 2 + normal 2 + tangent 2 + length 1 + known 1
CONTOUR_DIM = 11
# part one-hot 3 + displacement 2 + distance 1 + normal 2 + vn/vt 2 + known 1
RELATION_DIM = 11


def _one_hot(index: int, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    if 0 <= index < n:
        out[index] = 1.0
    return out


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return np.zeros(2, dtype=np.float64)
    return vector / length


def _edge_frames(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 2:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty, empty, np.zeros(0, dtype=np.float64)
    end = np.roll(p, -1, axis=0)
    direction = end - p
    length = np.linalg.norm(direction, axis=1)
    area = float(np.sum(p[:, 0] * end[:, 1] - end[:, 0] * p[:, 1]))
    normals = np.column_stack([direction[:, 1], -direction[:, 0]]) * (1.0 if area >= 0 else -1.0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return p, end, normals, length


def _in_window(points: np.ndarray, center: np.ndarray, half: float) -> np.ndarray:
    delta = np.abs(np.asarray(points, dtype=np.float64) - center)
    return np.all(delta <= half, axis=-1)


def _polygon_to_segment(
    poly: np.ndarray, a: np.ndarray, b: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    poly = np.asarray(poly, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(poly) < 2:
        return float("inf"), a, a
    direction = b - a
    length2 = float(direction @ direction)
    ap = poly - a
    t = np.clip((ap @ direction) / max(length2, 1e-12), 0.0, 1.0)
    on_seg = a + t[:, None] * direction
    dist_v = np.linalg.norm(poly - on_seg, axis=1)
    best_i = int(np.argmin(dist_v))
    best_dist = float(dist_v[best_i])
    best_on_poly = poly[best_i].copy()
    best_on_seg = on_seg[best_i].copy()

    p0 = poly
    p1 = np.roll(poly, -1, axis=0)
    edge = p1 - p0
    edge2 = np.sum(edge * edge, axis=1)
    for point in (a, b):
        w = np.clip(np.sum((point - p0) * edge, axis=1) / np.maximum(edge2, 1e-12), 0.0, 1.0)
        on_poly = p0 + w[:, None] * edge
        dist_e = np.linalg.norm(point - on_poly, axis=1)
        j = int(np.argmin(dist_e))
        if float(dist_e[j]) < best_dist:
            best_dist = float(dist_e[j])
            best_on_poly = on_poly[j].copy()
            best_on_seg = np.asarray(point, dtype=np.float64).copy()
    return best_dist, best_on_poly, best_on_seg


@dataclass
class SparseGeometryCache:
    """Reached-region terrain edges + local part shapes.

    Collider AABBs stay available so the current window can be queried.
    Contours and normals are materialized only for polygons that have entered
    that window; unreached terrain is not precomputed. Rebuild when colliders change.
    """

    index: TerrainContactIndex
    tip_local: np.ndarray | None
    body_local: np.ndarray | None
    pot_local: np.ndarray | None
    _edges: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict,
    )

    @classmethod
    def build(
        cls,
        polygons,
        tip_local: np.ndarray | None,
        body_local: np.ndarray | None,
        pot_local: np.ndarray | None,
    ) -> "SparseGeometryCache":
        index = polygons if isinstance(polygons, TerrainContactIndex) else TerrainContactIndex(polygons or [])
        return cls(index, tip_local, body_local, pot_local)

    def edge_frames(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self._edges.get(idx)
        if cached is None:
            cached = _edge_frames(self.index.polygons[idx])
            self._edges[idx] = cached
        return cached


def empty_sparse_frame(config: "TrainConfig") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = config.sparse_max_contour_tokens
    k = N_RELATION_PART * config.sparse_relation_k
    return (
        np.zeros((n, CONTOUR_DIM), dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros((k, RELATION_DIM), dtype=np.float32),
        np.zeros(k, dtype=np.float32),
    )


def _contour_token(
    cls: int,
    a: np.ndarray,
    b: np.ndarray,
    normal: np.ndarray,
    player: np.ndarray,
    pos_scale: float,
) -> np.ndarray:
    mid = 0.5 * (a + b)
    edge = b - a
    length = float(np.linalg.norm(edge))
    tangent = _unit(edge)
    feat = np.concatenate([
        _one_hot(cls, N_CONTOUR_CLASS),
        ((mid - player) / pos_scale).astype(np.float32),
        np.asarray(normal, dtype=np.float32),
        tangent.astype(np.float32),
        np.array([length / pos_scale, 1.0], dtype=np.float32),
    ])
    return feat


def _append_poly_tokens(
    tokens: list[np.ndarray],
    poly: np.ndarray | None,
    cls: int,
    player: np.ndarray,
    half: float,
    pos_scale: float,
) -> None:
    if poly is None or len(poly) < 2:
        return
    p0, p1, normals, lengths = _edge_frames(poly)
    for i in range(len(p0)):
        if lengths[i] < 1e-12:
            continue
        if not (_in_window(p0[i], player, half) or _in_window(p1[i], player, half)
                or _in_window(0.5 * (p0[i] + p1[i]), player, half)):
            continue
        tokens.append(_contour_token(cls, p0[i], p1[i], normals[i], player, pos_scale))


def _part_world(
    state: np.ndarray,
    cache: SparseGeometryCache,
    config: "TrainConfig",
) -> dict[int, np.ndarray | None]:
    px, py = float(state[0]), float(state[1])
    theta_b = player_hub_angle_deg(state) + config.body_angle_offset
    theta_t = rod_angle_deg(state, config.tip_angle_source) + config.tip_angle_offset
    tip = None
    if cache.tip_local is not None and len(cache.tip_local) >= 3:
        tip = reconstruct_polygon_world(
            (float(state[23]), float(state[24])), theta_t, cache.tip_local, config.tip_scale,
        )
    body = None
    if cache.body_local is not None and len(cache.body_local) >= 3:
        body = reconstruct_polygon_world((px, py), theta_b, cache.body_local, config.body_scale)
    pot = None
    if cache.pot_local is not None and len(cache.pot_local) >= 3:
        pot = reconstruct_polygon_world((px, py), theta_b, cache.pot_local, config.body_scale)
    return {PART_TIP: tip, PART_POT: pot, PART_BODY: body}


def _part_velocity(state: np.ndarray, part: int) -> np.ndarray:
    if part == PART_TIP:
        return np.array([float(state[25]), float(state[26])], dtype=np.float64)
    return np.array([float(state[2]), float(state[3])], dtype=np.float64)


def _relation_token(
    part: int,
    on_part: np.ndarray,
    on_surf: np.ndarray,
    normal: np.ndarray,
    velocity: np.ndarray,
    pos_scale: float,
    vel_scale: float,
) -> np.ndarray:
    delta = on_surf - on_part
    dist = float(np.linalg.norm(delta))
    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
    vn = float(velocity @ normal)
    vt = float(velocity @ tangent)
    feat = np.concatenate([
        _one_hot(part, N_RELATION_PART),
        (delta / pos_scale).astype(np.float32),
        np.array([dist / pos_scale], dtype=np.float32),
        np.asarray(normal, dtype=np.float32),
        np.array([vn / vel_scale, vt / vel_scale], dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    ])
    return feat


def build_sparse_geometry(
    state: np.ndarray,
    cache: SparseGeometryCache | None,
    config: "TrainConfig",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-frame tokens. Shared by rollout and TrajectoryDataset."""
    contour, cmask, relation, rmask = empty_sparse_frame(config)
    if cache is None or not config.sparse_geometry:
        return contour, cmask, relation, rmask

    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 1 or state.shape[0] not in (33, 35) or not np.isfinite(state).all():
        raise ValueError("build_sparse_geometry requires a finite 33D or 35D state")

    player = np.array([float(state[0]), float(state[1])], dtype=np.float64)
    half = 0.5 * float(config.sparse_window_meters)
    pos_scale = float(config.feature_position_scale)
    vel_scale = float(config.feature_velocity_scale)
    clip = float(config.feature_clip)
    max_n = int(config.sparse_max_contour_tokens)
    k_keep = int(config.sparse_relation_k)

    parts = _part_world(state, cache, config)
    player_tokens: list[np.ndarray] = []
    _append_poly_tokens(player_tokens, parts[PART_BODY], CLASS_PLAYER, player, half, pos_scale)
    _append_poly_tokens(player_tokens, parts[PART_POT], CLASS_PLAYER, player, half, pos_scale)
    tip_tokens: list[np.ndarray] = []
    _append_poly_tokens(tip_tokens, parts[PART_TIP], CLASS_TIP, player, half, pos_scale)

    terrain_scored: list[tuple[float, np.ndarray]] = []
    nearby = cache.index.nearby(player - half, player + half, 0.0)
    id_lookup = {id(p): i for i, p in enumerate(cache.index.polygons)}
    for poly in nearby:
        idx = id_lookup.get(id(poly))
        if idx is None:
            for i, cached in enumerate(cache.index.polygons):
                if cached is poly or (cached.shape == poly.shape and np.array_equal(cached, poly)):
                    idx = i
                    break
        if idx is None:
            continue
        p0, p1, normals, lengths = cache.edge_frames(idx)
        for i in range(len(p0)):
            if lengths[i] < 1e-12:
                continue
            mid = 0.5 * (p0[i] + p1[i])
            if not (_in_window(p0[i], player, half) or _in_window(p1[i], player, half)
                    or _in_window(mid, player, half)):
                continue
            token = _contour_token(CLASS_TERRAIN, p0[i], p1[i], normals[i], player, pos_scale)
            terrain_scored.append((float(np.linalg.norm(mid - player)), token))
    terrain_scored.sort(key=lambda item: item[0])

    reserved = player_tokens + tip_tokens
    remaining = max(0, max_n - len(reserved))
    selected = reserved + [item[1] for item in terrain_scored[:remaining]]
    n_use = min(len(selected), max_n)
    if n_use:
        stacked = np.clip(np.stack(selected[:n_use]), -clip, clip)
        contour[:n_use] = stacked
        cmask[:n_use] = 1.0

    for part in (PART_TIP, PART_POT, PART_BODY):
        poly = parts[part]
        slot0 = part * k_keep
        if poly is None or len(poly) < 3:
            continue
        velocity = _part_velocity(state, part)
        candidates: list[tuple[float, np.ndarray]] = []
        for env_poly in cache.index.nearby(poly.min(axis=0), poly.max(axis=0), half):
            idx = id_lookup.get(id(env_poly))
            if idx is None:
                for i, cached in enumerate(cache.index.polygons):
                    if cached is env_poly or (
                        cached.shape == env_poly.shape and np.array_equal(cached, env_poly)
                    ):
                        idx = i
                        break
            if idx is None:
                continue
            p0, p1, normals, lengths = cache.edge_frames(idx)
            for i in range(len(p0)):
                if lengths[i] < 1e-12:
                    continue
                mid = 0.5 * (p0[i] + p1[i])
                if not (_in_window(mid, player, half) or _in_window(p0[i], player, half)
                        or _in_window(p1[i], player, half)):
                    continue
                dist, on_part, on_surf = _polygon_to_segment(poly, p0[i], p1[i])
                if not np.isfinite(dist) or dist > 2.0 * half:
                    continue
                token = _relation_token(
                    part, on_part, on_surf, normals[i], velocity, pos_scale, vel_scale,
                )
                candidates.append((dist, token))
        candidates.sort(key=lambda item: item[0])
        for j, (_dist, token) in enumerate(candidates[:k_keep]):
            relation[slot0 + j] = np.clip(token, -clip, clip)
            rmask[slot0 + j] = 1.0

    return contour, cmask, relation, rmask


def parse_window_batch(batch):
    """Unpack 5-tuple, 9-tuple, or those plus SIL weight / value target."""
    dyn, pat, act, vm, target = batch[0], batch[1], batch[2], batch[3], batch[4]
    idx = 5
    contour = cmask = relation = rmask = None
    if len(batch) >= 9 and getattr(batch[5], "ndim", 0) >= 3:
        contour, cmask, relation, rmask = batch[5], batch[6], batch[7], batch[8]
        idx = 9
    weight = batch[idx] if len(batch) > idx else None
    vtarget = batch[idx + 1] if len(batch) > idx + 1 else None
    return dyn, pat, act, vm, target, contour, cmask, relation, rmask, weight, vtarget

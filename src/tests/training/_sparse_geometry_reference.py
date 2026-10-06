"""Frozen scalar selection/distance oracle from d3063bf (2026-10-04).

Test-only: do not optimize this reference alongside the production builder.
Unchanged pose/token helpers are shared; filtering, ranking and distance
arithmetic deliberately retain the pre-optimization implementation.
"""
from __future__ import annotations

import numpy as np

from training.sparse_geometry import (
    CLASS_PLAYER, CLASS_TERRAIN, CLASS_TIP, PART_BODY, PART_POT, PART_TIP,
    SparseGeometryCache, _append_poly_tokens, _contour_token, _in_window,
    _part_velocity, _part_world, _relation_token, empty_sparse_frame,
)


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



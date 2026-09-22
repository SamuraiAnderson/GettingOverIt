"""Ordered, coarse reference corridors. No video processing in the training loop.

Progress is paid only for new ground reached while both ends of the step are in
the corridor. Outside progress is consumed without payment, so re-entry cannot
cash in a shortcut. Missing video intervals remain separate, unrewarded gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def _simplify(points: np.ndarray, tolerance: float) -> list[int]:
    """Iterative Ramer–Douglas–Peucker; preserves order and segment endpoints."""
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        v = points[b] - points[a]
        denom = float(v @ v)
        t = np.clip((points[a + 1:b] - points[a]) @ v / max(denom, 1e-20), 0, 1)
        delta = points[a + 1:b] - (points[a] + t[:, None] * v)
        distances = np.linalg.norm(delta, axis=1)
        k = int(distances.argmax())
        if distances[k] > tolerance:
            m = a + 1 + k
            keep.add(m)
            stack.extend([(a, m), (m, b)])
    return sorted(keep)


@dataclass(frozen=True)
class Projection:
    progress: float
    distance: float
    confidence: float


class CorridorSegment:
    def __init__(self, points: np.ndarray, weights: np.ndarray):
        self.points = points
        self.vectors = np.diff(points, axis=0)
        self.lengths = np.linalg.norm(self.vectors, axis=1)
        cumulative = np.r_[0.0, np.cumsum(self.lengths)]
        self.starts = cumulative[:-1]
        self.length = float(cumulative[-1])
        self.weights = weights

    def project(self, point: np.ndarray, previous: float | None = None,
                window: float | None = None) -> Projection:
        # Clip actual line pieces to an arc-length window, not just vertex IDs.
        low, high = 0.0, self.length
        if previous is not None and window is not None:
            low, high = max(0.0, previous - window), min(self.length, previous + window)
        valid = (self.starts + self.lengths >= low - 1e-9) & (self.starts <= high + 1e-9)
        starts, lengths = self.starts[valid], self.lengths[valid]
        vectors, origins = self.vectors[valid], self.points[:-1][valid]
        t = np.sum((point - origins) * vectors, axis=1) / lengths**2
        t = np.clip(t, np.maximum(0.0, (low - starts) / lengths),
                    np.minimum(1.0, (high - starts) / lengths))
        distance = np.linalg.norm(point - (origins + t[:, None] * vectors), axis=1)
        progress = starts + t * lengths
        # At crossings prefer the branch nearest the current progress.
        near = np.flatnonzero(distance <= distance.min() + 1e-9)
        target = 0.0 if previous is None else previous
        k = near[np.argmin(np.abs(progress[near] - target))]
        return Projection(float(progress[k]), float(distance[k]), float(self.weights[valid][k]))


class ReferenceCorridor:
    def __init__(self, segments: list[CorridorSegment], config, metadata: dict | None = None):
        self.segments = segments
        self.config = config
        self.metadata = metadata or {}
        if not segments:
            raise ValueError("Reference trajectory has no nonzero continuous segments")
        for name in ("reference_corridor_radius", "reference_projection_window", "reference_max_step"):
            if not np.isfinite(getattr(config, name)) or getattr(config, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("reference_progress_weight", "reference_deviation_weight", "reference_simplify_tolerance"):
            if not np.isfinite(getattr(config, name)) or getattr(config, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= config.reference_interpolated_weight <= 1:
            raise ValueError("reference_interpolated_weight must be in [0, 1]")

    @classmethod
    def from_file(cls, path: str | Path, config) -> ReferenceCorridor:
        data = Path(path).read_bytes()
        payload = json.loads(data)
        segments = []
        raw_count = 0
        tolerance = config.reference_simplify_tolerance
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("reference_simplify_tolerance must be finite and nonnegative")
        for rows in payload["segments"]:
            if not rows:
                continue
            points = np.array([[p["x"], p["y"]] for p in rows], dtype=np.float64)
            if not np.isfinite(points).all():
                raise ValueError("Reference positions must be finite; split missing data into segments")
            # Fail rather than silently connect an explicitly missing frame.
            if all("frame" in p for p in rows) and any(
                b["frame"] != a["frame"] + 1 for a, b in zip(rows, rows[1:])
            ):
                raise ValueError("Non-contiguous frames inside a reference segment")
            raw_count += len(points)
            if len(points) < 2:
                continue
            sources = [p.get("source", "observed") for p in rows]
            if any(s not in ("observed", "interpolated") for s in sources):
                raise ValueError("Unknown reference point source")
            weights = np.array([1.0 if s == "observed" else config.reference_interpolated_weight
                                for s in sources])
            ids = _simplify(points, tolerance)
            # A stationary run can reduce to identical endpoints.
            keep = [ids[0]]
            for i in ids[1:]:
                if np.linalg.norm(points[i] - points[keep[-1]]) > 1e-8:
                    keep.append(i)
            if len(keep) < 2:
                continue
            edge_weights = np.array([weights[a:b + 1].min() for a, b in zip(keep, keep[1:])])
            segments.append(CorridorSegment(points[keep], edge_weights))
        return cls(segments, config, {
            "path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest(),
            "status": payload.get("status"), "training_ready": payload.get("training_ready"),
            "coordinate_warning": payload.get("coordinate_warning"),
            "raw_points": raw_count, "points": sum(len(s.points) for s in segments),
            "segments": len(segments), "length": sum(s.length for s in segments),
        })

    def tracker(self, position: np.ndarray) -> CorridorTracker:
        return CorridorTracker(self, position)

    def guidance_maps(self, shape: tuple[int, int], min_gx: int, min_gy: int,
                      resolution: float) -> list[np.ndarray]:
        """Cache one ordered corridor field per continuous segment in the existing grid.

        This is observation context only, not a second reward. Each map is zero
        beyond three corridor radii and increases along the reference direction.
        """
        maps = []
        radius = self.config.reference_corridor_radius
        for segment in self.segments:
            arr = np.zeros(shape, dtype=np.float32)
            low = np.floor((segment.points.min(axis=0) - 3 * radius) / resolution).astype(int)
            high = np.ceil((segment.points.max(axis=0) + 3 * radius) / resolution).astype(int)
            x0, x1 = max(0, low[0] - min_gx), min(shape[1], high[0] - min_gx + 1)
            y0, y1 = max(0, low[1] - min_gy), min(shape[0], high[1] - min_gy + 1)
            if x1 <= x0 or y1 <= y0:
                maps.append(arr)
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            positions = np.column_stack(((xx.ravel() + min_gx + .5) * resolution,
                                         (yy.ravel() + min_gy + .5) * resolution))
            values = np.zeros(len(positions), dtype=np.float32)
            for start in range(0, len(positions), 1024):
                delta = positions[start:start + 1024, None, :] - segment.points[None, :-1, :]
                t = np.clip(np.sum(delta * segment.vectors, axis=2) / segment.lengths**2, 0, 1)
                distance = np.linalg.norm(delta - t[:, :, None] * segment.vectors, axis=2)
                k = distance.argmin(axis=1)
                row = np.arange(len(k))
                d = distance[row, k]
                progress = segment.starts[k] + t[row, k] * segment.lengths[k]
                values[start:start + len(k)] = np.where(
                    d <= 3 * radius, np.exp(-.5 * (d / radius)**2) * (1 + progress / segment.length), 0,
                )
            arr[y0:y1, x0:x1] = values.reshape(y1 - y0, x1 - x0)
            maps.append(arr)
        return maps


def replay_corridor_fields(
    route: "ReferenceCorridor", raw_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """从状态序列确定性重放 CorridorTracker，恢复逐 step 走廊字段。

    返回 (corridor_progress, guidance_index)：
      - corridor_progress: (T,) float32，与 actions 对齐；progress[t] 是动作 t 把
        state[t] 推进到 state[t+1] 所获得的 credited 进度（CorridorStep.progress）。
        cumsum(corridor_progress)[t] = 第 t 步后的 total_progress（评估归档 trace 的
        credited_progress_m 口径）。
      - guidance_index: (T+1,) int8，与 states 对齐；guidance[t] 是处理完 state[t]
        之后 tracker 的引导段索引（patch 构建即用该时刻的值），无引导（末段之后
        的 gap）记 -1。

    tracker 投影窗口只依赖前序状态，重放与 rollout 在线记录逐点一致；
    与 local_evaluation 的 trace 重建共用同一 tracker 语义。
    """
    tracker = route.tracker(raw_states[0, :2])
    T = len(raw_states) - 1
    progress = np.zeros(T, dtype=np.float32)
    guidance = np.full(T + 1, -1, dtype=np.int8)
    if tracker.guidance_index is not None:
        guidance[0] = tracker.guidance_index
    for t in range(T):
        result = tracker.step(raw_states[t + 1, :2])
        progress[t] = result.progress
        if tracker.guidance_index is not None:
            guidance[t + 1] = tracker.guidance_index
    return progress, guidance


def corridor_cap_index(corridor_progress: np.ndarray) -> int:
    """credited 累计首次封顶的动作索引：最小 k 使 cumsum(progress)[k] ≈ total。

    即"最后一个有净推进的动作"；全零轨迹返回 0。剪枝/BC 截断/行为描述子共用此口径。
    容差取相对 1e-6：float32 逐步累加与一次性求和的截断顺序不同，绝对误差随 T 增长。
    """
    total = float(corridor_progress.sum(dtype=np.float64))
    if total <= 0.0:
        return 0
    cumulative = np.cumsum(corridor_progress, dtype=np.float64)
    return int(np.searchsorted(cumulative, total * (1 - 1e-6), side="left"))


@dataclass(frozen=True)
class CorridorStep:
    reward: float = 0.0
    progress: float = 0.0
    deviation_penalty: float = 0.0
    inside: bool = False
    gap: bool = False


class CorridorTracker:
    """One tracker per agent/episode; reset never pays for the deployment location."""
    def __init__(self, route: ReferenceCorridor, position: np.ndarray):
        self.route = route
        self.previous_position = np.asarray(position[:2], dtype=np.float64).copy()
        if self.previous_position.shape != (2,) or not np.isfinite(self.previous_position).all():
            raise ValueError("Agent position must contain two finite coordinates")
        projections = [s.project(self.previous_position) for s in route.segments]
        self.segment_index = min(range(len(projections)), key=lambda i: projections[i].distance)
        p = projections[self.segment_index]
        self.progress = self.best_progress = p.progress
        self.previous_inside = p.distance <= route.config.reference_corridor_radius
        self.waiting = False
        self.total_progress = 0.0
        self.last_step = CorridorStep()

    @property
    def guidance_index(self) -> int | None:
        idx = self.segment_index + int(self.waiting)
        return idx if idx < len(self.route.segments) else None

    def step(self, position: np.ndarray) -> CorridorStep:
        position = np.asarray(position[:2], dtype=np.float64)
        if not np.isfinite(position).all():
            raise ValueError("Agent position must be finite")
        cfg = self.route.config
        distance_moved = float(np.linalg.norm(position - self.previous_position))
        self.previous_position = position.copy()
        segment = self.route.segments[self.segment_index]
        if self.waiting:
            # Only enter the next segment at its start; no invented connecting edge.
            nxt = self.segment_index + 1
            if (nxt < len(self.route.segments)
                    and np.linalg.norm(position - self.route.segments[nxt].points[0]) <= cfg.reference_corridor_radius):
                self.segment_index = nxt
                p = self.route.segments[nxt].project(position, 0.0, cfg.reference_corridor_radius)
                self.progress = self.best_progress = p.progress
                self.previous_inside = p.distance <= cfg.reference_corridor_radius
                self.waiting = False
            self.last_step = CorridorStep(gap=True)
            return self.last_step

        # The displacement bound prevents stationary projections creeping forward
        # through a crossing. A large teleport re-baselines without giving reward.
        jumped = distance_moved > cfg.reference_max_step
        p = segment.project(position, None if jumped else self.progress,
                            None if jumped else min(cfg.reference_projection_window, 2 * distance_moved))
        inside = p.distance <= cfg.reference_corridor_radius
        gain = max(0.0, p.progress - self.best_progress)
        credited = gain * p.confidence if inside and self.previous_inside and not jumped else 0.0
        self.best_progress = max(self.best_progress, p.progress)
        self.progress = p.progress
        self.previous_inside = inside
        self.total_progress += credited
        penalty = cfg.reference_deviation_weight * min(1.0, max(0.0, p.distance / cfg.reference_corridor_radius - 1))
        self.last_step = CorridorStep(cfg.reference_progress_weight * credited - penalty,
                                      credited, penalty, inside)
        # A coarse guide must not require hitting the exact visual endpoint.
        # Enter the unobserved gap once the agent reaches the endpoint region;
        # the untraversed remainder is not paid out.
        if (inside and segment.length - p.progress <= cfg.reference_corridor_radius
                and np.linalg.norm(position - segment.points[-1]) <= cfg.reference_corridor_radius):
            self.waiting = True
        return self.last_step

"""
可着陆线段上的空投点几何采样（纯数学，无游戏依赖）。

输出坐标的 y 均为「表面 y + drop_height」，与 drop_point_physics 的 teleport 输入约定一致。
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def precompute_segment_arcs(
    segments: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """预计算各线段弧长和累积弧长，用于加权随机采样或弧上插值。"""
    arc_lengths: list[float] = []
    cum_lengths: list[np.ndarray] = []
    for seg in segments:
        d = np.diff(seg, axis=0)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
        arc_lengths.append(cum[-1])
        cum_lengths.append(cum)
    return np.array(arc_lengths), cum_lengths


def sample_from_segments(
    segments: list[np.ndarray],
    arc_lengths: np.ndarray,
    cum_lengths: list[np.ndarray],
    n: int,
    drop_height: float,
    rng: np.random.Generator,
) -> list[list[float]]:
    """
    从表面线段上随机采样 n 个投放点。

    按弧长加权选择线段，在线段上均匀采样位置，y 加上 drop_height。
    每次调用产生完全随机的位置，覆盖所有可着陆表面。
    """
    total = arc_lengths.sum()
    if total < 1e-6 or len(segments) == 0:
        return []

    weights = arc_lengths / total
    seg_indices = rng.choice(len(segments), size=n, p=weights)

    points: list[list[float]] = []
    for si in seg_indices:
        seg = segments[si]
        cum = cum_lengths[si]
        t = rng.uniform(0.0, cum[-1])
        x = float(np.interp(t, cum, seg[:, 0]))
        y = float(np.interp(t, cum, seg[:, 1]))
        points.append([x, y + drop_height])
    return points


def sample_from_fixed_pool(
    pool: list[list[float]],
    n: int,
    rng: np.random.Generator,
) -> list[list[float]]:
    """从固定点列表无放回随机抽取至多 n 个点（坐标已含 drop_height）。"""
    if not pool or n <= 0:
        return []
    k = min(n, len(pool))
    indices = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in indices]


def _allocate_drops(weights: list[float], n_drops: int) -> list[int]:
    """最大余数法：按权重比例公平分配投放点数。"""
    total = sum(weights)
    if total < 1e-6:
        return [0] * len(weights)
    exact = [n_drops * w / total for w in weights]
    floors = [int(f) for f in exact]
    remainders = [(exact[i] - floors[i], i) for i in range(len(exact))]
    deficit = n_drops - sum(floors)
    remainders.sort(reverse=True)
    for _, idx in remainders[:deficit]:
        floors[idx] += 1
    return floors


def compute_drop_points(
    segments: list[np.ndarray],
    n_drops: int,
    drop_height: float,
    height_decay: float = 0.0,
) -> np.ndarray:
    """
    在所有表面线段上分配 n_drops 个投放点（最大余数法）。

    权重 = arc_length × (1 - height_decay × normalized_height)
      height_decay=0  → 纯弧长（默认）
      height_decay=0.5 → 中等偏向低处
      height_decay=0.9 → 强烈偏向低处

    返回 shape (n_drops, 2) 的坐标数组 [x, y]（y 已加上 drop_height）。
    """
    if not segments:
        logger.error("无有效表面线段")
        return np.zeros((0, 2))

    seg_lengths: list[float] = []
    seg_avg_y: list[float] = []
    for seg in segments:
        d = np.diff(seg, axis=0)
        seg_lengths.append(float(np.sum(np.hypot(d[:, 0], d[:, 1]))))
        seg_avg_y.append(float(seg[:, 1].mean()))
    total_length = sum(seg_lengths)

    y_min = min(seg_avg_y)
    y_max = max(seg_avg_y)
    y_range = y_max - y_min if y_max > y_min else 1.0

    weights: list[float] = []
    for slen, avg_y in zip(seg_lengths, seg_avg_y):
        norm_h = (avg_y - y_min) / y_range
        w = slen * (1.0 - height_decay * norm_h)
        weights.append(max(w, 0.0))

    logger.info("共 %d 条表面线段，总弧长 %.1f, height_decay=%.2f",
                len(segments), total_length, height_decay)
    for i, (seg, slen, w) in enumerate(zip(segments, seg_lengths, weights)):
        logger.info("  seg[%d]: %d pts, len=%.1f, w=%.2f, x=[%.1f..%.1f], y=[%.1f..%.1f]",
                     i, len(seg), slen, w,
                     seg[:, 0].min(), seg[:, 0].max(),
                     seg[:, 1].min(), seg[:, 1].max())

    if sum(weights) < 1e-6:
        return np.zeros((0, 2))

    allocation = _allocate_drops(weights, n_drops)

    all_drops: list[np.ndarray] = []
    for i, (seg, n) in enumerate(zip(segments, allocation)):
        if n <= 0:
            continue

        d = np.diff(seg, axis=0)
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
        targets = np.linspace(cum[0], cum[-1], n + 2)[1:-1]
        dx = np.interp(targets, cum, seg[:, 0])
        dy = np.interp(targets, cum, seg[:, 1])
        all_drops.append(np.column_stack([dx, dy + drop_height]))
        logger.info("  seg[%d] 分配 %d 个投放点", i, n)

    if not all_drops:
        return np.zeros((0, 2))
    return np.vstack(all_drops)


def build_all_deploy_candidates(
    segments: list[np.ndarray],
    drop_height: float,
) -> np.ndarray:
    """
    所有可着陆线段折线顶点，y 加 drop_height 后的世界坐标投放点。
    shape (N, 2)。
    """
    if not segments:
        return np.zeros((0, 2), dtype=np.float64)
    blocks: list[np.ndarray] = []
    dh = float(drop_height)
    for seg in segments:
        s = np.asarray(seg, dtype=np.float64)
        if len(s) == 0:
            continue
        blocks.append(np.column_stack([s[:, 0], s[:, 1] + dh]))
    if not blocks:
        return np.zeros((0, 2), dtype=np.float64)
    raw = np.vstack(blocks)
    # 相邻线段共顶点会重复；按首次出现顺序去重，减少 all 模式重复探测
    seen: set[tuple[float, float]] = set()
    uniq_rows: list[np.ndarray] = []
    for row in raw:
        key = (float(row[0]), float(row[1]))
        if key not in seen:
            seen.add(key)
            uniq_rows.append(np.asarray(row, dtype=np.float64))
    if not uniq_rows:
        return np.zeros((0, 2), dtype=np.float64)
    return np.stack(uniq_rows, axis=0)


def sample_deploy_candidates_stratified(
    segments: list[np.ndarray],
    drop_height: float,
    max_samples: int,
    rng: np.random.Generator,
    *,
    height_decay: float = 0.0,
) -> np.ndarray:
    """
    若线段顶点总数 <= max_samples，返回 build_all_deploy_candidates；
    否则按弧长权重（与 compute_drop_points 一致）在弧上采 max_samples 个点。

    rng 预留用于后续随机弧长采样扩展；当前 compute_drop_points 为确定性 linspace。
    """
    _ = rng
    total_v = sum(len(s) for s in segments)
    if total_v <= max_samples:
        return build_all_deploy_candidates(segments, drop_height)
    return compute_drop_points(
        segments, max_samples, drop_height, height_decay,
    )

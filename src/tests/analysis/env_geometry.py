"""
环境几何原语：加载 environment.json、抽取可着陆台面、计算上升点间距。

纯离线，不依赖游戏进程 / player_contour。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_EXCLUDED_COLLIDERS = {"Snake"}
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ColliderStats:
    """碰撞体几何统计。"""
    diag_lengths: np.ndarray   # 包围盒对角线长度 (m)
    edge_lengths: np.ndarray   # 边长 (m)
    bbox_widths: np.ndarray
    bbox_heights: np.ndarray
    centers: np.ndarray        # (N, 2)


@dataclass
class LedgeGeometry:
    """可着陆台面几何。"""
    ledge_cells: np.ndarray          # (M, 2) 世界坐标 [x, y]
    ledge_segments: list[np.ndarray]  # 水平聚类后的台面段，各 (K, 2)
    segment_centers: np.ndarray       # (S, 2)
    # 相邻「向上」台面对：从较低到较高
    gaps_horizontal: np.ndarray       # 水平间距 (m)
    gaps_vertical: np.ndarray         # 竖直落差 (m)
    gaps_euclidean: np.ndarray        # 欧氏距离 (m)


def default_env_path() -> Path:
    return _REPO_ROOT / "checkpoints" / "environment.json"


def load_environment(env_path: Path | None = None) -> dict:
    path = env_path or default_env_path()
    if not path.exists():
        raise FileNotFoundError(f"environment.json 不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("已加载 %s: %d colliders", path, len(data.get("colliders", [])))
    return data


def extract_polygons(env_data: dict) -> list[np.ndarray]:
    """提取环境多边形顶点列表，跳过 Snake 等排除项。"""
    polys: list[np.ndarray] = []
    skipped = 0
    for collider in env_data.get("colliders", []):
        if collider.get("name") in _EXCLUDED_COLLIDERS:
            skipped += 1
            continue
        for path in collider.get("paths", []):
            pts = np.asarray(path, dtype=np.float64)
            if len(pts) >= 3:
                polys.append(pts)
    if skipped:
        logger.info("已过滤 %d 个排除碰撞体: %s", skipped, _EXCLUDED_COLLIDERS)
    return polys


def collider_stats(env_data: dict) -> ColliderStats:
    """统计每个 collider path 的包围盒尺寸与边长。"""
    diags, edges, widths, heights, centers = [], [], [], [], []
    for collider in env_data.get("colliders", []):
        if collider.get("name") in _EXCLUDED_COLLIDERS:
            continue
        for path in collider.get("paths", []):
            pts = np.asarray(path, dtype=np.float64)
            if len(pts) < 3:
                continue
            w = float(pts[:, 0].max() - pts[:, 0].min())
            h = float(pts[:, 1].max() - pts[:, 1].min())
            widths.append(w)
            heights.append(h)
            diags.append(float(np.hypot(w, h)))
            centers.append([float(pts[:, 0].mean()), float(pts[:, 1].mean())])
            d = np.diff(np.vstack([pts, pts[0]]), axis=0)
            edges.extend(np.hypot(d[:, 0], d[:, 1]).tolist())
    return ColliderStats(
        diag_lengths=np.asarray(diags, dtype=np.float64),
        edge_lengths=np.asarray(edges, dtype=np.float64),
        bbox_widths=np.asarray(widths, dtype=np.float64),
        bbox_heights=np.asarray(heights, dtype=np.float64),
        centers=np.asarray(centers, dtype=np.float64),
    )


def extract_ledge_cells(
    traversable: np.ndarray,
    min_gx: int,
    min_gy: int,
    resolution: float = 1.0,
) -> np.ndarray:
    """
    抽取可着陆台面格：可通行，且正下方一格为固体（不可通行）或越界。

    返回世界坐标 (M, 2)，取格子中心。
    """
    h, w = traversable.shape
    ys, xs = np.where(traversable)
    ledges: list[list[float]] = []
    for gy, gx in zip(ys, xs):
        below_solid = (gy == 0) or (not traversable[gy - 1, gx])
        if below_solid:
            wx = (min_gx + gx + 0.5) * resolution
            wy = (min_gy + gy + 0.5) * resolution
            ledges.append([wx, wy])
    if not ledges:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(ledges, dtype=np.float64)


def cluster_ledge_segments(
    ledge_cells: np.ndarray,
    resolution: float = 1.0,
    y_tol: float = 1.5,
    x_gap_max: float = 2.5,
) -> list[np.ndarray]:
    """
    将台面格按近似等高 + 水平连通聚类为台面段。

    y_tol: 同段允许的 y 差；x_gap_max: 同段相邻格最大 x 间隔。
    """
    if len(ledge_cells) == 0:
        return []

    order = np.lexsort((ledge_cells[:, 0], ledge_cells[:, 1]))
    pts = ledge_cells[order]

    segments: list[np.ndarray] = []
    cur: list[np.ndarray] = [pts[0]]
    for i in range(1, len(pts)):
        prev, p = cur[-1], pts[i]
        same_band = abs(p[1] - prev[1]) <= y_tol
        close_x = (p[0] - prev[0]) <= x_gap_max
        # 也允许与段平均高度接近
        band_ok = abs(p[1] - float(np.mean([c[1] for c in cur]))) <= y_tol
        if same_band and close_x and band_ok:
            cur.append(p)
        else:
            segments.append(np.asarray(cur, dtype=np.float64))
            cur = [p]
    segments.append(np.asarray(cur, dtype=np.float64))
    return segments


def compute_ascent_gaps(
    segments: list[np.ndarray],
    max_vertical: float = 80.0,
    max_horizontal: float = 60.0,
    k_nearest: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对每个台面段，找上方最近的若干更高台面，记录水平/竖直/欧氏间距。

    过滤过大的跳跃（非相邻阶段），避免把山顶与山脚配对。
    """
    if len(segments) < 2:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty

    centers = np.asarray(
        [[s[:, 0].mean(), s[:, 1].mean()] for s in segments],
        dtype=np.float64,
    )
    gaps_h, gaps_v, gaps_e = [], [], []
    for i, c in enumerate(centers):
        # 候选：更高的台面
        higher = np.where(centers[:, 1] > c[1] + 0.5)[0]
        if len(higher) == 0:
            continue
        dx = centers[higher, 0] - c[0]
        dy = centers[higher, 1] - c[1]
        dist = np.hypot(dx, dy)
        # 过滤不合理跳跃
        mask = (dy <= max_vertical) & (np.abs(dx) <= max_horizontal)
        if not np.any(mask):
            continue
        cand = higher[mask]
        d = dist[mask]
        order = np.argsort(d)[:k_nearest]
        for j in cand[order]:
            dx_ij = float(centers[j, 0] - c[0])
            dy_ij = float(centers[j, 1] - c[1])
            gaps_h.append(abs(dx_ij))
            gaps_v.append(dy_ij)
            gaps_e.append(float(np.hypot(dx_ij, dy_ij)))

    return (
        np.asarray(gaps_h, dtype=np.float64),
        np.asarray(gaps_v, dtype=np.float64),
        np.asarray(gaps_e, dtype=np.float64),
    )


def build_ledge_geometry(
    traversable: np.ndarray,
    min_gx: int,
    min_gy: int,
    resolution: float = 1.0,
) -> LedgeGeometry:
    """端到端：台面格 → 段 → 上升间距。"""
    cells = extract_ledge_cells(traversable, min_gx, min_gy, resolution)
    segs = cluster_ledge_segments(cells, resolution=resolution)
    centers = (
        np.asarray([[s[:, 0].mean(), s[:, 1].mean()] for s in segs], dtype=np.float64)
        if segs else np.zeros((0, 2), dtype=np.float64)
    )
    gh, gv, ge = compute_ascent_gaps(segs)
    logger.info(
        "台面: %d 格 → %d 段; 上升对 %d (euclid p50=%.1f p90=%.1f)",
        len(cells), len(segs), len(ge),
        float(np.median(ge)) if len(ge) else 0.0,
        float(np.percentile(ge, 90)) if len(ge) else 0.0,
    )
    return LedgeGeometry(
        ledge_cells=cells,
        ledge_segments=segs,
        segment_centers=centers,
        gaps_horizontal=gh,
        gaps_vertical=gv,
        gaps_euclidean=ge,
    )


def nearest_collider_gaps(centers: np.ndarray, k: int = 1) -> np.ndarray:
    """每个碰撞体中心到最近其他碰撞体的距离。"""
    if len(centers) < 2:
        return np.zeros(0, dtype=np.float64)
    # 朴素 O(N^2)，N≈278 可接受
    dmat = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(dmat, np.inf)
    nearest = np.partition(dmat, kth=0, axis=1)[:, 0]
    return nearest.astype(np.float64)

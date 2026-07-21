"""
可着陆线段上的空投点几何采样（纯数学，无游戏依赖）。

输出坐标的 y 均为「表面 y + drop_height」，与 drop_point_physics 的 teleport 输入约定一致。
"""

from __future__ import annotations

import logging
from collections import namedtuple

import numpy as np

logger = logging.getLogger(__name__)

# 33D 观测索引（见 PlayerState.ToFloatArray / entrypoints.md）
TIP_X_INDEX = 23
TIP_Y_INDEX = 24
POLE_X_INDEX = 19
POLE_Y_INDEX = 20
HAMMER_ANGLE_INDEX = 27

# 锤头世界多边形重建的标定参数（近似）：
#   world_poly = tip_center + scale * R(rod_dir + offset) @ local_poly
#   rod_dir: angle_source="pole" 用 pole→tip；"hub" 用 hammerAngle(hub→tip)
# 经 verify_tip_reconstruction 采样校验，pole / -90° / 1.0 使锤头颈部指向 pole、贴合地形。
TipReconConfig = namedtuple(
    "TipReconConfig", ["local_poly", "angle_source", "angle_offset_deg", "scale"]
)


def make_tip_recon_config(
    local_poly,
    angle_source: str = "pole",
    angle_offset_deg: float = -90.0,
    scale: float = 1.0,
) -> "TipReconConfig":
    """构造锤头重建配置；local_poly 为 tip 局部多边形 (N,2)。"""
    return TipReconConfig(
        np.asarray(local_poly, dtype=np.float64), angle_source, angle_offset_deg, scale
    )


def rod_angle_deg(obs_row: np.ndarray, angle_source: str) -> float:
    """锤杆世界朝向（度）。pole: pole→tip；hub: hammerAngle(hub→tip)。"""
    if angle_source == "hub":
        return float(obs_row[HAMMER_ANGLE_INDEX])
    dx = float(obs_row[TIP_X_INDEX]) - float(obs_row[POLE_X_INDEX])
    dy = float(obs_row[TIP_Y_INDEX]) - float(obs_row[POLE_Y_INDEX])
    return float(np.degrees(np.arctan2(dy, dx)))


# 身体部件位置索引：player(0,1), hub(5,6)
PLAYER_X_INDEX = 0
PLAYER_Y_INDEX = 1
HUB_X_INDEX = 5
HUB_Y_INDEX = 6


def extract_tip_local(player_data: dict | None) -> np.ndarray | None:
    """从 player_contour.json 取 tip 局部多边形 (N, 2)（自身坐标系）。"""
    if not player_data:
        return None
    tip = player_data.get("parts", {}).get("tip")
    if tip and tip.get("paths"):
        return np.asarray(tip["paths"][0], dtype=np.float64)
    return None


def extract_body_local(player_data: dict | None) -> list[np.ndarray] | None:
    """从 player_contour.json 取 body∪pot 局部多边形列表（player 共享坐标系）。

    body（躯干，+y 在上）与 pot（锅，在下）合成完整身体碰撞轮廓；缺失则返回 None。
    """
    if not player_data:
        return None
    parts = player_data.get("parts", {})
    out: list[np.ndarray] = []
    for name in ("body", "pot"):
        pr = parts.get(name, {}).get("paths")
        if pr:
            out.append(np.asarray(pr[0], dtype=np.float64))
    return out or None


def player_hub_angle_deg(obs_row: np.ndarray) -> float:
    """身体世界朝向基准（度）：player→hub 方向。

    body/pot 轮廓在 player 局部系中定义（+y=躯干朝上），其世界朝向纯由身体旋转决定；
    33D 无显式 body 绝对角，故用 player→hub 方向作代理（配合 body_angle_offset 标定）。
    """
    dx = float(obs_row[HUB_X_INDEX]) - float(obs_row[PLAYER_X_INDEX])
    dy = float(obs_row[HUB_Y_INDEX]) - float(obs_row[PLAYER_Y_INDEX])
    return float(np.degrees(np.arctan2(dy, dx)))


def reconstruct_polygon_world(
    center: tuple[float, float] | np.ndarray,
    angle_deg: float,
    local_poly: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """把局部坐标多边形按 (中心, 旋转角, 缩放) 摆到世界坐标。

    world_v = center + scale * R(angle) @ local_v，用于从 state 的部件位置 + 朝向
    重建碰撞体世界轮廓（近似，朝向/缩放由调用方给定）。返回 (N, 2)。
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    local = np.asarray(local_poly, dtype=np.float64) * float(scale)
    return local @ rot.T + np.asarray(center, dtype=np.float64)


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """射线法判断点 (x, y) 是否落在单个多边形内（顶点顺序无关，边界不保证）。"""
    pts = np.asarray(poly, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        yi, yj = pts[i, 1], pts[j, 1]
        if (yi > y) != (yj > y):
            x_cross = pts[i, 0] + (y - yi) * (pts[j, 0] - pts[i, 0]) / (yj - yi)
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def point_in_any_polygon(
    x: float, y: float, polygons: list[np.ndarray],
) -> bool:
    """点是否落在任意一个实心多边形内（与 L7 把每条 path 当实心处理一致）。"""
    for poly in polygons:
        if point_in_polygon(x, y, poly):
            return True
    return False


def tip_in_obstacle(obs_row: np.ndarray, solid_polygons, tip_cfg=None) -> bool:
    """settle 后锤头是否嵌进障碍物 → 该落点无效。

    判据：锤头中心点 (tip_x, tip_y) 落在任一实体多边形内即判无效。
    tip_cfg 参数保留仅为向后兼容，不再参与判定。
    solid_polygons 为 None / 空时跳过检查（保持旧行为）。
    """
    if solid_polygons is None or len(solid_polygons) == 0:
        return False

    tip_x = float(obs_row[TIP_X_INDEX])
    tip_y = float(obs_row[TIP_Y_INDEX])
    return point_in_any_polygon(tip_x, tip_y, solid_polygons)


# ── 人工涂抹排除区（deploy exclusion zones） ────────────────────────
# 用户在地图上用「圆形笔刷」涂抹出的排除区域，落点若落入任一圆内即被剔除。
# 存储于本项目 checkpoints/deploy_exclusion_zones.json（随仓库版本化，与游戏目录解耦）：
#   {"zones": [[cx, cy, r], ...], "brush_radius": <上次笔刷半径>}
# 坐标系与投放点一致（世界 x / y，y 已含 drop_height）。
EXCLUSION_ZONES_FILENAME = "deploy_exclusion_zones.json"


def default_exclusion_zones_path():
    """项目内排除区文件的默认路径：<repo_root>/checkpoints/<文件名>。"""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "checkpoints" / EXCLUSION_ZONES_FILENAME


# ── 候选点集（唯一文件，L7 → train） ────────────────────────────────
# L7 物理筛选产出的全量稳定候选点，作为向训练传递候选集的**唯一文件**，
# 随仓库版本化存于本项目 checkpoints/。训练侧读取它并做高度等简单过滤后投放。
# 坐标为 [x, y]，y 已含 drop_height（与投放点一致）。
CANDIDATE_POINTS_FILENAME = "drop_points_all_stable.json"


def default_candidate_points_path():
    """项目内候选点集文件的默认路径：<repo_root>/checkpoints/<文件名>。"""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "checkpoints" / CANDIDATE_POINTS_FILENAME


def load_candidate_points(path) -> list[list[float]]:
    """读取候选点集文件的 drop_points 列表，返回 [[x, y], ...]；缺失/空返回 []。"""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("候选点集读取失败 %s: %s", p, exc)
        return []
    pts = data.get("drop_points", []) if isinstance(data, dict) else data
    return [[float(x), float(y)] for x, y in pts] if pts else []


def filter_points_by_height(points, y_max: float | None = None,
                            y_min: float | None = None) -> list:
    """按点的 y 值做高度过滤：保留 y_min <= y <= y_max 的点（阈值 None 时该侧不限）。

    points: [x, y] 序列（y 与投放点一致，已含 drop_height）。返回保留点的 list。
    """
    if points is None:
        return []
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return []
    keep = np.ones(len(pts), dtype=bool)
    if y_max is not None:
        keep &= pts[:, 1] <= float(y_max)
    if y_min is not None:
        keep &= pts[:, 1] >= float(y_min)
    return [[float(p[0]), float(p[1])] for p in pts[keep]]


def load_exclusion_zones(path) -> np.ndarray | None:
    """读取排除区圆列表，返回 (M, 3) 的 [cx, cy, r] 数组；文件缺失/空则返回 None。"""
    import json
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("排除区文件读取失败 %s: %s", p, exc)
        return None
    zones = data.get("zones") if isinstance(data, dict) else data
    if not zones:
        return None
    arr = np.asarray(zones, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        logger.warning("排除区数据格式异常（应为 (M,3) 的 [cx,cy,r]）: %s", p)
        return None
    return arr


def point_in_exclusion(x: float, y: float, zones: np.ndarray | None) -> bool:
    """点 (x, y) 是否落入任一排除圆内（含边界）。zones 为 None/空时恒为 False。"""
    if zones is None or len(zones) == 0:
        return False
    dx = zones[:, 0] - float(x)
    dy = zones[:, 1] - float(y)
    return bool(np.any(dx * dx + dy * dy <= zones[:, 2] ** 2))


def filter_points_by_exclusion(points, zones: np.ndarray | None) -> list:
    """剔除落入排除区的投放点，保留其余点（保持输入顺序）。

    points: 可迭代的 [x, y] 序列（list / (N,2) 数组均可）。
    返回保留点的 list（元素为长度 2 的 list）。
    """
    if points is None:
        return []
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) == 0:
        return [list(map(float, p)) for p in points] if points is not None else []
    if zones is None or len(zones) == 0:
        return [[float(p[0]), float(p[1])] for p in pts]
    # 向量化：对每个点求到各圆心的平方距离，任一 <= r^2 即排除
    diff_x = pts[:, 0][:, None] - zones[None, :, 0]
    diff_y = pts[:, 1][:, None] - zones[None, :, 1]
    dist2 = diff_x * diff_x + diff_y * diff_y
    inside_any = np.any(dist2 <= (zones[None, :, 2] ** 2), axis=1)
    keep = pts[~inside_any]
    return [[float(p[0]), float(p[1])] for p in keep]


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

"""接触信号复刻——精确对齐游戏原生 HammerCollisions + PlayerSounds 判断（IL 反编译）。

## 复刻的 5 个物理量

均来自 Getting Over It 原生 Assembly-CSharp.dll 的 IL 逻辑，通过 Python 端多边形几何
在离线/rollout 时刻近似计算，不需要修改 C# 端 wire 协议。

| Python 派生维度   | 游戏原生对应                                             | 计算方法                          |
|-------------------|----------------------------------------------------------|-----------------------------------|
| `tip_contact`     | `HammerCollisions.collisionPoints` 有 Terrain layer entry | tip 世界多边形 ↔ Terrain 最小距离 < ε |
| `tip_grip`        | `HammerCollisions.slide == false`（静态摩擦/钩住）          | tip_contact ∧ v_tip<0.3 ∧ 连续帧 tip 移动 < 0.03m |
| `body_contact`    | `PlayerSounds` 里 `rb.GetPoint(contact).y > 0.4` 分支     | body（躯干，局部 +y 在上）世界多边形 ↔ Terrain 距离 < ε |
| `pot_contact`     | `PlayerSounds.isTouching`（OnCollisionStay Terrain，pot 稳态） | pot（锅底）世界多边形 ↔ Terrain 距离 < ε |
| `fall_distance`   | `PlayerSounds.Update` 的 `CircleCast(pot, 0.5, down)`     | player 中心向下几何射线到 Terrain 距离，clip[0,20]/20 |

## 为什么 body / pot 分开算

游戏 `PlayerSounds.OnCollisionEnter2D` 里存在**局部 y 分层**：
`rb.GetPoint(contacts[0].point).y > 0.4` 用于挑出"接触点打在躯干上部"分支（触发
`hurtThreshold=8` 时的 pain sound）；`y ≤ 0.4` 侧则对应 pot 底部触地 = 稳定支撑。
两者物理语义相反（body 触地 → 撞头 / 受伤；pot 触地 → 支点），合并后信号被稀释，
故此 Python 侧按 `player_contour.json` 里的 `parts.body` 与 `parts.pot` 分别重建后
各自输出 0/1。

## 关键常量（对齐游戏 IL 硬编码，勿改）

- `CONTACT_EPSILON = 0.03`：来自 `HammerCollisions.moveThreshold`（IL 构造函数 line 1110）
  游戏里是「接触点移动阈值」，用作距离判据的自然量级。同时也用作 grip 的位移阈值。
- `GRIP_VEL_THRESHOLD = 0.3`：`OnCollisionStay2D` 内硬编码 tip 速度上界（IL line 818）
- `FALL_CAST_RADIUS = 0.5`：`PlayerSounds.Update` 的 CircleCast 半径（IL line 162）
- `FALL_MAX_DISTANCE = 20.0`：归一化上界（游戏原生 100，20m 足够表征"离地程度"，超出即为"深度悬空"）

## 与已有代码的边界

- 多边形重建复用 `deploy_sampling.reconstruct_polygon_world / rod_angle_deg /
  player_hub_angle_deg`（与 `dataset.build_patch` 的 contact 通道使用**同一套**几何重建，
  确保 patch 与派生接触信号语义一致）。
- 环境多边形与 `traversable_mask.npz` 同源（`load_colliders` → `extract_polygons`）。

## 性能

- 单帧单 agent：~50-200 μs（tip 16 顶点 × 局部相关地形 ~10 个 × 平均 20 顶点）
- 与 rollout 主循环相比可忽略；不做空间索引（bbox 预筛已足够）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from matplotlib.path import Path as MplPath

from .deploy_sampling import (
    PLAYER_X_INDEX,
    PLAYER_Y_INDEX,
    TIP_X_INDEX,
    TIP_Y_INDEX,
    player_hub_angle_deg,
    reconstruct_polygon_world,
    rod_angle_deg,
)

# 33D 内的 tip 速度索引（VELOCITY_INDICES 的最后一对，除 cursor 外）
TIP_VX_INDEX = 25
TIP_VY_INDEX = 26

# ── 游戏 IL 硬编码常量（勿改）──
CONTACT_EPSILON = 0.03
GRIP_VEL_THRESHOLD = 0.3
FALL_CAST_RADIUS = 0.5
FALL_MAX_DISTANCE = 20.0

CONTACT_DIM = 5  # [tip_contact, tip_grip, body_contact, pot_contact, fall_distance_norm]


# ═══════════════════════════════════════════════════════════════════════
# 从 player_contour.json 分离 body 与 pot 局部多边形
# ═══════════════════════════════════════════════════════════════════════


def split_body_pot_local(
    player_data: dict | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """按名称分别拉取 body（躯干）与 pot（锅底）局部多边形，共用 player 局部坐标系。

    与 `deploy_sampling.extract_body_local` 返回 `[body, pot]` 合并 list 不同——本函数
    显式按名称索引，避免"某部件缺失"时顺序错位；供 `compute_contact_row` 独立计算
    body / pot 各自的 Terrain 触地信号（对应游戏 IL 里局部 y > 0.4 vs ≤ 0.4 的分支）。
    """
    if not player_data:
        return None, None
    parts = player_data.get("parts", {})
    body_paths = parts.get("body", {}).get("paths")
    pot_paths = parts.get("pot", {}).get("paths")
    body_arr = (
        np.asarray(body_paths[0], dtype=np.float64) if body_paths else None
    )
    pot_arr = np.asarray(pot_paths[0], dtype=np.float64) if pot_paths else None
    return body_arr, pot_arr


# ═══════════════════════════════════════════════════════════════════════
# 几何工具（point / segment / polygon）
# ═══════════════════════════════════════════════════════════════════════


def _point_segment_distance(
    px: float, py: float, ax: float, ay: float, bx: float, by: float,
) -> float:
    """点 (px, py) 到线段 [(ax,ay),(bx,by)] 的欧氏距离。退化线段（长度≈0）返回点到端点距离。"""
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return float(np.hypot(px - ax, py - ay))
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return float(np.hypot(px - cx, py - cy))


def _polygon_edge_distance(pts: np.ndarray, poly: np.ndarray) -> float:
    """点集 pts (N,2) 到 poly 边界的最小距离（不管点是否在多边形内）。

    向量化实现：对多边形每条边一次计算所有 pts 的最近距离，取全局最小。
    poly 假定闭合形式：最后一条边为 poly[-1] → poly[0]。
    """
    if len(poly) < 2:
        return float("inf")
    a = poly
    b = np.roll(poly, -1, axis=0)  # (M, 2)，b[i] = poly[(i+1) % M]
    edge = b - a  # (M, 2)
    denom = (edge * edge).sum(axis=1)  # (M,)
    # ap[i, m] = pts[i] - a[m]，广播成 (N, M, 2)
    ap = pts[:, None, :] - a[None, :, :]
    # t[i, m] = dot(ap[i,m], edge[m]) / denom[m]
    t_num = (ap * edge[None, :, :]).sum(axis=2)  # (N, M)
    t = np.where(denom[None, :] < 1e-12, 0.0, t_num / np.maximum(denom[None, :], 1e-12))
    t = np.clip(t, 0.0, 1.0)  # (N, M)
    closest = a[None, :, :] + t[:, :, None] * edge[None, :, :]  # (N, M, 2)
    diff = pts[:, None, :] - closest
    dist_sq = (diff * diff).sum(axis=2)  # (N, M)
    return float(np.sqrt(dist_sq.min()))


def _polygons_may_overlap(poly_a: np.ndarray, poly_b: np.ndarray, margin: float = 0.0) -> bool:
    """bbox 快速判：两多边形 AABB 是否重叠（含 margin 扩张）。"""
    a_min_x, a_min_y = poly_a[:, 0].min() - margin, poly_a[:, 1].min() - margin
    a_max_x, a_max_y = poly_a[:, 0].max() + margin, poly_a[:, 1].max() + margin
    b_min_x, b_min_y = poly_b[:, 0].min(), poly_b[:, 1].min()
    b_max_x, b_max_y = poly_b[:, 0].max(), poly_b[:, 1].max()
    return not (a_max_x < b_min_x or b_max_x < a_min_x or a_max_y < b_min_y or b_max_y < a_min_y)


def polygon_polygon_min_distance(
    poly_a: np.ndarray, poly_b: np.ndarray,
) -> float:
    """两个多边形间最小距离；相交/穿透返回 0。

    相交判定：任一多边形的任一顶点落入对方内部（射线法）。
    非相交：两组顶点到对方边的最近距离取最小。
    对凸/非凸均正确（边缘越过对方而顶点不入的病态非凸情况不会出现在
    Getting Over It 地形上，且距离 < ε 的门槛会被顶点入判先命中）。
    """
    poly_a = np.asarray(poly_a, dtype=np.float64)
    poly_b = np.asarray(poly_b, dtype=np.float64)
    if len(poly_a) < 3 or len(poly_b) < 3:
        return float("inf")

    if not _polygons_may_overlap(poly_a, poly_b, margin=CONTACT_EPSILON * 2):
        # 两多边形 bbox 都相隔较远，直接顶点 vs 边（不必判交）
        d1 = _polygon_edge_distance(poly_a, poly_b)
        d2 = _polygon_edge_distance(poly_b, poly_a)
        return min(d1, d2)

    path_a = MplPath(poly_a)
    path_b = MplPath(poly_b)
    if path_a.contains_points(poly_b).any() or path_b.contains_points(poly_a).any():
        return 0.0

    d1 = _polygon_edge_distance(poly_a, poly_b)
    d2 = _polygon_edge_distance(poly_b, poly_a)
    return min(d1, d2)


def raycast_down_to_polygons(
    ox: float, oy: float, polygons: Sequence[np.ndarray], max_dist: float,
) -> float:
    """从 (ox, oy) 向下（−y）发单点射线到多边形边的最近距离；无命中返回 max_dist。

    对每个多边形，扫描其边：若边跨过 x = ox（两端点 x 分居两侧或恰在 ox 上），
    线性插值得到该边在 x=ox 处的 y 值；取 y ≤ oy（严格下方）中的最大值 = 最近的地面。
    这是游戏 `PlayerSounds.CircleCast` 单点近似（半径 0.5 的"厚度"由 grip/contact 一并
    捕捉，此处只表征"离地多高"）。
    """
    best_y = -float("inf")
    for poly in polygons:
        p = np.asarray(poly, dtype=np.float64)
        n = len(p)
        if n < 2:
            continue
        if p[:, 0].max() < ox or p[:, 0].min() > ox:
            continue
        if p[:, 1].min() > oy:
            continue
        for i in range(n):
            ax, ay = p[i, 0], p[i, 1]
            bx, by = p[(i + 1) % n, 0], p[(i + 1) % n, 1]
            if (ax - ox) * (bx - ox) > 1e-12:
                continue
            if abs(bx - ax) < 1e-9:
                y_edge = max(ay, by)
            else:
                t = (ox - ax) / (bx - ax)
                y_edge = ay + t * (by - ay)
            if y_edge <= oy and y_edge > best_y:
                best_y = y_edge
    if best_y == -float("inf"):
        return max_dist
    return max(0.0, oy - best_y)


# ═══════════════════════════════════════════════════════════════════════
# 接触信号提取
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ContactPrevState:
    """跨帧状态：grip 需要"tip 在两帧间移动 < GRIP_MOVE_THRESHOLD"。

    tip_center: 上一帧 tip 世界中心（(x, y)）。首帧为 None。
    tip_contact_prev: 上一帧 tip_contact 值（0/1）。首帧为 0。
    """
    tip_center: tuple[float, float] | None = None
    tip_contact_prev: float = 0.0


@dataclass
class ContactConfig:
    """接触信号计算的几何参数，值默认与 `dataset.build_patch` 的 contact 通道一致。"""
    tip_angle_source: str = "pole"
    tip_angle_offset_deg: float = -90.0
    tip_scale: float = 1.0
    body_angle_offset_deg: float = -90.0
    body_scale: float = 1.0


def compute_contact_row(
    obs_row: np.ndarray,
    prev: ContactPrevState,
    *,
    tip_local: np.ndarray,
    body_local: np.ndarray | None,
    pot_local: np.ndarray | None,
    env_polygons: Sequence[np.ndarray],
    cfg: ContactConfig = ContactConfig(),
) -> tuple[np.ndarray, ContactPrevState]:
    """单帧计算 5 维接触信号，返回 (contact[5], new_prev)。

    输入：
      obs_row: 33D 原始观测（单帧）
      prev: 上一帧状态（首帧传空 ContactPrevState()）
      tip_local: tip 局部多边形 (N,2)，来自 extract_tip_local(player_data)
      body_local: 躯干（局部 +y 在上）局部多边形 (N,2)，来自 split_body_pot_local
      pot_local: 锅（局部 y ≤ 0.4）局部多边形 (N,2)，来自 split_body_pot_local
      env_polygons: 地形多边形列表（Terrain layer 的等价物）

    返回 contact = [tip_contact, tip_grip, body_contact, pot_contact, fall_distance_norm]，
    float32 (5,)。body / pot 部件缺失时对应维度置 0（等价于"该部件不参与接触判断"）。
    """
    obs_row = np.asarray(obs_row, dtype=np.float32)

    tip_center = (float(obs_row[TIP_X_INDEX]), float(obs_row[TIP_Y_INDEX]))
    tip_theta = rod_angle_deg(obs_row, cfg.tip_angle_source) + cfg.tip_angle_offset_deg
    tip_world = reconstruct_polygon_world(tip_center, tip_theta, tip_local, cfg.tip_scale)

    tip_dist = _min_distance_to_env(tip_world, env_polygons)
    tip_contact = 1.0 if tip_dist < CONTACT_EPSILON else 0.0

    tip_vx = float(obs_row[TIP_VX_INDEX])
    tip_vy = float(obs_row[TIP_VY_INDEX])
    tip_speed = float(np.hypot(tip_vx, tip_vy))
    tip_grip = 0.0
    if (
        tip_contact > 0.5
        and prev.tip_contact_prev > 0.5
        and tip_speed < GRIP_VEL_THRESHOLD
        and prev.tip_center is not None
    ):
        move = float(np.hypot(
            tip_center[0] - prev.tip_center[0],
            tip_center[1] - prev.tip_center[1],
        ))
        if move < CONTACT_EPSILON:
            tip_grip = 1.0

    player_center = (float(obs_row[PLAYER_X_INDEX]), float(obs_row[PLAYER_Y_INDEX]))
    body_theta = player_hub_angle_deg(obs_row) + cfg.body_angle_offset_deg

    body_contact = 0.0
    if body_local is not None and len(body_local) >= 3:
        body_world = reconstruct_polygon_world(
            player_center, body_theta, body_local, cfg.body_scale,
        )
        body_dist = _min_distance_to_env(body_world, env_polygons)
        if body_dist < CONTACT_EPSILON:
            body_contact = 1.0

    pot_contact = 0.0
    if pot_local is not None and len(pot_local) >= 3:
        pot_world = reconstruct_polygon_world(
            player_center, body_theta, pot_local, cfg.body_scale,
        )
        pot_dist = _min_distance_to_env(pot_world, env_polygons)
        if pot_dist < CONTACT_EPSILON:
            pot_contact = 1.0

    fall_dist = raycast_down_to_polygons(
        player_center[0], player_center[1], env_polygons, FALL_MAX_DISTANCE,
    )
    fall_dist_norm = min(fall_dist, FALL_MAX_DISTANCE) / FALL_MAX_DISTANCE

    contact = np.array(
        [tip_contact, tip_grip, body_contact, pot_contact, fall_dist_norm],
        dtype=np.float32,
    )
    new_prev = ContactPrevState(
        tip_center=tip_center,
        tip_contact_prev=tip_contact,
    )
    return contact, new_prev


def _min_distance_to_env(
    query_poly: np.ndarray, env_polygons: Sequence[np.ndarray],
) -> float:
    """query 多边形到 env 多边形集合的最小距离；bbox 预筛加速。"""
    if len(query_poly) < 3 or len(env_polygons) == 0:
        return float("inf")
    q = np.asarray(query_poly, dtype=np.float64)
    q_min_x, q_min_y = q[:, 0].min(), q[:, 1].min()
    q_max_x, q_max_y = q[:, 0].max(), q[:, 1].max()
    margin = max(CONTACT_EPSILON * 4, 0.2)  # 至少留 0.2m 预筛半径
    best = float("inf")
    for poly in env_polygons:
        p = np.asarray(poly, dtype=np.float64)
        if len(p) < 3:
            continue
        if (
            p[:, 0].max() < q_min_x - margin
            or p[:, 0].min() > q_max_x + margin
            or p[:, 1].max() < q_min_y - margin
            or p[:, 1].min() > q_max_y + margin
        ):
            continue
        d = polygon_polygon_min_distance(q, p)
        if d < best:
            best = d
            if best <= 0.0:
                return 0.0
    return best


def compute_contact_sequence(
    raw_states: np.ndarray,
    *,
    tip_local: np.ndarray,
    body_local: np.ndarray | None,
    pot_local: np.ndarray | None,
    env_polygons: Sequence[np.ndarray],
    cfg: ContactConfig = ContactConfig(),
) -> np.ndarray:
    """从 (T+1, 33) 原始状态序列离线一次性计算 (T+1, 5) 接触序列。

    首帧 prev 传空 ContactPrevState()；随时间顺序滚动更新 tip 中心/接触历史，
    保证与 rollout 在线计算的信号语义一致（都基于"上帧 → 本帧"两帧窗口）。
    """
    T_plus_1 = len(raw_states)
    out = np.zeros((T_plus_1, CONTACT_DIM), dtype=np.float32)
    prev = ContactPrevState()
    for i in range(T_plus_1):
        row, prev = compute_contact_row(
            raw_states[i], prev,
            tip_local=tip_local, body_local=body_local, pot_local=pot_local,
            env_polygons=env_polygons, cfg=cfg,
        )
        out[i] = row
    return out

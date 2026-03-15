"""
轨迹评分 + 攀爬效率图。

两层评分体系:
- base_score: 登顶高度 + 步数效率（不依赖效率图）
- score_trajectory: base_score + 路标奖励（依赖效率图，望远镜求和 → O(1)）

ClimbingEfficiencyMap: 地形感知的效率图
- 价值传播（跨迭代）: full_score 沿轨迹路径向出发点传播
- 地形感知扩散（每轮内）: 信号沿可通行区域扩散，不穿墙
- 后缀预计算: O(T) 更新效率图
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.path import Path as MplPath

if TYPE_CHECKING:
    from .config import TrainConfig


def base_score(states: np.ndarray, config: TrainConfig) -> float:
    """
    基础评分 — 爬得高 + 爬得快。

    states: (T+1, 29)，一段轨迹（完整或片段）。
    不依赖任何外部状态，可独立计算。
    """
    y = states[:, 1]
    start_y = y[0]
    summit = float(y.max() - start_y)
    peak_step = int(y.argmax()) + 1
    climb_speed = summit / peak_step
    return summit + config.efficiency_weight * climb_speed


def is_water_trajectory(states: np.ndarray, config: TrainConfig) -> bool:
    """检测轨迹是否落入水中。"""
    return float(states[:, 1].min()) < config.water_y_threshold


def score_trajectory(
    states: np.ndarray,
    efficiency_map: ClimbingEfficiencyMap | None,
    config: TrainConfig,
) -> float:
    """
    完整评分 = base_score + waypoint_weight * waypoint_reward。

    states: (T+1, 29) — 原始 29 维状态。
    waypoint_reward 利用望远镜求和简化为 E(终点) - E(起点)，O(1)。
    落水轨迹直接返回 -inf，确保在任何筛选中被淘汰。
    """
    if is_water_trajectory(states, config):
        return float("-inf")

    bs = base_score(states, config)

    waypoint_reward = 0.0
    if efficiency_map is not None:
        eff_end = efficiency_map.query(states[-1, 0], states[-1, 1])
        eff_start = efficiency_map.query(states[0, 0], states[0, 1])
        waypoint_reward = eff_end - eff_start

    return bs + config.waypoint_weight * waypoint_reward


# ---------------------------------------------------------------------------
# 攀爬效率图
# ---------------------------------------------------------------------------

def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """射线法判断点是否在多边形内。"""
    return MplPath(polygon).contains_point((x, y))


class ClimbingEfficiencyMap:
    """
    地形感知效率图 — 价值传播 + 路径扩散。

    将 2D 空间划分为网格，通过两重机制扩散信号：
    1. 价值传播（跨迭代）: 用 full_score 构建，沿轨迹路径传播
    2. 地形感知扩散（每轮内）: 信号沿可通行区域扩散，不穿墙
    """

    def __init__(
        self,
        polygons: list[np.ndarray],
        grid_resolution: float = 1.0,
        diffusion_iterations: int = 10,
        diffusion_alpha: float = 0.3,
    ):
        self.resolution = grid_resolution
        self.diffusion_iters = diffusion_iterations
        self.diffusion_alpha = diffusion_alpha

        self.traversable, self._min_gx, self._min_gy = (
            self._build_traversable_mask(polygons)
        )
        h, w = self.traversable.shape
        self._max_arr = np.full((h, w), -np.inf, dtype=np.float64)
        self._arr: np.ndarray | None = None

    # -- public helpers --

    def get_arr(self) -> np.ndarray | None:
        """返回扩散后的效率图数组（供 Dataset 引用）。"""
        return self._arr

    def copy(self) -> ClimbingEfficiencyMap:
        """深拷贝，用于保存 prev_eff_map。"""
        return copy.deepcopy(self)

    # -- 可通行性掩码 --

    def _build_traversable_mask(
        self, polygons: list[np.ndarray]
    ) -> tuple[np.ndarray, int, int]:
        """点在多边形内 = 固体 = 不可通行。"""
        all_pts = np.concatenate(polygons)
        x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
        y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())

        min_gx = int(np.floor(x_min / self.resolution)) - 1
        min_gy = int(np.floor(y_min / self.resolution)) - 1
        max_gx = int(np.ceil(x_max / self.resolution)) + 1
        max_gy = int(np.ceil(y_max / self.resolution)) + 1

        w = max_gx - min_gx + 1
        h = max_gy - min_gy + 1
        mask = np.ones((h, w), dtype=bool)

        mpl_paths = [MplPath(poly) for poly in polygons]
        for gy_idx in range(h):
            for gx_idx in range(w):
                cx = (min_gx + gx_idx + 0.5) * self.resolution
                cy = (min_gy + gy_idx + 0.5) * self.resolution
                if any(p.contains_point((cx, cy)) for p in mpl_paths):
                    mask[gy_idx, gx_idx] = False

        return mask, min_gx, min_gy

    # -- 增量更新 --

    def update(
        self,
        new_trajectories: list,
        config: TrainConfig,
        prev_eff_map: ClimbingEfficiencyMap | None = None,
    ) -> None:
        """
        增量更新效率图。
        后缀预计算将每条轨迹从 O(T^2) 降至 O(T)。
        """
        h, w = self.traversable.shape

        for traj in new_trajectories:
            states = traj.raw_states  # (T+1, 29)
            T = len(states)

            y = states[:, 1]

            # 后缀最大值: suffix_max[t] = max(y[t:])
            suffix_max = np.empty(T, dtype=np.float64)
            suffix_max[-1] = y[-1]
            for i in range(T - 2, -1, -1):
                suffix_max[i] = max(y[i], suffix_max[i + 1])

            # 后缀 argmax: suffix_argmax[t] = argmax(y[t:]) 相对于 t 的偏移
            suffix_argmax = np.zeros(T, dtype=np.int64)
            best_idx = T - 1
            for i in range(T - 2, -1, -1):
                if y[i] >= y[best_idx]:
                    best_idx = i
                suffix_argmax[i] = best_idx - i

            for t in range(T - 1):
                gx = int(states[t, 0] / self.resolution) - self._min_gx
                gy = int(states[t, 1] / self.resolution) - self._min_gy
                if not (0 <= gx < w and 0 <= gy < h):
                    continue

                summit = suffix_max[t] - y[t]
                peak_step = suffix_argmax[t] + 1
                climb_speed = summit / peak_step
                bs = summit + config.efficiency_weight * climb_speed

                wp = 0.0
                if prev_eff_map is not None:
                    eff_end = prev_eff_map.query(states[-1, 0], states[-1, 1])
                    eff_start = prev_eff_map.query(states[t, 0], states[t, 1])
                    wp = eff_end - eff_start

                fs = bs + config.waypoint_weight * wp
                if fs > self._max_arr[gy, gx]:
                    self._max_arr[gy, gx] = fs

        raw_arr = np.where(self._max_arr > -np.inf, self._max_arr, 0.0)

        self._arr = self._diffuse(raw_arr)

    # -- 地形感知扩散 --

    def _diffuse(self, arr: np.ndarray) -> np.ndarray:
        """迭代扩散: 4 邻域平均，只通过可通行格子传播。"""
        mask = self.traversable.astype(np.float64)
        arr = np.pad(arr, 1, mode="constant", constant_values=0)
        mask = np.pad(mask, 1, mode="constant", constant_values=0)

        for _ in range(self.diffusion_iters):
            n_sum = np.zeros_like(arr)
            n_cnt = np.zeros_like(arr)
            for shift, axis in [(-1, 0), (1, 0), (-1, 1), (1, 1)]:
                shifted_arr = np.roll(arr, shift, axis=axis)
                shifted_mask = np.roll(mask, shift, axis=axis)
                n_sum += shifted_arr * shifted_mask
                n_cnt += shifted_mask
            n_avg = np.where(n_cnt > 0, n_sum / n_cnt, 0.0)
            arr = np.where(
                mask,
                (1 - self.diffusion_alpha) * arr + self.diffusion_alpha * n_avg,
                0.0,
            )

        return arr[1:-1, 1:-1]

    # -- 查询 --

    def query(self, x: float, y: float) -> float:
        """查询扩散后的效率值。"""
        if self._arr is None:
            return 0.0
        gx = int(x / self.resolution) - self._min_gx
        gy = int(y / self.resolution) - self._min_gy
        h, w = self._arr.shape
        if 0 <= gx < w and 0 <= gy < h:
            return float(self._arr[gy, gx])
        return 0.0

    def query_gradient(self, x: float, y: float) -> tuple[float, float]:
        """
        查询效率图梯度方向 (dE/dx, dE/dy)，归一化。
        仅供调试/可视化使用，不作为模型输入。
        """
        if self._arr is None:
            return (0.0, 0.0)
        gx = int(x / self.resolution) - self._min_gx
        gy = int(y / self.resolution) - self._min_gy
        h, w = self._arr.shape
        if not (1 <= gx < w - 1 and 1 <= gy < h - 1):
            return (0.0, 0.0)
        dx = (self._arr[gy, gx + 1] - self._arr[gy, gx - 1]) / 2.0
        dy = (self._arr[gy + 1, gx] - self._arr[gy - 1, gx]) / 2.0
        norm = np.sqrt(dx**2 + dy**2) + 1e-8
        return (float(dx / norm), float(dy / norm))

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
import math
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.path import Path as MplPath

if TYPE_CHECKING:
    from .config import TrainConfig


def climb_score(summit: float, peak_step: int, config: TrainConfig) -> float:
    """相对攀爬核心分：爬得高（summit）+ 爬得快（climb_speed = summit / peak_step）。

    这是 base_score（整条轨迹算一次）与 ClimbingEfficiencyMap.update（逐后缀算一次）
    共享的唯一评分核心，只描述「相对出发点爬了多少、多快」。

    ⚠️ 本函数刻意 **不含任何绝对高度项**。绝对高度偏好由两套独立机制分工注入，
    二者互补、服务不同环节，切勿合并进本函数（否则效率图会对绝对高度双重计数）：
      - `abs_height_bonus`（config.abs_height_weight）：**每条轨迹**加一次，
        仅用于 base_score 的 BC 跨轨迹排序（登顶越高的轨迹越优先入库）。
      - `height_prior_weight`：**每个网格**在效率图初始化时按 world_y 注入 floor，
        用于效率图的空间价值（越高的可通行格子基线越高）。
    """
    climb_speed = summit / peak_step
    return summit + config.efficiency_weight * climb_speed


def base_score(states: np.ndarray, config: TrainConfig) -> float:
    """
    基础评分 — 相对攀爬核心 + 绝对高度排序加成。

    states: (T+1, 33)，一段轨迹（完整或片段）。
    不依赖任何外部状态，可独立计算。
    """
    y = states[:, 1]
    start_y = y[0]
    # summit = 全轨迹净爬升。对应 PPO step_reward 高度分量的望远镜和（勿误判为不一致）：
    # BC 对完整轨迹直接取 max 得标量 summit 用于排序；PPO 需其逐步差分形式做信用分配。
    summit = float(y.max() - start_y)
    peak_step = int(y.argmax()) + 1

    # 相对攀爬核心（与效率图 update 共享同一公式）
    score = climb_score(summit, peak_step, config)
    # 绝对高度排序加成：仅 BC 轨迹排序用；效率图侧的绝对高度改由 height_prior 承担，勿混用
    abs_height_bonus = float(y.max()) * config.abs_height_weight
    return score + abs_height_bonus


def is_water(y: float, config: TrainConfig) -> bool:
    """单点落水判定：y 低于 water_y_threshold 即视为落水。

    落水阈值的唯一判定入口（step 级）。step done / 落水扣分 / 轨迹级过滤
    （is_water_trajectory）均由此派生，避免各处 `y < threshold` 分散漂移。
    """
    return float(y) < config.water_y_threshold


def is_water_trajectory(states: np.ndarray, config: TrainConfig) -> bool:
    """轨迹级落水判定：轨迹最低点触水即整条判为落水（复用 is_water）。"""
    return is_water(float(states[:, 1].min()), config)


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


class RewardNormalizer:
    """Welford 在线算法跟踪 height / efficiency 分量的 running σ。

    warmup 阶段记录各分量统计量，达到 calibration_steps 后冻结 σ，
    之后 step_reward 切换到归一化模式 (α * h/σ_h + (1-α) * e/σ_e)。
    冻结后 reward scale 不再变化，防止 Critic 目标震荡。
    """

    def __init__(self, calibration_steps: int = 50_000):
        self._calibration_steps = calibration_steps
        self._n = 0
        self._mean_h = 0.0
        self._M2_h = 0.0
        self._mean_e = 0.0
        self._M2_e = 0.0
        self._frozen = False
        self.sigma_h = 1.0
        self.sigma_e = 1.0

    @property
    def is_active(self) -> bool:
        return self._frozen

    def observe(self, height_reward: float, eff_delta: float) -> None:
        if self._frozen:
            return
        self._n += 1
        d = height_reward - self._mean_h
        self._mean_h += d / self._n
        self._M2_h += d * (height_reward - self._mean_h)

        d = eff_delta - self._mean_e
        self._mean_e += d / self._n
        self._M2_e += d * (eff_delta - self._mean_e)

        if self._n >= self._calibration_steps:
            self._freeze()

    def _freeze(self) -> None:
        if self._n < 2:
            return
        self.sigma_h = max(math.sqrt(self._M2_h / (self._n - 1)), 1e-8)
        self.sigma_e = max(math.sqrt(self._M2_e / (self._n - 1)), 1e-8)
        self._frozen = True


def step_reward(
    obs_prev: np.ndarray,
    obs_curr: np.ndarray,
    efficiency_map: ClimbingEfficiencyMap | None,
    config: TrainConfig,
    running_max_y: float,
    normalizer: RewardNormalizer | None = None,
) -> tuple[float, float]:
    """
    PPO 逐步奖励（创新高模式）。

    只在超过历史最高点 running_max_y 时给正奖励，回落时仅给极轻微惩罚（或零），
    避免跳起后落地抵消正信号。

    与 BC 的对应关系（勿误以为两者是遗漏/不一致，见 base_score / climb_score）：
      本函数的高度分量是 BC `summit` 的**逐步望远镜分解**——沿一条轨迹累加所有正的
      创新高奖励恰好收敛到 `y_max - y_start = summit`。BC 因为能对完整轨迹直接取
      max，用标量 summit 排序即可；PPO 逐步优化则需要这种可加的差分形式来做信用分配。
      效率通道：BC 用 climb_speed（全局），PPO 用 eff_delta（逐步差分）；
      绝对高度通道：BC 用 abs_height_bonus，PPO 经效率图 height_prior 间接注入。
      三通道方向一致，故 BC→PPO 迁移不会互相推翻。

    当 normalizer 激活后，切换到归一化模式:
      reward = α * (height / σ_h) + (1-α) * (eff_delta / σ_e)

    返回 (reward, updated_max_y)。running_max_y 为必传参数（逐步创新高的历史最高点）。
    """
    curr_y = float(obs_curr[1])

    # ── height component（创新高）──
    if curr_y > running_max_y:
        height_reward = curr_y - running_max_y
        new_max_y = curr_y
    else:
        drop = running_max_y - curr_y
        height_reward = -config.neg_reward_scale * math.log1p(drop) if drop > 0 else 0.0
        new_max_y = running_max_y

    # ── efficiency component ──
    eff_delta = 0.0
    if efficiency_map is not None:
        eff_delta = (
            efficiency_map.query(float(obs_curr[0]), float(obs_curr[1]))
            - efficiency_map.query(float(obs_prev[0]), float(obs_prev[1]))
        )

    # ── combine ──
    if normalizer is not None:
        normalizer.observe(height_reward, eff_delta)
        if normalizer.is_active:
            alpha = config.reward_alpha
            reward = (
                alpha * height_reward / normalizer.sigma_h
                + (1 - alpha) * eff_delta / normalizer.sigma_e
            )
        else:
            reward = height_reward + config.waypoint_weight * eff_delta
    else:
        reward = height_reward + config.waypoint_weight * eff_delta

    if is_water(curr_y, config):
        reward = -10.0

    return reward, new_max_y


# ---------------------------------------------------------------------------
# 攀爬效率图
# ---------------------------------------------------------------------------

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
        cache_path: str | None = None,
        height_prior_weight: float = 0.0,
    ):
        self.resolution = grid_resolution
        self.diffusion_iters = diffusion_iterations
        self.diffusion_alpha = diffusion_alpha

        cached = self._load_cache(cache_path) if cache_path else None
        if cached is not None:
            self.traversable, self._min_gx, self._min_gy = cached
        else:
            self.traversable, self._min_gx, self._min_gy = (
                self._build_traversable_mask(polygons)
            )
            if cache_path:
                self._save_cache(cache_path)

        h, w = self.traversable.shape
        self._max_arr = np.full((h, w), -np.inf, dtype=np.float64)

        # 绝对高度偏好（网格级/空间价值）：给每个可通行格子按 world_y 注入基线 floor。
        # 这是效率图专属的绝对高度机制，与 base_score 的 abs_height_bonus（轨迹级/排序）分工，
        # 因此 update() 里的攀爬核心 climb_score 不再叠加绝对高度，避免双重计数。
        if height_prior_weight > 0:
            for gy_idx in range(h):
                world_y = (self._min_gy + gy_idx + 0.5) * self.resolution
                self._max_arr[gy_idx, :] = np.where(
                    self.traversable[gy_idx, :],
                    world_y * height_prior_weight,
                    -np.inf,
                )

        self._arr: np.ndarray | None = None

    # -- public helpers --

    def get_arr(self) -> np.ndarray | None:
        """返回扩散后的效率图数组（供 Dataset 引用）。"""
        return self._arr

    def copy(self) -> ClimbingEfficiencyMap:
        """深拷贝，用于保存 prev_eff_map。"""
        return copy.deepcopy(self)

    # -- 缓存 --

    @staticmethod
    def _load_cache(path: str) -> tuple[np.ndarray, int, int] | None:
        from pathlib import Path as _P
        p = _P(path)
        if not p.exists():
            return None
        data = np.load(p)
        return data["mask"].astype(bool), int(data["min_gx"]), int(data["min_gy"])

    def _save_cache(self, path: str) -> None:
        from pathlib import Path as _P
        p = _P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, mask=self.traversable,
            min_gx=np.array(self._min_gx), min_gy=np.array(self._min_gy),
        )

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
                peak_step = int(suffix_argmax[t]) + 1
                # 相对攀爬核心（与 base_score 共享）；绝对高度不在此叠加，
                # 效率图的绝对高度偏好由 height_prior_weight（网格 floor）承担，避免双重计数
                bs = climb_score(summit, peak_step, config)

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

    _DIFFUSION_CONV_EPS = 1e-4

    def _diffuse(self, arr: np.ndarray) -> np.ndarray:
        """迭代扩散: 4 邻域平均，只通过可通行格子传播。

        使用收敛检测 (max|Δ| < ε) 自动停止，
        diffusion_iters 作为 max_iterations 安全兜底。
        """
        mask = self.traversable.astype(np.float64)
        arr = np.pad(arr, 1, mode="constant", constant_values=0)
        mask = np.pad(mask, 1, mode="constant", constant_values=0)

        for i in range(self.diffusion_iters):
            old_arr = arr.copy()
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
            delta = float(np.abs(arr - old_arr).max())
            if delta < self._DIFFUSION_CONV_EPS:
                break

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

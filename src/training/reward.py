"""
轨迹评分 + 攀爬效率图。

两层评分体系:
- base_score: 攀升门控进度势 + 步数效率（不依赖效率图）
- score_trajectory: base_score + bc_waypoint_weight·路标（默认 0 → 纯 base_score 排序，
  效率图对 BC 仅经 patch 输入生效；效率图对 PPO 则以 PBRS 势场塑形 γΦ'−Φ 生效，见 step_reward）

ClimbingEfficiencyMap: 地形感知的效率图
- 价值传播（跨迭代）: full_score 沿轨迹路径向出发点传播
- 地形感知扩散（每轮内）: 信号沿可通行区域扩散，不穿墙
- 后缀预计算: O(T) 更新效率图
"""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.path import Path as MplPath

if TYPE_CHECKING:
    from .config import TrainConfig


def climb_score(progress: float, peak_step: int, config: TrainConfig) -> float:
    """相对攀爬核心分：进展得多（progress）+ 进展得快。

    量纲自洽（两项均为「米」）：
      progress[米] + efficiency_weight * (progress / peak_step * speed_ref_steps)[米]
    其中 climb_speed = progress / peak_step 为「米/步」，乘以参考步数 speed_ref_steps 换回
    「米」，efficiency_weight 遂为无量纲相对权重。等价于 progress * (1 + eff_w * ref / peak_step)。

    这是 base_score（整条轨迹算一次）与 ClimbingEfficiencyMap.update（逐后缀算一次）
    共享的唯一评分核心，只描述「相对出发点进展了多少、多快」。

    `progress` 由 `progress_metric` 给出 = 攀升门控的进度势（secured_dy + 门控横向 reach），
    **取代**旧的单一纵向 summit：让「右侧真实攀爬路径（有横向 reach）」压过「爬树死路（几乎零
    横向）」，并用 dwell 驻留过滤甩飞尖峰。

    ⚠️ 本函数刻意 **不含任何绝对高度项**。绝对高度偏好由两套独立机制分工注入，
    二者互补、服务不同环节，切勿合并进本函数（否则效率图会对绝对高度双重计数）：
      - `abs_height_bonus`（config.abs_height_weight）：**每条轨迹**加一次，
        仅用于 base_score 的 BC 跨轨迹排序（登顶越高的轨迹越优先入库）。
      - `height_prior_weight`：**每个网格**在效率图初始化时按 world_y 注入 floor，
        用于效率图的空间价值（默认 0.0，见 config 说明）。
    """
    climb_speed = progress / peak_step                     # [米/步]
    speed_term = climb_speed * config.speed_ref_steps      # [米]（参考步数换算回米）
    return progress + config.efficiency_weight * speed_term


def _secured_held(y: np.ndarray, dwell: int, tol: float) -> np.ndarray:
    """逐位置的「守住高度」held[t] = 从 t 起 dwell 步窗口内可持续的高度。

    窗口跨度 (max-min) ≤ tol 视为「守住」→ 取窗口最高点；否则窗口内发生真实回落
    → 取窗口最低点（保守）。据此甩飞尖峰（先冲高后坠落，跨度 > tol）会被压到低值，
    自然不会成为 argmax，从而滤除「甩到顶又滑回」的假进展。O(T·dwell)，dwell 小可忽略。
    """
    T = len(y)
    dwell = max(1, int(dwell))
    held = np.empty(T, dtype=np.float64)
    for t in range(T):
        w = y[t:t + dwell]
        wmax = float(w.max())
        wmin = float(w.min())
        held[t] = wmax if (wmax - wmin) <= tol else wmin
    return held


def progress_metric(
    states: np.ndarray, config: TrainConfig
) -> tuple[float, int, float, float]:
    """攀升门控的进度势（取代单一 summit）。

    states: (T+1, 33)。返回 (progress, peak_idx, secured_dy, reach)：
      - held = _secured_held(y)；secured 峰 peak_idx = argmax_t held[t]（并列取最早，越快越好）。
      - secured_dy = held[peak_idx] - y[0]（守得住的净爬升）。
      - reach = x[peak_idx] - x[0]（到达 secured 峰时的向右伸展）。
      - progress = secured_dy + progress_wx * max(0, reach)，仅当 secured_dy > 0；
        否则 progress = secured_dy（不爬升 → 横向不计；向左 → max(0,·) 不计）。

    peak_idx 同时用作 BC 截断点、score_trajectory 的 waypoint 终点，口径全线统一。
    """
    y = states[:, 1]
    x = states[:, 0]
    held = _secured_held(y, config.dwell_steps, config.dwell_drop_tol)
    peak_idx = int(held.argmax())
    secured_dy = float(held[peak_idx] - y[0])
    reach = float(x[peak_idx] - x[0])
    if secured_dy > 0:
        progress = secured_dy + config.progress_wx * max(0.0, reach)
    else:
        progress = secured_dy
    return progress, peak_idx, secured_dy, reach


def base_score(states: np.ndarray, config: TrainConfig) -> float:
    """
    基础评分 — 相对攀爬核心 + 绝对高度排序加成。

    states: (T+1, 33)，一段轨迹（完整或片段）。
    不依赖任何外部状态，可独立计算。
    """
    y = states[:, 1]
    # progress = 攀升门控的进度势（secured 纵向 + 门控横向），取代旧的单一 summit。
    # 对应 PPO step_reward 的逐步望远镜分解：BC 对完整轨迹直接取 secured 峰得标量用于排序；
    # PPO 需其逐步差分/确认式形式做信用分配。
    progress, peak_idx, _dy, _reach = progress_metric(states, config)
    peak_step = peak_idx + 1

    # 相对攀爬核心（与效率图 update 共享同一公式）
    score = climb_score(progress, peak_step, config)
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
    BC 轨迹排序分 = base_score + bc_waypoint_weight * waypoint_reward。

    states: (T+1, 33) — 原始 33 维状态。
    waypoint_reward = E(secured 峰) - E(起点)，O(1)。终点取 `progress_metric` 的 secured
    peak（而非轨迹末端或瞬时 y 峰），与数据集在 secured peak 截断、base_score 用 progress 排序
    的口径一致。

    ⚠️ `bc_waypoint_weight` 默认 0.0：效率图是 value-to-go，端点差 E(peak)-E(start) 沿真实
    攀爬多为负，直接进排序会把好轨迹压后；且势函数塑形（PBRS）是 RL 回报口径的手段、不适用于
    BC 监督排序。故默认只按 base_score 排序，效率图对 BC 仅经 patch wide 通道输入生效。
    如需恢复 waypoint 排序贡献，把 config.bc_waypoint_weight 调 > 0（一般不建议）。
    落水轨迹直接返回 -inf，确保在任何筛选中被淘汰。
    """
    if is_water_trajectory(states, config):
        return float("-inf")

    bs = base_score(states, config)

    if efficiency_map is None or config.bc_waypoint_weight == 0.0:
        return bs

    _progress, peak_idx, _dy, _reach = progress_metric(states, config)
    eff_peak = efficiency_map.query(states[peak_idx, 0], states[peak_idx, 1])
    eff_start = efficiency_map.query(states[0, 0], states[0, 1])
    waypoint_reward = eff_peak - eff_start

    return bs + config.bc_waypoint_weight * waypoint_reward


class RewardNormalizer:
    """Welford 在线算法跟踪 height / efficiency 分量的 running σ。

    σ 从第一步起就随 Welford 连续更新，达到 calibration_steps 后冻结、不再变化
    （防止 Critic 目标长期震荡）。step_reward **全程**使用同一归一化公式
    (α * h/σ_h + (1-α) * e/σ_e)，冻结只是把缓慢演化的 σ 定住，因此奖励尺度在
    冻结时刻**连续**，不会出现"warmup 用原始量纲、冻结后突然换成 σ 归一化"的跳变。
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
        """σ 是否已冻结（仅供日志判断，不再用于切换奖励公式）。"""
        return self._frozen

    def observe(self, height_reward: float, eff_shaping: float) -> None:
        if self._frozen:
            return
        self._n += 1
        d = height_reward - self._mean_h
        self._mean_h += d / self._n
        self._M2_h += d * (height_reward - self._mean_h)

        d = eff_shaping - self._mean_e
        self._mean_e += d / self._n
        self._M2_e += d * (eff_shaping - self._mean_e)

        # σ 连续更新（样本≥2 才有无偏方差），保证冻结前后奖励尺度平滑衔接
        if self._n >= 2:
            self.sigma_h = max(math.sqrt(self._M2_h / (self._n - 1)), 1e-8)
            self.sigma_e = max(math.sqrt(self._M2_e / (self._n - 1)), 1e-8)

        if self._n >= self._calibration_steps:
            self._frozen = True


class NewHighConfirmer:
    """确认式创新高（PPO 防甩飞）：新高需在 dwell 步窗口内守住才结算发奖。

    维护尾部滑窗（最近 dwell 步的 y）与已确认高度 confirmed。每步取窗口的「守住高度」
    held（窗口跨度 ≤ drop_tol 视为守住 → 取窗口最高点，否则取最低点，与 _secured_held 同口径），
    仅当 held 超过 confirmed 时才把差额作为奖励结算（并抬升 confirmed）。

    这是 BC `progress_metric` 中 secured_dy 的**逐步望远镜分解**：沿一条轨迹累加所有结算量，
    收敛到「守得住的净爬升」= held 峰 - 起点 y。甩飞尖峰（先冲高后坠落，窗口跨度 > tol）的
    held 被压到低值，不会抬升 confirmed → 不发奖，从而滤除「甩到顶又滑回」的假进展。
    """

    def __init__(self, start_y: float, dwell_steps: int, drop_tol: float):
        self.dwell = max(1, int(dwell_steps))
        self.tol = float(drop_tol)
        self.confirmed = float(start_y)
        self._buf: deque[float] = deque(maxlen=self.dwell)

    def confirm(self, curr_y: float) -> float:
        """喂入当前 y，返回本步确认结算的纵向奖励（≥0；未确认或未创新高则 0）。"""
        self._buf.append(float(curr_y))
        if len(self._buf) < self.dwell:
            return 0.0
        wmax = max(self._buf)
        wmin = min(self._buf)
        held = wmax if (wmax - wmin) <= self.tol else wmin
        if held > self.confirmed:
            gain = held - self.confirmed
            self.confirmed = held
            return float(gain)
        return 0.0


def step_reward(
    obs_prev: np.ndarray,
    obs_curr: np.ndarray,
    efficiency_map: ClimbingEfficiencyMap | None,
    config: TrainConfig,
    running_max_y: float,
    normalizer: RewardNormalizer | None = None,
    confirmer: NewHighConfirmer | None = None,
) -> tuple[float, float]:
    """
    PPO 逐步奖励（攀升门控进度势的逐步分解）。

    进度通道 = 纵向（secured 创新高）+ 前沿门控横向：
      - 纵向：提供 confirmer 时用**确认式创新高**（守住 dwell 步才结算，防甩飞，望远镜和 =
        secured_dy）；未提供时退化为即时创新高（回落按 -neg_reward_scale·log(1+drop)，
        neg_reward_scale 默认 0 → 回落零惩罚，望远镜和 = summit）。
      - 横向：`progress_wx · max(0, Δx)`，仅在攀升前沿计（curr_y ≥ running_max_y - dwell_drop_tol）。
        不在前沿（低处平地向右滑）不计、向左不计 → 与 BC progress_metric 的门控口径一致。

    效率通道：势函数塑形 PBRS（Ng et al. 1999）F = pbrs_gamma·Φ(s') − Φ(s)，Φ = 效率图（value-to-go）。
    pbrs_gamma=1（默认）→ 伸缩式 Φ'−Φ：沿轨迹求和 = Φ_end − Φ_start，纯进度信号、无漂移，
    上爬(Φ↑)给正、下滑给负，方向与主进度奖励一致，稠密引导攀爬。pbrs_gamma=gamma(0.99) 虽理论上
    折扣自洽，但带 −(1−γ)Φ 每步负漂移（Φ 越大惩罚越重、静止即被罚），实测主导奖励并压制攀爬（诊断#2）。

    与 BC 的对应关系（勿误以为遗漏/不一致，见 base_score / progress_metric）：
      纵向确认式创新高 = secured_dy 的逐步望远镜分解；横向前沿项 = 门控 reach 的逐步分解；
      效率通道 PPO 用 PBRS 塑形（pbrs_gamma·Φ'−Φ）。PBRS 是 RL 回报口径的塑形、不适用于 BC 监督排序，
      故 BC 侧不复用（score_trajectory 由 bc_waypoint_weight 控制，默认 0 → 图仅作 patch 输入）。

    当提供 normalizer 时，**全程**使用归一化模式（σ 连续演化，冻结时刻尺度不跳变）:
      reward = α * (progress / σ_h) + (1-α) * (F / σ_e)
    σ 冻结后 α/(1-α)、σ_h/σ_e 均为常数，等价于「主奖励 + 常数系数·PBRS」再乘全局正标量，
    故最优策略不变性在冻结后严格成立（常数倍势函数仍是合法势场，全局缩放不改 argmax）。
    未提供 normalizer 时退化为原始量纲加权 (progress + waypoint_weight * F)，waypoint_weight
    充当此处的常数塑形系数。

    返回 (reward, updated_max_y)。running_max_y 为必传参数（逐步创新高的历史最高点）。
    """
    curr_y = float(obs_curr[1])
    new_max_y = curr_y if curr_y > running_max_y else running_max_y

    # ── 纵向分量 ──
    if confirmer is not None:
        # 确认式创新高：守住 dwell 步才结算（防甩飞），已含「不发跌落惩罚」语义
        height_reward = confirmer.confirm(curr_y)
    elif curr_y > running_max_y:
        height_reward = curr_y - running_max_y
    else:
        drop = running_max_y - curr_y
        height_reward = -config.neg_reward_scale * math.log1p(drop) if drop > 0 else 0.0

    # ── 横向分量（前沿门控）──：只在攀升前沿计向右位移，低处平地滑动/向左不计
    dx = float(obs_curr[0]) - float(obs_prev[0])
    at_frontier = curr_y >= running_max_y - config.dwell_drop_tol
    horiz_reward = config.progress_wx * max(0.0, dx) if at_frontier else 0.0

    progress_reward = height_reward + horiz_reward

    # ── efficiency component: 势函数塑形 PBRS ──
    # F = pbrs_gamma·Φ(s') − Φ(s)，Φ = 效率图（value-to-go）。提供稠密引导。
    # pbrs_gamma=1（默认）→ 伸缩式 Φ'−Φ：沿轨迹求和=Φ_end−Φ_start，纯进度、无漂移，
    #   上爬(Φ↑)给正、下滑给负，方向与主进度奖励一致。
    # pbrs_gamma=gamma(0.99) → 理论折扣自洽，但带 −(1−γ)Φ 每步负漂移：Φ 越大惩罚越重、
    #   静止即被罚，实测主导奖励并把 mean_reward 拖成随 Φ 增长的负漂移、压制攀爬（见诊断#2）。
    eff_shaping = 0.0
    if efficiency_map is not None:
        phi_curr = efficiency_map.query(float(obs_curr[0]), float(obs_curr[1]))
        phi_prev = efficiency_map.query(float(obs_prev[0]), float(obs_prev[1]))
        eff_shaping = config.pbrs_gamma * phi_curr - phi_prev

    # ── combine ──
    # 全程用同一归一化公式：σ 从第一步起连续演化、冻结时定住，奖励尺度无跳变。
    if normalizer is not None:
        normalizer.observe(progress_reward, eff_shaping)
        alpha = config.reward_alpha
        reward = (
            alpha * progress_reward / normalizer.sigma_h
            + (1 - alpha) * eff_shaping / normalizer.sigma_e
        )
    else:
        reward = progress_reward + config.waypoint_weight * eff_shaping

    if is_water(curr_y, config):
        reward = config.water_penalty

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
        mask_subsample: int = 3,
    ):
        self.resolution = grid_resolution
        self.diffusion_iters = diffusion_iterations
        self.diffusion_alpha = diffusion_alpha
        # 掩码子采样：每格 k×k 采样，任一子点在多边形内即判固体。k>1 修复 1m 中心采样
        # 漏判薄墙/薄树（薄特征中心落空 → 误判可通行 → 扩散穿墙，见 verify_mask_aliasing）。
        self._mask_subsample = max(1, int(mask_subsample))

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

        # 扩散用的静态邻域结构（仅依赖 traversable mask，与格值无关）：预计算一次，
        # 供 _diffuse 每轮迭代/每次 update 复用，避免在迭代循环里重复计算 n_cnt/shifted_mask。
        self._init_diffusion_kernels()

    def _init_diffusion_kernels(self) -> None:
        """预计算扩散的静态量：padded mask、4 邻域 shifted mask、邻居计数 n_cnt。"""
        pad_mask = np.pad(
            self.traversable.astype(np.float64), 1, mode="constant", constant_values=0
        )
        self._diff_pad_mask = pad_mask                      # 自身可通行掩码（float 1/0）
        self._diff_shifted_masks = [
            np.roll(pad_mask, shift, axis=axis)
            for shift, axis in ((-1, 0), (1, 0), (-1, 1), (1, 1))
        ]
        n_cnt = np.zeros_like(pad_mask)
        for sm in self._diff_shifted_masks:
            n_cnt += sm
        self._diff_n_cnt = n_cnt                            # 每格可通行邻居数（静态）

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
        """点在多边形内 = 固体 = 不可通行。

        每格 k×k 子采样（k=self._mask_subsample）：任一子点落入多边形即判固体，对薄墙/薄树
        保守，修复 1m 中心采样漏判（薄特征中心落空 → 误判可通行 → 扩散穿墙）。k=1 退化为
        旧的格中心单点判定。
        """
        all_pts = np.concatenate(polygons)
        x_min, x_max = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
        y_min, y_max = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())

        min_gx = int(np.floor(x_min / self.resolution)) - 1
        min_gy = int(np.floor(y_min / self.resolution)) - 1
        max_gx = int(np.ceil(x_max / self.resolution)) + 1
        max_gy = int(np.ceil(y_max / self.resolution)) + 1

        w = max_gx - min_gx + 1
        h = max_gy - min_gy + 1

        k = self._mask_subsample
        if k <= 1:
            offsets = [(0.5, 0.5)]
        else:
            offsets = [
                ((i + 0.5) / k, (j + 0.5) / k)
                for i in range(k) for j in range(k)
            ]

        mpl_paths = [MplPath(poly) for poly in polygons]
        GX, GY = np.meshgrid(np.arange(w), np.arange(h))
        solid = np.zeros((h, w), dtype=bool)
        for ox, oy in offsets:
            cx = ((min_gx + GX) + ox) * self.resolution
            cy = ((min_gy + GY) + oy) * self.resolution
            pts = np.column_stack([cx.ravel(), cy.ravel()])
            inside = np.zeros(len(pts), dtype=bool)
            for p in mpl_paths:
                inside |= p.contains_points(pts)
            solid |= inside.reshape(h, w)

        return ~solid, min_gx, min_gy

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
            states = traj.raw_states  # (T+1, 33)
            T = len(states)

            y = states[:, 1]
            x = states[:, 0]

            # secured held：先把每个位置折算成「守得住的高度」（滤甩飞尖峰），
            # 后缀口径全部基于 held 而非裸 y，与 base_score/progress_metric 统一。
            held = _secured_held(y, config.dwell_steps, config.dwell_drop_tol)

            # held 的后缀最大值与其绝对 argmax（并列取最早 → 越快到达越优）。
            suffix_held = np.empty(T, dtype=np.float64)
            suffix_arg = np.empty(T, dtype=np.int64)
            best_idx = T - 1
            suffix_held[-1] = held[-1]
            suffix_arg[-1] = best_idx
            for i in range(T - 2, -1, -1):
                if held[i] >= held[best_idx]:
                    best_idx = i
                suffix_held[i] = held[best_idx]
                suffix_arg[i] = best_idx

            for t in range(T - 1):
                gx = int(np.floor(states[t, 0] / self.resolution)) - self._min_gx
                gy = int(np.floor(states[t, 1] / self.resolution)) - self._min_gy
                if not (0 <= gx < w and 0 <= gy < h):
                    continue

                peak_idx = int(suffix_arg[t])
                secured_dy = suffix_held[t] - y[t]
                reach = x[peak_idx] - x[t]
                # 攀升门控进度势（与 progress_metric/base_score 共享口径）：不爬升不计横向、向左不计
                if secured_dy > 0:
                    progress = secured_dy + config.progress_wx * max(0.0, reach)
                else:
                    progress = secured_dy
                peak_step = (peak_idx - t) + 1
                # 相对攀爬核心（与 base_score 共享）；绝对高度不在此叠加，
                # 效率图的绝对高度偏好由 height_prior_weight（网格 floor）承担，避免双重计数
                bs = climb_score(progress, peak_step, config)

                wp = 0.0
                if prev_eff_map is not None:
                    # 终点取「t 之后的 secured 峰」，与本处 progress/peak_step 的后缀语义一致，
                    # 也与 score_trajectory(t=0→全局 secured 峰) 口径统一；复用 suffix_arg，仍 O(1)。
                    eff_peak = prev_eff_map.query(states[peak_idx, 0], states[peak_idx, 1])
                    eff_start = prev_eff_map.query(states[t, 0], states[t, 1])
                    wp = eff_peak - eff_start

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

        静态量（padded mask / 4 邻域 shifted mask / 邻居计数 n_cnt）由 _init_diffusion_kernels
        预计算并复用；本循环内只随 arr 变化滚动，避免逐迭代重算掩码结构。
        """
        pad_mask = self._diff_pad_mask
        shifted_masks = self._diff_shifted_masks
        n_cnt = self._diff_n_cnt
        shifts = ((-1, 0), (1, 0), (-1, 1), (1, 1))

        mask_bool = pad_mask > 0
        arr = np.pad(arr, 1, mode="constant", constant_values=0)

        for _ in range(self.diffusion_iters):
            old_arr = arr.copy()
            n_sum = np.zeros_like(arr)
            for k, (shift, axis) in enumerate(shifts):
                n_sum += np.roll(arr, shift, axis=axis) * shifted_masks[k]
            n_avg = np.divide(
                n_sum, n_cnt, out=np.zeros_like(n_sum), where=n_cnt > 0
            )
            arr = np.where(
                mask_bool,
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
        # 用 floor 而非 int 截断：与掩码构建（_build_traversable_mask）、patch 裁剪
        # （dataset.crop_centered）保持同一取格语义，避免负坐标处错位一格。
        gx = int(np.floor(x / self.resolution)) - self._min_gx
        gy = int(np.floor(y / self.resolution)) - self._min_gy
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
        gx = int(np.floor(x / self.resolution)) - self._min_gx
        gy = int(np.floor(y / self.resolution)) - self._min_gy
        h, w = self._arr.shape
        if not (1 <= gx < w - 1 and 1 <= gy < h - 1):
            return (0.0, 0.0)
        dx = (self._arr[gy, gx + 1] - self._arr[gy, gx - 1]) / 2.0
        dy = (self._arr[gy + 1, gx] - self._arr[gy - 1, gx]) / 2.0
        norm = np.sqrt(dx**2 + dy**2) + 1e-8
        return (float(dx / norm), float(dy / norm))

"""
轨迹数据集 — 极简存储 + 即时 patch 构建 + peak 截断 + 滑动窗口管理。

轨迹只存原始 29D 状态 + 动作（~61KB/条）。
地形图静态共享，效率图使用最新版，patch 在 __getitem__ 中即时构建。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .config import TrainConfig

# 17D 动力学索引: vel_x/y, ang_vel, hub_vx/vy/angle, slider_vx/vy/angle,
# handle_vx/vy, pole_vx/vy, tip_vx/vy, hammer_angle
DYNAMICS_INDICES = [2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 21, 22, 25, 26, 27]

# 身体部件位置索引: player(0,1), hub(5,6), slider(10,11)
BODY_POS_INDICES = [(0, 1), (5, 6), (10, 11)]
# 锤子部件位置索引: handle(15,16), pole(19,20), tip(23,24)
HAMMER_POS_INDICES = [(15, 16), (19, 20), (23, 24)]


@dataclass
class Trajectory:
    raw_states: np.ndarray           # (T+1, 29)
    actions: np.ndarray              # (T, 2)
    score: float = 0.0
    iteration: int = 0


def crop_centered(
    arr: np.ndarray,
    x: float,
    y: float,
    min_gx: int,
    min_gy: int,
    size: int = 32,
    patch_res: float = 0.5,
    arr_res: float = 1.0,
) -> np.ndarray:
    """
    从全局数组裁剪局部 patch，处理分辨率差异。

    全局数组分辨率 arr_res=1.0m，patch 分辨率 patch_res=0.5m。
    先裁剪 arr_size×arr_size 区域，再最近邻上采样到 size×size。
    """
    arr_size = int(size * patch_res / arr_res)
    scale = size // arr_size

    gx_center = x / arr_res - min_gx
    gy_center = y / arr_res - min_gy

    half = arr_size / 2.0
    gx_start = int(np.floor(gx_center - half))
    gy_start = int(np.floor(gy_center - half))

    h, w = arr.shape
    crop = np.zeros((arr_size, arr_size), dtype=np.float32)

    sy_lo = max(0, gy_start)
    sy_hi = min(h, gy_start + arr_size)
    sx_lo = max(0, gx_start)
    sx_hi = min(w, gx_start + arr_size)

    if sy_lo < sy_hi and sx_lo < sx_hi:
        crop[sy_lo - gy_start : sy_hi - gy_start,
             sx_lo - gx_start : sx_hi - gx_start] = arr[sy_lo:sy_hi, sx_lo:sx_hi]

    if scale > 1:
        crop = np.repeat(np.repeat(crop, scale, axis=0), scale, axis=1)
    return crop


def render_gaussian(
    channel: np.ndarray,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    patch_res: float = 0.5,
    sigma: float = 0.6,
) -> None:
    """在通道上渲染以 (bx, by) 为目标的 Gaussian blob，(cx, cy) 为 patch 中心。"""
    size = channel.shape[0]
    px = (bx - cx) / patch_res + size / 2.0
    py = (by - cy) / patch_res + size / 2.0

    ys = np.arange(size, dtype=np.float32)
    xs = np.arange(size, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    sigma_px = sigma / patch_res
    blob = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma_px**2))
    channel += blob


class TrajectoryDataset(Dataset):
    """
    轨迹数据集。

    持有全局地形 mask 和效率图引用，在 __getitem__ 中即时构建 4ch patch。
    双层过滤: add_trajectories 做 top-K% 筛选，trim_oldest 做 FIFO 滑动窗口。
    """

    def __init__(
        self,
        config: TrainConfig,
        terrain_mask: np.ndarray,
        efficiency_arr: np.ndarray | None,
        min_gx: int = 0,
        min_gy: int = 0,
    ):
        self.config = config
        self.terrain_mask = terrain_mask
        self.efficiency_arr = efficiency_arr
        self._min_gx = min_gx
        self._min_gy = min_gy
        self.trajectories: list[Trajectory] = []
        self._index: list[tuple[int, int]] = []  # (traj_idx, window_start)

    def update_efficiency_arr(self, new_arr: np.ndarray | None) -> None:
        """每轮迭代后更新效率图引用。"""
        self.efficiency_arr = new_arr

    def set_trajectories(self, trajs: list[Trajectory]) -> None:
        """设置初始轨迹集（冷启动用）。"""
        self.trajectories = list(trajs)
        self._rebuild_index()

    def add_trajectories(
        self,
        trajs: list[Trajectory],
        scores: list[float],
        keep_ratio: float = 0.2,
        iteration: int = 0,
    ) -> None:
        """追加新轨迹，先做 top-K% 筛选，只有好轨迹才入库。"""
        if not trajs:
            return
        threshold = float(np.percentile(scores, 100 * (1 - keep_ratio)))
        for traj, sc in zip(trajs, scores):
            if sc >= threshold:
                traj.score = sc
                traj.iteration = iteration
                self.trajectories.append(traj)
        self._rebuild_index()

    def trim_oldest(self, max_n: int) -> None:
        """滑动窗口：按 iteration 淘汰最旧的轨迹直到不超过 max_n 条。"""
        if len(self.trajectories) <= max_n:
            return
        self.trajectories.sort(key=lambda t: t.iteration, reverse=True)
        self.trajectories = self.trajectories[:max_n]
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """
        重建 (traj_idx, window_start) 索引。

        每个窗口需要:
        - states[start : start+ctx]  → ctx 个观测 (dynamics + patch)
        - actions[start : start+ctx] → ctx 个历史动作 (模型输入)
        - actions[start+ctx]         → 目标动作 (监督信号)

        peak 截断: 只在 [0, peak_t) 范围内取 actions，排除跌落段。
        """
        ctx = self.config.context_len
        self._index = []
        for ti, traj in enumerate(self.trajectories):
            peak_t = int(traj.raw_states[:, 1].argmax())
            max_target = min(peak_t, len(traj.actions) - 1)
            for start in range(max(0, max_target - ctx + 1)):
                self._index.append((ti, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        ti, start = self._index[idx]
        traj = self.trajectories[ti]
        ctx = self.config.context_len

        # states[start : start+ctx] → observations
        obs_states = traj.raw_states[start : start + ctx]  # (ctx, 29)
        # actions[start : start+ctx] → input actions
        input_actions = traj.actions[start : start + ctx]  # (ctx, 2)
        # actions[start+ctx] → target
        target_action = traj.actions[start + ctx]           # (2,)

        dynamics = obs_states[:, DYNAMICS_INDICES].astype(np.float32)  # (ctx, 17)

        ch = self.config.patch_channels
        ps = self.config.patch_size
        patches = np.zeros((ctx, ch, ps, ps), dtype=np.float32)
        for t in range(ctx):
            patches[t] = self._build_patch(obs_states[t])

        return (
            torch.from_numpy(dynamics),
            torch.from_numpy(patches),
            torch.from_numpy(input_actions.astype(np.float32)),
            torch.from_numpy(target_action.astype(np.float32)),
        )

    def _build_patch(self, step_state: np.ndarray) -> np.ndarray:
        """即时构建 4ch 32x32 patch。"""
        px, py = float(step_state[0]), float(step_state[1])
        size = self.config.patch_size
        patch = np.zeros((self.config.patch_channels, size, size), dtype=np.float32)

        if self.terrain_mask is not None:
            patch[0] = crop_centered(
                self.terrain_mask.astype(np.float32),
                px, py, self._min_gx, self._min_gy,
                size=size, patch_res=self.config.patch_resolution,
                arr_res=self.config.grid_resolution,
            )

        if self.efficiency_arr is not None:
            patch[1] = crop_centered(
                self.efficiency_arr.astype(np.float32),
                px, py, self._min_gx, self._min_gy,
                size=size, patch_res=self.config.patch_resolution,
                arr_res=self.config.grid_resolution,
            )

        for bx_idx, by_idx in BODY_POS_INDICES:
            render_gaussian(
                patch[2],
                float(step_state[bx_idx]), float(step_state[by_idx]),
                px, py,
                patch_res=self.config.patch_resolution,
            )

        for hx_idx, hy_idx in HAMMER_POS_INDICES:
            render_gaussian(
                patch[3],
                float(step_state[hx_idx]), float(step_state[hy_idx]),
                px, py,
                patch_res=self.config.patch_resolution,
            )

        return patch

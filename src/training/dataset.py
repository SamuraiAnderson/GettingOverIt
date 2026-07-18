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

# ── 动力学特征布局（平移等变，不含绝对坐标）──
# 原始状态 33 维 = 基础 29 + fakeCursor 4（cursorX/Y=29/30, cursorVelX/Y=31/32，均为绝对量）
# 速度/角速度分量 (15): player vel(2,3) + ang_vel(4) + hub vel(7,8)
# + slider vel(12,13) + handle vel(17,18) + pole vel(21,22) + tip vel(25,26) + cursor vel(31,32)
VELOCITY_INDICES = [2, 3, 4, 7, 8, 12, 13, 17, 18, 21, 22, 25, 26, 31, 32]
# 角度 (3): hubAngle(9), sliderAngle(14), hammerAngle(27)
# C# 侧单位为「度」(eulerAngles.z / Atan2*Rad2Deg)，编码前需转弧度再取 sin/cos
ANGLE_INDICES = [9, 14, 27]
# 部件相对 player(0,1) 的位置 (6): hub, slider, handle, pole, tip, cursor
# cursor(29,30) 相对坐标 = 弹簧力臂，恢复对锤子受力的马尔可夫可观测性
REL_POS_INDICES = [(5, 6), (10, 11), (15, 16), (19, 20), (23, 24), (29, 30)]
# 动力学特征总维度: 15 速度 + 3 角度×2(sin/cos) + 6 部件×2(相对坐标) = 33
DYNAMICS_DIM = len(VELOCITY_INDICES) + len(ANGLE_INDICES) * 2 + len(REL_POS_INDICES) * 2

# 身体部件位置索引: player(0,1), hub(5,6), slider(10,11)
BODY_POS_INDICES = [(0, 1), (5, 6), (10, 11)]
# 锤子部件位置索引: handle(15,16), pole(19,20), tip(23,24)
HAMMER_POS_INDICES = [(15, 16), (19, 20), (23, 24)]


def build_dynamics(raw: np.ndarray) -> np.ndarray:
    """从原始 33D 状态构建平移等变的动力学特征向量。

    末轴组成:
      - 15 维速度/角速度（原始值，含 cursor 速度）
      - 3 个角度 → (sin, cos)，共 6 维（角度先由度转弧度，消除环绕不连续）
      - 6 个部件相对 player 的坐标 (part - player)，共 12 维（含 cursor，保留相对几何精度）

    支持任意前置维度，末轴须为 33。返回末轴 = DYNAMICS_DIM。
    """
    raw = np.asarray(raw, dtype=np.float32)
    vel = raw[..., VELOCITY_INDICES]
    ang = np.deg2rad(raw[..., ANGLE_INDICES])
    ang_feat = np.concatenate([np.sin(ang), np.cos(ang)], axis=-1)

    px = raw[..., 0:1]
    py = raw[..., 1:2]
    rel_parts = []
    for xi, yi in REL_POS_INDICES:
        rel_parts.append(raw[..., xi:xi + 1] - px)
        rel_parts.append(raw[..., yi:yi + 1] - py)
    rel = np.concatenate(rel_parts, axis=-1)

    return np.concatenate([vel, ang_feat, rel], axis=-1).astype(np.float32)


def compute_dynamics_stats(trajectories: list) -> tuple[np.ndarray, np.ndarray]:
    """从轨迹集合统计动力学特征的 per-dim mean/std，用于观测归一化标定。

    返回 (mean, std)，均为 (DYNAMICS_DIM,)。方差过小的维度 std 置 1.0 防止除零放大。
    空输入时退化为 (0, 1)（等价恒等归一化）。
    """
    feats = [build_dynamics(traj.raw_states) for traj in trajectories]
    if not feats:
        return (
            np.zeros(DYNAMICS_DIM, dtype=np.float32),
            np.ones(DYNAMICS_DIM, dtype=np.float32),
        )
    stacked = np.concatenate(feats, axis=0)  # (ΣT, DYNAMICS_DIM)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


@dataclass
class Trajectory:
    raw_states: np.ndarray           # (T+1, 33)
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


def build_patch(
    state: np.ndarray,
    terrain_mask: np.ndarray | None,
    efficiency_arr: np.ndarray | None,
    config: TrainConfig,
    min_gx: int,
    min_gy: int,
) -> np.ndarray:
    """构建单样本 4ch patch（以 player 为中心）。

    通道: ch0 地形可通行 mask, ch1 攀爬效率图, ch2 身体部件 blob, ch3 锤子部件 blob。

    离线数据集（__getitem__）与在线采集/推理（rollout._build_patches_batch）
    共享此唯一实现，保证两侧 patch 逐通道完全一致，避免训练/推理分叉。
    `efficiency_arr` 为效率图数组（在线侧由 eff_map.get_arr() 取得），可为 None。
    """
    px, py = float(state[0]), float(state[1])
    size = config.patch_size
    patch = np.zeros((config.patch_channels, size, size), dtype=np.float32)

    if terrain_mask is not None:
        patch[0] = crop_centered(
            terrain_mask.astype(np.float32),
            px, py, min_gx, min_gy,
            size=size, patch_res=config.patch_resolution,
            arr_res=config.grid_resolution,
        )

    if efficiency_arr is not None:
        patch[1] = crop_centered(
            efficiency_arr.astype(np.float32),
            px, py, min_gx, min_gy,
            size=size, patch_res=config.patch_resolution,
            arr_res=config.grid_resolution,
        )

    for bx_idx, by_idx in BODY_POS_INDICES:
        render_gaussian(
            patch[2],
            float(state[bx_idx]), float(state[by_idx]),
            px, py, patch_res=config.patch_resolution,
        )

    for hx_idx, hy_idx in HAMMER_POS_INDICES:
        render_gaussian(
            patch[3],
            float(state[hx_idx]), float(state[hy_idx]),
            px, py, patch_res=config.patch_resolution,
        )

    return patch


def left_pad_sequence(
    entries: np.ndarray, ctx: int
) -> tuple[np.ndarray, np.ndarray]:
    """将时间维右对齐的序列左填充零到定长 ctx。

    entries: (k, *feat)，k 为真实步数（最后一个 = 最新），k 可为 0；
             k > ctx 时只保留最近 ctx 步。
    返回 (padded (ctx, *feat), valid_mask (ctx,))：valid_mask 中 1=真实步、0=左填充位。

    离线数据集（`TrajectoryDataset.__getitem__`）与在线采集
    （`rollout._build_history_window`）共享此唯一的填充/掩码实现，从结构上消除
    历史上出现过的训练/推理 off-by-one 错位。填充位恒为零、且右侧最后一步恒有效。
    """
    entries = np.asarray(entries, dtype=np.float32)
    k = entries.shape[0]
    feat_shape = entries.shape[1:]

    if k >= ctx:
        padded = np.array(entries[-ctx:], dtype=np.float32)
        valid = np.ones(ctx, dtype=np.float32)
        return padded, valid

    pad = np.zeros((ctx - k, *feat_shape), dtype=np.float32)
    padded = np.concatenate([pad, entries], axis=0)
    valid = np.zeros(ctx, dtype=np.float32)
    valid[ctx - k:] = 1.0
    return padded, valid


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
        重建 (traj_idx, tau) 索引。tau 为「待预测动作」的时间步。

        对齐语义与在线推理完全一致（见 __getitem__）：预测 action_tau 时模型可见
        观测 obs_0..obs_tau（含当前 obs_tau）与历史动作 act_0..act_{tau-1}。
        早期步 (tau < ctx-1) 通过左填充 + key_padding_mask 处理，与推理起步阶段一致。

        peak 截断: tau ∈ [0, min(peak_t, T-1)]，排除越过最高点后的跌落段。
        """
        self._index = []
        for ti, traj in enumerate(self.trajectories):
            peak_t = int(traj.raw_states[:, 1].argmax())
            max_target = min(peak_t, len(traj.actions) - 1)
            for tau in range(max_target + 1):
                self._index.append((ti, tau))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        ti, tau = self._index[idx]
        traj = self.trajectories[ti]
        ctx = self.config.context_len
        raw = traj.raw_states       # (T+1, 33)
        acts = traj.actions          # (T, 2)

        # 观测窗口: obs_{tau-ctx+1 .. tau}（含当前步 tau）的真实切片，再经 left_pad_sequence 左填充
        obs_real = raw[max(0, tau - ctx + 1): tau + 1]      # (k_obs, 33)
        dyn_real = build_dynamics(obs_real)                 # (k_obs, DYNAMICS_DIM)
        dynamics, valid_mask = left_pad_sequence(dyn_real, ctx)

        pat_real = np.stack([self._build_patch(s) for s in obs_real])  # (k_obs, ch, ps, ps)
        patches, _ = left_pad_sequence(pat_real, ctx)

        # 动作历史窗口: act_{tau-ctx .. tau-1}（当前步之前）的真实切片，同样左填充
        if tau == 0:
            act_real = np.zeros((0, 2), dtype=np.float32)
        else:
            act_real = acts[max(0, tau - ctx): tau].astype(np.float32)  # (k_act, 2)
        input_actions, _ = left_pad_sequence(act_real, ctx)

        # 有效性由观测驱动（与推理 _build_history_window 共用 left_pad_sequence，天然一致）
        target_action = acts[tau].astype(np.float32)

        return (
            torch.from_numpy(dynamics),
            torch.from_numpy(patches),
            torch.from_numpy(input_actions),
            torch.from_numpy(valid_mask),
            torch.from_numpy(target_action),
        )

    def _build_patch(self, step_state: np.ndarray) -> np.ndarray:
        """即时构建 4ch patch（委托模块级 build_patch，保证离线/在线一致）。"""
        return build_patch(
            step_state,
            self.terrain_mask,
            self.efficiency_arr,
            self.config,
            self._min_gx,
            self._min_gy,
        )

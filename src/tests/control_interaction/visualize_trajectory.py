"""
通用轨迹可视化工具

提供四类绘图函数，供控制有效性验证脚本调用：
  - plot_trajectories      : 多条轨迹在指定状态维度上的折线叠加图
  - plot_distance_over_time: 两条轨迹逐步欧式距离折线图
  - plot_dose_response     : 幅值-响应曲线（含误差棒与饱和点标注）
  - plot_dim_max_diff      : 各状态维度最大差异柱状图（用于可重复性测试）
  - plot_dist_heatmap      : 序列终态配对距离矩阵热图（用于区分性测试）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional

matplotlib.rcParams["font.family"] = "sans-serif"

STATE_LABELS = [
    "player_x", "player_y", "vel_x", "vel_y", "ang_vel",
    "hub_x", "hub_y", "hub_vx", "hub_vy", "hub_angle",
    "slider_x", "slider_y", "slider_vx", "slider_vy", "slider_angle",
    "handle_x", "handle_y", "handle_vx", "handle_vy",
    "pole_x", "pole_y", "pole_vx", "pole_vy",
    "tip_x", "tip_y", "tip_vx", "tip_vy",
    "hammer_angle", "timestamp",
]


def plot_trajectories(
    trajs: list,
    labels: list,
    dims: list = (0, 1),
    title: str = "Trajectories",
    save_path: Optional[str] = None,
) -> None:
    """
    绘制多条轨迹在指定状态维度上的折线叠加图。

    参数：
        trajs     : list of ndarray，每个元素 shape (T, STATE_DIM)
        labels    : 每条轨迹对应的图例标签
        dims      : 要绘制的状态维度索引列表
        title     : 图标题
        save_path : 若非 None，则保存到文件而非显示
    """
    n_dims = len(dims)
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 3 * n_dims), squeeze=False)
    fig.suptitle(title, fontsize=13)

    colors = plt.cm.tab10.colors

    for row, dim in enumerate(dims):
        ax = axes[row, 0]
        dim_label = STATE_LABELS[dim] if dim < len(STATE_LABELS) else f"dim_{dim}"
        for i, (traj, lbl) in enumerate(zip(trajs, labels)):
            traj = np.asarray(traj)
            ax.plot(traj[:, dim], label=lbl, color=colors[i % len(colors)], linewidth=1.2)
        ax.set_ylabel(dim_label)
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_distance_over_time(
    obs_a: np.ndarray,
    obs_b: np.ndarray,
    title: str = "State Distance Over Time",
    threshold: Optional[float] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    绘制两条轨迹逐步欧式距离折线图。

    参数：
        obs_a, obs_b : ndarray shape (T, STATE_DIM)
        threshold    : 若非 None，绘制水平参考线
        save_path    : 若非 None，则保存到文件而非显示
    """
    obs_a = np.asarray(obs_a)
    obs_b = np.asarray(obs_b)
    dists = np.linalg.norm(obs_a - obs_b, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dists, color="steelblue", linewidth=1.4, label="‖obs_a − obs_b‖")
    if threshold is not None:
        ax.axhline(threshold, color="tomato", linestyle="--", linewidth=1,
                   label=f"threshold={threshold}")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("step")
    ax.set_ylabel("Euclidean distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_dose_response(
    amplitudes: list,
    responses_mean: list,
    responses_std: Optional[list] = None,
    saturation_point: Optional[float] = None,
    title: str = "Dose-Response Curve",
    ylabel: str = "Δplayer_y",
    save_path: Optional[str] = None,
) -> None:
    """
    绘制幅值-响应曲线（含误差棒与饱和点标注）。

    参数：
        amplitudes       : 动作幅值列表
        responses_mean   : 每个幅值对应的均值响应量
        responses_std    : 每个幅值对应的标准差（可选）
        saturation_point : 饱和点幅值（可选，绘制竖线标注）
        ylabel           : Y 轴标签
        save_path        : 若非 None，则保存到文件而非显示
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)

    # 左图：幅值 → 响应量
    ax = axes[0]
    ax.errorbar(
        amplitudes, responses_mean,
        yerr=responses_std if responses_std is not None else None,
        fmt="o-", color="steelblue", linewidth=1.4, capsize=4,
    )
    if saturation_point is not None:
        ax.axvline(saturation_point, color="tomato", linestyle="--", linewidth=1,
                   label=f"saturation≈{saturation_point}")
        ax.legend()
    ax.set_xlabel("Action amplitude")
    ax.set_ylabel(ylabel)
    ax.set_title("Amplitude → Response")
    ax.grid(True, alpha=0.3)

    # 右图：边际增益率
    ax2 = axes[1]
    responses_mean = np.asarray(responses_mean)
    if len(responses_mean) > 1:
        marginal = np.diff(responses_mean) / (np.abs(responses_mean[:-1]) + 1e-9) * 100
        mid_amps = [(amplitudes[i] + amplitudes[i + 1]) / 2 for i in range(len(amplitudes) - 1)]
        ax2.bar(mid_amps, marginal, width=np.diff(amplitudes) * 0.6,
                color="steelblue", alpha=0.7)
        ax2.axhline(5, color="tomato", linestyle="--", linewidth=1, label="5% threshold")
        ax2.set_xlabel("Action amplitude (midpoint)")
        ax2.set_ylabel("Marginal gain (%)")
        ax2.set_title("Marginal Gain Rate")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_dim_max_diff(
    per_dim_diff: np.ndarray,
    threshold: float = 1e-4,
    title: str = "Per-Dimension Max Difference Across Trials",
    save_path: Optional[str] = None,
) -> None:
    """
    绘制各状态维度跨轮最大差异柱状图，超过阈值的维度用红色高亮。

    参数：
        per_dim_diff : ndarray shape (STATE_DIM,)
        threshold    : 警告阈值（超过则红色显示）
        save_path    : 若非 None，则保存到文件而非显示
    """
    n = len(per_dim_diff)
    labels = [STATE_LABELS[i] if i < len(STATE_LABELS) else f"dim_{i}" for i in range(n)]
    colors = ["tomato" if v > threshold else "steelblue" for v in per_dim_diff]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.5), 4))
    ax.bar(range(n), per_dim_diff, color=colors)
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1,
               label=f"threshold={threshold}")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Max diff across trials")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_dist_heatmap(
    dist_matrix: np.ndarray,
    labels: list,
    threshold: Optional[float] = None,
    title: str = "Pairwise Final-State Distance Matrix",
    save_path: Optional[str] = None,
) -> None:
    """
    绘制序列终态配对欧式距离矩阵热图（用于 L4-B 区分性测试）。

    参数：
        dist_matrix : ndarray shape (N, N)，对角线为 0
        labels      : N 个序列名称
        threshold   : 若非 None，在色条旁标注最小通过阈值
        save_path   : 若非 None，则保存到文件而非显示
    """
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = len(labels)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.2), max(4, n * 1.0)))
    im = ax.imshow(dist_matrix, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=dist_matrix.max() or 1.0)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = dist_matrix[i, j]
            text_color = "white" if val > (dist_matrix.max() or 1.0) * 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Euclidean distance (physics dims)")

    if threshold is not None:
        ax.set_title(f"{title}\n(threshold={threshold}, blue=min required)", fontsize=11)
    else:
        ax.set_title(title, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

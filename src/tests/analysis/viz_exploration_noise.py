"""可视化对比三种探索噪声：白噪声(iid 高斯) / OU 过程 / pink 噪声。

用途：直观说明为什么"逐帧 iid 高斯"（项目当前 PPO 探索）难以撞出连贯攀爬动作，
而 OU / pink 噪声具有时间相关性，更容易产生连贯的锤子运动基元。

运行：
    python -m src.tests.analysis.viz_exploration_noise
产物：
    src/Data/analysis/exploration_noise_compare.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def white_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """逐帧独立高斯（项目当前用法）：每帧一个互不相关的采样。"""
    return rng.normal(0.0, sigma, size=n)


def ou_process(
    n: int, sigma: float, theta: float, dt: float, rng: np.random.Generator
) -> np.ndarray:
    """Ornstein-Uhlenbeck 过程（均值回复的相关噪声）。

    离散更新: x_{t+1} = x_t - theta * x_t * dt + sigma * sqrt(dt) * N(0,1)
    - theta 越小 → 相关时间越长（越"黏"、越连贯）
    - 均值回复保证不会无界漂移，稳态标准差 ≈ sigma / sqrt(2*theta)
    机器人/连续控制 RL 常用（如 DDPG 原论文）来做时间相关探索。
    """
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = x[t - 1] - theta * x[t - 1] * dt + sigma * np.sqrt(dt) * rng.normal()
    return x


def pink_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """pink(1/f) 噪声：功率谱 ∝ 1/f，低频(慢)成分强、高频(抖)成分弱。

    用谱整形实现：对白噪声做 FFT，按 1/sqrt(f) 缩放幅度后逆变换。
    介于白噪声(平谱)与布朗噪声(1/f^2)之间，兼具"连贯"与"仍在探索多尺度"。
    """
    white = rng.normal(0.0, 1.0, size=n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])  # 1/sqrt(f)：功率 ∝ 1/f
    shaped = spectrum * scale
    out = np.fft.irfft(shaped, n=n)
    out = out - out.mean()
    out = out / (out.std() + 1e-12) * sigma  # 归一到目标 std，便于公平对比
    return out


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """归一化自相关（lag=0 处为 1），衡量"时间上有多黏"。"""
    x = x - x.mean()
    full = np.correlate(x, x, mode="full")
    mid = len(full) // 2
    ac = full[mid : mid + max_lag + 1]
    return ac / ac[0]


def main() -> None:
    rng = np.random.default_rng(0)

    # 场景参数：对齐项目控制频率
    fs = 50.0            # stepFrames=1 → ~50Hz 控制
    dt = 1.0 / fs
    n = 250              # 5 秒 = 250 帧
    t = np.arange(n) * dt

    # 目标动作幅度（贴合项目：中性噪声 std≈20 单位，动作范围 ±100）
    sigma = 20.0
    action_scale = 100.0

    # 三种噪声（都归一到相近幅度，公平对比"形状"而非"大小"）
    w = white_noise(n, sigma, rng)
    ou = ou_process(n, sigma * np.sqrt(2 * 3.0), theta=3.0, dt=dt, rng=rng)
    pk = pink_noise(n, sigma, rng)

    series = [
        ("White 白噪声 (iid 高斯) — 项目当前", w, "#d62728"),
        ("OU 过程 (theta=3, ~0.33s 相关)", ou, "#1f77b4"),
        ("Pink 噪声 (1/f)", pk, "#2ca02c"),
    ]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, width_ratios=[2.0, 1.0], hspace=0.45, wspace=0.25)

    # 左列：三条时间序列（作为 dx 动作分量）
    for i, (name, x, c) in enumerate(series):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(t, x, color=c, lw=1.2)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.fill_between(t, -action_scale, action_scale, color="gray", alpha=0.05)
        ax.set_ylim(-action_scale, action_scale)
        ax.set_ylabel("dx 动作")
        ax.set_title(name, fontsize=11, loc="left")
        if i == len(series) - 1:
            ax.set_xlabel("时间 (秒)  —  50Hz 控制，共 5 秒")
        # 标注"连贯性"直觉
        zero_cross = np.sum(np.abs(np.diff(np.sign(x)))) / 2
        ax.text(
            0.99,
            0.04,
            f"过零 {int(zero_cross)} 次（越少越连贯）",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=c, alpha=0.8),
        )

    # 右上：自相关对比（核心：时间相关性）
    ax_ac = fig.add_subplot(gs[0:2, 1])
    max_lag = 60
    lags = np.arange(max_lag + 1) * dt
    for name, x, c in series:
        ax_ac.plot(lags, autocorr(x, max_lag), color=c, lw=1.8, label=name.split(" —")[0].split(" (")[0])
    ax_ac.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax_ac.set_title("自相关：动作在时间上有多"黏"", fontsize=11, loc="left")
    ax_ac.set_xlabel("时间间隔 lag (秒)")
    ax_ac.set_ylabel("归一化自相关")
    ax_ac.legend(fontsize=8, loc="upper right")
    ax_ac.grid(alpha=0.2)

    # 右下：累积轨迹（把噪声当速度积分 → 直观看"走出多远的连贯位移"）
    ax_cum = fig.add_subplot(gs[2, 1])
    for name, x, c in series:
        ax_cum.plot(t, np.cumsum(x) * dt, color=c, lw=1.5)
    ax_cum.set_title("噪声积分（连贯位移）", fontsize=11, loc="left")
    ax_cum.set_xlabel("时间 (秒)")
    ax_cum.set_ylabel("∫ 动作 dt")
    ax_cum.grid(alpha=0.2)

    fig.suptitle(
        "探索噪声对比：白噪声每帧乱跳(难凑出连贯攀爬基元) vs OU/Pink 时间相关(易产生连贯锤子运动)",
        fontsize=13,
    )

    out_dir = Path(__file__).resolve().parents[3] / "src" / "Data" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exploration_noise_compare.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()

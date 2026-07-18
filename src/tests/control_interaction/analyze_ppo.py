"""分析 PPO 训练 checkpoints — 提取参数统计和训练趋势。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

def load_checkpoints(ckpt_dir: Path) -> list[tuple[int, dict]]:
    """加载所有 checkpoint，按迭代号排序。"""
    results = []
    for f in sorted(ckpt_dir.glob("ppo_iter_*.pt")):
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
        iteration = ckpt.get("iteration", int(f.stem.split("_")[-1]))
        results.append((iteration, ckpt))
    return results


def analyze_checkpoints(checkpoints: list[tuple[int, dict]]) -> dict:
    """从 checkpoints 中提取参数统计。"""
    stats = {
        "iterations": [],
        "actor_log_std": [],
        "actor_std": [],
        "param_norm": [],
        "actor_mean_weight_norm": [],
        "critic_weight_norm": [],
        "backbone_weight_norm": [],
        "grad_momentum_norm": [],
        "grad_variance_norm": [],
    }

    for iteration, ckpt in checkpoints:
        model_state = ckpt["model"]
        stats["iterations"].append(iteration)

        log_std = model_state["actor_log_std"].numpy()
        stats["actor_log_std"].append(log_std.copy())
        stats["actor_std"].append(np.exp(log_std).copy())

        total_norm = 0.0
        actor_norm = 0.0
        critic_norm = 0.0
        backbone_norm = 0.0
        for key, param in model_state.items():
            p = param.float().numpy()
            n = np.sum(p ** 2)
            total_norm += n
            if key.startswith("actor_"):
                actor_norm += n
            elif key.startswith("critic_"):
                critic_norm += n
            else:
                backbone_norm += n

        stats["param_norm"].append(np.sqrt(total_norm))
        stats["actor_mean_weight_norm"].append(np.sqrt(actor_norm))
        stats["critic_weight_norm"].append(np.sqrt(critic_norm))
        stats["backbone_weight_norm"].append(np.sqrt(backbone_norm))

        opt = ckpt.get("optimizer")
        if opt and "state" in opt:
            mom_norm = 0.0
            var_norm = 0.0
            for s in opt["state"].values():
                if "exp_avg" in s:
                    mom_norm += float(s["exp_avg"].float().norm() ** 2)
                if "exp_avg_sq" in s:
                    var_norm += float(s["exp_avg_sq"].float().norm() ** 2)
            stats["grad_momentum_norm"].append(np.sqrt(mom_norm))
            stats["grad_variance_norm"].append(np.sqrt(var_norm))
        else:
            stats["grad_momentum_norm"].append(0.0)
            stats["grad_variance_norm"].append(0.0)

    return stats


def compute_param_diffs(checkpoints: list[tuple[int, dict]]) -> dict:
    """计算相邻 checkpoint 之间参数差异。"""
    diffs = {"iterations": [], "total_delta": [], "actor_delta": [], "critic_delta": [], "backbone_delta": []}
    for i in range(1, len(checkpoints)):
        _, ckpt_prev = checkpoints[i - 1]
        _, ckpt_curr = checkpoints[i]
        diffs["iterations"].append(checkpoints[i][0])

        total_d = 0.0
        actor_d = 0.0
        critic_d = 0.0
        backbone_d = 0.0
        for key in ckpt_curr["model"]:
            delta = (ckpt_curr["model"][key].float() - ckpt_prev["model"][key].float()).numpy()
            d = np.sum(delta ** 2)
            total_d += d
            if key.startswith("actor_"):
                actor_d += d
            elif key.startswith("critic_"):
                critic_d += d
            else:
                backbone_d += d

        diffs["total_delta"].append(np.sqrt(total_d))
        diffs["actor_delta"].append(np.sqrt(actor_d))
        diffs["critic_delta"].append(np.sqrt(critic_d))
        diffs["backbone_delta"].append(np.sqrt(backbone_d))

    return diffs


def plot_analysis(stats: dict, diffs: dict, out_path: Path) -> None:
    """绘制 6 子图分析面板。"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    iters = stats["iterations"]

    # 1. Actor log_std 随训练变化
    ax = axes[0, 0]
    log_stds = np.array(stats["actor_log_std"])
    for dim in range(log_stds.shape[1]):
        ax.plot(iters, log_stds[:, dim], "o-", label=f"dim {dim}")
    ax.set_title("Actor log_std (exploration noise)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("log_std")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Actor std（指数空间）
    ax = axes[0, 1]
    stds = np.array(stats["actor_std"])
    for dim in range(stds.shape[1]):
        ax.plot(iters, stds[:, dim], "o-", label=f"dim {dim}")
    ax.set_title("Actor std (action noise magnitude)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("std")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 参数范数
    ax = axes[1, 0]
    ax.plot(iters, stats["backbone_weight_norm"], "s-", label="Backbone", alpha=0.8)
    ax.plot(iters, stats["critic_weight_norm"], "^-", label="Critic", alpha=0.8)
    ax.plot(iters, stats["actor_mean_weight_norm"], "D-", label="Actor", alpha=0.8)
    ax.set_title("Parameter L2 Norm by Component")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 相邻 checkpoint 参数变化量
    ax = axes[1, 1]
    if diffs["iterations"]:
        ax.plot(diffs["iterations"], diffs["total_delta"], "o-", label="Total", alpha=0.8)
        ax.plot(diffs["iterations"], diffs["backbone_delta"], "s-", label="Backbone", alpha=0.8)
        ax.plot(diffs["iterations"], diffs["critic_delta"], "^-", label="Critic", alpha=0.8)
        ax.plot(diffs["iterations"], diffs["actor_delta"], "D-", label="Actor", alpha=0.8)
    ax.set_title("Parameter Delta between Checkpoints")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 Delta")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Optimizer 动量 & 方差范数
    ax = axes[2, 0]
    ax.plot(iters, stats["grad_momentum_norm"], "o-", label="exp_avg (momentum)", alpha=0.8)
    ax.plot(iters, stats["grad_variance_norm"], "s-", label="exp_avg_sq (variance)", alpha=0.8)
    ax.set_title("AdamW Optimizer State Norms")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 总参数范数趋势
    ax = axes[2, 1]
    ax.plot(iters, stats["param_norm"], "o-", color="purple", alpha=0.8)
    ax.set_title("Total Model Parameter L2 Norm")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 Norm")
    ax.grid(True, alpha=0.3)

    fig.suptitle("PPO Training Checkpoint Analysis", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"分析图已保存: {out_path}")


def print_summary(stats: dict, diffs: dict) -> None:
    """打印文本摘要。"""
    print("=" * 70)
    print("PPO 训练 Checkpoint 分析摘要")
    print("=" * 70)
    print(f"  Checkpoints 数量: {len(stats['iterations'])}")
    print(f"  迭代范围: {stats['iterations'][0]} → {stats['iterations'][-1]}")
    print()

    print("── Actor 探索噪声 (std) ──")
    stds = np.array(stats["actor_std"])
    for dim in range(stds.shape[1]):
        print(f"  dim {dim}: {stds[0, dim]:.6f} → {stds[-1, dim]:.6f}  "
              f"(变化 {(stds[-1, dim] / stds[0, dim] - 1) * 100:+.1f}%)")

    print()
    print("── 参数范数趋势 ──")
    print(f"  Backbone: {stats['backbone_weight_norm'][0]:.2f} → {stats['backbone_weight_norm'][-1]:.2f}")
    print(f"  Critic:   {stats['critic_weight_norm'][0]:.2f} → {stats['critic_weight_norm'][-1]:.2f}")
    print(f"  Actor:    {stats['actor_mean_weight_norm'][0]:.4f} → {stats['actor_mean_weight_norm'][-1]:.4f}")
    print(f"  Total:    {stats['param_norm'][0]:.2f} → {stats['param_norm'][-1]:.2f}")

    if diffs["iterations"]:
        print()
        print("── 参数变化幅度 (相邻 checkpoint 间) ──")
        deltas = np.array(diffs["total_delta"])
        print(f"  平均 delta: {deltas.mean():.4f}")
        print(f"  最大 delta: {deltas.max():.4f} (iter {diffs['iterations'][np.argmax(deltas)]})")
        print(f"  最小 delta: {deltas.min():.4f} (iter {diffs['iterations'][np.argmin(deltas)]})")

    if stats["grad_momentum_norm"][-1] > 0:
        print()
        print("── Optimizer 状态 (最新 checkpoint) ──")
        print(f"  Momentum norm:  {stats['grad_momentum_norm'][-1]:.4f}")
        print(f"  Variance norm:  {stats['grad_variance_norm'][-1]:.4f}")

    print("=" * 70)


def main():
    ckpt_dir = REPO_ROOT / "checkpoints"
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"扫描 checkpoint 目录: {ckpt_dir}")
    checkpoints = load_checkpoints(ckpt_dir)
    if not checkpoints:
        print("未找到 checkpoint 文件！")
        return

    print(f"找到 {len(checkpoints)} 个 checkpoint: "
          f"iter {[c[0] for c in checkpoints]}")

    stats = analyze_checkpoints(checkpoints)
    diffs = compute_param_diffs(checkpoints)
    print_summary(stats, diffs)
    plot_analysis(stats, diffs, log_dir / "ppo_checkpoint_analysis.png")


if __name__ == "__main__":
    main()

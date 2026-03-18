"""
PPO 训练入口 — 策略梯度迭代循环。

每轮: 启动游戏 → collect_ppo → 关闭游戏 → PPO update → 日志/checkpoint

用法:
  python -m src.training.main_ppo
  python -m src.training.main_ppo --num-agents 5 --max-iterations 200
  python -m src.training.main_ppo --steps-per-rollout 300
  python -m src.training.main_ppo --resume checkpoints/ppo_iter_0050.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.ppo_trainer import PPOTrainer
from training.reward import ClimbingEfficiencyMap
from training.rollout import RolloutWorker

sys.path.insert(0, str(_REPO_ROOT / "src" / "tests" / "control_interaction"))
from test_l7_surface_airdrop import (
    compute_landable_surfaces,
    compute_player_height,
    extract_polygons,
    load_colliders,
)

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def save_checkpoint(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    checkpoint_dir: str,
) -> None:
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ppo_iter_{iteration:04d}.pt"
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    state["iteration"] = iteration
    torch.save(state, path)
    logger.info("Checkpoint saved: %s", path)


def log_metrics(iteration: int, metrics: dict[str, float], extras: dict | None = None) -> None:
    parts = [f"iter={iteration}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v:.6f}")
    if extras:
        for k, v in extras.items():
            parts.append(f"{k}={v:.4f}")
    logger.info("  ".join(parts))


def save_efficiency_map(
    eff_map: ClimbingEfficiencyMap,
    env_data: dict,
    iteration: int,
    log_dir: str,
) -> None:
    """绘制全局效率图并保存到 log_dir（复用 main_train 的逻辑）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection

    arr = eff_map.get_arr()
    if arr is None:
        return

    res = eff_map.resolution
    min_gx = eff_map._min_gx
    min_gy = eff_map._min_gy
    h, w = arr.shape

    x_lo = (min_gx + 0.5) * res
    x_hi = (min_gx + w - 0.5) * res
    y_lo = (min_gy + 0.5) * res
    y_hi = (min_gy + h - 0.5) * res
    extent = [x_lo, x_hi, y_lo, y_hi]

    fig, ax = plt.subplots(figsize=(20, 12))

    excluded = {"Snake"}
    env_patches = []
    for collider in env_data.get("colliders", []):
        if collider.get("name") in excluded:
            continue
        for path in collider.get("paths", []):
            pts = np.array(path, dtype=np.float64)
            if len(pts) >= 3:
                env_patches.append(plt.Polygon(pts, closed=True))
    if env_patches:
        pc = PatchCollection(
            env_patches, facecolor="#d0d0d0", edgecolor="#555555",
            linewidth=0.3, alpha=0.6, zorder=1,
        )
        ax.add_collection(pc)

    masked = np.where(eff_map.traversable & (arr > 0), arr, np.nan)
    im = ax.imshow(
        masked, origin="lower", extent=extent,
        cmap="inferno", interpolation="nearest",
        alpha=0.85, zorder=2,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Efficiency Value", fontsize=10)

    vmax = float(np.nanmax(masked)) if np.any(~np.isnan(masked)) else 0
    ax.set_title(
        f"PPO Climbing Efficiency Map — iter {iteration}  "
        f"(max={vmax:.1f}, grid={res}m)",
        fontsize=13,
    )
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ppo_effmap_iter_{iteration:04d}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("效率图已保存: %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO 训练 — 策略梯度迭代循环")
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--steps-per-rollout", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从指定 checkpoint 恢复训练（如 checkpoints/ppo_iter_0050.pt）",
    )
    parser.add_argument("--no-launch", action="store_true", help="不启动游戏（假设已运行）")
    parser.add_argument(
        "--random-deploy", action="store_true",
        help="随机投放 agent 到地图表面（默认所有 agent 留在初始位置）",
    )
    parser.add_argument(
        "--resume-bc", type=str, default=None,
        help="从 BC（ActionPredictor）checkpoint 迁移 backbone 权重初始化 PPO",
    )
    return parser.parse_args()


def main() -> None:
    _setup_logging()
    args = parse_args()

    config = TrainConfig()
    if args.num_agents is not None:
        config.num_agents = args.num_agents
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.steps_per_rollout is not None:
        config.steps_per_rollout = args.steps_per_rollout
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    if args.lr is not None:
        config.lr = args.lr
    if args.ent_coef is not None:
        config.ent_coef = args.ent_coef
    if args.random_deploy:
        config.random_deploy = True

    logger.info("PPO Config: %s", config)

    # ── 初始化 ──
    model = ActorCritic(config)
    ppo_trainer = PPOTrainer(config)
    rollout_worker = RolloutWorker(config)
    start_iteration = 0

    if args.resume:
        ckpt = torch.load(args.resume, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_iteration = ckpt.get("iteration", 0)
        if "optimizer" in ckpt:
            ppo_trainer.set_pending_optimizer_state(ckpt["optimizer"])
        logger.info("从 PPO checkpoint 恢复 (iteration=%d): %s", start_iteration, args.resume)
    elif args.resume_bc:
        bc_state = torch.load(args.resume_bc, weights_only=True)
        model.load_from_bc(bc_state)
        logger.info("从 BC checkpoint 迁移 backbone 权重: %s", args.resume_bc)

    # ── 加载地形数据 ──
    game_root = Path(config.game_root)
    env_data, player_data = load_colliders(game_root)
    polygons = extract_polygons(env_data)

    cache_dir = Path(config.checkpoint_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 效率图 ──
    mask_cache = str(cache_dir / "traversable_mask.npz")
    eff_map = ClimbingEfficiencyMap(
        polygons,
        grid_resolution=config.grid_resolution,
        diffusion_iterations=config.diffusion_iterations,
        diffusion_alpha=config.diffusion_alpha,
        cache_path=mask_cache,
        height_prior_weight=config.height_prior_weight,
    )

    rollout_worker.set_terrain_info(
        eff_map.traversable, eff_map._min_gx, eff_map._min_gy,
    )

    if config.random_deploy:
        segments_cache = cache_dir / "landable_segments.npz"
        if segments_cache.exists():
            seg_data = np.load(segments_cache, allow_pickle=True)
            segments = list(seg_data["segments"])
            logger.info("可着陆表面 (缓存): %d 条线段", len(segments))
        else:
            player_height = compute_player_height(player_data)
            min_clearance = player_height + config.surface_padding
            segments = compute_landable_surfaces(
                polygons, min_clearance,
                resolution=0.1,
                y_max_cutoff=config.y_max_cutoff,
            )
            np.savez_compressed(
                segments_cache,
                segments=np.array(segments, dtype=object),
                min_clearance=np.array(min_clearance),
            )
            logger.info("可着陆表面: %d 条线段 (已缓存)", len(segments))
        rollout_worker.set_surface_segments(segments)
    else:
        logger.info("初始位置模式: 跳过表面线段加载")

    # ── 主循环 ──
    try:
        for iteration in range(start_iteration, config.max_iterations):
            logger.info("=" * 60)
            logger.info("=== PPO 迭代 %d/%d ===", iteration + 1, config.max_iterations)

            # Phase 1: 采集
            if not args.no_launch:
                rollout_worker.launch_game()
            else:
                if rollout_worker.env is None:
                    from env import GoiEnv
                    rollout_worker.env = GoiEnv(
                        port=config.port, num_agents=config.num_agents,
                    )
                    rollout_worker.env.connect()
                    rollout_worker.env.reset()

            logger.info("Phase 1: PPO 采集 (%d 步 × %d agents)",
                        config.steps_per_rollout, config.num_agents)
            buffer, trajectories = rollout_worker.collect_ppo(
                model, config.steps_per_rollout, eff_map=eff_map,
            )
            logger.info("  采集完成: buffer size=%d, trajectories=%d",
                        buffer.size, len(trajectories))

            if not args.no_launch:
                rollout_worker.close_game()

            if buffer.size == 0:
                logger.warning("buffer 为空，跳过本轮 PPO 更新")
                continue

            # Phase 1.5: 更新效率图
            if trajectories:
                prev_eff_map = eff_map.copy()
                eff_map.update(trajectories, config, prev_eff_map)

            # Phase 2: PPO 更新
            logger.info("Phase 2: PPO 更新 (%d epochs, batch_size=%d)",
                        config.ppo_epochs, config.ppo_batch_size)
            metrics = ppo_trainer.update(model, buffer)

            # Phase 3: 日志 & 保存
            reward_arr = np.array(buffer._rewards)
            extras = {
                "mean_reward": float(reward_arr.mean()),
                "sum_reward": float(reward_arr.sum()),
                "buffer_size": float(buffer.size),
            }
            log_metrics(iteration + 1, metrics, extras)

            if (iteration + 1) % config.save_interval == 0:
                save_checkpoint(
                    model, ppo_trainer.optimizer,
                    iteration + 1, config.checkpoint_dir,
                )
                save_efficiency_map(eff_map, env_data, iteration + 1, config.log_dir)

        # 最终保存
        save_checkpoint(
            model, ppo_trainer.optimizer,
            config.max_iterations, config.checkpoint_dir,
        )
        logger.info("PPO 训练完成")

    finally:
        rollout_worker.teardown()


if __name__ == "__main__":
    main()

"""
入口 — 迭代式训练主循环。

Phase 0: 冷启动 — 多轮随机采集凑够好轨迹
Phase 1..N: 模型采集 → 效率图更新 → 评分筛选 → Fine-tune → 保存

用法:
  python -m src.training.main_train
  python -m src.training.main_train --num-agents 5 --max-iterations 50
  python -m src.training.main_train --resume              # 跳过冷启动
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig
from training.dataset import TrajectoryDataset, compute_dynamics_stats
from training.model import ActionPredictor
from training.reward import ClimbingEfficiencyMap, base_score, score_trajectory
from training.rollout import RolloutWorker
from training.trainer import Trainer

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
    model: ActionPredictor,
    iteration: int,
    checkpoint_dir: str,
) -> None:
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"model_iter_{iteration:04d}.pt"
    torch.save(model.state_dict(), path)
    logger.info("Checkpoint saved: %s", path)


def calibrate_normalizer(model, trajectories: list) -> None:
    """用轨迹集合标定观测归一化器（仅首次，之后冻结）。"""
    if model.dynamics_stats_ready:
        return
    mean, std = compute_dynamics_stats(trajectories)
    model.set_dynamics_stats(mean, std)
    logger.info(
        "观测归一化标定完成: dim=%d, mean|·|~%.4f, std~%.4f",
        len(mean), float(np.abs(mean).mean()), float(std.mean()),
    )


def log_metrics(iteration: int, metrics: dict[str, float]) -> None:
    parts = [f"iter={iteration}"]
    for k, v in metrics.items():
        parts.append(f"{k}={v:.6f}")
    logger.info("  ".join(parts))


def save_efficiency_map(
    eff_map: ClimbingEfficiencyMap,
    env_data: dict,
    iteration: int,
    log_dir: str,
) -> None:
    """绘制全局效率图并保存到 log_dir。"""
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

    # 地形多边形（排除 Snake 等非地形碰撞体）
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

    # 效率图热力图（只显示有值的区域）
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
        f"Climbing Efficiency Map — iter {iteration}  "
        f"(max={vmax:.1f}, grid={res}m)",
        fontsize=13,
    )
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    out_dir = Path(log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"effmap_iter_{iteration:04d}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    data_path = out_dir / f"effmap_iter_{iteration:04d}.npz"
    np.savez_compressed(
        data_path,
        arr=arr,
        max_arr=eff_map._max_arr,
        min_gx=np.array(eff_map._min_gx),
        min_gy=np.array(eff_map._min_gy),
        resolution=np.array(eff_map.resolution),
    )
    logger.info("效率图已保存: %s + %s", path, data_path)


_WARMUP_FILENAME = "warmup_state.pkl"


def save_warmup_state(
    checkpoint_dir: str,
    good_trajs: list,
) -> None:
    """冷启动完成后将轨迹序列化到磁盘。"""
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / _WARMUP_FILENAME

    traj_data = [
        {
            "raw_states": t.raw_states,
            "actions": t.actions,
            "score": t.score,
            "iteration": t.iteration,
        }
        for t in good_trajs
    ]
    state = {"trajectories": traj_data}
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("冷启动状态已保存: %s (%.1f MB)", path, path.stat().st_size / 1e6)


def load_warmup_state(checkpoint_dir: str) -> list:
    """从磁盘加载冷启动轨迹。"""
    from training.dataset import Trajectory

    path = Path(checkpoint_dir) / _WARMUP_FILENAME
    with open(path, "rb") as f:
        state = pickle.load(f)

    good_trajs = [
        Trajectory(
            raw_states=d["raw_states"],
            actions=d["actions"],
            score=d["score"],
            iteration=d["iteration"],
        )
        for d in state["trajectories"]
    ]

    logger.info("冷启动轨迹已恢复: %d 条", len(good_trajs))
    return good_trajs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL 训练框架 — 迭代式训练")
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--warmup-rollouts", type=int, default=None)
    parser.add_argument("--steps-per-rollout", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument(
        "--resume", action="store_true",
        help="跳过冷启动，从上次保存的 warmup_state.pkl 恢复",
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
    if args.warmup_rollouts is not None:
        config.warmup_rollouts = args.warmup_rollouts
    if args.steps_per_rollout is not None:
        config.steps_per_rollout = args.steps_per_rollout
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir

    logger.info("Config: %s", config)

    # ── 初始化组件 ──
    model = ActionPredictor(config)
    trainer = Trainer(config)
    rollout_worker = RolloutWorker(config)

    game_root = Path(config.game_root)
    env_data, player_data = load_colliders(game_root)
    polygons = extract_polygons(env_data)

    # 计算可着陆表面（缓存到磁盘，首次 ~11s，后续 <1s）
    cache_dir = Path(config.checkpoint_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    segments_cache = cache_dir / "landable_segments.npz"

    if segments_cache.exists():
        seg_data = np.load(segments_cache, allow_pickle=True)
        segments = list(seg_data["segments"])
        min_clearance = float(seg_data["min_clearance"])
        logger.info("可着陆表面 (缓存): %d 条线段 (min_clearance=%.2f)", len(segments), min_clearance)
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
        logger.info("可着陆表面: %d 条线段 (min_clearance=%.2f, 已缓存)", len(segments), min_clearance)

    # 构建效率图（traversable mask 缓存到磁盘，首次 ~87s，后续 <1s）
    mask_cache = str(cache_dir / "traversable_mask.npz")
    eff_map = ClimbingEfficiencyMap(
        polygons,
        grid_resolution=config.grid_resolution,
        diffusion_iterations=config.diffusion_iterations,
        diffusion_alpha=config.diffusion_alpha,
        cache_path=mask_cache,
        height_prior_weight=config.height_prior_weight,
    )
    prev_eff_map = None

    terrain_mask = eff_map.traversable
    dataset = TrajectoryDataset(
        config, terrain_mask, eff_map.get_arr(),
        min_gx=eff_map._min_gx, min_gy=eff_map._min_gy,
    )

    rollout_worker.set_terrain_info(terrain_mask, eff_map._min_gx, eff_map._min_gy)
    rollout_worker.set_surface_segments(segments)

    warmup_path = Path(config.checkpoint_dir) / _WARMUP_FILENAME
    resume = args.resume and warmup_path.exists()

    try:
        if resume:
            # ── 恢复冷启动状态（无需启动游戏）──
            logger.info("=== 恢复模式: 从 %s 加载冷启动状态 ===", warmup_path)
            good_trajs = load_warmup_state(config.checkpoint_dir)
            eff_map.update(good_trajs, config, prev_eff_map=None)
            prev_eff_map = eff_map.copy()
            dataset.update_efficiency_arr(eff_map.get_arr())
            save_efficiency_map(eff_map, env_data, 0, config.log_dir)

            dataset.set_trajectories(good_trajs)

            ckpt_path = Path(config.checkpoint_dir) / "model_iter_0000.pt"
            if ckpt_path.exists():
                model.load_state_dict(torch.load(ckpt_path, weights_only=True))
                logger.info("模型权重已恢复: %s", ckpt_path)
            else:
                logger.info("未找到 model_iter_0000.pt，使用新模型进行首次训练")
                calibrate_normalizer(model, dataset.trajectories)
                logger.info("=== Phase 0: 首次训练 (数据集大小=%d) ===", len(dataset))
                metrics = trainer.train(model, dataset, config)
                log_metrics(0, metrics)
                save_checkpoint(model, 0, config.checkpoint_dir)
        else:
            # ── Phase 0: 冷启动 — 启动游戏 → 采集 → 关闭游戏 → 训练 ──
            rollout_worker.setup()

            logger.info("=== Phase 0: 冷启动 (%d 轮随机采集) ===", config.warmup_rollouts)
            all_random_trajs = []
            for r in range(config.warmup_rollouts):
                trajs = rollout_worker.collect_random(config.steps_per_rollout)
                all_random_trajs.extend(trajs)
                logger.info("  冷启动 %d/%d: 采集 %d 条轨迹", r + 1, config.warmup_rollouts, len(trajs))

            # 采集完成，关闭游戏
            rollout_worker.close_game()

            for t in all_random_trajs:
                t.score = base_score(t.raw_states, config)
                t.iteration = 0

            scores = [t.score for t in all_random_trajs]
            threshold = float(np.percentile(scores, 100 * (1 - config.keep_ratio)))
            good_trajs = [t for t in all_random_trajs if t.score >= threshold]
            good_trajs.sort(key=lambda t: t.score, reverse=True)
            good_trajs = good_trajs[: config.max_good_trajectories]

            logger.info(
                "冷启动完成: %d 条总轨迹 → %d 条好轨迹 (阈值=%.2f)",
                len(all_random_trajs), len(good_trajs), threshold,
            )

            # 初始化效率图
            eff_map.update(all_random_trajs, config, prev_eff_map=None)
            prev_eff_map = eff_map.copy()
            dataset.update_efficiency_arr(eff_map.get_arr())
            save_efficiency_map(eff_map, env_data, 0, config.log_dir)

            # 保存冷启动状态到磁盘
            save_warmup_state(config.checkpoint_dir, good_trajs)

            # 首次训练（游戏已关闭）
            dataset.set_trajectories(good_trajs)
            calibrate_normalizer(model, dataset.trajectories)
            logger.info("=== Phase 0: 首次训练 (数据集大小=%d) ===", len(dataset))
            metrics = trainer.train(model, dataset, config)
            log_metrics(0, metrics)
            save_checkpoint(model, 0, config.checkpoint_dir)

        # ── 正式迭代循环 ──
        for iteration in range(1, config.max_iterations):
            logger.info("=== 迭代 %d/%d ===", iteration, config.max_iterations)

            # Phase 1: 启动游戏 → 多轮模型采集 → 关闭游戏
            rollout_worker.launch_game()
            noise = config.explore_noise_std * (config.noise_decay ** iteration)
            logger.info(
                "Phase 1: 模型采集 (%d 轮, noise_std=%.4f)",
                config.rollouts_per_iteration, noise,
            )
            trajectories = []
            for r in range(config.rollouts_per_iteration):
                batch = rollout_worker.collect_with_model(
                    model, config.steps_per_rollout, noise, eff_map=eff_map,
                )
                trajectories.extend(batch)
                logger.info("  采集 %d/%d: %d 条轨迹", r + 1, config.rollouts_per_iteration, len(batch))
            rollout_worker.close_game()

            # Phase 2a: 增量更新效率图
            logger.info("Phase 2a: 更新效率图")
            eff_map.update(trajectories, config, prev_eff_map)
            prev_eff_map = eff_map.copy()
            dataset.update_efficiency_arr(eff_map.get_arr())

            # Phase 2b: 评分 → top-K% 筛选 → 滑动窗口管理
            scores = [
                score_trajectory(t.raw_states, eff_map, config)
                for t in trajectories
            ]
            dataset.add_trajectories(
                trajectories, scores,
                keep_ratio=config.keep_ratio,
                iteration=iteration,
            )
            dataset.trim_oldest(config.max_good_trajectories)
            logger.info(
                "Phase 2b: %d 条新轨迹, 数据集=%d 条 (%d 训练窗口)",
                len(trajectories), len(dataset.trajectories), len(dataset),
            )

            # Phase 3: Fine-tune（游戏已关闭）
            logger.info("Phase 3: Fine-tune")
            metrics = trainer.train(model, dataset, config)

            # Phase 4: 日志 & 保存
            log_metrics(iteration, metrics)
            if iteration % config.save_interval == 0:
                save_checkpoint(model, iteration, config.checkpoint_dir)
                save_efficiency_map(eff_map, env_data, iteration, config.log_dir)

        # 最终保存
        save_checkpoint(model, config.max_iterations, config.checkpoint_dir)
        logger.info("训练完成")

    finally:
        rollout_worker.teardown()


if __name__ == "__main__":
    main()

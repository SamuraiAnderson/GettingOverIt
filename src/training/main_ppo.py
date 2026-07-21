"""
PPO 训练入口 — 策略梯度迭代循环。

每轮: 启动游戏 → collect_ppo → 关闭游戏 → PPO update → 日志/checkpoint

用法:
  python -m src.training.main_ppo
  python -m src.training.main_ppo --num-agents 5 --max-iterations 200
  python -m src.training.main_ppo --steps-per-rollout 300
  python -m src.training.main_ppo --resume checkpoints/ppo_iter_0050.pt
  # 仅在 125m 以下区域随机投放 + 每轮 500 步（低处课程与步长呼应）
  python -m src.training.main_ppo --random-deploy --y-max-cutoff 125 --steps-per-rollout 500
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
from training.dataset import TrajectoryDataset, compute_dynamics_stats
from training.ppo_trainer import PPOTrainer
from training.deploy_sampling import (
    default_candidate_points_path,
    extract_body_local,
    extract_tip_local,
    filter_points_by_height,
    load_candidate_points,
)
from training.reward import (
    ClimbingEfficiencyMap,
    RewardNormalizer,
    progress_metric,
    score_trajectory,
)
from training.rollout import RolloutWorker

sys.path.insert(0, str(_REPO_ROOT / "src" / "tests" / "control_interaction"))
from test_l7_surface_airdrop import (
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
    parser.add_argument(
        "--y-max-cutoff", type=float, default=None,
        help=(
            "随机投放的可着陆表面高度上限（屏蔽 y 高于此值的表面）；"
            "与 --steps-per-rollout 配套限定低处训练区，如 --y-max-cutoff 125 --steps-per-rollout 500"
        ),
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument(
        "--init-log-std", type=float, default=None,
        help="PPO 探索噪声初值上限 log_std（覆盖 config.ppo_init_log_std；越小噪声越小）",
    )
    parser.add_argument(
        "--log-std-max", type=float, default=None,
        help="PPO 探索噪声 clamp 上限 log_std（覆盖 config.ppo_log_std_max）",
    )
    parser.add_argument(
        "--value-warmup-iters", type=int, default=None,
        help="开局只预热 critic 的迭代数（覆盖 config.value_warmup_iters；0=禁用）",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从指定 checkpoint 恢复训练（如 checkpoints/ppo_iter_0050.pt）",
    )
    parser.add_argument("--no-launch", action="store_true", help="不启动游戏（假设已运行）")
    parser.add_argument(
        "--persist-game", action="store_true",
        help="游戏跨轮常驻：只在开局启动一次,循环内仅 env.reset()（省每轮 ~20s 启停）",
    )
    parser.add_argument(
        "--random-deploy", dest="random_deploy", action="store_true", default=None,
        help="随机投放 agent 到候选点集（覆盖 config.random_deploy=True）",
    )
    parser.add_argument(
        "--no-random-deploy", dest="random_deploy", action="store_false",
        help="所有 agent 留在初始起点（开局专注课程）",
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
    if args.y_max_cutoff is not None:
        config.y_max_cutoff = args.y_max_cutoff
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    if args.lr is not None:
        config.lr = args.lr
    if args.ent_coef is not None:
        config.ent_coef = args.ent_coef
    if args.init_log_std is not None:
        config.ppo_init_log_std = args.init_log_std
    if args.log_std_max is not None:
        config.ppo_log_std_max = args.log_std_max
    if args.value_warmup_iters is not None:
        config.value_warmup_iters = args.value_warmup_iters
    if args.random_deploy is not None:
        config.random_deploy = args.random_deploy

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
        # 候选点集为唯一投放来源（pool-only）：读 L7 唯一候选文件 → train 侧高度过滤
        cand_path = default_candidate_points_path()
        candidates = load_candidate_points(cand_path)
        if not candidates:
            logger.error(
                "候选点集为空/缺失: %s；请先运行 "
                "test_l7_surface_airdrop.py --physics-filter-all 生成",
                cand_path,
            )
        # 高度过滤：候选 y 含 drop_height，故阈值取 y_max_cutoff + drop_height 与 L7 表面口径对齐
        n_before = len(candidates)
        candidates = filter_points_by_height(
            candidates, y_max=config.y_max_cutoff + config.drop_height,
        )
        logger.info(
            "候选点集: %s 载入 %d 个 → 高度过滤(y_max_cutoff=%g) 保留 %d 个",
            cand_path.name, n_before, config.y_max_cutoff, len(candidates),
        )
        rollout_worker.set_candidate_points(candidates)
    else:
        logger.info("初始位置模式: 跳过候选点集加载")

    # patch 接触分支几何（dualscale 需要，无论是否随机投放）；也用于运行时锤头卡障碍过滤
    rollout_worker.set_deploy_obstacle_geometry(polygons, player_data)

    # ── Self-Imitation 精英轨迹池 ──（复用 BC 侧 TrajectoryDataset：top-K% 过滤 + secured-peak
    # 窗口截断 + 用当前 eff_map 即时重建 patch；只存 raw_states/actions，内存开销近零）
    sil_dataset = (
        TrajectoryDataset(
            config, eff_map.traversable, eff_map.get_arr(),
            min_gx=eff_map._min_gx, min_gy=eff_map._min_gy,
            solid_polygons=polygons,
            tip_local=extract_tip_local(player_data),
            body_local=extract_body_local(player_data),
        )
        if config.sil_enabled else None
    )

    # ── 奖励归一化器 ──
    calibration_steps = config.steps_per_rollout * config.num_agents * 10
    reward_normalizer = RewardNormalizer(calibration_steps=calibration_steps)
    logger.info(
        "RewardNormalizer: %d calibration steps (≈%d rollouts)",
        calibration_steps, 10,
    )

    # ── 观测归一化标定 ──
    # resume / resume-bc 已随 checkpoint 携带归一化统计；全新模型需先跑一轮标定 rollout。
    if not model.dynamics_stats_ready:
        logger.info("观测归一化: 全新模型，执行标定 rollout (%d 步)", config.steps_per_rollout)
        _calib_launched = False
        if not args.no_launch:
            rollout_worker.launch_game()
            _calib_launched = True
        elif rollout_worker.env is None:
            from env import GoiEnv
            rollout_worker.env = GoiEnv(port=config.port, num_agents=config.num_agents)
            rollout_worker.env.connect()
            rollout_worker.env.reset()

        calib_trajs = rollout_worker.collect_random(config.steps_per_rollout)
        mean, std = compute_dynamics_stats(calib_trajs)
        model.set_dynamics_stats(mean, std)
        logger.info(
            "观测归一化标定完成: dim=%d, mean|·|~%.4f, std~%.4f",
            len(mean), float(np.abs(mean).mean()), float(std.mean()),
        )
        if _calib_launched:
            rollout_worker.close_game()
    else:
        logger.info("观测归一化: 沿用 checkpoint 携带的统计量")

    # 游戏常驻模式：开局启动一次，循环内跳过每轮启停，仅靠 collect_ppo 的 env.reset() 复位。
    # manage_per_iter=True 时维持原有"每轮启停"行为。
    manage_per_iter = not args.no_launch and not args.persist_game
    if args.persist_game and not args.no_launch and rollout_worker.env is None:
        logger.info("持久游戏模式: 启动一次游戏，后续跨轮常驻（每轮仅 env.reset()，省启停开销）")
        rollout_worker.launch_game()

    # ── 停滞探测器状态（精英池峰值竖直高度停滞 → 自适应提方差）──
    _best_pool_sdy = float("-inf")
    _stall = 0
    _boost = 0.0

    # ── 主循环 ──
    try:
        for iteration in range(start_iteration, config.max_iterations):
            logger.info("=" * 60)
            logger.info("=== PPO 迭代 %d/%d ===", iteration + 1, config.max_iterations)

            # Phase 1: 采集
            if manage_per_iter:
                rollout_worker.launch_game()
            elif rollout_worker.env is None:
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
                normalizer=reward_normalizer,
            )
            logger.info("  采集完成: buffer size=%d, trajectories=%d",
                        buffer.size, len(trajectories))
            if reward_normalizer.is_active:
                logger.info(
                    "  RewardNormalizer active: σ_h=%.4f, σ_e=%.4f, α=%.2f",
                    reward_normalizer.sigma_h, reward_normalizer.sigma_e,
                    config.reward_alpha,
                )

            if manage_per_iter:
                rollout_worker.close_game()

            if buffer.size == 0:
                logger.warning("buffer 为空，跳过本轮 PPO 更新")
                continue

            # Phase 1.5: 更新效率图
            if trajectories:
                prev_eff_map = eff_map.copy()
                eff_map.update(trajectories, config, prev_eff_map)

            # Phase 1.6: 更新 Self-Imitation 精英池（每轮 top-K% 入池 → 按分数全局保留 top-N）
            if sil_dataset is not None and trajectories:
                sil_dataset.update_efficiency_arr(eff_map.get_arr())
                sil_scores = [
                    score_trajectory(t.raw_states, eff_map, config)
                    for t in trajectories
                ]
                sil_dataset.add_trajectories(
                    trajectories, sil_scores,
                    keep_ratio=config.sil_keep_ratio, iteration=iteration,
                )
                # 按分数保留 top-N（非 FIFO，历史最优开局不被时间淘汰），set_trajectories 重建索引
                top = sorted(
                    sil_dataset.trajectories, key=lambda t: t.score, reverse=True,
                )[: config.sil_traj_cap]
                sil_dataset.set_trajectories(top)

            # Phase 2: PPO 更新（前 value_warmup_iters 轮只预热 critic，冻结策略防灾难性遗忘）
            value_only = iteration < config.value_warmup_iters
            if value_only:
                logger.info(
                    "Phase 2: Critic 预热 %d/%d (只训 critic_head, %d epochs, 策略冻结)",
                    iteration + 1, config.value_warmup_iters, config.value_warmup_epochs,
                )
                metrics = ppo_trainer.update_critic_only(model, buffer)
            else:
                logger.info("Phase 2: PPO 更新 (%d epochs, batch_size=%d)",
                            config.ppo_epochs, config.ppo_batch_size)
                metrics = ppo_trainer.update(model, buffer)

            # Phase 2.5: Self-Imitation（BC 监督精英池，锚定历史最优动作；仅非预热轮）
            if (sil_dataset is not None and not value_only
                    and iteration >= config.sil_start_iter and len(sil_dataset) > 0):
                logger.info(
                    "Phase 2.5: Self-Imitation (%d 轨迹, %d 窗口, %d epochs)",
                    len(sil_dataset.trajectories), len(sil_dataset), config.sil_epochs,
                )
                # boost 越高 → SIL 越松（避免精英池把被提方差的探索又拽回贴地开局）
                _sil_scale = 1.0 - min(
                    1.0, _boost / max(config.explore_boost_max, 1e-9)
                ) * config.explore_sil_relax
                metrics.update(ppo_trainer.sil_update(
                    model, sil_dataset,
                    loss_coef=config.sil_loss_coef * _sil_scale,
                ))

            # Phase 2.6: 停滞探测器 — 按精英池 max secured_dy(竖直)判断是否停滞，
            # 停滞则抬 boost、有进展则回落；设 model.exploration_boost 供【下一轮】采集+更新一致生效。
            if (config.explore_boost_enabled and sil_dataset is not None
                    and iteration >= config.sil_start_iter):
                sdys = [
                    float(progress_metric(t.raw_states, config)[2])
                    for t in sil_dataset.trajectories
                ]
                cur_best = max(sdys) if sdys else 0.0
                if cur_best > _best_pool_sdy + config.explore_stall_eps:
                    _best_pool_sdy = cur_best
                    _stall = 0
                    _boost = max(0.0, _boost - config.explore_boost_step)
                else:
                    _stall += 1
                    if _stall >= config.explore_stall_patience:
                        _boost = min(
                            config.explore_boost_max,
                            _boost + config.explore_boost_step,
                        )
                        _stall = 0
                model.exploration_boost = _boost
                logger.info(
                    "[explore] pool_best_sdy=%.2f stall=%d boost=%.2f (std≈%.3f)",
                    _best_pool_sdy, _stall, _boost,
                    float((model.actor_log_std + _boost).clamp(
                        config.ppo_log_std_min, config.ppo_log_std_max
                    ).exp().mean()),
                )

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

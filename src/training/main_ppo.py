"""
PPO 训练入口 — 策略梯度迭代循环。

每轮: 启动游戏 → collect_ppo → 关闭游戏 → PPO update → 日志/checkpoint

完整 CLI / 推荐配方 / 废弃清单见 doc/main_ppo_usage.md。

用法（当前最小任务 — 关 deploy、短 episode）:
  python -m src.training.main_ppo --no-random-deploy --steps-per-rollout 400 --persist-game
  python -m src.training.main_ppo --no-random-deploy --resume-bc checkpoints/model_iter_0010.pt
  python -m src.training.main_ppo --resume checkpoints/ppo_iter_0050.pt

低处课程投放（基座过树后再用）:
  python -m src.training.main_ppo --random-deploy --y-max-cutoff 125 --steps-per-rollout 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.actor_critic import ActorCritic
from training.config import TrainConfig
from training.geometry_profile import configure_geometry
from training.experimental_policy import create_actor_critic
from training.experiment_options import add_experiment_arguments, apply_experiment_arguments, restore_initialization_shape, ppo_config
from training.dataset import (
    Trajectory,
    TrajectoryDataset,
    compute_dynamics_stats,
    prune_trajectory,
)
from training.model import (
    expand_optimizer_state_for_dynamics_dim,
    expand_state_dict_for_dynamics_dim,
)
from training.ppo_trainer import PPOTrainer
from training.deploy_sampling import (
    default_candidate_points_path,
    extract_body_local,
    extract_tip_local,
    filter_points_by_height,
    load_candidate_points,
)
from training.reference_corridor import replay_corridor_fields
from training.reward import (
    ClimbingEfficiencyMap,
    RewardNormalizer,
    corridor_score,
    progress_metric,
    score_trajectory,
)
from training.rollout import RolloutWorker
from training.run_logging import RunRecorder, resolve_run_paths, load_elites

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
    config: TrainConfig | None = None,
    reference_metadata: dict | None = None,
    value_warmup_end: int | None = None,
    *, trainer=None, deployment_rng=None, loop_state=None,
) -> None:
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ppo_iter_{iteration:04d}.pt"
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    state["iteration"] = iteration
    if config is not None:
        state["config"] = asdict(config)
    if reference_metadata is not None:
        state["reference_corridor"] = reference_metadata
    if value_warmup_end is not None:
        state["value_warmup_end"] = value_warmup_end
    from training.random_state import capture_random_state
    state["rng_state"] = capture_random_state(deployment_rng)
    if trainer is not None:
        state["trainer_state"] = trainer.checkpoint_state()
        if optimizer is None and trainer._pending_optim_state is not None:
            state["optimizer"] = trainer._pending_optim_state
    if loop_state is not None:
        state["loop_state"] = loop_state
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    logger.info("Checkpoint saved: %s", path)


def _sil_score(traj: "Trajectory", eff_map: "ClimbingEfficiencyMap", config: TrainConfig) -> float:
    """精英池排序分分派：走廊轨迹用 credited 进度口径（corridor_score），
    其余沿用高度口径（score_trajectory）。与 prune_trajectory / _rebuild_index 同步。"""
    if traj.corridor_progress is not None or getattr(traj, "corridor_credit_events", None) is not None:
        from training.reward import corridor_score_of
        return corridor_score_of(traj, config)
    return score_trajectory(traj.raw_states, eff_map, config)


def log_metrics(iteration: int, metrics: dict[str, float], extras: dict | None = None) -> None:
    parts = [f"iter={iteration}"]
    for k, v in metrics.items():
        if isinstance(v, str):
            parts.append(f"{k}={v}")
        else:
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO 训练 — 策略梯度迭代循环")
    add_experiment_arguments(parser)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Initial seed (fresh default 0); resume retains the saved random streams")
    parser.add_argument("--reset-rng", action="store_true",
                        help="On resume, explicitly start new random streams with --seed instead of restoring them")
    parser.add_argument("--geometry-profile", choices=("native-pose-v1", "legacy"), default=None,
                        help="Explicitly select contact/patch pose calibration; resume otherwise preserves checkpoint geometry")
    parser.add_argument("--save-rollout-inputs", action="store_true",
                        help="Archive complete PPO inputs under checkpoint_dir for exact replay")
    parser.add_argument("--recipe", choices=("ppo", "ppo-ou", "ppo-ou-sil"), default=None,
                        help="Fixed ablation; disables adaptive exploration and experimental contact heads")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=None,
                        help="墙钟小时上限。到点后不再开始新一轮，当前轮会跑完")
    parser.add_argument(
        "--delayed-exploration", action=argparse.BooleanOptionalAction, default=None,
        help="8 个 player 按 0/200/400/600 四档延迟探索；延迟期保留 15%% 噪声，再经 50 步线性升到满幅度。"
             "续训继承 checkpoint；--no-delayed-exploration 显式关闭",
    )
    parser.add_argument("--target-physical-samples", type=int, default=None,
                        help="Fresh run: stop after the first complete iteration reaching this sample count (includes critic warmup)")
    parser.add_argument("--additional-physical-samples", type=int, default=None,
                        help="Resume: stop after this many additional physical samples, at a complete iteration")
    parser.add_argument("--sil-resume-pool", type=str, default=None,
                        help="Restore a saved SIL elite pool, including corridor fields, scores and source iterations")
    parser.add_argument("--sil-init-pool", type=str, default=None,
                        help="Cold-start SIL from an elite-pool npz already scored with the current progress mode; requires --init-from")
    parser.add_argument("--steps-per-rollout", type=int, default=None)
    parser.add_argument("--ppo-batch-size", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None,
                        help="Checkpoint interval in completed training iterations")
    parser.add_argument("--elite-archive", action=argparse.BooleanOptionalAction, default=None,
                        help="Archive complete elite episodes independently of SIL (reference v2 only)")
    parser.add_argument("--elite-archive-capacity", type=int, default=None)
    parser.add_argument("--gru-sil", action=argparse.BooleanOptionalAction, default=None,
                        help="Positive-advantage recurrent SIL from complete elite episodes; enables full archive")
    parser.add_argument("--elite-archive-resume", type=str, default=None,
                        help="Restore a full-episode archive; otherwise use the resumed checkpoint's matching snapshot")
    parser.add_argument("--stop-file", type=str, default=None,
                        help="Finish the current iteration and save when this file exists")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="CPU threads used by PyTorch in this process")
    parser.add_argument(
        "--y-max-cutoff", type=float, default=None,
        help=(
            "随机投放的可着陆表面高度上限（屏蔽 y 高于此值的表面）；"
            "与 --steps-per-rollout 配套限定低处训练区，如 --y-max-cutoff 125 --steps-per-rollout 500"
        ),
    )
    parser.add_argument(
        "--run-dir", type=str, default=None,
        help=(
            "实验归档目录（推荐）。给个裸名字即落到 runs/<名字>/，checkpoint、效率图、"
            "控制台、指标、精英池全部收在这一个目录里；与 --checkpoint-dir/--log-dir 互斥"
        ),
    )
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="（旧式）checkpoint 输出目录；新实验请改用 --run-dir")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="（旧式）效率图输出目录；新实验请改用 --run-dir")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None,
                        help="PPO return discount per physical step; retained on resume")
    parser.add_argument("--gae-lambda", type=float, default=None,
                        help="GAE trace decay per physical step; retained on resume")
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
        "--sil-seed", type=str, nargs="+", default=None, metavar="NPZ",
        help="精英池冷启动种子轨迹（.npz，含 raw_states/actions/contact），可多个；"
             "入池时按 score_trajectory 现算分数，后续与采集轨迹一同参与 MAP-Elites 竞争",
    )
    parser.add_argument(
        "--corridor-sil", action="store_true",
        help="走廊模式下启用 SIL 与 P3.2 优势加权（覆盖 configure_reference / experimental "
             "路径的强制停用，均为走廊 SIL 机械就绪前的遗留）。explore_boost 仍保持停用"
             "（其停滞判据是高度口径）。",
    )
    parser.add_argument(
        "--phase-noise", dest="phase_noise", action=argparse.BooleanOptionalAction,
        default=None,
        help="空中相放大探索噪声（默认关闭，未验证机制 opt-in）；--no-phase-noise 显式关闭",
    )
    parser.add_argument(
        "--sil-archive-mode", choices=("map_elites", "td_pareto"), default=None,
        help="SIL 精英归档：map_elites 对照；td_pareto 为终点 (t,D) 前沿（默认 map_elites）",
    )
    parser.add_argument(
        "--sil-td-sampling-mode",
        choices=("legacy_min_length", "balanced_stratified", "coverage_share"),
        default=None,
        help="td_pareto 的 SIL 窗口采样。legacy 保持原最短轨迹预算与单一 D 门控；"
             "balanced_stratified / coverage_share 使用完整动作区间并关闭该门控",
    )
    parser.add_argument(
        "--sil-td-sampling-horizon-steps", type=int, default=None,
        help="新采样模式的固定比较时限 H。省略时从 steps_per_rollout 解析并写入 checkpoint",
    )
    parser.add_argument(
        "--sil-td-windows-per-traj", type=int, default=None,
        help="legacy：每条轨迹窗口上限，0 对齐最短轨迹。新模式：平均计算预算 K，0 解析为 32",
    )
    parser.add_argument(
        "--sil-td-sampling-seed", type=int, default=None,
        help="SIL 采样专用种子。省略时使用 --seed，并在新模式 checkpoint 中保存解析值",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只解析并打印采样配置，不建模型、不启动游戏、不写 run 目录",
    )
    parser.add_argument(
        "--sil-only", action="store_true",
        help="只对已恢复的精英池做模仿更新，不启动游戏、不采集、不跑 PPO",
    )
    parser.add_argument(
        "--sil-only-minutes", type=float, default=None,
        help="--sil-only 的墙钟时间上限（分钟）。到点后完成当前这次模仿更新再退出",
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
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--reference-trajectory", nargs="?", const=str(
            _REPO_ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory_game_world.json"
        ), default=None,
        help="启用走廊进度奖励；可指定 JSON，省略路径使用实机相机重投影的速通轨迹",
    )
    reference_group.add_argument(
        "--no-reference-trajectory", dest="reference_trajectory", action="store_const", const="",
        help="显式停用参考走廊（包括从 checkpoint 恢复的走廊设置）",
    )
    parser.add_argument(
        "--reference-corridor-radius", type=float, default=None,
        help="走廊半宽，世界米（默认 1.5）；内部只奖励前进，不惩罚偏离中心线",
    )
    parser.add_argument("--reference-deviation-weight", type=float, default=None,
                        help="Maximum corridor deviation penalty per physical step")
    args = parser.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.reset_rng and not args.resume:
        parser.error("--reset-rng requires --resume")
    for name in ("ppo_batch_size", "save_interval", "torch_threads", "elite_archive_capacity"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.target_physical_samples is not None:
        if args.target_physical_samples < 1 or args.resume:
            parser.error("--target-physical-samples requires a positive count and a fresh run")
    if args.additional_physical_samples is not None:
        if args.additional_physical_samples < 1 or not args.resume or args.target_physical_samples is not None:
            parser.error("--additional-physical-samples requires a positive count and --resume")
    if args.sil_resume_pool and (not args.resume or args.sil_seed or args.sil_init_pool):
        parser.error("--sil-resume-pool requires --resume and cannot be combined with --sil-seed or --sil-init-pool")
    if args.sil_init_pool and (not args.init_from or args.resume or args.sil_seed):
        parser.error("--sil-init-pool requires --init-from and cannot be combined with --resume or --sil-seed")
    if args.sil_only and (not args.resume or not args.sil_resume_pool):
        parser.error("--sil-only requires --resume and --sil-resume-pool")
    if args.sil_only and (args.sil_only_minutes is None or args.sil_only_minutes <= 0):
        parser.error("--sil-only requires a positive --sil-only-minutes")
    return args


def configure_credit(config: TrainConfig, args: argparse.Namespace, checkpoint: dict) -> None:
    """Keep return semantics fixed within a resumed run; ablations start fresh."""
    saved = checkpoint.get("config", {})
    for name in ("gamma", "gae_lambda"):
        previous = saved.get(name, getattr(config, name))
        value = getattr(args, name)
        value = previous if value is None else value
        if not np.isfinite(value) or not 0 <= value <= 1 or (name == "gamma" and value == 0):
            raise ValueError(f"Invalid {name}: {value}")
        if checkpoint and value != previous:
            raise ValueError(f"Changing {name} on resume requires --init-from and a fresh run")
        setattr(config, name, value)


def configure_reference(config: TrainConfig, args: argparse.Namespace, checkpoint: dict) -> None:
    """Resume reference semantics before constructing observations or rewards."""
    saved = checkpoint.get("config", {})
    for field in fields(config):
        if field.name.startswith("reference_") and field.name in saved:
            setattr(config, field.name, saved[field.name])
    if args.reference_trajectory is not None:
        config.reference_trajectory = args.reference_trajectory or None
    if args.reference_corridor_radius is not None:
        config.reference_corridor_radius = args.reference_corridor_radius
    if args.reference_deviation_weight is not None:
        config.reference_deviation_weight = args.reference_deviation_weight
    if not np.isfinite(config.reference_deviation_weight) or config.reference_deviation_weight < 0:
        raise ValueError("reference_deviation_weight must be finite and nonnegative")
    if config.reference_trajectory:
        config.reference_trajectory = str(Path(config.reference_trajectory).resolve())
        if args.random_deploy is None:
            config.random_deploy = bool(saved.get("random_deploy", False)) if saved.get("reference_trajectory") else False
        # Existing SIL ranks/truncates by height, and rebuilds a single global map.
        # Keep that BC contract intact; corridor PPO uses its stored input windows.
        # 注（2026-09-23）：此停用是走廊 SIL 机械就绪前的遗留；--corridor-sil 在
        # main() 的配置收尾处统一覆盖启用（见该处注释）。
        config.sil_enabled = False
        config.explore_boost_enabled = False


def configure_sil_sampling(config: TrainConfig, args: argparse.Namespace, resume_state: dict) -> None:
    """Resolve sampling mode, horizon, K, and seed. Old resumes stay on the legacy sampler."""
    from training.reference_corridor import PROGRESS_MODE_V2
    from training.sil_archive import NEW_SAMPLING_MODES, SAMPLING_LEGACY, SAMPLING_MODES

    saved = resume_state.get("config", {}) if resume_state else {}
    is_resume = bool(resume_state)
    saved_has_mode = "sil_td_sampling_mode" in saved
    requested_mode = getattr(args, "sil_td_sampling_mode", None)
    requested_horizon = getattr(args, "sil_td_sampling_horizon_steps", None)
    requested_k = getattr(args, "sil_td_windows_per_traj", None)
    requested_seed = getattr(args, "sil_td_sampling_seed", None)

    def _reject_saved_change(name: str, requested, saved_value) -> None:
        if requested is None or not (is_resume and saved_has_mode):
            return
        if requested != saved_value:
            raise ValueError(
                f"恢复中的实验已保存 {name}={saved_value}，不能改成 {requested}；"
                "切换采样配置需要新的 --run-dir 和显式起点"
            )

    if is_resume and not saved_has_mode:
        if requested_mode not in (None, SAMPLING_LEGACY):
            if not getattr(args, "run_dir", None):
                raise ValueError(
                    "旧 checkpoint 没有 sil_td_sampling_mode，不能在原实验里切换采样模式；"
                    "请使用新的 --run-dir"
                )
            config.sil_td_sampling_mode = requested_mode
        else:
            config.sil_td_sampling_mode = SAMPLING_LEGACY
    elif is_resume and saved_has_mode:
        _reject_saved_change("sil_td_sampling_mode", requested_mode, saved.get("sil_td_sampling_mode"))
    elif requested_mode is not None:
        config.sil_td_sampling_mode = requested_mode
    if config.sil_td_sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"Unknown sil_td_sampling_mode: {config.sil_td_sampling_mode}")

    if config.sil_td_sampling_mode == SAMPLING_LEGACY:
        if requested_k is not None:
            _reject_saved_change(
                "sil_td_windows_per_traj", requested_k, saved.get("sil_td_windows_per_traj", 0))
            config.sil_td_windows_per_traj = int(requested_k)
        return

    if config.sil_archive_mode != "td_pareto":
        raise ValueError(f"{config.sil_td_sampling_mode} 需要 sil_archive_mode=td_pareto")
    if getattr(config, "reference_progress_mode", None) != PROGRESS_MODE_V2:
        raise ValueError(f"{config.sil_td_sampling_mode} 需要 reference_progress_mode={PROGRESS_MODE_V2}")
    if config.sil_td_sampling_mode not in NEW_SAMPLING_MODES:
        raise ValueError(f"Unknown sil_td_sampling_mode: {config.sil_td_sampling_mode}")

    if requested_horizon is not None:
        _reject_saved_change(
            "sil_td_sampling_horizon_steps", requested_horizon,
            saved.get("sil_td_sampling_horizon_steps"))
        config.sil_td_sampling_horizon_steps = int(requested_horizon)
    elif not (is_resume and saved_has_mode):
        if int(config.sil_td_sampling_horizon_steps) <= 0:
            config.sil_td_sampling_horizon_steps = int(config.steps_per_rollout)
    if int(config.sil_td_sampling_horizon_steps) < 1:
        raise ValueError("sil_td_sampling_horizon_steps 必须是正整数")

    if requested_k is not None:
        _reject_saved_change(
            "sil_td_windows_per_traj", requested_k, saved.get("sil_td_windows_per_traj"))
        config.sil_td_windows_per_traj = int(requested_k)
    elif int(config.sil_td_windows_per_traj) <= 0:
        config.sil_td_windows_per_traj = 32
    if int(config.sil_td_windows_per_traj) < 1:
        raise ValueError("新采样模式的 sil_td_windows_per_traj 解析后必须为正")

    if is_resume and saved_has_mode and "sil_td_sampling_seed" in saved:
        _reject_saved_change(
            "sil_td_sampling_seed", requested_seed, saved.get("sil_td_sampling_seed"))
        config.sil_td_sampling_seed = int(saved["sil_td_sampling_seed"])
    elif requested_seed is not None:
        config.sil_td_sampling_seed = int(requested_seed)
    elif int(config.sil_td_sampling_seed) <= 0:
        config.sil_td_sampling_seed = int(getattr(args, "seed", 0) or 0)
    if not config.sil_td_sampling_rule_version:
        config.sil_td_sampling_rule_version = "coverage_v1"


def run_sil_only(
    args, config, model, ppo_trainer, sil_dataset, recorder, start_iteration, reference_meta,
    warmup_end,
) -> None:
    """Imitation updates on the restored elite pool. No game, no PPO, pool stays frozen."""
    if sil_dataset is None or not sil_dataset.trajectories:
        raise ValueError("--sil-only 需要已经恢复的非空精英池")
    if not model.dynamics_stats_ready:
        raise ValueError("--sil-only 需要 checkpoint 里已有观测归一化统计，不能为了标定去启动游戏")
    deadline = time.perf_counter() + float(args.sil_only_minutes) * 60.0
    iteration = int(start_iteration)
    logger.info(
        "只做模仿更新 %.1f 分钟：不启动游戏、不采集、不跑 PPO；精英池保持 %d 条",
        args.sil_only_minutes, len(sil_dataset.trajectories),
    )
    try:
        while time.perf_counter() < deadline:
            iteration += 1
            logger.info("=" * 60)
            logger.info("=== 模仿更新 %d ===", iteration)
            metrics = ppo_trainer.sil_update(
                model, sil_dataset,
                loss_coef=config.sil_loss_coef,
                budget_updates=None,
                behavior_progress=None,
                iteration=iteration,
            )
            log_metrics(iteration, metrics)
            if recorder is not None:
                recorder.append_metrics(iteration, metrics)
                sampling_record = getattr(ppo_trainer, "last_sil_sampling", None)
                if sampling_record is not None:
                    recorder.save_sil_sampling(iteration, sampling_record)
            save_checkpoint(
                model, ppo_trainer.optimizer, iteration, config.checkpoint_dir,
                config=config, reference_metadata=reference_meta,
                value_warmup_end=warmup_end, trainer=ppo_trainer,
            )
            if recorder is not None:
                recorder.save_elites(sil_dataset.trajectories, iteration)
            if float(metrics.get("sil_updates", 0.0)) <= 0.0:
                logger.warning("本轮没有模仿更新: %s", metrics.get("sil_sampling_mode", metrics))
        if recorder is not None:
            recorder.finalize("completed", last_iteration=iteration)
        logger.info("模仿更新完成，最后一轮 %d", iteration)
    except BaseException as exc:
        if recorder is not None:
            recorder.finalize(
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                last_iteration=iteration,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise


def main() -> None:
    _setup_logging()
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    logger.info("PyTorch CPU threads: %d", torch.get_num_threads())

    if args.run_dir and (args.checkpoint_dir or args.log_dir):
        raise SystemExit("--run-dir 与 --checkpoint-dir/--log-dir 互斥，二选一")

    resume_state = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else {}
    config = ppo_config(resume_state, legacy_bc=bool(args.resume_bc))
    if args.init_from and (args.resume or args.resume_bc):
        raise ValueError("--init-from is mutually exclusive with --resume / --resume-bc")
    init_state = torch.load(args.init_from, map_location="cpu", weights_only=False) if args.init_from else {}
    restore_initialization_shape(config, init_state)
    if args.resume and "reference_progress_mode" not in resume_state.get("config", {}):
        from training.reference_corridor import PROGRESS_MODE_V1
        config.reference_progress_mode = PROGRESS_MODE_V1
    if args.resume and "actor_mean_contact.weight" not in resume_state["model"]:
        config.contact_heads_enabled = False
    geometry_changed = configure_geometry(config, resume_state or init_state or ({"bc": True} if args.resume_bc else None),
                                           args.geometry_profile)
    if (resume_state.get("rng_state") and not args.reset_rng and args.seed is not None
            and args.seed != config.training_seed):
        raise ValueError("Changing --seed on resume requires --reset-rng; otherwise the saved RNG is restored")
    args.seed = config.training_seed if args.seed is None else args.seed
    config.training_seed = args.seed
    from training.random_state import seed_training, restore_random_state
    seed_training(args.seed)
    if args.recipe:
        if args.resume and "actor_mean_contact.weight" in resume_state["model"]:
            raise ValueError("Fixed single-head baseline requires a single-head checkpoint")
        config.ou_enabled = args.recipe != "ppo"
        config.sil_enabled = args.recipe == "ppo-ou-sil"
        config.explore_boost_enabled = False
        config.contact_heads_enabled = False
        config.phase_weighting_enabled = False
        config.sil_advantage_weighting_enabled = False
    if args.num_agents is not None:
        config.num_agents = args.num_agents
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.delayed_exploration is not None:
        config.delayed_exploration_enabled = args.delayed_exploration
    if args.max_hours is not None and args.max_hours <= 0:
        raise ValueError("--max-hours 必须是正数")
    if args.steps_per_rollout is not None:
        config.steps_per_rollout = args.steps_per_rollout
    if config.delayed_exploration_enabled and (
            config.num_agents != 8 or config.steps_per_rollout <= 600
            or not 0.0 < float(config.delayed_exploration_floor) <= 1.0
            or int(config.delayed_exploration_ramp_steps) < 1):
        raise ValueError("延迟探索需要 8 个 player、每轮多于 600 步，以及合法的残留倍率与爬升长度")
    if args.ppo_batch_size is not None:
        config.ppo_batch_size = args.ppo_batch_size
    if args.save_interval is not None:
        config.save_interval = args.save_interval
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
    if args.phase_noise is not None:
        config.phase_noise_enabled = args.phase_noise
    if args.sil_archive_mode is not None:
        config.sil_archive_mode = args.sil_archive_mode
    if args.random_deploy is not None:
        config.random_deploy = args.random_deploy
    configure_credit(config, args, resume_state)
    configure_reference(config, args, resume_state)
    apply_experiment_arguments(config, args, resume_state)
    from training.policy_features import experimental_enabled
    if experimental_enabled(config) and (args.resume_bc or args.recipe == "ppo-ou-sil"):
        raise ValueError("GRU/representation policies support PPO --init-from/--resume and disable legacy BC/SIL")
    if args.corridor_sil and config.policy_backbone == "gru":
        raise ValueError("GRU does not support legacy corridor SIL; use --gru-sil")
    if args.corridor_sil and config.reference_trajectory:
        # 走廊 SIL 机械（corridor_score / prune_trajectory 走廊分支 / MAP-Elites bd /
        # 种子 corridor 字段重放 / P3.2 优势加权）已就绪；configure_reference 与
        # experimental 路径里的 sil 强制停用均为就绪前遗留，在此统一覆盖。
        # explore_boost 保持停用（其停滞判据是高度口径，走廊语义需另行论证）。
        config.sil_enabled = True
        config.sil_advantage_weighting_enabled = True
        logger.info("--corridor-sil: 走廊模式启用 SIL（评分/剪枝/入池/优势加权均走走廊口径）")
    configure_sil_sampling(config, args, resume_state)
    if args.elite_archive is not None:
        config.elite_archive_enabled = args.elite_archive
    if args.elite_archive_capacity is not None:
        config.elite_archive_capacity = args.elite_archive_capacity
    if args.gru_sil is not None:
        config.gru_sil_enabled = args.gru_sil
    if config.gru_sil_enabled:
        from training.gru_sil import validate_gru_sil
        validate_gru_sil(config)
        if args.elite_archive is False:
            raise ValueError("--gru-sil requires full elite archival; cannot use --no-elite-archive")
        if args.sil_seed or args.sil_resume_pool or args.sil_init_pool or args.sil_only:
            raise ValueError("GRU SIL uses --elite-archive-resume, not legacy SIL pools or SIL-only mode")
        config.elite_archive_enabled = True
        if args.reset_rng or not resume_state.get("config", {}).get("gru_sil_enabled", False):
            config.gru_sil_seed = args.seed
    if args.elite_archive_resume and not config.elite_archive_enabled:
        raise ValueError("--elite-archive-resume requires --elite-archive")
    if config.elite_archive_enabled and (not args.run_dir or not config.reference_trajectory
                                       or config.reference_progress_mode != "confirmed_endpoint_v2"):
        raise ValueError("--elite-archive requires --run-dir and confirmed_endpoint_v2 reference scoring")
    if args.dry_run:
        payload = {
            "policy_backbone": config.policy_backbone,
            "training_seed": config.training_seed,
            "rng_resume_mode": ("restore" if resume_state.get("rng_state") and not args.reset_rng else "seed"),
            "delayed_exploration_enabled": config.delayed_exploration_enabled,
            "normalize_action_history": config.normalize_action_history,
            "recurrent_seq_len": config.recurrent_seq_len,
            "random_deploy": config.random_deploy,
            "sil_enabled": config.sil_enabled,
            "gru_sil_enabled": config.gru_sil_enabled,
            "gru_sil_max_updates": config.gru_sil_max_updates,
            "gru_sil_budget_ratio": config.gru_sil_budget_ratio,
            "gru_sil_target_kl": config.gru_sil_target_kl,
            "elite_archive_enabled": config.elite_archive_enabled,
            "elite_archive_capacity": config.elite_archive_capacity,
            "sil_td_sampling_mode": config.sil_td_sampling_mode,
            "sil_td_sampling_horizon_steps": config.sil_td_sampling_horizon_steps,
            "sil_td_windows_per_traj": config.sil_td_windows_per_traj,
            "sil_td_sampling_seed": config.sil_td_sampling_seed,
            "sil_td_sampling_rule_version": config.sil_td_sampling_rule_version,
            "sil_archive_mode": config.sil_archive_mode,
            "reference_progress_mode": config.reference_progress_mode,
            "steps_per_rollout": config.steps_per_rollout,
            "sil_batch_size": config.sil_batch_size,
            "sil_epochs": config.sil_epochs,
            "sil_max_batches": config.sil_max_batches,
            "sil_budget_ratio": config.sil_budget_ratio,
            "max_iterations": config.max_iterations,
            "resume_iteration": int(resume_state.get("iteration", 0)) if resume_state else 0,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    # 走廊模式 SIL 已迁移到走廊口径（剪枝/排序/bd/BC 截断全部基于 credited 进度，
    # patch wide 通道按 guidance_index 重建引导场），ppo-ou-sil 不再与走廊互斥。

    # ── 运行归档 ──
    # --run-dir 把产物收敛到 runs/<tag>/ 并开启元数据/指标/控制台/精英池落盘；
    # 走旧式 --checkpoint-dir/--log-dir 时 recorder 为 None，行为与从前完全一致。
    recorder: RunRecorder | None = None
    if args.run_dir:
        paths = resolve_run_paths(args.run_dir, repo_root=_REPO_ROOT)
        config.checkpoint_dir = str(paths.checkpoints)
        config.log_dir = str(paths.logs)
        recorder = RunRecorder(paths, repo_root=_REPO_ROOT)
        recorder.attach_console()
        logger.info("实验归档目录: %s", paths.root)

    logger.info("PPO Config: %s", config)
    # ── 初始化 ──
    model = create_actor_critic(config)
    ppo_trainer = PPOTrainer(config)
    rollout_worker = RolloutWorker(config)
    rollout_worker._rng = np.random.default_rng(args.seed)
    reference = rollout_worker.reference_corridor
    reference_meta = reference.metadata if reference is not None else None
    if reference is not None:
        logger.info(
            "参考走廊: %d 连续段, %d→%d 点, 半宽 %.2fm；内部新进度 %.2f/m，外部惩罚上限 %.3f/step",
            reference_meta["segments"], reference_meta["raw_points"], reference_meta["points"],
            config.reference_corridor_radius, config.reference_progress_weight,
            config.reference_deviation_weight,
        )
        logger.info("参考来源状态: %s；%s", reference_meta["status"], reference_meta["coordinate_warning"])
        if config.sil_enabled:
            logger.info("走廊模式 + SIL：替代高度/向右/效率图引导；精英池走走廊评分/剪枝/优势加权口径")
        else:
            logger.info("走廊模式: 替代高度/向右/效率图引导，停用基于高度的 SIL 和精英池停滞加噪")
        saved_hash = resume_state.get("reference_corridor", {}).get("sha256")
        if saved_hash and args.reference_trajectory is None and saved_hash != reference_meta["sha256"]:
            raise ValueError("Checkpoint reference file has changed; select --reference-trajectory explicitly to switch")
    from training.reference_anchor import ANCHOR_POT_CALIBRATED
    from training.reference_fingerprint import (
        assert_fingerprint_compatible, load_fingerprint_sidecar, reference_fingerprint,
    )
    if config.reference_anchor_id == ANCHOR_POT_CALIBRATED:
        if not config.reference_anchor_verified:
            raise ValueError("校准锅锚点尚未独立验证，拒绝正式训练")
        if config.tcp_protocol != "anchor35":
            raise ValueError("校准锅锚点任务必须 tcp_protocol=anchor35，禁止降级 legacy33")
    reference_fp = reference_fingerprint(
        config,
        route_meta=reference_meta,
        schema_id="anchor35" if config.tcp_protocol == "anchor35" else "legacy33",
        raw_dim=35 if config.tcp_protocol == "anchor35" else 33,
    )
    if recorder is not None:
        recorder.write_meta(config, argv=sys.argv, extra={
            "resume": args.resume, "resume_bc": args.resume_bc, "init_from": args.init_from,
            "reference_corridor": reference_meta, "reference_fingerprint": reference_fp,
        })
    start_iteration = 0

    if args.resume:
        ckpt = resume_state
        # 老 34D checkpoint → 38D 兼容：dynamics_proj 加 4 新列（初始化 0）+ dyn_mean/std 追加
        # 4 个元素（0/1）。见 model.expand_state_dict_for_dynamics_dim 文档。
        expanded = expand_state_dict_for_dynamics_dim(ckpt["model"], model.state_dict())
        model.load_state_dict(expanded)
        start_iteration = ckpt.get("iteration", 0)
        if "optimizer" in ckpt:
            # 权重扩了 34D→39D 后，恢复的 Adam 动量仍是旧宽度，需同步扩展新列（补 0），
            # 否则 optimizer.step() 会因 exp_avg (d,34) vs grad (d,39) 尺寸冲突崩溃。
            ppo_trainer.set_pending_optimizer_state(
                expand_optimizer_state_for_dynamics_dim(ckpt["optimizer"], model)
            )
        logger.info("从 PPO checkpoint 恢复 (iteration=%d): %s", start_iteration, args.resume)
    elif args.init_from:
        if hasattr(model, "initialize_from"):
            loaded = model.initialize_from(init_state["model"])
            logger.info("显式暖启动：加载 %d 个兼容张量；新模块初始化，不恢复 optimizer/iteration", len(loaded))
        else:
            model.load_state_dict(init_state["model"])
        # 默认不重设：继承源 checkpoint 的 log_std。打开 ppo_reset_log_std_on_init
        # 才按 ppo_init_log_std 拉回。
        if config.ppo_reset_log_std_on_init and hasattr(model, "reset_log_std"):
            model.reset_log_std(config.ppo_init_log_std)
            logger.info("init-from σ 重设: log_std → %.2f (std=%.3f)",
                        config.ppo_init_log_std, float(model.actor_log_std.detach().exp().mean()))
        from training.reference_corridor import progress_mode_of
        source_mode = progress_mode_of(init_state.get("config", {}))
        if source_mode != config.reference_progress_mode and hasattr(model, "reset_critic_head"):
            model.reset_critic_head()
            config.progress_mode_warm_start = True
            logger.info("进度口径 %s → %s：重新初始化 critic，不恢复旧优化器",
                        source_mode, config.reference_progress_mode)
    elif args.resume_bc:
        bc_state = torch.load(args.resume_bc, weights_only=True)
        model.load_from_bc(bc_state)
        logger.info("从 BC checkpoint 迁移 backbone 权重: %s", args.resume_bc)

    old_reference_config = {
        k: v for k, v in resume_state.get("config", {}).items() if k.startswith("reference_")
    }
    # Collection and update must use the same device from the first rollout.
    model = model.to(ppo_trainer.device)
    reference_changed = bool(args.resume) and (
        bool(old_reference_config.get("reference_trajectory")) != bool(config.reference_trajectory)
        or (bool(config.reference_trajectory) and (
            old_reference_config != {k: v for k, v in asdict(config).items() if k.startswith("reference_")}
            or resume_state.get("reference_corridor", {}).get("sha256") != reference_meta["sha256"]
        ))
    )
    semantics_changed = reference_changed or geometry_changed or bool(getattr(config, "progress_mode_warm_start", False))
    warmup_end = start_iteration + config.value_warmup_iters if semantics_changed else config.value_warmup_iters
    if not semantics_changed and args.value_warmup_iters is None:
        warmup_end = max(warmup_end, int(resume_state.get("value_warmup_end", 0)))
    if reference_changed:
        logger.info("奖励目标改变：从当前迭代重新预热 critic %d 轮", config.value_warmup_iters)
    if geometry_changed:
        logger.info("几何输入标定改变：从当前迭代重新预热 critic %d 轮", config.value_warmup_iters)
    if not semantics_changed and resume_state.get("trainer_state"):
        ppo_trainer.restore_checkpoint_state(resume_state["trainer_state"], model)

    # Independent archive preserves the entire episode, never supplies SIL batches.
    elite_archive = None
    if config.elite_archive_enabled:
        from training.elite_archive import EliteTrajectoryArchive
        elite_archive = EliteTrajectoryArchive(
            config, recorder.paths.elites / "full_episodes", reference_fp)
        archive_source = Path(args.elite_archive_resume) if args.elite_archive_resume else (
            Path(args.resume).resolve().parent.parent / "elites" / "full_episodes"
            / f"elites_iter_{start_iteration:04d}.npz" if args.resume else None)
        if archive_source is not None and (args.elite_archive_resume or archive_source.exists()
                                          or resume_state.get("config", {}).get("elite_archive_enabled", False)):
            logger.info("Full-episode elite archive restored: %d trajectories from %s",
                        elite_archive.restore(archive_source), archive_source)
        logger.info("Full-episode elite archive enabled (capacity=%d, legacy_SIL=%s, GRU_SIL=%s)",
                    config.elite_archive_capacity, config.sil_enabled, config.gru_sil_enabled)

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

    # ── Self-Imitation 精英轨迹池 ──（复用 BC 侧 TrajectoryDataset：top-K% 过滤 + 推进点
    # 窗口截断 + 即时重建 patch；只存 raw_states/actions，内存开销近零。
    # 走廊模式：轨迹自带 corridor_progress/guidance_index（rollout 逐帧记录），
    # 剪枝/排序/BC 截断走 credited 口径，patch wide 通道按 guidance_index 从
    # reference_maps 选引导段地图重建，与在线采集逐点一致）
    reference_maps = None
    if reference is not None and (config.sil_enabled or config.gru_sil_enabled):
        reference_maps = reference.guidance_maps(
            eff_map.traversable.shape, eff_map._min_gx, eff_map._min_gy,
            config.grid_resolution,
        )
        # 共享给 rollout_worker，避免在线采集时重复栅格化
        if rollout_worker._reference_maps is None:
            rollout_worker._reference_maps = reference_maps
    sequence_encoder = (
        TrajectoryDataset(
            config, eff_map.traversable, eff_map.get_arr(),
            min_gx=eff_map._min_gx, min_gy=eff_map._min_gy,
            solid_polygons=polygons,
            tip_local=extract_tip_local(player_data),
            body_local=extract_body_local(player_data),
            reference_maps=reference_maps,
        )
        if config.sil_enabled or config.gru_sil_enabled else None
    )
    sil_dataset = sequence_encoder if config.sil_enabled else None

    # ── 精英池冷启动种子（可选）：外部剪枝后的优质轨迹直接入池，iteration=-1 标记；
    # 首轮 add_trajectories_map_elites 会用当前池重建 cell_map，种子照常参与竞争/被替换。
    # 走廊模式下若种子未携带 corridor 字段，从 raw_states 确定性重放 tracker 补齐。
    if sil_dataset is not None and args.sil_seed:
        seed_trajs: list[Trajectory] = []
        for seed_path in args.sil_seed:
            assert_fingerprint_compatible(
                reference_fp, load_fingerprint_sidecar(seed_path), context=str(seed_path),
            )
            z = np.load(seed_path)
            if "raw_states" not in z.files or "actions" not in z.files:
                raise ValueError(f"{seed_path}: 没有完整状态/动作，不能转成确认距离")
            traj = Trajectory(
                raw_states=np.asarray(z["raw_states"], dtype=np.float32),
                actions=np.asarray(z["actions"], dtype=np.float32),
                contact=(np.asarray(z["contact"], dtype=np.float32)
                         if "contact" in z.files else None),
                iteration=-1,
                termination_known="terminated" in z.files,
            )
            if reference is not None:
                from training.reference_corridor import replay_corridor_result
                played = replay_corridor_result(reference, traj.raw_states)
                traj.corridor_progress = np.asarray(played["legacy_credit"], dtype=np.float32)
                traj.guidance_index = played["guidance_index"]
                traj.corridor_credit_events = played["credit_events"]
                traj.corridor_confirmed_progress = played["confirmed_progress"]
                traj.step_rewards = played["rewards"]
                traj.progress_mode = played["mode"]
                traj.trajectory_valid = played["valid"]
                if "terminated" in z.files:
                    traj.terminated = np.asarray(z["terminated"], dtype=bool)
                logger.info("SIL 种子按 %s 重放: %s (D=%.3fm)",
                            played["mode"], seed_path, played["distance"])
            traj.score = _sil_score(traj, eff_map, config)
            seed_trajs.append(traj)
            logger.info("SIL 种子: %s (T=%d, score=%.3f)",
                        seed_path, len(traj.actions), traj.score)
        if seed_trajs:
            sil_dataset.load_sil_pool(seed_trajs)
            logger.info("SIL 精英池冷启动: %d 条种子轨迹入池", len(sil_dataset.trajectories))

    if args.sil_resume_pool:
        if sil_dataset is None:
            raise ValueError("Restoring an elite pool requires SIL to be enabled")
        assert_fingerprint_compatible(
            reference_fp, load_fingerprint_sidecar(args.sil_resume_pool),
            context=str(args.sil_resume_pool),
        )
        restored_trajs = [Trajectory(**row) for row in load_elites(args.sil_resume_pool)]
        from training.sil_archive import recover_v2_termination
        recovered = sum(recover_v2_termination(traj) for traj in restored_trajs)
        if recovered:
            logger.info("SIL 恢复池补齐 %d 条 v2 轨迹的终止标记（全 false）", recovered)
        if not restored_trajs:
            raise ValueError("Cannot resume an empty SIL pool")
        if reference is not None and any(t.corridor_progress is None or t.guidance_index is None
                                         for t in restored_trajs):
            raise ValueError("Corridor SIL pool is missing saved corridor fields")
        sil_dataset.load_sil_pool(restored_trajs)
        logger.info("SIL 精英池完整恢复: %s (%d 条轨迹)", args.sil_resume_pool, len(sil_dataset.trajectories))

    if args.sil_init_pool:
        if sil_dataset is None:
            raise ValueError("An initial SIL pool requires SIL to be enabled")
        assert_fingerprint_compatible(
            reference_fp, load_fingerprint_sidecar(args.sil_init_pool),
            context=str(args.sil_init_pool),
        )
        from training.reference_corridor import PROGRESS_MODE_V2, trajectory_endpoint
        from training.sil_archive import recover_v2_termination
        init_trajs = []
        for row in load_elites(args.sil_init_pool):
            traj = Trajectory(**row)
            if traj.progress_mode != PROGRESS_MODE_V2 or traj.corridor_confirmed_progress is None:
                raise ValueError(f"{args.sil_init_pool}: every row needs confirmed_endpoint_v2 fields")
            if not trajectory_endpoint(traj).valid:
                raise ValueError(f"{args.sil_init_pool}: a row has no valid confirmed endpoint")
            recover_v2_termination(traj)
            init_trajs.append(traj)
        if not init_trajs:
            raise ValueError("Cannot init from an empty SIL pool")
        sil_dataset.load_sil_pool(init_trajs)
        logger.info("SIL 精英池按当前口径冷启动: %s (%d 条轨迹)",
                    args.sil_init_pool, len(sil_dataset.trajectories))

    # ── 奖励归一化器 ──
    calibration_steps = config.steps_per_rollout * config.num_agents * 10
    reward_normalizer = RewardNormalizer(calibration_steps=calibration_steps) if reference is None else None
    saved_loop = resume_state.get("loop_state", {}) if not semantics_changed else {}
    if reward_normalizer is not None and saved_loop.get("reward_normalizer"):
        reward_normalizer.__dict__.update(saved_loop["reward_normalizer"])
    if reward_normalizer is not None:
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
            from env import make_goi_env
            rollout_worker.env = make_goi_env(config)
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

    def restore_training_rng():
        if args.resume and resume_state.get("rng_state") and not args.reset_rng:
            restore_random_state(resume_state["rng_state"], rollout_worker._rng)
            logger.info("恢复 Python/NumPy/Torch/CUDA/投放随机数状态，继续 checkpoint 随机序列")
        elif args.resume:
            logger.warning("%s：随机序列从 seed=%d 重新开始，不能视为不中断训练的逐步复现",
                           "显式 --reset-rng" if args.reset_rng else "旧 checkpoint 缺少 RNG 状态", args.seed)

    if args.sil_only:
        restore_training_rng()
        run_sil_only(
            args, config, model, ppo_trainer, sil_dataset, recorder,
            start_iteration, reference_meta, warmup_end,
        )
        return

    # 游戏常驻模式：开局启动一次，循环内跳过每轮启停，仅靠 collect_ppo 的 env.reset() 复位。
    # manage_per_iter=True 时维持原有"每轮启停"行为。
    manage_per_iter = not args.no_launch and not args.persist_game
    if args.persist_game and not args.no_launch and rollout_worker.env is None:
        logger.info("持久游戏模式: 启动一次游戏，后续跨轮常驻（每轮仅 env.reset()，省启停开销）")
        rollout_worker.launch_game()

    # ── 停滞探测器状态（精英池峰值竖直高度停滞 → 自适应提方差）──
    _best_pool_sdy = float(saved_loop.get("best_pool_sdy", float("-inf")))
    _stall = int(saved_loop.get("stall", 0))
    _boost = float(saved_loop.get("boost", 0.0)) if config.explore_boost_enabled else 0.0
    model.exploration_boost = _boost

    def checkpoint_loop_state():
        return {"best_pool_sdy": _best_pool_sdy, "stall": _stall, "boost": _boost,
                "reward_normalizer": dict(vars(reward_normalizer)) if reward_normalizer is not None else None}

    restore_training_rng()  # After model creation, normalization and game startup consume their own work.

    # ── 主循环 ──
    last_iteration = start_iteration
    completed_iterations = start_iteration
    physical_samples = 0
    sample_target = args.additional_physical_samples or args.target_physical_samples
    hour_limit = None if args.max_hours is None else time.monotonic() + args.max_hours * 3600
    if config.delayed_exploration_enabled:
        from training.delayed_exploration import OPEN_STEPS
        logger.info(
            "延迟探索: player 开启步 %s，延迟期倍率 %.2f，之后 %d 步线性升到 1",
            list(OPEN_STEPS), config.delayed_exploration_floor, config.delayed_exploration_ramp_steps,
        )
    try:
        for iteration in range(start_iteration, config.max_iterations):
            if args.stop_file and Path(args.stop_file).exists():
                logger.info("Stop file received; saving completed iteration %d", completed_iterations)
                break
            if hour_limit is not None and time.monotonic() >= hour_limit:
                logger.info("达到 %.2f 小时墙钟上限，停止于第 %d 轮之前",
                            args.max_hours, iteration + 1)
                break
            last_iteration = iteration
            logger.info("=" * 60)
            logger.info("=== PPO 迭代 %d/%d ===", iteration + 1, config.max_iterations)

            # Phase 1: 采集
            if manage_per_iter:
                rollout_worker.launch_game()
            elif rollout_worker.env is None:
                from env import make_goi_env
                rollout_worker.env = make_goi_env(config)
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
            if elite_archive is not None:
                archive_stats = elite_archive.ingest(trajectories, iteration + 1)
                recorder.stage_metrics(**archive_stats)
            if hasattr(rollout_worker, "last_feature_scales"):
                diagnostics_dir = Path(config.checkpoint_dir) / "feature_diagnostics"
                diagnostics_dir.mkdir(parents=True, exist_ok=True)
                (diagnostics_dir / f"iter_{iteration + 1:04d}.json").write_text(json.dumps({
                    "feature_scales": rollout_worker.last_feature_scales,
                    "segments": rollout_worker.last_segment_trace,
                    "profile": rollout_worker.last_profile}, indent=2), encoding="utf-8")
            if args.save_rollout_inputs and buffer.size:
                archive_dir = Path(config.checkpoint_dir) / "rollouts"
                archive_dir.mkdir(parents=True, exist_ok=True)
                buffer.save(archive_dir / f"iter_{iteration + 1:04d}.npz")
                torch.save({"model": model.state_dict(), "config": asdict(config),
                            "reference_corridor": reference_meta or {},
                            "policy_context": buffer.policy_context},
                           archive_dir / f"behavior_{iteration + 1:04d}.pt")
            if reward_normalizer is not None and reward_normalizer.is_active:
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
            if trajectories and reference is None:
                prev_eff_map = eff_map.copy()
                eff_map.update(trajectories, config, prev_eff_map)

            # Phase 1.6: 更新 Self-Imitation 精英池
            # 默认走 MAP-Elites（P1.2）：bd=secured peak (x,y)，每 cell 保留分数最高一条；
            # 是否达到 sil_traj_cap 由 cell 数决定，仍以分数 top-N 兜底裁剪防止过大。
            # sil_map_elites_enabled=False 时回退到旧 top-K% 相对门槛路径。
            if sil_dataset is not None and trajectories:
                sil_dataset.update_efficiency_arr(eff_map.get_arr())
                # 入池前物理剪枝：截到 secured peak（+dwell 观察窗），峰后跌落/震荡段
                # 不进精英池存储；score/bd/BC 窗口口径与完整轨迹一致（见 prune_trajectory）。
                # 注：效率图 update（Phase 1.5）仍用完整轨迹——峰后段是"这些点曾到达峰"
                # 的信用记录，属效率图后缀口径的一部分，不剪。
                if config.sil_prune_enabled:
                    pool_trajs = [prune_trajectory(t, config) for t in trajectories]
                else:
                    pool_trajs = trajectories
                sil_scores = [_sil_score(t, eff_map, config) for t in pool_trajs]
                if config.sil_archive_mode == "td_pareto":
                    archive_stats = sil_dataset.ingest_sil_archive(
                        pool_trajs, sil_scores, iteration=iteration)
                    logger.info(
                        "[SIL/td-pareto] size=%d pareto=%d behavior=%d bucketed=%s capped=%s",
                        int(archive_stats["sil_archive_size"]),
                        int(archive_stats["sil_archive_pareto"]),
                        int(archive_stats["sil_archive_behavior"]),
                        bool(archive_stats["sil_archive_bucketed"]),
                        bool(archive_stats["sil_archive_capped"]),
                    )
                    if recorder is not None:
                        recorder.stage_metrics(**archive_stats)
                elif config.sil_map_elites_enabled:
                    me_stats = sil_dataset.add_trajectories_map_elites(
                        pool_trajs, sil_scores,
                        iteration=iteration,
                        cell_size=config.sil_cell_size,
                        cold_start_min=config.sil_cold_start_min,
                    )
                    logger.info(
                        "[SIL/MAP-Elites] cells=%d, admitted=%d, replaced=%d "
                        "(cell_size=%.1fm)",
                        me_stats["n_cells"], me_stats["n_admitted"],
                        me_stats["n_replaced"], config.sil_cell_size,
                    )
                    if recorder is not None:
                        recorder.stage_metrics(
                            me_cells=me_stats["n_cells"],
                            me_admitted=me_stats["n_admitted"],
                            me_replaced=me_stats["n_replaced"],
                        )
                else:
                    sil_dataset.add_trajectories(
                        pool_trajs, sil_scores,
                        keep_ratio=config.sil_keep_ratio, iteration=iteration,
                    )
                # map_elites / top-K 仍可按分数兜底。td_pareto 容量在 ingest 内处理，
                # 禁止再按 traj.score 截断。
                if (config.sil_archive_mode != "td_pareto"
                        and len(sil_dataset.trajectories) > config.sil_traj_cap):
                    top = sorted(
                        sil_dataset.trajectories, key=lambda t: t.score, reverse=True,
                    )[: config.sil_traj_cap]
                    sil_dataset.set_trajectories(top)

            # Phase 2: PPO 更新（前 value_warmup_iters 轮只预热 critic，冻结策略防灾难性遗忘）
            update_started = time.perf_counter()
            value_only = iteration < warmup_end
            if value_only:
                logger.info(
                    "Phase 2: Critic 预热 %d/%d (只训 critic_head, %d epochs, 策略冻结)",
                    iteration + 1, warmup_end, config.value_warmup_epochs,
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
                # 行为门控水平：当前轮 per-agent 平均 credited（原始米）。
                # corridor_progress_m 是全 agent 逐步累加和，除以 agent 数得均值。
                _corridor = getattr(rollout_worker, "last_corridor_metrics", None) or {}
                _valid_agents = _corridor.get("corridor_valid_agents", config.num_agents)
                _behavior = (_corridor.get("corridor_progress_m", 0.0) / _valid_agents
                             if _corridor and _valid_agents else None)
                metrics.update(ppo_trainer.sil_update(
                    model, sil_dataset,
                    loss_coef=config.sil_loss_coef * _sil_scale,
                    # 预算从属于 PPO：当轮 PPO 被 KL 早停饥饿时 SIL 同步收敛，
                    # 防止模仿梯度压倒在线更新（20260923 塌陷期曾 297:43）。
                    budget_updates=int(metrics.get("n_updates", 0)),
                    behavior_progress=_behavior,
                    iteration=iteration + 1,
                ))
                sampling_record = getattr(ppo_trainer, "last_sil_sampling", None)
                if recorder is not None and sampling_record is not None:
                    recorder.save_sil_sampling(iteration + 1, sampling_record)

            if config.gru_sil_enabled and not value_only and iteration >= config.sil_start_iter:
                from training.gru_sil import gru_sil_update
                gru_metrics = gru_sil_update(
                    ppo_trainer, model, elite_archive, sequence_encoder, buffer,
                    budget_updates=int(metrics.get("n_updates", 0)), iteration=iteration + 1)
                metrics.update(gru_metrics)
                logger.info("GRU SIL: %d/%d updates, positive=%d/%d, KL=%.6f, rejected=%d",
                            gru_metrics["gru_sil_updates"], gru_metrics["gru_sil_budget"],
                            gru_metrics["gru_sil_positive_frames"], gru_metrics["gru_sil_sampled_frames"],
                            gru_metrics["gru_sil_kl"], gru_metrics["gru_sil_rejected_steps"])
                recorder.save_sil_sampling(iteration + 1, ppo_trainer.last_gru_sil_sampling)

            # Phase 2.55: σ 退火（M2 修复）：巩固进展 → 探索噪声线性退火，
            # 下限 ppo_log_std_min；预热轮不动（策略未训练）。forward 内 clamp 兜底。
            if config.ppo_log_std_anneal_rate > 0 and not value_only:
                model.anneal_log_std(config.ppo_log_std_anneal_rate, config.ppo_log_std_min)

            update_seconds = time.perf_counter() - update_started
            # Phase 2.6: 停滞探测器 — 按精英池最佳进度判断是否停滞（走廊轨迹取 credited
            # 总进度，其余取 secured_dy 竖直口径），停滞则抬 boost、有进展则回落；
            # 设 model.exploration_boost 供【下一轮】采集+更新一致生效。
            if (config.explore_boost_enabled and sil_dataset is not None
                    and iteration >= config.sil_start_iter):
                from training.reference_corridor import trajectory_endpoint
                sdys = [
                    trajectory_endpoint(t).distance if t.corridor_progress is not None
                    or getattr(t, "corridor_credit_events", None) is not None
                    else float(progress_metric(t.raw_states, config)[2])
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
                _std_now = float((model.actor_log_std + _boost).clamp(
                    config.ppo_log_std_min, config.ppo_log_std_max
                ).exp().mean())
                logger.info(
                    "[explore] pool_best_sdy=%.2f stall=%d boost=%.2f (std≈%.3f)",
                    _best_pool_sdy, _stall, _boost, _std_now,
                )
                if recorder is not None:
                    recorder.stage_metrics(
                        explore_pool_best_sdy=_best_pool_sdy,
                        explore_stall=_stall,
                        explore_boost=_boost,
                        explore_std=_std_now,
                    )

            # Phase 3: 日志 & 保存
            reward_arr = np.array(buffer._rewards)
            extras = {
                "mean_reward": float(reward_arr.mean()),
                "sum_reward": float(reward_arr.sum()),
                "buffer_size": float(buffer.size),
            }
            extras.update(rollout_worker.last_corridor_metrics)
            extras.update(rollout_worker.last_profile)
            extras["update_and_sil_seconds"] = update_seconds
            log_metrics(iteration + 1, metrics, extras)
            if recorder is not None:
                recorder.append_metrics(iteration + 1, metrics, extras)
            completed_iterations = iteration + 1
            physical_samples += int(extras.get("rollout_environment_steps", buffer.size))

            if (iteration + 1) % config.save_interval == 0:
                save_checkpoint(
                    model, ppo_trainer.optimizer,
                    iteration + 1, config.checkpoint_dir,
                    config=config, reference_metadata=reference_meta,
                    value_warmup_end=warmup_end,
                    trainer=ppo_trainer, deployment_rng=rollout_worker._rng,
                    loop_state=checkpoint_loop_state(),
                )
                if reference is None:
                    save_efficiency_map(eff_map, env_data, iteration + 1, config.log_dir)
                # 精英池只活在内存里，不随 checkpoint 走。不在这里快照，"策略当时到底
                # 走出了什么行为"就只能靠事后重新拉游戏在环采样去猜。
                if recorder is not None and sil_dataset is not None:
                    recorder.save_elites(sil_dataset.trajectories, iteration + 1)
                if elite_archive is not None:
                    elite_archive.save(iteration + 1)

            if sample_target is not None and physical_samples >= sample_target:
                logger.info("实际物理样本终点: %d >= %d，完成 %d 轮（含 critic 预热）",
                            physical_samples, sample_target, completed_iterations)
                break

        # 最终保存
        save_checkpoint(
            model, ppo_trainer.optimizer,
            completed_iterations, config.checkpoint_dir,
            config=config, reference_metadata=reference_meta,
            value_warmup_end=warmup_end,
            trainer=ppo_trainer, deployment_rng=rollout_worker._rng,
            loop_state=checkpoint_loop_state(),
        )
        if recorder is not None:
            if sil_dataset is not None:
                recorder.save_elites(sil_dataset.trajectories, completed_iterations)
            if elite_archive is not None:
                elite_archive.save(completed_iterations)
            recorder.finalize("completed", last_iteration=completed_iterations)
        logger.info("PPO 训练完成")

    except BaseException as exc:
        # 中断（Ctrl-C）和异常都要把 run_meta 落到终态，否则归档里留下一个永远
        # "running" 的记录，事后分不清是跑完了还是崩了。
        if recorder is not None:
            recorder.finalize(
                "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                last_iteration=last_iteration,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise
    finally:
        rollout_worker.teardown()


if __name__ == "__main__":
    main()

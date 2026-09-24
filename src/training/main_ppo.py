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
import random
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
from training.experiment_options import add_experiment_arguments, apply_experiment_arguments, restore_initialization_shape
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
    torch.save(state, path)
    logger.info("Checkpoint saved: %s", path)


def _sil_score(traj: "Trajectory", eff_map: "ClimbingEfficiencyMap", config: TrainConfig) -> float:
    """精英池排序分分派：走廊轨迹用 credited 进度口径（corridor_score），
    其余沿用高度口径（score_trajectory）。与 prune_trajectory / _rebuild_index 同步。"""
    if traj.corridor_progress is not None:
        return corridor_score(traj.corridor_progress, config)
    return score_trajectory(traj.raw_states, eff_map, config)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO 训练 — 策略梯度迭代循环")
    add_experiment_arguments(parser)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--geometry-profile", choices=("native-pose-v1", "legacy"), default=None,
                        help="Explicitly select contact/patch pose calibration; resume otherwise preserves checkpoint geometry")
    parser.add_argument("--save-rollout-inputs", action="store_true",
                        help="Archive complete PPO inputs under checkpoint_dir for exact replay")
    parser.add_argument("--recipe", choices=("ppo", "ppo-ou", "ppo-ou-sil"), default=None,
                        help="Fixed ablation; disables adaptive exploration and experimental contact heads")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--target-physical-samples", type=int, default=None,
                        help="Fresh run: stop after the first complete iteration reaching this sample count (includes critic warmup)")
    parser.add_argument("--additional-physical-samples", type=int, default=None,
                        help="Resume: stop after this many additional physical samples, at a complete iteration")
    parser.add_argument("--sil-resume-pool", type=str, default=None,
                        help="Restore a saved SIL elite pool, including corridor fields, scores and source iterations")
    parser.add_argument("--steps-per-rollout", type=int, default=None)
    parser.add_argument("--ppo-batch-size", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None,
                        help="Checkpoint interval in completed training iterations")
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
            _REPO_ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory.json"
        ), default=None,
        help="启用走廊进度奖励；可指定 JSON，省略路径使用当前速通视频轨迹",
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
    for name in ("ppo_batch_size", "save_interval", "torch_threads"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.target_physical_samples is not None:
        if args.target_physical_samples < 1 or args.resume:
            parser.error("--target-physical-samples requires a positive count and a fresh run")
    if args.additional_physical_samples is not None:
        if args.additional_physical_samples < 1 or not args.resume or args.target_physical_samples is not None:
            parser.error("--additional-physical-samples requires a positive count and --resume")
    if args.sil_resume_pool and (not args.resume or args.sil_seed):
        parser.error("--sil-resume-pool requires --resume and cannot be combined with --sil-seed")
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


def main() -> None:
    _setup_logging()
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    logger.info("PyTorch CPU threads: %d", torch.get_num_threads())

    if args.run_dir and (args.checkpoint_dir or args.log_dir):
        raise SystemExit("--run-dir 与 --checkpoint-dir/--log-dir 互斥，二选一")

    config = TrainConfig()
    resume_state = torch.load(args.resume, weights_only=False) if args.resume else {}
    if args.init_from and (args.resume or args.resume_bc):
        raise ValueError("--init-from is mutually exclusive with --resume / --resume-bc")
    init_state = torch.load(args.init_from, map_location="cpu", weights_only=False) if args.init_from else {}
    restore_initialization_shape(config, init_state)
    # Restore the actual network/input contract, not today's defaults.
    for field in fields(config):
        if field.name in resume_state.get("config", {}) and field.name not in ("game_root", "port"):
            setattr(config, field.name, resume_state["config"][field.name])
    if args.resume and "actor_mean_contact.weight" not in resume_state["model"]:
        config.contact_heads_enabled = False
    geometry_changed = configure_geometry(config, resume_state or init_state or ({"bc": True} if args.resume_bc else None),
                                           args.geometry_profile)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    if args.steps_per_rollout is not None:
        config.steps_per_rollout = args.steps_per_rollout
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
    if args.random_deploy is not None:
        config.random_deploy = args.random_deploy
    configure_credit(config, args, resume_state)
    configure_reference(config, args, resume_state)
    apply_experiment_arguments(config, args, resume_state)
    if args.corridor_sil and config.reference_trajectory:
        # 走廊 SIL 机械（corridor_score / prune_trajectory 走廊分支 / MAP-Elites bd /
        # 种子 corridor 字段重放 / P3.2 优势加权）已就绪；configure_reference 与
        # experimental 路径里的 sil 强制停用均为就绪前遗留，在此统一覆盖。
        # explore_boost 保持停用（其停滞判据是高度口径，走廊语义需另行论证）。
        config.sil_enabled = True
        config.sil_advantage_weighting_enabled = True
        logger.info("--corridor-sil: 走廊模式启用 SIL（评分/剪枝/入池/优势加权均走走廊口径）")
    from training.policy_features import experimental_enabled
    if experimental_enabled(config) and (args.resume_bc or args.recipe == "ppo-ou-sil"):
        raise ValueError("Representation experiments support PPO --init-from/--resume and disable legacy BC/SIL")
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
    if recorder is not None:
        recorder.write_meta(config, argv=sys.argv, extra={
            "resume": args.resume, "resume_bc": args.resume_bc, "init_from": args.init_from, "reference_corridor": reference_meta,
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
        # σ 重设（M2 修复）：PPO→PPO 暖启动会原样继承源 checkpoint 的 log_std
        #（实测 σ≈0.52 全程钉死，技能住在噪声里）。按 ppo_init_log_std 拉回设计初值。
        if config.ppo_reset_log_std_on_init and hasattr(model, "reset_log_std"):
            model.reset_log_std(config.ppo_init_log_std)
            logger.info("init-from σ 重设: log_std → %.2f (std=%.3f)",
                        config.ppo_init_log_std, float(model.actor_log_std.detach().exp().mean()))
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
    semantics_changed = reference_changed or geometry_changed
    warmup_end = start_iteration + config.value_warmup_iters if semantics_changed else config.value_warmup_iters
    if not semantics_changed and args.value_warmup_iters is None:
        warmup_end = max(warmup_end, int(resume_state.get("value_warmup_end", 0)))
    if reference_changed:
        logger.info("奖励目标改变：从当前迭代重新预热 critic %d 轮", config.value_warmup_iters)
    if geometry_changed:
        logger.info("几何输入标定改变：从当前迭代重新预热 critic %d 轮", config.value_warmup_iters)

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
    if reference is not None and config.sil_enabled:
        reference_maps = reference.guidance_maps(
            eff_map.traversable.shape, eff_map._min_gx, eff_map._min_gy,
            config.grid_resolution,
        )
        # 共享给 rollout_worker，避免在线采集时重复栅格化
        if rollout_worker._reference_maps is None:
            rollout_worker._reference_maps = reference_maps
    sil_dataset = (
        TrajectoryDataset(
            config, eff_map.traversable, eff_map.get_arr(),
            min_gx=eff_map._min_gx, min_gy=eff_map._min_gy,
            solid_polygons=polygons,
            tip_local=extract_tip_local(player_data),
            body_local=extract_body_local(player_data),
            reference_maps=reference_maps,
        )
        if config.sil_enabled else None
    )

    # ── 精英池冷启动种子（可选）：外部剪枝后的优质轨迹直接入池，iteration=-1 标记；
    # 首轮 add_trajectories_map_elites 会用当前池重建 cell_map，种子照常参与竞争/被替换。
    # 走廊模式下若种子未携带 corridor 字段，从 raw_states 确定性重放 tracker 补齐。
    if sil_dataset is not None and args.sil_seed:
        seed_trajs: list[Trajectory] = []
        for seed_path in args.sil_seed:
            z = np.load(seed_path)
            traj = Trajectory(
                raw_states=np.asarray(z["raw_states"], dtype=np.float32),
                actions=np.asarray(z["actions"], dtype=np.float32),
                contact=(np.asarray(z["contact"], dtype=np.float32)
                         if "contact" in z.files else None),
                iteration=-1,
                corridor_progress=(np.asarray(z["corridor_progress"], dtype=np.float32)
                                   if "corridor_progress" in z.files else None),
                guidance_index=(np.asarray(z["guidance_index"], dtype=np.int8)
                                if "guidance_index" in z.files else None),
            )
            if reference is not None and traj.corridor_progress is None:
                traj.corridor_progress, traj.guidance_index = replay_corridor_fields(
                    reference, traj.raw_states,
                )
                logger.info("SIL 种子走廊字段重放: %s (credited=%.3fm)",
                            seed_path, float(traj.corridor_progress.sum()))
            traj.score = _sil_score(traj, eff_map, config)
            seed_trajs.append(traj)
            logger.info("SIL 种子: %s (T=%d, score=%.3f)",
                        seed_path, len(traj.actions), traj.score)
        if seed_trajs:
            sil_dataset.set_trajectories(seed_trajs)
            logger.info("SIL 精英池冷启动: %d 条种子轨迹入池", len(seed_trajs))

    if args.sil_resume_pool:
        if sil_dataset is None:
            raise ValueError("Restoring an elite pool requires SIL to be enabled")
        restored_trajs = [Trajectory(**row) for row in load_elites(args.sil_resume_pool)]
        if not restored_trajs:
            raise ValueError("Cannot resume an empty SIL pool")
        if reference is not None and any(t.corridor_progress is None or t.guidance_index is None
                                         for t in restored_trajs):
            raise ValueError("Corridor SIL pool is missing saved corridor fields")
        sil_dataset.set_trajectories(restored_trajs)
        logger.info("SIL 精英池完整恢复: %s (%d 条轨迹)", args.sil_resume_pool, len(restored_trajs))

    # ── 奖励归一化器 ──
    calibration_steps = config.steps_per_rollout * config.num_agents * 10
    reward_normalizer = RewardNormalizer(calibration_steps=calibration_steps) if reference is None else None
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
    last_iteration = start_iteration
    completed_iterations = start_iteration
    physical_samples = 0
    sample_target = args.additional_physical_samples or args.target_physical_samples
    try:
        for iteration in range(start_iteration, config.max_iterations):
            last_iteration = iteration
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
                if config.sil_map_elites_enabled:
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
                # 按分数保留 top-N 兜底（防 cell 数长期无界增长；MAP-Elites 下同 cell
                # 至多 1 条 → 天然稀疏，通常远低于 cap，此裁剪几乎 no-op 但保底）
                if len(sil_dataset.trajectories) > config.sil_traj_cap:
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
                _behavior = (_corridor.get("corridor_progress_m", 0.0)
                             / max(1, config.num_agents)) if _corridor else None
                metrics.update(ppo_trainer.sil_update(
                    model, sil_dataset,
                    loss_coef=config.sil_loss_coef * _sil_scale,
                    # 预算从属于 PPO：当轮 PPO 被 KL 早停饥饿时 SIL 同步收敛，
                    # 防止模仿梯度压倒在线更新（20260923 塌陷期曾 297:43）。
                    budget_updates=int(metrics.get("n_updates", 0)),
                    behavior_progress=_behavior,
                ))

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
                sdys = [
                    float(t.corridor_progress.sum()) if t.corridor_progress is not None
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
                )
                if reference is None:
                    save_efficiency_map(eff_map, env_data, iteration + 1, config.log_dir)
                # 精英池只活在内存里，不随 checkpoint 走。不在这里快照，"策略当时到底
                # 走出了什么行为"就只能靠事后重新拉游戏在环采样去猜。
                if recorder is not None and sil_dataset is not None:
                    recorder.save_elites(sil_dataset.trajectories, iteration + 1)

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
        )
        if recorder is not None:
            if sil_dataset is not None:
                recorder.save_elites(sil_dataset.trajectories, completed_iterations)
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

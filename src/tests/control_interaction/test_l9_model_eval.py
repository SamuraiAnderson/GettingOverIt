"""
层次 9：模型评估 — 加载 checkpoint，单 agent 从初始位置无噪声推理

加载训练好的模型权重和效率图，用单个 player 从游戏默认起点开始，
模型纯推理（无噪声）驱动 N 步，记录完整轨迹并保存 JSON。
自动识别 BC（ActionPredictor）和 PPO（ActorCritic）checkpoint。

用法:
  python src/tests/control_interaction/test_l9_model_eval.py
  python src/tests/control_interaction/test_l9_model_eval.py --checkpoint checkpoints/model_iter_0005.pt
  python src/tests/control_interaction/test_l9_model_eval.py --checkpoint checkpoints/ppo_iter_0050.pt
  python src/tests/control_interaction/test_l9_model_eval.py --steps 2000
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig
from training.dataset import (
    build_dynamics,
    build_patch,
)
from training.deploy_sampling import extract_body_local, extract_tip_local
from training.model import ActionPredictor
from training.actor_critic import ActorCritic
from training.reward import ClimbingEfficiencyMap
from training.rollout import _build_history_window

sys.path.insert(0, str(_REPO_ROOT / "src" / "tests" / "control_interaction"))
from test_l7_surface_airdrop import extract_polygons, load_colliders

logger = logging.getLogger(__name__)

_PROJECT_CONFIG = _REPO_ROOT / "src" / "config" / "project.json"


def _get_game_root() -> Path:
    with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return Path(cfg["game"]["executable_path"]).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L9: 模型评估 — 单 agent 无噪声推理")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/model_iter_0010.pt",
        help="模型权重路径（自动识别 BC / PPO checkpoint）",
    )
    parser.add_argument("--steps", type=int, default=1000, help="推理步数")
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="同一 checkpoint 复跑次数（>1 时报 secured 高度分布，抵消游戏方差）",
    )
    parser.add_argument(
        "--output", type=str, default="runs/_eval/eval_trajectory.json",
        help=(
            "输出轨迹 JSON 路径。默认落到评估归档目录；跑完用 "
            "`python -m src.training.archive_evals` 刷新可入库的 summary.csv"
        ),
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--no-launch", action="store_true", help="不启动游戏（假设已运行）")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    config = TrainConfig()
    config.num_agents = 1

    # ── 加载模型（自动识别 BC / PPO checkpoint）──
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error("Checkpoint 不存在: %s", ckpt_path)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_ppo = False

    ckpt_data = torch.load(ckpt_path, weights_only=False)
    if isinstance(ckpt_data, dict) and "model" in ckpt_data:
        # PPO checkpoint: {"model": state_dict, "optimizer": ..., "iteration": ...}
        model = ActorCritic(config)
        model.load_state_dict(ckpt_data["model"])
        is_ppo = True
        logger.info("PPO 模型已加载: %s (device=%s)", ckpt_path, device)
    else:
        # BC checkpoint: 直接是 state_dict
        state_dict = ckpt_data if isinstance(ckpt_data, dict) else torch.load(ckpt_path, weights_only=True)
        model = ActionPredictor(config)
        model.load_state_dict(state_dict)
        logger.info("BC 模型已加载: %s (device=%s)", ckpt_path, device)

    model = model.to(device)
    model.eval()

    # ── 构建 dualscale patch 所需几何（与训练一致：build_patch 共享实现）──
    game_root = _get_game_root()
    env_data, player_data = load_colliders(game_root)
    polygons = extract_polygons(env_data)
    tip_local = extract_tip_local(player_data)
    body_local = extract_body_local(player_data)

    cache_dir = Path(config.checkpoint_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mask_cache = str(cache_dir / "traversable_mask.npz")
    eff_map = ClimbingEfficiencyMap(
        polygons,
        grid_resolution=config.grid_resolution,
        diffusion_iterations=config.diffusion_iterations,
        diffusion_alpha=config.diffusion_alpha,
        cache_path=mask_cache,
        height_prior_weight=config.height_prior_weight,
    )
    # 用 warmup_state.pkl 缓存的轨迹回放价值，复现训练时 wide 通道看到的效率图
    warmup_path = Path(config.checkpoint_dir) / "warmup_state.pkl"
    if warmup_path.exists():
        try:
            from training.dataset import Trajectory
            with open(warmup_path, "rb") as f:
                ws = pickle.load(f)
            trajs = [
                Trajectory(
                    raw_states=d["raw_states"], actions=d["actions"],
                    score=d.get("score", 0.0), iteration=d.get("iteration", 0),
                )
                for d in ws.get("trajectories", [])
            ]
            if trajs:
                eff_map.update(trajs, config, prev_eff_map=None)
                logger.info("效率图已按 %d 条缓存轨迹回放", len(trajs))
        except Exception as e:
            logger.warning("效率图轨迹回放失败（wide 通道退化为零）: %s", e)

    terrain_mask = eff_map.traversable
    eff_arr = eff_map.get_arr()
    min_gx, min_gy = eff_map._min_gx, eff_map._min_gy
    logger.info(
        "地形/效率图就绪: mask=%s, eff_max=%.3f, patch_mode=%s",
        terrain_mask.shape,
        float(np.nanmax(eff_arr)) if eff_arr is not None else 0.0,
        config.patch_mode,
    )

    # ── 启动游戏 ──
    from start.game_launcher import GameLauncher
    from start.game_mode_controller import GameModeController
    from env.goi_env import GoiEnv
    from training.rollout import _write_num_duplicates

    game_root = _get_game_root()
    process = None

    if not args.no_launch:
        _write_num_duplicates(game_root, 1)
        GameModeController(str(game_root)).set_game_runtime_mode()
        process = GameLauncher().launch(wait=False)
        logger.info("游戏已启动")

    env = GoiEnv(port=config.port, num_agents=1)
    env.connect()
    env.reset()

    zero_actions = np.zeros((1, 2), dtype=np.float32)
    for _ in range(args.warmup_steps):
        env.step(zero_actions)
    env.new_snapshot()
    env.toggle_collider_visual(False)
    logger.info("Warmup 完成，开始推理")

    # ── 推理循环（可复跑 N 次以抵消游戏方差）──
    from training.reward import progress_metric

    ctx = config.context_len
    scale = config.action_scale
    n_steps = args.steps
    repeat = max(1, args.repeat)

    def run_episode(verbose: bool):
        """跑一回合：env.reset() → n_steps 确定性推理，返回 (states_arr, actions_arr)。"""
        obs = env.reset()
        all_states = [obs[0].copy()]
        all_actions: list[np.ndarray] = []
        dyn_history: list[np.ndarray] = []
        act_history: list[np.ndarray] = []
        patch_history: list[np.ndarray] = []
        max_y = float(obs[0, 1])

        for step in range(n_steps):
            cur_obs = obs[0]
            patch = build_patch(
                cur_obs, terrain_mask, eff_arr, config, min_gx, min_gy,
                solid_polygons=polygons, tip_local=tip_local, body_local=body_local,
            )
            dynamics = build_dynamics(cur_obs)
            dyn_history.append(dynamics.copy())
            patch_history.append(patch.copy())

            dyn_window, pat_window, act_window, valid_mask = _build_history_window(
                dyn_history, patch_history, act_history, ctx,
            )
            if is_ppo:
                pred = model.predict_deterministic(
                    dyn_window, pat_window, act_window, valid_mask=valid_mask
                )
            else:
                pred = model.predict(
                    dyn_window, pat_window, act_window, valid_mask=valid_mask
                )
            action = np.clip(pred, -scale, scale).astype(np.float32)
            obs, _ = env.step(action.reshape(1, 2))

            all_states.append(obs[0].copy())
            all_actions.append(action.copy())
            act_history.append(action.copy())

            cur_y = float(obs[0, 1])
            if cur_y > max_y:
                max_y = cur_y

            if verbose:
                logger.info(
                    "  step %4d/%d | pos=(%.2f, %.2f) | action=(%.4f, %.4f) | max_y=%.2f",
                    step + 1, n_steps, float(obs[0, 0]), cur_y,
                    float(action[0]), float(action[1]), max_y,
                )

        return np.array(all_states), (
            np.array(all_actions) if all_actions else np.zeros((0, 2))
        )

    logger.info("=" * 60)
    logger.info("开始评估: checkpoint=%s, %d 步 × %d 回合", ckpt_path, n_steps, repeat)
    logger.info("=" * 60)

    records: list[dict] = []
    last_states = last_actions = None
    try:
        for ep in range(repeat):
            states_arr, actions_arr = run_episode(verbose=(repeat == 1))
            last_states, last_actions = states_arr, actions_arr

            start_y = float(states_arr[0][1])
            final_y = float(states_arr[-1][1])
            max_y = float(states_arr[:, 1].max())
            # secured 口径（与训练/数据集排序一致）：dwell 确认后的净爬升 + 门控右向 reach
            _progress, _peak_idx, secured_dy, reach = progress_metric(states_arr, config)

            records.append({
                "start_y": start_y, "final_y": final_y, "max_y": max_y,
                "secured_dy": float(secured_dy), "reach": float(reach),
            })
            logger.info(
                "回合 %d/%d: secured_dy=%.2f  reach=%.2f  peak_Δ=%.2f  final_Δ=%.2f",
                ep + 1, repeat, secured_dy, reach, max_y - start_y, final_y - start_y,
            )
    except KeyboardInterrupt:
        logger.info("用户中断，已完成 %d 回合", len(records))
    finally:
        env.close()
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                pass

    # ── 保存最后一回合轨迹 ──
    if last_states is not None:
        result = {
            "checkpoint": str(ckpt_path),
            "steps": int(last_actions.shape[0]),
            "repeat": repeat,
            "records": records,
            "states": last_states.tolist(),
            "actions": last_actions.tolist(),
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        logger.info("轨迹已保存: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # ── 打印报告（secured 为主指标）──
    def _stats(key: str) -> str:
        vals = np.array([r[key] for r in records], dtype=np.float64)
        return (
            f"mean={vals.mean():.2f}  std={vals.std():.2f}  "
            f"min={vals.min():.2f}  median={np.median(vals):.2f}  max={vals.max():.2f}"
        )

    logger.info("=" * 60)
    logger.info("评估报告  Checkpoint: %s", ckpt_path)
    logger.info("  回合数: %d  (每回合 %d 步)", len(records), n_steps)
    if records:
        logger.info("  secured_dy (主指标): %s", _stats("secured_dy"))
        logger.info("  reach     (右向)   : %s", _stats("reach"))
        peak_deltas = np.array([r["max_y"] - r["start_y"] for r in records])
        final_deltas = np.array([r["final_y"] - r["start_y"] for r in records])
        logger.info(
            "  raw peak_Δ (诊断)  : mean=%.2f  std=%.2f  min=%.2f  max=%.2f",
            peak_deltas.mean(), peak_deltas.std(), peak_deltas.min(), peak_deltas.max(),
        )
        logger.info(
            "  raw final_Δ(诊断)  : mean=%.2f  std=%.2f  min=%.2f  max=%.2f",
            final_deltas.mean(), final_deltas.std(), final_deltas.min(), final_deltas.max(),
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

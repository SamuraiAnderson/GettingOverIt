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
    BODY_POS_INDICES,
    HAMMER_POS_INDICES,
    build_dynamics,
    crop_centered,
    render_gaussian,
)
from training.model import ActionPredictor
from training.actor_critic import ActorCritic
from training.reward import ClimbingEfficiencyMap
from training.rollout import _build_history_window

logger = logging.getLogger(__name__)

_PROJECT_CONFIG = _REPO_ROOT / "src" / "config" / "project.json"


def _get_game_root() -> Path:
    with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return Path(cfg["game"]["executable_path"]).parent


def _build_patch(
    obs_1d: np.ndarray,
    config: TrainConfig,
    terrain_mask: np.ndarray | None,
    eff_arr: np.ndarray | None,
    min_gx: int,
    min_gy: int,
) -> np.ndarray:
    """为单个观测构建 4ch 32x32 patch。"""
    ps = config.patch_size
    patch = np.zeros((config.patch_channels, ps, ps), dtype=np.float32)
    px, py = float(obs_1d[0]), float(obs_1d[1])

    if terrain_mask is not None:
        patch[0] = crop_centered(
            terrain_mask.astype(np.float32),
            px, py, min_gx, min_gy,
            size=ps, patch_res=config.patch_resolution,
            arr_res=config.grid_resolution,
        )

    if eff_arr is not None:
        patch[1] = crop_centered(
            eff_arr.astype(np.float32),
            px, py, min_gx, min_gy,
            size=ps, patch_res=config.patch_resolution,
            arr_res=config.grid_resolution,
        )

    for bx_idx, by_idx in BODY_POS_INDICES:
        render_gaussian(
            patch[2],
            float(obs_1d[bx_idx]), float(obs_1d[by_idx]),
            px, py, patch_res=config.patch_resolution,
        )

    for hx_idx, hy_idx in HAMMER_POS_INDICES:
        render_gaussian(
            patch[3],
            float(obs_1d[hx_idx]), float(obs_1d[hy_idx]),
            px, py, patch_res=config.patch_resolution,
        )

    return patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="L9: 模型评估 — 单 agent 无噪声推理")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/model_iter_0010.pt",
        help="模型权重路径（自动识别 BC / PPO checkpoint）",
    )
    parser.add_argument("--steps", type=int, default=1000, help="推理步数")
    parser.add_argument(
        "--output", type=str, default="logs/eval_trajectory.json",
        help="输出轨迹 JSON 路径",
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

    # ── 加载地形 mask ──
    mask_cache = Path(config.checkpoint_dir) / "cache" / "traversable_mask.npz"
    terrain_mask = None
    min_gx, min_gy = 0, 0
    if mask_cache.exists():
        data = np.load(mask_cache)
        terrain_mask = data["mask"].astype(bool)
        min_gx, min_gy = int(data["min_gx"]), int(data["min_gy"])
        logger.info("地形 mask 已加载: shape=%s", terrain_mask.shape)
    else:
        logger.warning("未找到地形缓存 %s，patch 通道 0 将为空", mask_cache)

    # ── 加载效率图 ──
    warmup_path = Path(config.checkpoint_dir) / "warmup_state.pkl"
    eff_arr = None
    if warmup_path.exists():
        with open(warmup_path, "rb") as f:
            warmup_state = pickle.load(f)
        eff_arr = warmup_state.get("eff_arr")
        if eff_arr is not None:
            logger.info("效率图已加载: shape=%s", eff_arr.shape)
    else:
        logger.warning("未找到 warmup_state.pkl，patch 通道 1 将为空")

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

    # ── 推理循环 ──
    ctx = config.context_len
    scale = config.action_scale
    n_steps = args.steps

    obs = env.reset()

    all_states = [obs[0].copy()]
    all_actions = []
    dyn_history = []
    act_history = []
    patch_history = []

    max_y = float(obs[0, 1])
    start_y = float(obs[0, 1])

    logger.info("=" * 60)
    logger.info("开始推理: %d 步, 起始 y=%.2f", n_steps, start_y)
    logger.info("=" * 60)

    try:
        for step in range(n_steps):
            cur_obs = obs[0]

            patch = _build_patch(cur_obs, config, terrain_mask, eff_arr, min_gx, min_gy)
            dynamics = build_dynamics(cur_obs)

            dyn_history.append(dynamics.copy())
            patch_history.append(patch.copy())

            dyn_window, pat_window, act_window, valid_mask = _build_history_window(
                dyn_history, patch_history, act_history,
                ctx, config.state_dim,
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

            actions_batch = action.reshape(1, 2)
            obs, _ = env.step(actions_batch)

            all_states.append(obs[0].copy())
            all_actions.append(action.copy())
            act_history.append(action.copy())

            cur_y = float(obs[0, 1])
            if cur_y > max_y:
                max_y = cur_y

            logger.info(
                "  step %4d/%d | pos=(%.2f, %.2f) | pred=(%.4f, %.4f) | "
                "action=(%.4f, %.4f) | dyn=[%s] | max_y=%.2f",
                step + 1, n_steps,
                float(obs[0, 0]), cur_y,
                float(pred[0]), float(pred[1]),
                float(action[0]), float(action[1]),
                ", ".join(f"{d:.3f}" for d in dynamics[:6]),
                max_y,
            )

    except KeyboardInterrupt:
        logger.info("用户中断，已完成 %d 步", len(all_actions))

    finally:
        env.close()
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                pass

    # ── 保存轨迹 ──
    final_y = float(all_states[-1][1])
    states_arr = np.array(all_states)
    actions_arr = np.array(all_actions) if all_actions else np.zeros((0, 2))

    result = {
        "checkpoint": str(ckpt_path),
        "steps": len(all_actions),
        "start_y": round(start_y, 3),
        "max_y": round(max_y, 3),
        "final_y": round(final_y, 3),
        "states": states_arr.tolist(),
        "actions": actions_arr.tolist(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    logger.info("轨迹已保存: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # ── 打印报告 ──
    logger.info("=" * 60)
    logger.info("评估报告")
    logger.info("  Checkpoint: %s", ckpt_path)
    logger.info("  步数: %d", len(all_actions))
    logger.info("  起始 y: %.2f", start_y)
    logger.info("  最高 y: %.2f  (Δ=%.2f)", max_y, max_y - start_y)
    logger.info("  最终 y: %.2f  (Δ=%.2f)", final_y, final_y - start_y)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

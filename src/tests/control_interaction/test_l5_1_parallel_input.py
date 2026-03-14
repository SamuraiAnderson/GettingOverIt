"""
层次 5.1：Per-Agent 输入隔离验证

验证 RewiredMouseOverride 的 per-agent 隔离机制是否工作：
  - Agent 0 持续右推 (+amplitude, 0)
  - Agent 1 持续左推 (-amplitude, 0)
  - 执行 n_steps 步后，两个 Agent 的 player_x 应出现可观测的分离
  - 同时检查两个 Agent 的 vel_x 符号是否在多数步骤中相反

前置条件：
  - C# 端 RuntimeConfig.json 中 numDuplicates >= 2
  - 已部署包含 per-agent 输入隔离的 GameRuntime_v2 DLL

用法：
  python src/tests/control_interaction/test_l5_1_parallel_input.py
  python src/tests/control_interaction/test_l5_1_parallel_input.py --no-launch --steps 300
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from env.goi_env import GoiEnv, STATE_DIM
from start.game_launcher import GameLauncher
from start.game_mode_controller import GameModeController

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "test_config.json"

STATE_LABELS = [
    "player_x", "player_y", "vel_x", "vel_y", "ang_vel",
    "hub_x", "hub_y", "hub_vx", "hub_vy", "hub_angle",
    "slider_x", "slider_y", "slider_vx", "slider_vy", "slider_angle",
    "handle_x", "handle_y", "handle_vx", "handle_vy",
    "pole_x", "pole_y", "pole_vx", "pole_vy",
    "tip_x", "tip_y", "tip_vx", "tip_vy",
    "hammer_angle", "timestamp",
]


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def launch_game() -> object:
    logger.info("设置游戏为 GameRuntime 模式...")
    GameModeController().set_game_runtime_mode()
    proc = GameLauncher().launch(wait=False)
    logger.info("游戏已启动 (PID=%d)", proc.pid)
    return proc


def warmup_and_snapshot(env: GoiEnv, n_agents: int, n_steps: int = 500) -> None:
    logger.info("Warmup: %d 步（%d agents，零动作）...", n_steps, n_agents)
    zero_action = np.zeros((n_agents, 2), dtype=np.float32)
    for _ in range(n_steps):
        env.step(zero_action)
    env.new_snapshot()
    logger.info("Warmup 完成，已设置新快照")


def _detect_num_agents(env: GoiEnv) -> int:
    """通过 reset 响应推断 C# 端实际 agent 数量。"""
    obs = env.reset()
    n_detected = obs.shape[0]
    if n_detected != env.num_agents:
        logger.warning(
            "C# 返回 %d 个 agent，Python 预期 %d，自动校正",
            n_detected, env.num_agents,
        )
        env.num_agents = n_detected
    return n_detected


def run_parallel_input_test(
    env: GoiEnv,
    n_agents: int,
    n_steps: int,
    amplitude: float,
    sep_threshold: float,
) -> dict:
    """
    Agent 0 持续右推，Agent 1 持续左推，验证轨迹分离。
    """
    logger.info("=" * 60)
    logger.info("Per-Agent 输入隔离测试: %d 步, amplitude=%.1f", n_steps, amplitude)
    logger.info("  Agent 0: action=(+%.1f, 0)  Agent 1: action=(-%.1f, 0)", amplitude, amplitude)
    logger.info("=" * 60)

    actions = np.zeros((n_agents, 2), dtype=np.float32)
    actions[0, 0] = amplitude
    actions[1, 0] = -amplitude

    obs0 = env.reset()
    traj = {i: [obs0[i].copy()] for i in range(n_agents)}

    for step in range(1, n_steps + 1):
        obs, _ = env.step(actions)
        for i in range(n_agents):
            traj[i].append(obs[i].copy())

        if step % 50 == 0 or step == 1:
            parts = []
            for i in range(n_agents):
                s = obs[i]
                parts.append(
                    f"A{i}[px={s[0]:.2f} vx={s[2]:.3f}]"
                )
            logger.info("  step %3d: %s", step, "  |  ".join(parts))

    traj_arrays = {i: np.stack(traj[i]) for i in range(n_agents)}

    # player_x 列
    px_0 = traj_arrays[0][:, 0]
    px_1 = traj_arrays[1][:, 0]
    final_sep = abs(float(px_0[-1] - px_1[-1]))

    # vel_x 列
    vx_0 = traj_arrays[0][:, 2]
    vx_1 = traj_arrays[1][:, 2]

    # 中后期步骤中 vel_x 符号相反的比例（跳过前 10 步让物理稳定）
    skip = min(10, n_steps // 5)
    sign_opposite = np.sum(np.sign(vx_0[skip:]) != np.sign(vx_1[skip:]))
    total_check = len(vx_0[skip:])
    sign_ratio = sign_opposite / max(total_check, 1)

    # 任一 agent 是否有非零速度
    max_abs_vx_0 = float(np.max(np.abs(vx_0)))
    max_abs_vx_1 = float(np.max(np.abs(vx_1)))

    logger.info("-" * 60)
    logger.info("结果:")
    logger.info("  Agent 0 最终 player_x: %.4f", float(px_0[-1]))
    logger.info("  Agent 1 最终 player_x: %.4f", float(px_1[-1]))
    logger.info("  位置分离: %.4f  (阈值 > %.2f)", final_sep, sep_threshold)
    logger.info("  vel_x 符号相反比例: %.1f%% (%d/%d)", sign_ratio * 100, sign_opposite, total_check)
    logger.info("  Agent 0 max|vel_x|: %.4f", max_abs_vx_0)
    logger.info("  Agent 1 max|vel_x|: %.4f", max_abs_vx_1)

    pass_sep = final_sep > sep_threshold
    pass_motion_0 = max_abs_vx_0 > 1e-3
    pass_motion_1 = max_abs_vx_1 > 1e-3
    all_pass = pass_sep and pass_motion_0 and pass_motion_1

    logger.info("-" * 60)
    logger.info("  位置分离: %s", "PASS" if pass_sep else "FAIL")
    logger.info("  Agent 0 运动: %s", "PASS" if pass_motion_0 else "FAIL (vel_x 始终为 0)")
    logger.info("  Agent 1 运动: %s", "PASS" if pass_motion_1 else "FAIL (vel_x 始终为 0)")
    logger.info("  总体: %s", "PASS — 输入隔离有效" if all_pass else "FAIL — 输入隔离无效或 agent 未运动")

    return {
        "passed": all_pass,
        "pass_sep": pass_sep,
        "pass_motion_0": pass_motion_0,
        "pass_motion_1": pass_motion_1,
        "final_sep": final_sep,
        "sign_ratio": sign_ratio,
        "traj_0": traj_arrays[0],
        "traj_1": traj_arrays[1],
    }


def main():
    cfg = _load_config()
    default_port = cfg.get("port", 9000)
    default_timeout = cfg.get("connect_timeout", 60)
    l5_1_cfg = cfg.get("l5_1", {})
    default_agents = l5_1_cfg.get("num_agents", 2)
    default_steps = l5_1_cfg.get("n_steps", 200)
    default_amp = l5_1_cfg.get("amplitude", 80)
    default_warmup = l5_1_cfg.get("warmup_steps", 500)
    default_thresh = l5_1_cfg.get("sep_threshold", 0.5)

    parser = argparse.ArgumentParser(description="L5.1 Per-Agent 输入隔离验证")
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--timeout", type=float, default=default_timeout)
    parser.add_argument("--agents", type=int, default=default_agents)
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--amplitude", type=float, default=default_amp)
    parser.add_argument("--warmup", type=int, default=default_warmup)
    parser.add_argument("--threshold", type=float, default=default_thresh)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.agents < 2:
        logger.error("需要至少 2 个 Agent（--agents >= 2）")
        sys.exit(1)

    if not args.no_launch:
        launch_game()
        logger.info("等待游戏启动（最多 %.0fs）...", args.timeout)

    with GoiEnv(port=args.port, num_agents=args.agents, timeout=args.timeout) as env:
        logger.info("连接成功，Agent 数量: %d", args.agents)

        n_agents = _detect_num_agents(env)
        if n_agents < 2:
            logger.error("C# 端只有 %d 个 agent，无法测试输入隔离", n_agents)
            sys.exit(1)

        warmup_and_snapshot(env, n_agents, args.warmup)

        result = run_parallel_input_test(
            env,
            n_agents=n_agents,
            n_steps=args.steps,
            amplitude=args.amplitude,
            sep_threshold=args.threshold,
        )

    if not args.no_plot and result.get("traj_0") is not None:
        try:
            from tests.control_interaction.visualize_trajectory import (
                plot_trajectories,
                plot_distance_over_time,
            )

            plot_trajectories(
                [result["traj_0"], result["traj_1"]],
                ["agent_0 (+X)", "agent_1 (-X)"],
                dims=[0, 1],
                title="L5.1: Per-Agent Input Isolation (should diverge)",
            )
            plot_distance_over_time(
                result["traj_0"],
                result["traj_1"],
                title="L5.1: Position Distance Over Time",
                threshold=args.threshold,
            )
        except ImportError:
            logger.warning("visualize_trajectory 不可用，跳过绘图")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

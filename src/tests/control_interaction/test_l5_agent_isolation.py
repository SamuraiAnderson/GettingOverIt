"""
层次 5：多 Agent 独立性验证（方式 A — 单实例多复制体）

验证同一游戏进程内的多个 Agent 的控制完全独立，动作不跨 Agent 干扰。

前置条件：游戏已配置 numDuplicates >= 2（即至少 2 个 Agent）。
  可在 GoiData/runtime_config.json 中设置 numDuplicates = 2。

三种情形：
  情形 X（同动作一致性）：actions[0] == actions[1]，期望轨迹完全一致
  情形 Y（不同动作可区分性）：actions[0] 向右, actions[1] 向左，期望轨迹持续分离
  情形 Z（物理干扰测试）：Agent 0 施加动作，Agent 1 零动作，验证 Agent 1 速度不受影响

用法：
  python src/tests/control_interaction/test_l5_agent_isolation.py
  python src/tests/control_interaction/test_l5_agent_isolation.py --no-launch --steps 100
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
from tests.control_interaction.visualize_trajectory import (
    plot_trajectories,
    plot_distance_over_time,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "test_config.json"


# ── Warmup & 诊断 ──────────────────────────────────────────────────────

def warmup_and_snapshot(env: GoiEnv, n_agents: int, n_steps: int = 300) -> None:
    """
    让所有 Agent 以零动作热身 n_steps 步，等待动画阶段结束、物理稳定，
    然后调用 new_snapshot() 将当前状态设为复位基准。
    """
    logger.info("Warmup: 运行 %d 步（%d agents，零动作）...", n_steps, n_agents)
    zero_action = np.zeros((n_agents, 2), dtype=np.float32)
    for _ in range(n_steps):
        env.step(zero_action)
    env.new_snapshot()
    logger.info("Warmup 完成，已设置新快照")


def diagnose_initial_state(env: GoiEnv, n_agents: int, n_probe: int = 30) -> bool:
    """
    重置环境，逐步运行 n_probe 步（零动作），每隔 5 步打印关键状态，
    观察 vel_y 是否随步骤变化（判断 rigidbody 是否正在运动）。
    同时检测 NaN/Inf 首次出现步骤。
    """
    logger.info("--- 诊断：step-0~%d 原始状态（零动作探针，每5步采样）---", n_probe)
    zero_action = np.zeros((n_agents, 2), dtype=np.float32)
    env.reset()

    first_nan_step = {i: None for i in range(n_agents)}

    for step in range(n_probe):
        obs, _ = env.step(zero_action)
        for i in range(n_agents):
            if first_nan_step[i] is not None:
                continue
            s = obs[i]
            if bool(np.any(np.isnan(s) | np.isinf(s))):
                bad = [STATE_LABELS[d] for d in range(len(s)) if not np.isfinite(s[d])]
                logger.warning("  Agent %d 第 %d 步出现 NaN/Inf — 维度: %s", i, step, bad)
                first_nan_step[i] = step

        # 每隔 5 步打印一次，便于观察趋势
        if step % 5 == 4 or step == n_probe - 1:
            parts = []
            for i in range(n_agents):
                s = obs[i]
                parts.append(
                    f"A{i}[px={s[0]:.2f} py={s[1]:.2f} vx={s[2]:.3f} vy={s[3]:.3f} "
                    f"hub_x={s[5]:.2f} tip_x={s[23]:.2f} ang={s[27]:.1f}°]"
                )
            logger.info("  step%3d: %s", step + 1, "  |  ".join(parts))

    all_ok = all(v is None for v in first_nan_step.values())
    if all_ok:
        logger.info("  诊断通过：%d 步零动作内无 NaN/Inf", n_probe)
    else:
        logger.warning("  诊断失败：存在 NaN/Inf")

    # 结论推断
    last_obs = obs
    for i in range(n_agents):
        s = last_obs[i]
        if np.all(s == 0):
            logger.warning("  Agent %d: 全零 → StateService 未正确初始化（isReady=False）", i)
        elif abs(s[0]) < 1e-6 and abs(s[5]) < 1e-6:
            logger.warning("  Agent %d: player_x≈0 且 hub_x≈0 → root transform 在原点，"
                           "物理位置可能在 rigidbody.position 而非 transform.position", i)
        else:
            logger.info("  Agent %d: 状态正常，player_x=%.3f, hub_x=%.3f", i, s[0], s[5])
    return all_ok

STATE_LABELS = [
    "player_x", "player_y", "vel_x", "vel_y", "ang_vel",
    "hub_x", "hub_y", "hub_vx", "hub_vy", "hub_angle",
    "slider_x", "slider_y", "slider_vx", "slider_vy", "slider_angle",
    "handle_x", "handle_y", "handle_vx", "handle_vy",
    "pole_x", "pole_y", "pole_vx", "pole_vy",
    "tip_x", "tip_y", "tip_vx", "tip_vy",
    "hammer_angle", "timestamp",
]

# 判据阈值
SCENARIO_X_MAX_DIFF = 1e-3    # 同动作时 max|obs[0]-obs[1]| 应 < 此值
SCENARIO_Y_MIN_DIST = 1.0     # 不同动作时 step 100 处欧式距离应 > 此值
SCENARIO_Z_VEL_THRESH = 1e-3  # 零动作 Agent 的速度变化应 < 此值（排除重力影响用差分）


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


# ── 情形 X：同动作一致性 ─────────────────────────────────────────────

def scenario_x(env: GoiEnv, n_steps: int) -> dict:
    """两个 Agent 给相同动作，轨迹应完全一致（差异 < 1e-3）。"""
    logger.info("--- 情形 X：同动作一致性（%d 步）---", n_steps)

    action = np.array([[80.0, 0.0], [80.0, 0.0]], dtype=np.float32)

    obs0 = env.reset()
    traj_0, traj_1 = [obs0[0].copy()], [obs0[1].copy()]

    for _ in range(n_steps):
        obs, _ = env.step(action)
        traj_0.append(obs[0].copy())
        traj_1.append(obs[1].copy())

    traj_0 = np.stack(traj_0)
    traj_1 = np.stack(traj_1)

    max_diff = float(np.abs(traj_0 - traj_1).max())
    passed   = max_diff < SCENARIO_X_MAX_DIFF

    logger.info("  max|obs[0]-obs[1]|: %.2e  期望 < %.0e  [%s]",
                max_diff, SCENARIO_X_MAX_DIFF, "OK" if passed else "FAIL")
    if not passed:
        worst_dim = int(np.abs(traj_0 - traj_1).max(axis=0).argmax())
        logger.warning("  最差维度: [%d] %s", worst_dim,
                       STATE_LABELS[worst_dim] if worst_dim < len(STATE_LABELS) else f"dim_{worst_dim}")

    return {"passed": passed, "max_diff": max_diff, "traj_0": traj_0, "traj_1": traj_1}


# ── 情形 Y：不同动作可区分性 ─────────────────────────────────────────

def scenario_y(env: GoiEnv, n_steps: int) -> dict:
    """两个 Agent 给相反方向动作，轨迹应持续分离（step 100 时距离 > 1.0）。"""
    logger.info("--- 情形 Y：不同动作可区分性（%d 步）---", n_steps)

    action = np.array([[100.0, 0.0], [-100.0, 0.0]], dtype=np.float32)

    obs0 = env.reset()
    traj_0, traj_1 = [obs0[0].copy()], [obs0[1].copy()]

    for _ in range(n_steps):
        obs, _ = env.step(action)
        traj_0.append(obs[0].copy())
        traj_1.append(obs[1].copy())

    traj_0 = np.stack(traj_0)
    traj_1 = np.stack(traj_1)

    # 逐步欧式距离
    dists = np.linalg.norm(traj_0 - traj_1, axis=1)
    check_step = min(100, n_steps)
    dist_at_check = float(dists[check_step])
    passed = dist_at_check > SCENARIO_Y_MIN_DIST

    logger.info("  step %d 时欧式距离: %.4f  期望 > %.1f  [%s]",
                check_step, dist_at_check, SCENARIO_Y_MIN_DIST, "OK" if passed else "FAIL")
    logger.info("  最终距离（step %d）: %.4f", n_steps, float(dists[-1]))

    return {
        "passed": passed,
        "dist_at_step100": dist_at_check,
        "dist_final": float(dists[-1]),
        "dists": dists,
        "traj_0": traj_0,
        "traj_1": traj_1,
    }


# ── 情形 Z：物理干扰测试 ─────────────────────────────────────────────

def scenario_z(env: GoiEnv, n_steps: int) -> dict:
    """
    Agent 0 持续施加动作，Agent 1 保持零动作。
    验证 Agent 1 的速度变化不超过阈值（排除重力本身的加速度影响，
    通过与纯零动作基线对比）。
    """
    logger.info("--- 情形 Z：物理干扰测试（%d 步）---", n_steps)

    # 先采集纯零动作基线（单 Agent 或 Agent 1 零动作）
    action_test = np.array([[100.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    action_zero = np.array([[  0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    # 基线：两个 Agent 均零动作
    obs0_base = env.reset()
    vel_base_1 = []
    for _ in range(n_steps):
        obs, _ = env.step(action_zero)
        vel_base_1.append(obs[1, 2:4].copy())   # vel_x, vel_y of agent 1

    # 测试：Agent 0 向右，Agent 1 零动作
    obs0_test = env.reset()
    vel_test_1 = []
    for _ in range(n_steps):
        obs, _ = env.step(action_test)
        vel_test_1.append(obs[1, 2:4].copy())

    vel_base_1 = np.stack(vel_base_1)   # (T, 2)
    vel_test_1 = np.stack(vel_test_1)   # (T, 2)

    # 差异：测试 vs 基线（剔除重力本身的速度变化）
    vel_diff = np.abs(vel_test_1 - vel_base_1)
    max_vel_diff = float(vel_diff.max())
    passed = max_vel_diff < SCENARIO_Z_VEL_THRESH

    logger.info("  Agent 1 速度差（vs 纯零基线）最大值: %.2e  期望 < %.0e  [%s]",
                max_vel_diff, SCENARIO_Z_VEL_THRESH, "OK" if passed else "FAIL")
    if not passed:
        logger.warning("  [诊断] 检查 PlayerDuplicateManager 中 IgnoreCollision 配置")

    return {
        "passed": passed,
        "max_vel_diff": max_vel_diff,
        "vel_base_1": vel_base_1,
        "vel_test_1": vel_test_1,
    }


# ── 主函数 ────────────────────────────────────────────────────────────

def main():
    cfg = _load_config()
    default_port    = cfg.get("port", 9000)
    default_timeout = cfg.get("connect_timeout", 60)
    l5_cfg          = cfg.get("l5", {})
    default_agents  = l5_cfg.get("num_agents", 2)
    default_steps   = l5_cfg.get("n_steps", 200)

    parser = argparse.ArgumentParser(description="L5 多 Agent 独立性验证")
    parser.add_argument("--port",      type=int,   default=default_port)
    parser.add_argument("--timeout",   type=float, default=default_timeout)
    parser.add_argument("--agents",    type=int,   default=default_agents)
    parser.add_argument("--steps",     type=int,   default=default_steps)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--no-plot",   action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.agents < 2:
        logger.error("需要至少 2 个 Agent（--agents >= 2），请先在 runtime_config.json 中设置 numDuplicates >= 2")
        sys.exit(1)

    if not args.no_launch:
        launch_game()
        logger.info("等待游戏启动（最多 %.0fs）...", args.timeout)

    with GoiEnv(port=args.port, num_agents=args.agents, timeout=args.timeout) as env:
        logger.info("连接成功，Agent 数量: %d", args.agents)
        logger.info("=" * 60)

        # ── 热身 & 诊断 ──────────────────────────────────────────
        warmup_steps = cfg.get("l5", {}).get("warmup_steps", 300)
        warmup_and_snapshot(env, args.agents, warmup_steps)
        state_ok = diagnose_initial_state(env, args.agents)
        if not state_ok:
            logger.error("step-0 存在 NaN/Inf，后续测试结果无意义，请检查 C# 复制体初始化逻辑")
        logger.info("=" * 60)

        rx = scenario_x(env, args.steps)
        logger.info("")
        ry = scenario_y(env, args.steps)
        logger.info("")
        rz = scenario_z(env, min(args.steps, 200))
        logger.info("")

        # 综合结论
        all_pass = rx["passed"] and ry["passed"] and rz["passed"]
        logger.info("=" * 60)
        logger.info("=== 综合结论 ===")
        logger.info("  情形 X（同动作一致性）:  %s", "通过" if rx["passed"] else "失败")
        logger.info("  情形 Y（不同动作区分性）: %s", "通过" if ry["passed"] else "失败")
        logger.info("  情形 Z（物理干扰测试）:   %s", "通过" if rz["passed"] else "失败")
        logger.info("  总体: %s", "多 Agent 控制完全独立" if all_pass else "存在 Agent 间干扰，需排查")

    if not args.no_plot:
        # 情形 X：轨迹叠加图 + 状态差
        plot_trajectories(
            [rx["traj_0"], rx["traj_1"]], ["agent_0", "agent_1"],
            dims=[0, 1], title="L5 Scenario X: Same Action (should overlap)",
        )
        plot_distance_over_time(
            rx["traj_0"], rx["traj_1"],
            title="L5 Scenario X: State Distance (should be ~0)",
            threshold=SCENARIO_X_MAX_DIFF,
        )

        # 情形 Y：分离图 + 欧式距离
        plot_trajectories(
            [ry["traj_0"], ry["traj_1"]], ["agent_0 (right)", "agent_1 (left)"],
            dims=[0, 1], title="L5 Scenario Y: Different Actions (should diverge)",
        )
        plot_distance_over_time(
            ry["traj_0"], ry["traj_1"],
            title="L5 Scenario Y: State Distance Over Time (should grow)",
            threshold=SCENARIO_Y_MIN_DIST,
        )

        # 情形 Z：Agent 1 速度对比（基线 vs 干扰测试）
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle("L5 Scenario Z: Agent 1 Velocity (zero action, with/without Agent 0 moving)")
        for i, (label, color) in enumerate([("vel_x", "steelblue"), ("vel_y", "tomato")]):
            ax = axes[i]
            ax.plot(rz["vel_base_1"][:, i], label=f"baseline {label}", color=color, linestyle="--")
            ax.plot(rz["vel_test_1"][:, i], label=f"test {label}", color=color)
            ax.set_ylabel(label)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("step")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

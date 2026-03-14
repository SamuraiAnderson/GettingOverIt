"""
层次 1+2：协议层 + 因果层验证

层次 1 — TCP 协议验证：
  - 连接是否成功
  - reset() / step() 返回正确 shape
  - 物理引擎在运行（零动作下状态仍在变化）

层次 2 — 因果层验证：
  - 6 个方向动作对 player_x / player_y / hammer_angle 的影响方向正确
  - 不同方向动作产生可区分的状态变化

用法：
  python src/tests/control_interaction/test_l1_l2_protocol_causality.py
  python src/tests/control_interaction/test_l1_l2_protocol_causality.py --no-launch --steps 50
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
from tests.control_interaction.visualize_trajectory import plot_trajectories

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

DIRECTIONS = {
    "right":      np.array([[ 100.0,    0.0]], dtype=np.float32),
    "left":       np.array([[-100.0,    0.0]], dtype=np.float32),
    "up":         np.array([[   0.0,  100.0]], dtype=np.float32),
    "down":       np.array([[   0.0, -100.0]], dtype=np.float32),
    "right_up":   np.array([[  70.0,   70.0]], dtype=np.float32),
    "right_down": np.array([[  70.0,  -70.0]], dtype=np.float32),
    "zero":       np.array([[   0.0,    0.0]], dtype=np.float32),
}


def _load_config() -> dict:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def launch_game() -> object:
    logger.info("设置游戏为 GameRuntime 模式...")
    GameModeController().set_game_runtime_mode()
    launcher = GameLauncher()
    proc = launcher.launch(wait=False)
    logger.info("游戏已启动 (PID=%d)", proc.pid)
    return proc


# ── 层次 1：协议验证 ─────────────────────────────────────────────────

def verify_protocol(env: GoiEnv) -> dict:
    """验证 TCP 协议：shape、物理运行、零动作惯性"""
    logger.info("=== 层次 1：协议验证 ===")
    results = {}

    # shape 检查
    obs0 = env.reset()
    shape_ok = obs0.shape == (1, STATE_DIM)
    results["shape_ok"] = shape_ok
    logger.info("  reset() shape: %s  期望 (1, %d)  [%s]",
                obs0.shape, STATE_DIM, "OK" if shape_ok else "FAIL")

    # 零动作下状态变化（物理在运行）
    zero = DIRECTIONS["zero"]
    deltas = []
    obs_prev = obs0.copy()
    for _ in range(20):
        obs, _ = env.step(zero)
        deltas.append(np.linalg.norm(obs[0] - obs_prev[0]))
        obs_prev = obs.copy()

    mean_delta = float(np.mean(deltas))
    physics_ok = mean_delta > 1e-3
    results["physics_running"] = physics_ok
    results["zero_action_mean_delta"] = mean_delta
    logger.info("  零动作均值状态变化: %.6f  期望 > 1e-3  [%s]",
                mean_delta, "OK" if physics_ok else "FAIL")

    return results


# ── 层次 2：因果验证 ─────────────────────────────────────────────────

def run_direction(env: GoiEnv, action: np.ndarray, n_steps: int) -> tuple:
    """
    执行 n_steps 步，返回 (初始状态, 最终状态, 完整轨迹)。
    """
    obs0 = env.reset()
    traj = [obs0[0].copy()]
    for _ in range(n_steps):
        obs, _ = env.step(action)
        traj.append(obs[0].copy())
    return obs0[0], obs[0], np.stack(traj)


def verify_causality(env: GoiEnv, n_steps: int) -> dict:
    """验证各方向动作的因果影响方向"""
    logger.info("=== 层次 2：因果验证（各 %d 步）===", n_steps)

    records = {}
    trajs = []
    traj_labels = []

    for name, action in DIRECTIONS.items():
        obs0, obs_final, traj = run_direction(env, action, n_steps)
        delta_x     = float(obs_final[0] - obs0[0])
        delta_y     = float(obs_final[1] - obs0[1])
        delta_hammer = float(obs_final[27] - obs0[27])
        records[name] = {"delta_x": delta_x, "delta_y": delta_y, "delta_hammer": delta_hammer}
        trajs.append(traj)
        traj_labels.append(name)
        logger.info("  %-12s: Δx=%+7.3f  Δy=%+7.3f  Δhammer=%+7.3f°",
                    name, delta_x, delta_y, delta_hammer)

    # 方向性判断
    checks = {
        "right_moves_right":  records["right"]["delta_x"]    > 0,
        "left_moves_left":    records["left"]["delta_x"]     < 0,
        "up_positive_delta":  records["up"]["delta_y"]       > records["zero"]["delta_y"],
        "down_vs_up_y":       records["down"]["delta_y"]     < records["up"]["delta_y"],
        "hammer_right_vs_left": (
            records["right"]["delta_hammer"] != records["left"]["delta_hammer"]
        ),
    }
    logger.info("--- 方向性判断 ---")
    all_pass = True
    for check_name, passed in checks.items():
        status = "OK" if passed else "FAIL"
        logger.info("  %-30s [%s]", check_name, status)
        if not passed:
            all_pass = False

    results = {"direction_checks": checks, "all_pass": all_pass, "per_direction": records}
    return results, trajs, traj_labels


# ── 主函数 ────────────────────────────────────────────────────────────

def main():
    cfg = _load_config()
    default_port = cfg.get("port", 9000)
    default_timeout = cfg.get("connect_timeout", 60)
    default_steps = cfg.get("l1l2", {}).get("n_steps", 100)

    parser = argparse.ArgumentParser(description="L1+L2 协议与因果验证")
    parser.add_argument("--port",      type=int,   default=default_port)
    parser.add_argument("--timeout",   type=float, default=default_timeout)
    parser.add_argument("--steps",     type=int,   default=default_steps)
    parser.add_argument("--no-launch", action="store_true", help="跳过游戏启动")
    parser.add_argument("--no-plot",   action="store_true", help="跳过可视化")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.no_launch:
        launch_game()
        logger.info("等待游戏启动（最多 %.0fs）...", args.timeout)

    with GoiEnv(port=args.port, num_agents=1, timeout=args.timeout) as env:
        logger.info("连接成功，开始验证")
        logger.info("-" * 60)

        # 层次 1
        l1_result = verify_protocol(env)
        logger.info("-" * 60)

        # 层次 2
        l2_result, trajs, traj_labels = verify_causality(env, args.steps)
        logger.info("-" * 60)

        # 综合结论
        l1_ok = l1_result["shape_ok"] and l1_result["physics_running"]
        l2_ok = l2_result["all_pass"]
        logger.info("=== 综合结论 ===")
        logger.info("  层次 1（协议）: %s", "通过" if l1_ok else "失败")
        logger.info("  层次 2（因果）: %s", "通过" if l2_ok else "失败")
        if not l1_ok:
            logger.warning("  [诊断] 检查 TCP 连接与 C# 日志，确认 PlayerInputService 反射初始化正常")
        if not l2_ok:
            logger.warning("  [诊断] 动作方向性异常，检查 PlayerInputService.SetMouseInput 方向映射")

        # 可视化
        if not args.no_plot:
            plot_trajectories(
                trajs, traj_labels,
                dims=[0, 1, 27],
                title="L1+L2: Player Trajectory by Direction",
            )


if __name__ == "__main__":
    main()

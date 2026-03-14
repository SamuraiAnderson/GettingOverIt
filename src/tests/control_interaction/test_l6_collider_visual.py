"""
层次 6：碰撞箱可视化验证

在游戏窗口中实时高亮描边 Player 的所有 Collider2D，
同时执行 15 秒正弦波运动，供人工目视确认碰撞箱跟随 Player 移动。

用法：
  python src/tests/control_interaction/test_l6_collider_visual.py
  python src/tests/control_interaction/test_l6_collider_visual.py --no-launch --duration 20
"""

import argparse
import json
import logging
import math
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

PHYSICS_HZ = 50.0


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _detect_num_agents(env: GoiEnv) -> int:
    """发送一次 reset 来探测 C# 侧实际的 agent 数量"""
    obs = env.reset()
    n = obs.shape[0]
    logger.info("探测到 C# 侧 agent 数量: %d（Python 侧 num_agents=%d）", n, env.num_agents)
    if n != env.num_agents:
        logger.warning(
            "MISMATCH: Python num_agents=%d vs C# agents=%d — 将以 C# 为准",
            env.num_agents, n,
        )
        env.num_agents = n
    return n


def launch_game():
    logger.info("设置 GameRuntime 模式并启动游戏...")
    GameModeController().set_game_runtime_mode()
    GameLauncher().launch(wait=False)


def warmup_and_snapshot(env: GoiEnv, n_agents: int, warmup_steps: int) -> None:
    logger.info("Warmup: %d 步零动作（%d agents）...", warmup_steps, n_agents)
    zero = np.zeros((n_agents, 2), dtype=np.float32)
    env.reset()
    for _ in range(warmup_steps):
        env.step(zero)
    env.new_snapshot()
    obs = env.reset()
    logger.info(
        "Warmup 完成 — player=(%.2f, %.2f) vel=(%.4f, %.4f)",
        obs[0, 0], obs[0, 1], obs[0, 2], obs[0, 3],
    )


def run_visual_test(
    env: GoiEnv, n_agents: int, duration_sec: float, amplitude: float,
) -> None:
    total_steps = int(duration_sec * PHYSICS_HZ)
    logger.info(
        "开启碰撞箱描边，执行 %.1f 秒运动（%d 步，amplitude=%.0f, %d agents）",
        duration_sec, total_steps, amplitude, n_agents,
    )

    env.toggle_collider_visual(True)

    init_pos = None
    max_displacement = 0.0

    for step in range(total_steps):
        t = step / PHYSICS_HZ
        ax = amplitude * math.sin(2.0 * math.pi * 0.2 * t)
        ay = amplitude * math.sin(2.0 * math.pi * 0.15 * t)

        actions = np.zeros((n_agents, 2), dtype=np.float32)
        actions[0] = [ax, ay]
        obs, dones = env.step(actions)

        px, py = obs[0, 0], obs[0, 1]
        vx, vy = obs[0, 2], obs[0, 3]

        if init_pos is None:
            init_pos = (px, py)

        displacement = math.sqrt((px - init_pos[0])**2 + (py - init_pos[1])**2)
        max_displacement = max(max_displacement, displacement)

        if step % 50 == 0:
            logger.info(
                "  step %4d/%d  t=%5.1fs  action=(%7.1f,%7.1f)  "
                "pos=(%8.2f,%8.2f)  vel=(%7.2f,%7.2f)  disp=%.3f",
                step, total_steps, t, ax, ay, px, py, vx, vy, displacement,
            )

    env.toggle_collider_visual(False)

    if max_displacement < 0.01:
        logger.error(
            "FAIL: Player 未移动（max_displacement=%.4f）— "
            "检查 C# 侧 input_enabled / numDuplicates 配置",
            max_displacement,
        )
    else:
        logger.info("Player 最大位移: %.3f", max_displacement)

    logger.info("碰撞箱描边已关闭")


def main():
    parser = argparse.ArgumentParser(description="L6: 碰撞箱可视化验证")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--duration", type=float, default=None, help="运动时长（秒）")
    parser.add_argument("--amplitude", type=float, default=None, help="动作幅度")
    parser.add_argument("--warmup", type=int, default=None, help="warmup 步数")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_config()
    l6 = cfg.get("l6", {})
    port = args.port or cfg.get("port", 9000)
    timeout = args.timeout or cfg.get("connect_timeout", 60)
    duration = args.duration or l6.get("duration_sec", 15)
    amplitude = args.amplitude or l6.get("amplitude", 200)
    warmup_steps = args.warmup or l6.get("warmup_steps", 500)

    if not args.no_launch:
        launch_game()

    with GoiEnv(port=port, num_agents=1, timeout=timeout) as env:
        n_agents = _detect_num_agents(env)

        warmup_and_snapshot(env, n_agents, warmup_steps)
        run_visual_test(env, n_agents, duration, amplitude)

        stats = env.get_stats()
        logger.info("步进统计: %s", stats)

    logger.info("=" * 60)
    logger.info("L6 完成 — 请在游戏窗口确认碰撞箱描边是否正确跟随 Player")
    logger.info("  - 绿色 = Pot（锅）")
    logger.info("  - 红色 = Tip（锤头）")
    logger.info("  - 黄色 = Body（身体）")
    logger.info("  - 青色 = 其他部件")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

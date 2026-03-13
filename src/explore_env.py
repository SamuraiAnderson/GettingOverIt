"""
环境能力探测脚本

目标：在真正开始 RL 训练之前，验证并量化：
  1. 帧级别同步是否真正工作（每个 STEP 对应精确的物理帧数）
  2. 单步延迟分布（决定训练速度的上限）
  3. 状态一致性（相同输入是否产生相同状态变化）
  4. 多 agent 行为（N 个复制体是否独立响应不同输入）
  5. Reset 是否真正恢复到初始状态

运行前提：
  - 游戏已以 GameRuntime 模式启动（通过 game_mode_controller.py 设置信号文件后启动）
  - C# 插件正在运行并监听 TCP 端口（默认 9000，可在 project.json 中配置）

用法：
  python src/explore_env.py
  python src/explore_env.py --agents 3 --steps 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

if Path(__file__).parent.as_posix() not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from env.goi_env import GoiEnv, STATE_DIM
from start.game_launcher import GameLauncher
from start.game_mode_controller import GameModeController

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "config" / "project.json"

STATE_LABELS = [
    "player_x", "player_y", "vel_x", "vel_y", "ang_vel",
    "hub_x", "hub_y", "hub_vx", "hub_vy", "hub_angle",
    "slider_x", "slider_y", "slider_vx", "slider_vy", "slider_angle",
    "handle_x", "handle_y", "handle_vx", "handle_vy",
    "pole_x", "pole_y", "pole_vx", "pole_vy",
    "tip_x", "tip_y", "tip_vx", "tip_vy",
    "hammer_angle", "timestamp",
]


def _load_tcp_port() -> int:
    try:
        with open(_CONFIG_PATH) as f:
            return json.load(f).get("tcp_port", 9000)
    except Exception:
        return 9000


# ────────────────────────────────────────────────────────────────────
# 探测实验
# ────────────────────────────────────────────────────────────────────

def probe_step_latency(env: GoiEnv, n_steps: int = 200):
    """实验一：步进延迟基准测试"""
    logger.info("=== 实验一：步进延迟（%d 步，零动作）===", n_steps)
    env.reset()

    zero_actions = np.zeros((env.num_agents, 2), dtype=np.float32)
    for _ in range(n_steps):
        env.step(zero_actions)

    stats = env.get_stats()
    logger.info("  平均延迟: %.2f ms", stats["mean_ms"])
    logger.info("  标准差:   %.2f ms", stats["std_ms"])
    logger.info("  最小/最大: %.2f / %.2f ms", stats["min_ms"], stats["max_ms"])
    logger.info("  等效最大 Hz: %.1f", 1000 / stats["mean_ms"])
    return stats


def probe_determinism(env: GoiEnv, n_trials: int = 5, n_steps: int = 30):
    """实验二：确定性验证"""
    logger.info("=== 实验二：确定性（%d 次试验 × %d 步）===", n_trials, n_steps)

    test_action = np.array([[30.0, 0.0]] * env.num_agents, dtype=np.float32)
    trajectories = []

    for _ in range(n_trials):
        env.reset()
        states_seq = []
        for _ in range(n_steps):
            obs, _ = env.step(test_action)
            states_seq.append(obs[0].copy())
        trajectories.append(np.stack(states_seq))

    final_states = np.stack([t[-1] for t in trajectories])
    max_diff = np.abs(final_states - final_states[0]).max(axis=0)
    worst_dim = int(np.argmax(max_diff))

    logger.info("  各维度最大差异（跨 %d 次试验）:", n_trials)
    for i, (label, diff) in enumerate(zip(STATE_LABELS, max_diff)):
        if diff > 1e-4:
            logger.info("    [%2d] %-20s: %.6f  <-", i, label, diff)
    logger.info("  最大差异维度: [%d] %s = %.6f", worst_dim, STATE_LABELS[worst_dim], max_diff[worst_dim])
    logger.info("  全局最大差异: %.6f", max_diff.max())

    is_deterministic = max_diff.max() < 1e-4
    logger.info("  结论: %s", "确定性 [OK]" if is_deterministic else "非确定性 [注意]")
    return {"max_diff": float(max_diff.max()), "is_deterministic": is_deterministic}


def probe_input_response(env: GoiEnv, n_steps: int = 50):
    """实验三：输入响应性"""
    logger.info("=== 实验三：输入响应（%d 步/方向）===", n_steps)

    directions = {
        "右 (+x)": np.array([[50.0,   0.0]] * env.num_agents, dtype=np.float32),
        "左 (-x)": np.array([[-50.0,  0.0]] * env.num_agents, dtype=np.float32),
        "上 (+y)": np.array([[0.0,   50.0]] * env.num_agents, dtype=np.float32),
        "下 (-y)": np.array([[0.0,  -50.0]] * env.num_agents, dtype=np.float32),
        "零输入":   np.zeros((env.num_agents, 2), dtype=np.float32),
    }

    results = {}
    for label, action in directions.items():
        obs0 = env.reset()
        for _ in range(n_steps):
            obs, _ = env.step(action)
        delta = obs[0] - obs0[0]
        results[label] = {
            "Δplayer_x":     float(delta[0]),
            "Δplayer_y":     float(delta[1]),
            "Δhammer_angle": float(delta[27]),
        }
        logger.info("  %-10s: Δx=%+.3f  Δy=%+.3f  Δhammer=%+.3f°", label, delta[0], delta[1], delta[27])

    return results


def probe_multi_agent(env: GoiEnv, n_steps: int = 60):
    """实验四：多 agent 独立性"""
    if env.num_agents < 2:
        logger.info("=== 实验四：跳过（num_agents=1）===")
        return None

    logger.info("=== 实验四：多 agent 独立性（%d agent，%d 步）===", env.num_agents, n_steps)

    actions = np.zeros((env.num_agents, 2), dtype=np.float32)
    for i in range(env.num_agents):
        angle = (i / env.num_agents) * 360
        actions[i, 0] = np.cos(np.radians(angle)) * 50.0
        actions[i, 1] = np.sin(np.radians(angle)) * 50.0
    logger.info("  各 agent 动作: %s", actions.tolist())

    obs0 = env.reset()
    for _ in range(n_steps):
        obs, _ = env.step(actions)

    deltas = obs - obs0
    for i in range(env.num_agents):
        logger.info("    agent %d: Δx=%+.3f  Δy=%+.3f  Δhammer=%+.3f°", i, deltas[i, 0], deltas[i, 1], deltas[i, 27])

    pairwise_diffs = []
    for i in range(env.num_agents):
        for j in range(i + 1, env.num_agents):
            d = float(np.abs(obs[i] - obs[j]).max())
            pairwise_diffs.append(d)
            logger.info("    agent %d vs agent %d: 最大状态差异 = %.4f", i, j, d)

    all_different = all(d > 0.01 for d in pairwise_diffs)
    logger.info("  结论: %s", "各 agent 状态独立 [OK]" if all_different else "agent 状态高度相似 [注意]")
    return {"pairwise_diffs": pairwise_diffs, "all_different": all_different}


def probe_reset_fidelity(env: GoiEnv, n_resets: int = 5):
    """实验五：Reset 保真性"""
    logger.info("=== 实验五：Reset 保真性（%d 次）===", n_resets)

    initial_states = []
    for i in range(n_resets):
        if i > 0:
            action = np.array([[40.0, 20.0]] * env.num_agents, dtype=np.float32)
            for _ in range(20):
                env.step(action)
        obs = env.reset()
        initial_states.append(obs[0].copy())

    initial_states = np.stack(initial_states)
    max_diff = np.abs(initial_states - initial_states[0]).max(axis=0)
    logger.info("  reset 后状态最大差异: %.6f", max_diff.max())

    consistent = max_diff.max() < 1e-3
    logger.info("  结论: %s", "Reset 一致 [OK]" if consistent else "Reset 不一致 [注意]，物理初态有偏差")
    return {"max_diff": float(max_diff.max()), "consistent": consistent}


# ────────────────────────────────────────────────────────────────────
# 启动 + 主流程
# ────────────────────────────────────────────────────────────────────

def launch_game_runtime():
    logger.info("设置游戏运行模式...")
    ctrl = GameModeController()
    ctrl.set_game_runtime_mode()
    launcher = GameLauncher()
    process = launcher.launch(wait=False)
    logger.info("游戏已启动 (PID=%d)", process.pid)
    return process


def main():
    default_port = _load_tcp_port()

    parser = argparse.ArgumentParser(description="Getting Over It 环境能力探测")
    parser.add_argument("--agents",       type=int,   default=1,            help="agent 数量")
    parser.add_argument("--port",         type=int,   default=default_port, help="TCP 端口")
    parser.add_argument("--steps",        type=int,   default=200,          help="延迟测试步数")
    parser.add_argument("--no-launch",    action="store_true",               help="不自动启动游戏")
    parser.add_argument("--connect-wait", type=float, default=60,           help="等待游戏启动的最长秒数")
    args = parser.parse_args()

    process = None
    if not args.no_launch:
        process = launch_game_runtime()
        logger.info("等待游戏启动并加载场景（最多 %.0fs）...", args.connect_wait)
    else:
        logger.info("跳过游戏启动（--no-launch）")

    report = {}
    with GoiEnv(host="127.0.0.1", port=args.port, num_agents=args.agents, timeout=args.connect_wait) as env:
        logger.info("连接成功！开始探测（%d agent）", args.agents)
        logger.info("-" * 60)

        obs0 = env.reset()
        logger.info("初始观测 shape: %s  (期望: [%d, %d])", obs0.shape, args.agents, STATE_DIM)
        logger.info("初始 agent 0 position: (%.3f, %.3f)", obs0[0, 0], obs0[0, 1])
        logger.info("-" * 60)

        report["latency"]     = probe_step_latency(env, n_steps=args.steps)
        logger.info("-" * 60)
        report["determinism"] = probe_determinism(env)
        logger.info("-" * 60)
        report["input_resp"]  = probe_input_response(env)
        logger.info("-" * 60)
        report["multi_agent"] = probe_multi_agent(env)
        logger.info("-" * 60)
        report["reset_fid"]   = probe_reset_fidelity(env)
        logger.info("-" * 60)

    logger.info("=== 探测完成 — 总结 ===")
    lat = report["latency"]
    logger.info("  步进延迟: 均值 %.2fms  最大 %.2fms  等效 %.0f Hz", lat["mean_ms"], lat["max_ms"], 1000 / lat["mean_ms"])
    det = report["determinism"]
    logger.info("  确定性:   %s (最大差异=%.2e)", "OK" if det["is_deterministic"] else "失败", det["max_diff"])
    rst = report["reset_fid"]
    logger.info("  Reset:    %s (最大差异=%.2e)", "OK" if rst["consistent"] else "失败", rst["max_diff"])
    if report["multi_agent"]:
        ma = report["multi_agent"]
        logger.info("  多 agent: %s", "独立 OK" if ma["all_different"] else "待观察")

    if process is not None:
        logger.info("探测完毕，游戏仍在运行。Ctrl+C 退出后游戏进程将继续。")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()

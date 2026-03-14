"""
层次 8：空投部署测试 — 根据投放点缓存批量复制并传送 Player

读取 L7 产出的 drop_points.json 缓存，自动配置 runtime_config.json 的
numDuplicates，启动游戏，将每个 agent 传送到对应投放点，
并启用自由相机（滚轮缩放 + 中键拖动）方便观察。

用法：
  python src/tests/control_interaction/test_l8_airdrop_deploy.py
  python src/tests/control_interaction/test_l8_airdrop_deploy.py --no-camera-free
  python src/tests/control_interaction/test_l8_airdrop_deploy.py --settle-steps 100
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "test_config.json"
_PROJECT_CONFIG = _REPO_ROOT / "src" / "config" / "project.json"


def _get_game_root(override: str | None = None) -> Path:
    if override:
        return Path(override)
    with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return Path(cfg["game"]["executable_path"]).parent


def load_drop_points(game_root: Path) -> list[list[float]]:
    """从 L7 缓存的 drop_points.json 中加载投放点坐标。"""
    dp_path = game_root / "GoiData" / "Colliders" / "drop_points.json"
    if not dp_path.exists():
        raise FileNotFoundError(
            f"投放点缓存不存在: {dp_path}\n"
            "请先运行 test_l7_surface_airdrop.py 生成 drop_points.json"
        )
    with open(dp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = data["drop_points"]
    logger.info("从缓存加载了 %d 个投放点", len(points))
    for i, (x, y) in enumerate(points):
        logger.info("  #%02d  (%.2f, %.2f)", i, x, y)
    return points


def write_num_duplicates(game_root: Path, n: int) -> None:
    """
    在 runtime_config.json 中写入 numDuplicates。
    若文件已存在则合并更新，否则创建新文件。
    """
    config_path = game_root / "GoiData" / "runtime_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {}

    old_val = cfg.get("numDuplicates", "?")
    cfg["numDuplicates"] = n
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    logger.info("runtime_config.json: numDuplicates %s → %d", old_val, n)


def warmup_and_snapshot(env, num_agents: int, warmup_steps: int) -> None:
    """零动作 warmup 后拍快照，使物理稳定。"""
    logger.info("Warmup: %d 步零动作...", warmup_steps)
    zero_actions = np.zeros((num_agents, 2), dtype=np.float32)
    for _ in range(warmup_steps):
        env.step(zero_actions)
    env.new_snapshot()
    logger.info("Warmup 完成，快照已拍摄")


def deploy_agents(env, drop_points: list[list[float]], settle_steps: int,
                   num_agents: int) -> np.ndarray:
    """
    将复制体 agent 传送到对应投放点。
    agent 0 是原始 Player，保持原位不动；agent 1..N-1 对应 drop_points[0..N-2]。
    返回 settle 后的状态 shape (num_agents, STATE_DIM)。
    """
    for i, (x, y) in enumerate(drop_points):
        agent_idx = i + 1
        env.teleport(x, y, agent_index=agent_idx)
        logger.info("Agent %d 已传送到 (%.2f, %.2f)", agent_idx, x, y)

    if settle_steps > 0:
        logger.info("物理稳定: %d 步零动作...", settle_steps)
        zero_actions = np.zeros((num_agents, 2), dtype=np.float32)
        for _ in range(settle_steps):
            obs, _ = env.step(zero_actions)
    else:
        zero_actions = np.zeros((num_agents, 2), dtype=np.float32)
        obs, _ = env.step(zero_actions)

    logger.info("物理稳定完成")
    return obs


def report_landing(drop_points: list[list[float]], obs: np.ndarray) -> None:
    """对比目标投放点与 settle 后的实际位置，打印报告。"""
    logger.info("-" * 60)
    logger.info("投放结果 (目标 → 实际落点):")
    logger.info("%-8s  %-22s  %-22s  %s", "Agent", "目标 (x, y)", "实际 (x, y)", "偏移距离")
    logger.info("-" * 60)

    drifts = []
    for i, (tx, ty) in enumerate(drop_points):
        agent_idx = i + 1
        ax = float(obs[agent_idx, 0])
        ay = float(obs[agent_idx, 1])
        drift = ((ax - tx) ** 2 + (ay - ty) ** 2) ** 0.5
        drifts.append(drift)
        logger.info("Agent %-3d  (%7.2f, %7.2f)  →  (%7.2f, %7.2f)  Δ=%.3f",
                     agent_idx, tx, ty, ax, ay, drift)

    # agent 0 原始 Player 位置
    a0x, a0y = float(obs[0, 0]), float(obs[0, 1])
    logger.info("Agent 0    (原始 Player)          →  (%7.2f, %7.2f)", a0x, a0y)
    logger.info("-" * 60)
    logger.info("偏移统计: mean=%.3f  max=%.3f  min=%.3f",
                np.mean(drifts), np.max(drifts), np.min(drifts))


def run_sine_motion(env, num_agents: int, duration: float, amplitude: float,
                    freq: float = 0.5, dt: float = 0.02) -> None:
    """
    向所有 agent 发送正弦波动作，持续 duration 秒。
    每个 agent 有随机相位偏移，避免完全同步。
    """
    n_steps = int(duration / dt)
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, size=num_agents)

    logger.info("正弦波运动: %d 步 (%.1fs), amplitude=%.1f, freq=%.2fHz",
                n_steps, duration, amplitude, freq)

    for step in range(n_steps):
        t = step * dt
        actions = np.zeros((num_agents, 2), dtype=np.float32)
        for i in range(num_agents):
            actions[i, 0] = amplitude * np.sin(2 * np.pi * freq * t + phases[i])
        env.step(actions)

        if (step + 1) % 100 == 0:
            logger.info("  运动进度: %d/%d (%.1fs)", step + 1, n_steps, (step + 1) * dt)

    logger.info("正弦波运动完成")


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="L8: 空投部署测试")
    parser.add_argument("--game-root", type=str, default=None, help="游戏根目录")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None, help="热身步数")
    parser.add_argument("--settle-steps", type=int, default=None, help="传送后物理稳定步数")
    parser.add_argument("--no-camera-free", action="store_true", help="不启用自由相机")
    parser.add_argument("--no-launch", action="store_true",
                        help="不启动游戏（假设游戏已运行）")
    parser.add_argument("--motion-duration", type=float, default=None,
                        help="正弦波运动持续秒数")
    parser.add_argument("--motion-amplitude", type=float, default=None,
                        help="正弦波动作幅度")
    parser.add_argument("--no-motion", action="store_true",
                        help="跳过正弦波运动阶段")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_config()
    l8 = cfg.get("l8", {})
    port = args.port or cfg.get("port", 9000)
    timeout = args.timeout or cfg.get("connect_timeout", 60)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else l8.get("warmup_steps", 300)
    settle_steps = args.settle_steps if args.settle_steps is not None else l8.get("settle_steps", 200)
    motion_duration = args.motion_duration if args.motion_duration is not None else l8.get("motion_duration", 10.0)
    motion_amplitude = args.motion_amplitude if args.motion_amplitude is not None else l8.get("motion_amplitude", 50.0)

    game_root = _get_game_root(args.game_root)
    logger.info("游戏根目录: %s", game_root)

    # 1. 加载投放点缓存
    drop_points = load_drop_points(game_root)
    n_drops = len(drop_points)
    if n_drops == 0:
        logger.error("投放点数量为 0，无法部署")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("L8: 投放 %d 个 agent 到预计算位置", n_drops)
    logger.info("=" * 60)

    # 2. 写入 numDuplicates 并启动游戏
    from start.game_launcher import GameLauncher
    from start.game_mode_controller import GameModeController
    from env.goi_env import GoiEnv

    # agent 0 = 原始 Player（保持原位），agent 1..n_drops = 复制体
    num_agents = n_drops + 1
    write_num_duplicates(game_root, num_agents)

    if not args.no_launch:
        GameModeController(str(game_root)).set_game_runtime_mode()
        GameLauncher().launch(wait=False)
        logger.info("等待游戏启动（最多 %.0fs）...", timeout)

    # 3. 连接
    env = GoiEnv(port=port, num_agents=num_agents, timeout=timeout)
    env.connect()

    try:
        # 4. Reset + warmup
        env.reset()
        warmup_and_snapshot(env, num_agents, warmup_steps)

        # 5. 启用自由相机
        if not args.no_camera_free:
            env.set_camera_free(True)
            logger.info("自由相机已启用 — 滚轮缩放, 左键拖动")

        # 6. 传送复制体 agent（agent 0 原始 Player 保持原位）
        obs = deploy_agents(env, drop_points, settle_steps, num_agents)

        # 7. 落点检查
        report_landing(drop_points, obs)

        # 8. 开启碰撞体描边可视化 + 正弦波运动
        if not args.no_motion:
            env.toggle_collider_visual(True)
            logger.info("碰撞体描边可视化已开启")
            run_sine_motion(env, num_agents, motion_duration, motion_amplitude)

        # 9. 保持连接，等待用户观察
        logger.info("=" * 60)
        logger.info("部署完成 — %d 个复制体已就位（原始 Player 保持原位）", n_drops)
        logger.info("游戏窗口中: 滚轮缩放, 左键拖动平移")
        logger.info("按 Enter 关闭连接并退出...")
        logger.info("=" * 60)
        input()

    finally:
        env.close()
        logger.info("L8 测试结束")


if __name__ == "__main__":
    main()

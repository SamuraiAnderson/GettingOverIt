"""
层次 9：落水重置测试 — 验证 player / 复制体自然落水是否触发游戏重置

测试方法（游戏物理与 step 同步，不 step 游戏也暂停）：
  1. 将 agent teleport 到水面正上方（无地形支撑）
  2. 持续 step（零输入）让 agent 自然坠落入水，追踪 y 轨迹（坠落阶段）
  3. agent 触底后继续 step N 步（等待阶段），观察游戏是否触发重置

测试场景：
  1. 复制体自然落水 → 继续 step 等待 → 看是否重置
  2. 原始 Player 自然落水 → 继续 step 等待 → 看是否重置

判定标准：
  - 坠落阶段 y 触底后回升 → 游戏在坠落过程中就触发了重生
  - 等待阶段位置发生显著变化 → 游戏延迟触发了重置
  - 其他 agent 位置发生突变 → 重置是全局的
  - dones 标志变为 True → C# 端有终止检测

用法：
  python src/tests/control_interaction/test_l9_water_reset.py
  python src/tests/control_interaction/test_l9_water_reset.py --no-launch
  python src/tests/control_interaction/test_l9_water_reset.py --wait-steps 600
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

from env.goi_env import GoiEnv
from start.game_launcher import GameLauncher
from start.game_mode_controller import GameModeController

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "test_config.json"
_PROJECT_CONFIG = _REPO_ROOT / "src" / "config" / "project.json"


def _get_game_root(override=None):
    if override:
        return Path(override)
    with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return Path(cfg["game"]["executable_path"]).parent


def _load_config():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_num_duplicates(game_root, n):
    config_path = game_root / "GoiData" / "runtime_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    cfg["numDuplicates"] = n
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def log_agent_pos(label, obs, agents=None):
    if agents is None:
        agents = range(obs.shape[0])
    for i in agents:
        logger.info("  %s agent %d: (%.2f, %.2f)", label, i,
                     float(obs[i, 0]), float(obs[i, 1]))


def fall_then_wait(env, num_agents, target_agent, fall_steps,
                   wait_steps, other_initial):
    """
    阶段 A: 持续 step 让 agent 自然坠落入水，追踪 y 轨迹。
    阶段 B: 触底后继续 step（零输入）wait_steps 步，观察游戏是否延迟触发重置。

    返回 dict:
      fall_reset  — 坠落阶段是否检测到回弹
      wait_reset  — 等待阶段是否检测到位置显著变化
      global_hit  — 其他 agent 是否受影响
      y_history   — 全程 y 值序列（坠落 + 等待）
      fall_len    — 坠落阶段步数
    """
    zero_actions = np.zeros((num_agents, 2), dtype=np.float32)

    # ── 阶段 A：step 让 agent 坠落 ──
    logger.info("── 阶段 A：持续 step %d 步，让 agent 自然坠落 ──", fall_steps)
    y_history = []
    fall_reset = False
    global_hit = False
    y_min_so_far = float('inf')

    for step in range(fall_steps):
        obs, dones = env.step(zero_actions)
        y = float(obs[target_agent, 1])
        y_history.append(y)
        y_min_so_far = min(y_min_so_far, y)

        if dones.any():
            logger.warning("  step %d: dones=%s", step, dones.tolist())

        if len(y_history) >= 2 and y_min_so_far < -5.0:
            if y - y_min_so_far > 3.0:
                logger.warning(
                    "  step %d: 回弹! y_min=%.2f → 当前y=%.2f (回升%.2f)",
                    step, y_min_so_far, y, y - y_min_so_far,
                )
                fall_reset = True
                _check_global(obs, num_agents, target_agent,
                              other_initial, global_hit)
                break

        if step % 50 == 0:
            logger.info("  step %d: agent %d y=%.2f", step, target_agent, y)

    fall_len = len(y_history)
    last_x = float(obs[target_agent, 0])
    last_y = float(obs[target_agent, 1])
    logger.info("  坠落阶段结束 (%d 步): agent %d 位置 (%.2f, %.2f)  y_min=%.2f",
                fall_len, target_agent, last_x, last_y, y_min_so_far)
    log_agent_pos("坠落后全体", obs)

    # ── 阶段 B：继续 step 等待，观察是否延迟重置 ──
    logger.info("── 阶段 B：继续 step %d 步，等待游戏是否触发重置 ──", wait_steps)
    wait_start_pos = (last_x, last_y)
    wait_reset = False

    for step in range(wait_steps):
        obs, dones = env.step(zero_actions)
        y = float(obs[target_agent, 1])
        y_history.append(y)

        if dones.any():
            logger.warning("  wait step %d: dones=%s", step, dones.tolist())

        cur_x = float(obs[target_agent, 0])
        dx = abs(cur_x - wait_start_pos[0])
        dy = abs(y - wait_start_pos[1])
        if dx > 1.0 or dy > 1.0:
            logger.warning(
                "  wait step %d: 位置显著变化! (%.2f,%.2f)→(%.2f,%.2f) "
                "Δx=%.2f Δy=%.2f → 游戏触发了延迟重置",
                step, wait_start_pos[0], wait_start_pos[1],
                cur_x, y, dx, dy,
            )
            wait_reset = True
            if not global_hit:
                for i in range(num_agents):
                    if i == target_agent:
                        continue
                    dy_i = abs(float(obs[i, 1]) - other_initial[i])
                    if dy_i > 3.0:
                        logger.warning(
                            "  agent %d 位置偏移 Δy=%.2f ← 全局影响", i, dy_i)
                        global_hit = True
            break

        if step % 100 == 0:
            logger.info("  wait step %d: agent %d (%.2f, %.2f)",
                        step, target_agent, cur_x, y)

    if not wait_reset:
        logger.info("  等待阶段结束: 位置无显著变化 → 游戏未触发延迟重置")
    log_agent_pos("等待后全体", obs)

    return {
        "fall_reset": fall_reset,
        "wait_reset": wait_reset,
        "global": global_hit,
        "y_history": np.array(y_history),
        "fall_len": fall_len,
    }


def _check_global(obs, num_agents, target_agent, other_initial, global_hit):
    for i in range(num_agents):
        if i == target_agent:
            continue
        dy_i = abs(float(obs[i, 1]) - other_initial[i])
        if dy_i > 3.0:
            logger.warning(
                "  agent %d 位置偏移 Δy=%.2f ← 全局影响", i, dy_i)
            global_hit = True


# ── 测试 1: 复制体自然落水 ──

def test_duplicate_fall(env, num_agents, drop_x, drop_y,
                        fall_steps, wait_steps):
    logger.info("=" * 60)
    logger.info("测试 1: 复制体自然落水")
    logger.info("=" * 60)

    obs = env.reset()
    zero_actions = np.zeros((num_agents, 2), dtype=np.float32)
    for _ in range(30):
        obs, _ = env.step(zero_actions)

    log_agent_pos("初始", obs)
    other_y = {i: float(obs[i, 1]) for i in range(num_agents)}

    target = 1
    logger.info("将 agent %d teleport 到 (%.1f, %.1f) — 水面上方无支撑",
                target, drop_x, drop_y)
    obs = env.teleport(drop_x, drop_y, agent_index=target)
    log_agent_pos("teleport后", obs, [target])

    r = fall_then_wait(env, num_agents, target, fall_steps,
                       wait_steps, other_y)
    y_hist = r["y_history"]

    logger.info("y 轨迹: start=%.2f → min=%.2f → end=%.2f "
                "(坠落%d步 + 等待%d步)",
                y_hist[0], y_hist.min(), y_hist[-1],
                r["fall_len"], len(y_hist) - r["fall_len"])

    overall = r["fall_reset"] or r["wait_reset"]
    if overall:
        logger.info("结论: 复制体落水 → 触发重置%s (坠落阶段=%s, 等待阶段=%s)",
                     "（全局影响）" if r["global"] else "（仅自身）",
                     r["fall_reset"], r["wait_reset"])
    else:
        logger.info("结论: 复制体落水 → 未触发重置 (最终 y=%.2f)", y_hist[-1])

    return {"reset": overall, "fall_reset": r["fall_reset"],
            "wait_reset": r["wait_reset"], "global": r["global"],
            "y_min": float(y_hist.min())}


# ── 测试 2: 原始 Player 自然落水 ──

def test_player_fall(env, num_agents, drop_x, drop_y,
                     fall_steps, wait_steps):
    logger.info("=" * 60)
    logger.info("测试 2: 原始 Player 自然落水")
    logger.info("=" * 60)

    obs = env.reset()
    zero_actions = np.zeros((num_agents, 2), dtype=np.float32)
    for _ in range(30):
        obs, _ = env.step(zero_actions)

    log_agent_pos("初始", obs)
    other_y = {i: float(obs[i, 1]) for i in range(num_agents)}

    target = 0
    logger.info("将 agent %d (原始 Player) teleport 到 (%.1f, %.1f) — 水面上方无支撑",
                target, drop_x, drop_y)
    obs = env.teleport(drop_x, drop_y, agent_index=target)
    log_agent_pos("teleport后", obs, [target])

    r = fall_then_wait(env, num_agents, target, fall_steps,
                       wait_steps, other_y)
    y_hist = r["y_history"]

    logger.info("y 轨迹: start=%.2f → min=%.2f → end=%.2f "
                "(坠落%d步 + 等待%d步)",
                y_hist[0], y_hist.min(), y_hist[-1],
                r["fall_len"], len(y_hist) - r["fall_len"])

    overall = r["fall_reset"] or r["wait_reset"]
    if overall:
        logger.info("结论: 原始 Player 落水 → 触发重置%s (坠落阶段=%s, 等待阶段=%s)",
                     "（全局影响）" if r["global"] else "（仅自身）",
                     r["fall_reset"], r["wait_reset"])
    else:
        logger.info("结论: 原始 Player 落水 → 未触发重置 (最终 y=%.2f)", y_hist[-1])

    return {"reset": overall, "fall_reset": r["fall_reset"],
            "wait_reset": r["wait_reset"], "global": r["global"],
            "y_min": float(y_hist.min())}


# ── 主流程 ──

def main():
    parser = argparse.ArgumentParser(description="L9: 落水重置测试 — 自然坠落方式")
    parser.add_argument("--game-root", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--no-launch", action="store_true", help="不启动游戏")
    parser.add_argument("--drop-x", type=float, default=-60.0,
                        help="投放 x 坐标 (默认 -60, 地形外侧)")
    parser.add_argument("--drop-y", type=float, default=5.0,
                        help="投放 y 坐标 (默认 5, 水面上方)")
    parser.add_argument("--fall-steps", type=int, default=300,
                        help="坠落阶段最大 step 数 (默认 300)")
    parser.add_argument("--wait-steps", type=int, default=500,
                        help="触底后继续 step 等待数 (默认 500, 约 10 秒)")
    parser.add_argument("--warmup-steps", type=int, default=300)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_config()
    port = args.port or cfg.get("port", 9000)
    timeout = args.timeout or cfg.get("connect_timeout", 60)

    game_root = _get_game_root(args.game_root)
    logger.info("游戏根目录: %s", game_root)

    num_agents = 3
    write_num_duplicates(game_root, num_agents)

    if not args.no_launch:
        GameModeController(str(game_root)).set_game_runtime_mode()
        GameLauncher().launch(wait=False)
        logger.info("等待游戏启动...")

    env = GoiEnv(port=port, num_agents=num_agents, timeout=timeout)
    env.connect()

    try:
        env.reset()
        logger.info("Warmup: %d 步...", args.warmup_steps)
        zero_actions = np.zeros((num_agents, 2), dtype=np.float32)
        for _ in range(args.warmup_steps):
            env.step(zero_actions)
        env.new_snapshot()
        logger.info("快照已拍摄")

        results = {}

        results["duplicate_fall"] = test_duplicate_fall(
            env, num_agents, args.drop_x, args.drop_y,
            args.fall_steps, args.wait_steps,
        )

        results["player_fall"] = test_player_fall(
            env, num_agents, args.drop_x, args.drop_y,
            args.fall_steps, args.wait_steps,
        )

        # ── 总结 ──
        logger.info("=" * 60)
        logger.info("L9 测试总结")
        logger.info("=" * 60)
        logger.info("  投放位置: (%.1f, %.1f)  坠落:%d步  等待:%d步",
                    args.drop_x, args.drop_y,
                    args.fall_steps, args.wait_steps)
        for name, r in results.items():
            status = "触发重置" if r["reset"] else "未触发重置"
            scope = "（全局）" if r["global"] else ""
            logger.info("  %-20s: %s%s  (坠落=%s, 等待=%s, y_min=%.2f)",
                        name, status, scope,
                        r.get("fall_reset", "?"),
                        r.get("wait_reset", "?"),
                        r["y_min"])

        any_reset = any(r["reset"] for r in results.values())
        any_global = any(r["global"] for r in results.values())

        logger.info("-" * 60)
        if any_global:
            logger.warning(
                "结论: 落水会触发全局重置 — 训练中必须避免任何 agent 落水"
            )
        elif any_reset:
            logger.warning(
                "结论: 落水会触发单体重置 — 训练中需检测并丢弃落水轨迹"
            )
        else:
            logger.info(
                "结论: 落水不触发重置 — RL 环境安全，但落水轨迹仍应被评分淘汰"
            )

    finally:
        env.close()
        logger.info("L9 测试结束")


if __name__ == "__main__":
    main()

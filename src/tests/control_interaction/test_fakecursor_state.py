"""
fakeCursor 状态接入验证（档 B 定式测试）

前提：GameRuntime_v2 已重编译部署，回包状态维度 = 33
      （基础 29 + fakeCursor 4：cursorX/Y=29/30, cursorVelX/Y=31/32，均为绝对量）。

验证 4 件事（对应 33 维改造的正确性）：
  1. 可读 & 非退化：所有 agent（含复制体）的 cursor 维度有限，且非零动作下会移动
     （若一直为 0 说明命中了 C# 侧 fakeCursorRB==null 的回退，即未取到光标）。
  2. 独立性：
     - 不同动作 → 各 agent 的 cursor 相对坐标持续分离；
     - 相同动作 → 各 agent 的 cursor 相对坐标一致（确认无跨 agent 串扰、非共享对象）。
  3. teleport 协同平移：teleport 后 (cursor - player) 基本不变、cursor 速度≈0
     （验证 StepController.Teleport 对 fakeCursorRB 的 co-translate 正确）。
  4. reset 一致性：reset 后 cursor 相对坐标回到快照基准。

cursor 相对坐标定义与 build_dynamics 一致：cursor_rel = (cursorX,cursorY) - (playerX,playerY)

用法：
  python src/tests/control_interaction/test_fakecursor_state.py
  python src/tests/control_interaction/test_fakecursor_state.py --no-launch --agents 2
"""

import argparse
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

# 原始状态中的 cursor 索引（与 C# StepController.CollectAllStates 及 dataset.build_dynamics 对齐）
CURSOR_X, CURSOR_Y = 29, 30
CURSOR_VX, CURSOR_VY = 31, 32

# 判据阈值
MOVE_EPS = 0.3      # 非零动作下 cursor_rel 相对初值的最大位移应 > 此值（否则疑似未取到光标）
SEP_EPS = 0.3       # 不同动作时两 agent cursor_rel 距离应 > 此值
SAME_EPS = 0.5      # 相同动作时两 agent cursor_rel 距离应 < 此值（cursor 会移动数单位，容许复制体间微小累积差）
TELE_REL_EPS = 1.0  # teleport 前后 cursor_rel 变化应 < 此值（co-translate 保持相对构型）
TELE_VEL_EPS = 2.0  # teleport 后 cursor 速度应 < 此值（清零 + 单帧 settle）
RESET_EPS = 0.5     # reset 后 cursor_rel 与基准差应 < 此值


def cursor_rel(obs: np.ndarray) -> np.ndarray:
    """obs (N,33) → cursor 相对 player 坐标 (N,2)。"""
    return obs[:, [CURSOR_X, CURSOR_Y]] - obs[:, [0, 1]]


def cursor_vel(obs: np.ndarray) -> np.ndarray:
    """obs (N,33) → cursor 速度 (N,2)。"""
    return obs[:, [CURSOR_VX, CURSOR_VY]]


def warmup_and_snapshot(env: GoiEnv, n_agents: int, n_steps: int) -> None:
    logger.info("Warmup: %d 步零动作（%d agents）...", n_steps, n_agents)
    zero = np.zeros((n_agents, 2), dtype=np.float32)
    for _ in range(n_steps):
        env.step(zero)
    env.new_snapshot()
    logger.info("Warmup 完成，已拍新快照")


def phase_readable(env: GoiEnv, n: int) -> dict:
    """相位 1：可读性 & 基准。"""
    logger.info("--- 相位1：可读性 & 基准 ---")
    obs = env.reset()
    finite = bool(np.isfinite(obs[:, CURSOR_X:CURSOR_VY + 1]).all())
    rel0 = cursor_rel(obs)
    for i in range(n):
        logger.info("  agent%d: cursor_abs=(%.3f,%.3f) player=(%.3f,%.3f) cursor_rel=(%.3f,%.3f)",
                    i, obs[i, CURSOR_X], obs[i, CURSOR_Y], obs[i, 0], obs[i, 1],
                    rel0[i, 0], rel0[i, 1])
    logger.info("  维度有限性: %s", "OK" if finite else "FAIL(存在 NaN/Inf)")
    return {"passed": finite, "rel0": rel0}


def phase_response_independence(env: GoiEnv, n: int, steps: int) -> dict:
    """相位 2a：响应性 + 不同动作独立性。"""
    logger.info("--- 相位2a：响应性 + 不同动作独立性（%d 步）---", steps)
    obs = env.reset()
    rel_init = cursor_rel(obs)

    # 运动探测基准：player(0,1)、tip(23,24)、cursor_abs(29,30)
    player0 = obs[:, [0, 1]].copy()
    tip0 = obs[:, [23, 24]].copy()
    cur0 = obs[:, [CURSOR_X, CURSOR_Y]].copy()

    # 每个 agent 一个不同方向的动作（对角，确保锤子摆动）
    dirs = [(80.0, 60.0), (-80.0, 60.0), (60.0, 80.0), (-60.0, 80.0)]
    action = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        action[i] = dirs[i % len(dirs)]

    max_move = np.zeros(n)
    max_player = np.zeros(n)
    max_tip = np.zeros(n)
    max_cur = np.zeros(n)
    rels = []
    for _ in range(steps):
        obs, _ = env.step(action)
        rel = cursor_rel(obs)
        rels.append(rel.copy())
        max_move = np.maximum(max_move, np.linalg.norm(rel - rel_init, axis=1))
        max_player = np.maximum(max_player, np.linalg.norm(obs[:, [0, 1]] - player0, axis=1))
        max_tip = np.maximum(max_tip, np.linalg.norm(obs[:, [23, 24]] - tip0, axis=1))
        max_cur = np.maximum(max_cur, np.linalg.norm(obs[:, [CURSOR_X, CURSOR_Y]] - cur0, axis=1))

    rels = np.stack(rels)  # (steps, n, 2)

    logger.info("  [运动探测] 各 agent %d 步内最大位移:", steps)
    for i in range(n):
        logger.info("    agent%d: player=%.4f  tip(锤尖)=%.4f  cursor_abs=%.4f",
                    i, max_player[i], max_tip[i], max_cur[i])

    # 响应性：每个 agent 都应有明显位移
    responsive = bool((max_move > MOVE_EPS).all())
    for i in range(n):
        # 方向相关性（仅报告）：Δrel 与动作的点积符号
        d_rel = rels[-1, i] - rel_init[i]
        corr = float(np.dot(d_rel, action[i]))
        logger.info("  agent%d: action=(%.0f,%.0f) max|Δcursor_rel|=%.3f  Δrel=(%.3f,%.3f) dot(action)=%.1f",
                    i, action[i, 0], action[i, 1], max_move[i], d_rel[0], d_rel[1], corr)
    logger.info("  响应性(所有 agent 位移>%.2f): %s", MOVE_EPS, "OK" if responsive else "FAIL")

    # 不同动作独立性：相邻 agent 的 cursor_rel 末步距离
    sep_ok = True
    if n >= 2:
        for i in range(n - 1):
            dist = float(np.linalg.norm(rels[-1, i] - rels[-1, i + 1]))
            ok = dist > SEP_EPS
            sep_ok = sep_ok and ok
            logger.info("  agent%d vs agent%d 末步 cursor_rel 距离=%.3f (>%.2f? %s)",
                        i, i + 1, dist, SEP_EPS, "OK" if ok else "FAIL")
    else:
        logger.info("  仅 1 agent，跳过分离性检查")

    return {"passed": responsive and sep_ok, "responsive": responsive, "sep_ok": sep_ok}


def phase_same_action(env: GoiEnv, n: int, steps: int) -> dict:
    """相位 2b：相同动作一致性（无跨 agent 串扰）。"""
    logger.info("--- 相位2b：相同动作一致性（%d 步）---", steps)
    if n < 2:
        logger.info("  仅 1 agent，跳过")
        return {"passed": True}
    env.reset()
    action = np.tile(np.array([80.0, 0.0], dtype=np.float32), (n, 1))
    max_diff = 0.0
    for _ in range(steps):
        obs, _ = env.step(action)
        rel = cursor_rel(obs)
        diff = float(np.abs(rel[0] - rel[1]).max())
        max_diff = max(max_diff, diff)
    passed = max_diff < SAME_EPS
    logger.info("  相同动作下 agent0 vs agent1 cursor_rel 最大差=%.2e (<%.0e? %s)",
                max_diff, SAME_EPS, "OK" if passed else "FAIL")
    return {"passed": passed, "max_diff": max_diff}


def phase_teleport(env: GoiEnv, n: int, warm_steps: int = 5) -> dict:
    """相位 3：teleport 协同平移。"""
    logger.info("--- 相位3：teleport 协同平移 ---")
    obs = env.reset()
    # 先让 agent1 动起来，制造非零 cursor_rel
    tgt = 1 if n >= 2 else 0
    action = np.zeros((n, 2), dtype=np.float32)
    action[tgt] = (80.0, 0.0)
    for _ in range(warm_steps):
        obs, _ = env.step(action)
    rel_before = cursor_rel(obs)[tgt].copy()
    px, py = float(obs[tgt, 0]), float(obs[tgt, 1])

    # teleport 到附近（小位移，避免坠落）
    target = (px, py + 2.0)
    obs = env.teleport(target[0], target[1], agent_index=tgt)
    rel_after = cursor_rel(obs)[tgt]
    vel_after = cursor_vel(obs)[tgt]

    d_rel = float(np.linalg.norm(rel_after - rel_before))
    v_mag = float(np.linalg.norm(vel_after))
    rel_ok = d_rel < TELE_REL_EPS
    vel_ok = v_mag < TELE_VEL_EPS
    logger.info("  agent%d teleport→(%.2f,%.2f)", tgt, target[0], target[1])
    logger.info("  cursor_rel 前=(%.3f,%.3f) 后=(%.3f,%.3f) |Δ|=%.3f (<%.2f? %s)",
                rel_before[0], rel_before[1], rel_after[0], rel_after[1],
                d_rel, TELE_REL_EPS, "OK" if rel_ok else "FAIL")
    logger.info("  teleport 后 cursor 速度=|%.3f| (<%.2f? %s)",
                v_mag, TELE_VEL_EPS, "OK" if vel_ok else "FAIL")
    return {"passed": rel_ok and vel_ok, "d_rel": d_rel, "v_mag": v_mag}


def phase_reset_consistency(env: GoiEnv, n: int, rel0: np.ndarray) -> dict:
    """相位 4：reset 一致性。"""
    logger.info("--- 相位4：reset 一致性 ---")
    obs = env.reset()
    rel = cursor_rel(obs)
    max_diff = float(np.abs(rel - rel0).max())
    passed = max_diff < RESET_EPS
    logger.info("  reset 后 cursor_rel 与基准最大差=%.3f (<%.2f? %s)",
                max_diff, RESET_EPS, "OK" if passed else "FAIL")
    return {"passed": passed, "max_diff": max_diff}


def main() -> int:
    parser = argparse.ArgumentParser(description="fakeCursor 状态接入验证（档 B）")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if STATE_DIM != 33:
        logger.error("goi_env.STATE_DIM=%d，期望 33。请确认已同步协议维度。", STATE_DIM)
        return 2

    if not args.no_launch:
        logger.info("设置 GameRuntime 模式并启动游戏...")
        GameModeController().set_game_runtime_mode()
        proc = GameLauncher().launch(wait=False)
        logger.info("游戏已启动 (PID=%d)，等待连接（最多 %.0fs）", proc.pid, args.timeout)

    results = {}
    with GoiEnv(port=args.port, num_agents=args.agents, timeout=args.timeout) as env:
        logger.info("已连接，agent 数=%d, STATE_DIM=%d", args.agents, STATE_DIM)
        logger.info("=" * 64)

        warmup_and_snapshot(env, args.agents, args.warmup)
        logger.info("=" * 64)

        r1 = phase_readable(env, args.agents)
        logger.info("")
        r2a = phase_response_independence(env, args.agents, args.steps)
        logger.info("")
        r2b = phase_same_action(env, args.agents, args.steps)
        logger.info("")
        r3 = phase_teleport(env, args.agents)
        logger.info("")
        r4 = phase_reset_consistency(env, args.agents, r1["rel0"])
        logger.info("=" * 64)

        results = {
            "可读&有限": r1["passed"],
            "响应+不同动作独立": r2a["passed"],
            "相同动作一致": r2b["passed"],
            "teleport协同平移": r3["passed"],
            "reset一致": r4["passed"],
        }

    logger.info("=== 综合结论 ===")
    for k, v in results.items():
        logger.info("  %-20s %s", k, "通过" if v else "失败")
    all_pass = all(results.values())
    logger.info("  总体: %s", "fakeCursor 已正确接入 33 维状态" if all_pass else "存在问题，见上方 FAIL 项")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

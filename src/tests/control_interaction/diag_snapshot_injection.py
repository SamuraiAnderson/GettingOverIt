"""
new_snapshot 前后注入链路差分诊断（对拍）

目的：定位「输入注入在 warmup + new_snapshot + reset 后失效」的根因。
方法：在同一游戏进程、同一 TCP 连接内，仅改变一个变量，跑三个相位并测量
      每个 agent 在相同动作下的实际位移。

相位设计（单变量隔离）：
  A  初始快照注入：       reset()  → action×K
  B  仅 warmup（不拍快照）：reset()  → zero×W → action×K   （在 warmed 的活跃状态上注入）
  C  new_snapshot 后注入： new_snapshot() → reset() → action×K

若 A 有位移、C 无位移：
  - B 也无位移  → warmup 本身破坏注入（与快照无关）
  - B 有位移    → new_snapshot / 从新快照 reset 破坏注入

同时打印每个 agent 的 player / tip / cursor_abs / hammer_angle 位移，
配合 C# 侧 [StepDiag]/[Rewired.GetAxis] 日志交叉印证 fcRB 与 GetAxis 拦截行为。

用法：
  python src/tests/control_interaction/diag_snapshot_injection.py
  python src/tests/control_interaction/diag_snapshot_injection.py --no-launch --agents 2
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

PX, PY = 0, 1
TIPX, TIPY = 23, 24
HAMMER = 27
CURX, CURY = 29, 30
CURVX, CURVY = 31, 32


def _actions(n: int) -> np.ndarray:
    """每个 agent 一个不同方向的大动作，确保锤子明显摆动。"""
    dirs = [(80.0, 60.0), (-80.0, 60.0), (60.0, 80.0), (-60.0, 80.0)]
    a = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        a[i] = dirs[i % len(dirs)]
    return a


def _measure(env: GoiEnv, obs0: np.ndarray, action: np.ndarray, k: int, tag: str) -> None:
    """从当前状态 obs0 起，施加 action k 步，测量每个 agent 各部件最大位移。"""
    n = obs0.shape[0]
    p0 = obs0[:, [PX, PY]].copy()
    t0 = obs0[:, [TIPX, TIPY]].copy()
    c0 = obs0[:, [CURX, CURY]].copy()
    h0 = obs0[:, HAMMER].copy()
    crel0 = (obs0[:, [CURX, CURY]] - obs0[:, [PX, PY]]).copy()

    mp = np.zeros(n); mt = np.zeros(n); mc = np.zeros(n); mh = np.zeros(n); mcrel = np.zeros(n)
    for _ in range(k):
        obs, _ = env.step(action)
        mp = np.maximum(mp, np.linalg.norm(obs[:, [PX, PY]] - p0, axis=1))
        mt = np.maximum(mt, np.linalg.norm(obs[:, [TIPX, TIPY]] - t0, axis=1))
        mc = np.maximum(mc, np.linalg.norm(obs[:, [CURX, CURY]] - c0, axis=1))
        mh = np.maximum(mh, np.abs(obs[:, HAMMER] - h0))
        crel = obs[:, [CURX, CURY]] - obs[:, [PX, PY]]
        mcrel = np.maximum(mcrel, np.linalg.norm(crel - crel0, axis=1))

    logger.info("[%s] 施加动作 %d 步后各 agent 最大位移：", tag, k)
    for i in range(n):
        logger.info(
            "    agent%d  action=(%+.0f,%+.0f)  player=%.4f  tip=%.4f  cursor_abs=%.4f  cursor_rel=%.4f  |Δhammer|=%.4f",
            i, action[i, 0], action[i, 1], mp[i], mt[i], mc[i], mcrel[i], mh[i],
        )
        # 同时打印首帧 cursor 绝对/相对坐标，判断 fcRB 是否命中回退
        logger.info(
            "            起点 player=(%.3f,%.3f) cursor_abs=(%.3f,%.3f) cursor_rel=(%.3f,%.3f)",
            p0[i, 0], p0[i, 1], c0[i, 0], c0[i, 1], crel0[i, 0], crel0[i, 1],
        )


def _trace(env: GoiEnv, action: np.ndarray, k: int, tag: str) -> None:
    """逐步追踪 agent0 的 hammer/cursor_abs 相对起点的位移，定位注入何时"唤醒"。"""
    obs0 = env.step(action)[0]  # 第一步也算入
    h0 = obs0[:, HAMMER].copy()
    c0 = obs0[:, [CURX, CURY]].copy()
    marks = [1, 2, 3, 5, 8, 12, 20, 30, k]
    for s in range(2, k + 1):
        obs, _ = env.step(action)
        if s in marks:
            dh = float(abs(obs[0, HAMMER] - h0[0]))
            dc = float(np.linalg.norm(obs[0, [CURX, CURY]] - c0[0]))
            logger.info("    [%s] step=%2d  agent0 |Δhammer|=%8.4f  |Δcursor_abs|=%8.4f", tag, s, dh, dc)


def main() -> int:
    parser = argparse.ArgumentParser(description="new_snapshot 前后注入链路差分诊断")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--steps", type=int, default=15, help="每相位施加动作步数 K")
    parser.add_argument("--warmup", type=int, default=100, help="warmup 零动作步数 W")
    parser.add_argument("--no-launch", action="store_true")
    parser.add_argument("--trace", action="store_true", help="逐步追踪模式：定位注入唤醒时机")
    parser.add_argument("--waketest", action="store_true", help="区分唤醒是墙钟时间还是累计步数")
    parser.add_argument("--sleep", type=float, default=8.0, help="waketest 中纯等待秒数")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if STATE_DIM != 33:
        logger.error("goi_env.STATE_DIM=%d，期望 33", STATE_DIM)
        return 2

    if not args.no_launch:
        logger.info("设置 GameRuntime 模式并启动游戏...")
        GameModeController().set_game_runtime_mode()
        proc = GameLauncher().launch(wait=False)
        logger.info("游戏已启动 (PID=%d)，等待连接（最多 %.0fs）", proc.pid, args.timeout)

    action = _actions(args.agents)
    zero = np.zeros((args.agents, 2), dtype=np.float32)

    # startup_settle_s=0：本诊断需复现"冷启动"状态，故禁用连接后的默认预热等待
    with GoiEnv(port=args.port, num_agents=args.agents, timeout=args.timeout,
                startup_settle_s=0.0) as env:
        logger.info("已连接，agents=%d STATE_DIM=%d", args.agents, STATE_DIM)
        logger.info("=" * 72)

        if args.waketest:
            import time as _time
            # W1：连接后立刻注入（最冷，尚未 step、尚未额外等待）
            logger.info(">>> W1：连接后立刻 reset → action×30（最冷基线）")
            env.reset(); _trace(env, action, 30, "W1最冷")
            logger.info("-" * 72)
            # W2：纯等待 sleep 秒（不 step）后再注入
            logger.info(">>> W2：reset → 纯 sleep %.1fs（不 step）→ action×30", args.sleep)
            env.reset()
            _time.sleep(args.sleep)
            _trace(env, action, 30, "W2纯等待")
            logger.info("=" * 72)
            logger.info("waketest 完成：若 W2 明显强于 W1 → 唤醒取决于墙钟时间(游戏初始化)；"
                        "若 W2 仍弱 → 取决于累计步数/物理推进。")
            return 0

        if args.trace:
            # 逐步追踪：定位注入"唤醒"是取决于步数还是快照态，并验证可复现性
            logger.info(">>> T1：reset → action×40（冷启动，逐步追踪）")
            env.reset(); _trace(env, action, 40, "T1冷启动")
            logger.info("-" * 72)
            logger.info(">>> T2：reset → zero×%d → action×40（warmup 后追踪）", args.warmup)
            env.reset()
            for _ in range(args.warmup):
                env.step(zero)
            _trace(env, action, 40, "T2warmup后")
            logger.info("-" * 72)
            logger.info(">>> T3：再次 reset → zero×%d → action×40（可复现性）", args.warmup)
            env.reset()
            for _ in range(args.warmup):
                env.step(zero)
            _trace(env, action, 40, "T3复现")
            logger.info("-" * 72)
            logger.info(">>> T4：reset → action×40（再次冷启动，验证是否被前序 warmup 影响）")
            env.reset(); _trace(env, action, 40, "T4冷启动2")
            logger.info("=" * 72)
            logger.info("追踪完成。")
            return 0

        # --- 相位 A：初始快照注入 ---
        logger.info(">>> 相位 A：初始快照 reset → action（基线，应有位移）")
        obsA = env.reset()
        if obsA.shape[0] != args.agents:
            logger.warning("!!! reset 返回 %d 个 agent，与请求 %d 不符（协议/复制体数不一致）",
                           obsA.shape[0], args.agents)
        _measure(env, obsA, action, args.steps, "A-初始快照")
        logger.info("-" * 72)

        # --- 相位 B：仅 warmup，不拍快照，在活跃 warmed 状态上注入 ---
        logger.info(">>> 相位 B：reset → zero×%d（warmup，不拍快照）→ action", args.warmup)
        env.reset()
        obsB = None
        for _ in range(args.warmup):
            obsB, _ = env.step(zero)
        _measure(env, obsB, action, args.steps, "B-仅warmup活跃态")
        logger.info("-" * 72)

        # --- 相位 C：new_snapshot 后注入 ---
        logger.info(">>> 相位 C：new_snapshot（拍下当前 warmed 态）→ reset → action")
        env.new_snapshot()
        obsC = env.reset()
        _measure(env, obsC, action, args.steps, "C-快照后reset")
        logger.info("=" * 72)

        # --- 补充相位 D：快照后再 warmup 再注入（排查是否需要额外 warmup 唤醒）---
        logger.info(">>> 相位 D：快照后 reset → zero×%d → action（排查唤醒需求）", args.warmup)
        env.reset()
        obsD = None
        for _ in range(args.warmup):
            obsD, _ = env.step(zero)
        _measure(env, obsD, action, args.steps, "D-快照后再warmup")
        logger.info("=" * 72)

    logger.info("诊断完成。请结合 C# BepInEx 日志中 [StepDiag] / [Rewired.GetAxis] 交叉分析。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
方案 B 前提验证：把「游戏 + N 复制体」当作并行世界模型规划器是否成立。

方案 B（Expert Iteration / 游戏即模型的 CEM-MPC）依赖四条物理前提，本脚本逐条实测：

  P1 同起点     : reset() 后 N 个复制体的完整 33D 状态逐位一致（都回滚到同一快照）。
  P2 确定性+零串扰: 从同起点让所有 agent 执行**完全相同**的动作序列，长时程轨迹逐位相同。
                   任何分叉 = 非确定性 或 复制体互相干扰（共享可变状态/碰撞未隔离）。
  P3 隔离性     : agent0 执行序列 X；其他 agent 分别在「全零」与「随机动作」两种情况下运行，
                   agent0 的轨迹必须完全不受其他 agent 行为影响（并行独立 rollout 的真正保证）。
  P4 跨reset复现 : 同起点 + 同序列跑两次（中间 reset），轨迹一致（规划器评估必须可复现）。

四条全部成立 → N 个复制体可作为「同一起点、不同动作序列」的并行 rollout，方案 B 前提成立。

用法：
  python src/tests/control_interaction/test_planb_premise.py --agents 6 --horizon 150
  python src/tests/control_interaction/test_planb_premise.py --no-launch --agents 6
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

# 33D 状态里随时间单调递增、不参与"轨迹是否一致"判定的字段：index 28 = timestamp。
TIMESTAMP_IDX = 28
# 判据阈值（世界单位：位置 m、速度 m/s）。同机同构建的 Box2D 在相同输入下应逐位可复现，
# 故阈值取物理上可忽略的小量；同时始终报告实际最大差值，便于判断是"真一致"还是"接近"。
IDENTICAL_EPS = 1e-2      # P1/P2/P3/P4 判定"逐位一致"的最大允许偏差
DIVERGE_EPS = 0.3         # P3 中不同动作的 agent 之间应当出现的最小分叉（确认动作确实生效）


def _state_cols() -> np.ndarray:
    """参与一致性比较的列（剔除 timestamp）。"""
    cols = list(range(STATE_DIM))
    cols.remove(TIMESTAMP_IDX)
    return np.array(cols, dtype=int)


_COLS = _state_cols()


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """两个 (…,33) 状态数组在非 timestamp 列上的最大绝对差。"""
    return float(np.abs(a[..., _COLS] - b[..., _COLS]).max())


def _worst_field(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    """返回 (差异最大的原始状态索引, 差值)。用于诊断分叉发生在哪个物理量。"""
    d = np.abs(a[..., _COLS] - b[..., _COLS])
    flat = d.reshape(-1, d.shape[-1])
    col = int(flat.max(axis=0).argmax())
    return int(_COLS[col]), float(flat[:, col].max())


def _seeded_action_seq(n_agents: int, horizon: int, seed: int) -> np.ndarray:
    """确定性动作序列 (horizon, n_agents, 2)，幅度覆盖大范围以诱发摆动与接触。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(-100.0, 100.0, size=(horizon, n_agents, 2)).astype(np.float32)


def _trace_steps(horizon: int) -> list[int]:
    """分歧增长曲线的采样步（用于区分混沌指数增长 vs 恒定偏置）。"""
    cands = [1, 5, 10, 20, 50, 100, 150, 200, 300]
    return [s for s in cands if s <= horizon]


def p0_zero_action(env: GoiEnv, n: int, horizon: int) -> dict:
    """P0：零动作决定性诊断——把混沌降到最弱，隔离 reset/串扰问题。

    P0a 跨 agent：reset 后所有 agent 零动作，agent 间是否保持一致。
    P0b 跨 reset ：零动作序列跑两次，agent0 是否可复现。
    零动作下若仍分叉/不可复现，则问题在状态恢复不全或非确定性，而非动作诱发的混沌。
    """
    logger.info("--- P0：零动作决定性诊断（混沌最弱下的一致性/可复现性）---")
    trace = _trace_steps(horizon)

    # P0a：跨 agent（单 episode 内）
    env.reset()
    zero = np.zeros((n, 2), dtype=np.float32)
    a_curve, last = {}, None
    for t in range(1, horizon + 1):
        obs, _ = env.step(zero)
        last = obs
        if t in trace:
            a_curve[t] = _max_abs_diff(obs, np.repeat(obs[0:1], n, axis=0))
    p0a_max = max(a_curve.values()) if a_curve else 0.0
    logger.info("  P0a 跨 agent（零动作）分歧增长: %s",
                " ".join(f"s{t}={a_curve[t]:.2e}" for t in trace))

    # P0b：跨 reset（agent0，零动作跑两次）
    def run_zero() -> np.ndarray:
        env.reset()
        traj = []
        for _ in range(horizon):
            obs, _ = env.step(zero)
            traj.append(obs[0].copy())
        return np.stack(traj)

    z1, z2 = run_zero(), run_zero()
    b_curve = {t: _max_abs_diff(z1[t - 1], z2[t - 1]) for t in trace}
    p0b_max = max(b_curve.values()) if b_curve else 0.0
    logger.info("  P0b 跨 reset（零动作）agent0 分歧增长: %s",
                " ".join(f"s{t}={b_curve[t]:.2e}" for t in trace))

    idx_a, _ = _worst_field(last, np.repeat(last[0:1], n, axis=0))
    logger.info("  P0a 末步最大分歧字段 state[%d]；P0a_max=%.3e P0b_max=%.3e",
                idx_a, p0a_max, p0b_max)
    return {"p0a_max": p0a_max, "p0b_max": p0b_max, "a_curve": a_curve, "b_curve": b_curve}


def p_chaos_regimes(env: GoiEnv, n: int, horizon: int) -> dict:
    """混沌率 vs 动作平滑度扫描：测出方案 B 的「可用规划时域」。

    所有 agent 从同一起点执行**同一动作序列**，测跨 agent 分歧增长。分歧越慢，
    在该动作风格下可无损并行/规划的时域越长。三种动作风格代表不同平滑度：
      hold    : 恒定中等动作（最平滑）
      smooth  : 低频正弦（平滑连贯挥动，接近"基元"）
      violent : 每步 ±100 白噪声（最坏情况，已在 P2 测过，这里复测对照）
    报告每种风格下分歧首次超过 τ=0.1 / 1.0 / 10.0 的步数（= 该精度下的可用时域）。
    """
    logger.info("--- 混沌率 vs 动作平滑度（可用规划时域）---")
    trace = _trace_steps(horizon)
    thresholds = [0.1, 1.0, 10.0]

    def regime_seq(kind: str) -> np.ndarray:
        ts = np.arange(horizon)
        if kind == "hold":
            a = np.tile(np.array([40.0, 30.0], np.float32), (horizon, 1))
        elif kind == "smooth":
            # 低频正弦：周期 40 步，幅度 60 —— 连贯挥动
            a = np.stack([60.0 * np.sin(2 * np.pi * ts / 40.0),
                          60.0 * np.cos(2 * np.pi * ts / 40.0)], axis=1).astype(np.float32)
        else:  # violent
            a = _seeded_action_seq(1, horizon, seed=42)[:, 0, :]
        return a

    out = {}
    for kind in ("hold", "smooth", "violent"):
        env.reset()
        seq = regime_seq(kind)
        curve, cross = {}, {th: None for th in thresholds}
        for t in range(1, horizon + 1):
            obs, _ = env.step(np.tile(seq[t - 1], (n, 1)))
            d = _max_abs_diff(obs, np.repeat(obs[0:1], n, axis=0))
            for th in thresholds:
                if cross[th] is None and d > th:
                    cross[th] = t
            if t in trace:
                curve[t] = d
        out[kind] = {"curve": curve, "cross": cross}
        logger.info("  [%-7s] 分歧: %s", kind,
                    " ".join(f"s{t}={curve[t]:.2e}" for t in trace))
        logger.info("  [%-7s] 首超阈值步: >0.1@%s  >1.0@%s  >10@%s", kind,
                    cross[0.1], cross[1.0], cross[10.0])
    return out


def warmup_and_snapshot(env: GoiEnv, n_agents: int, n_steps: int) -> None:
    logger.info("Warmup: %d 步零动作（%d agents）...", n_steps, n_agents)
    zero = np.zeros((n_agents, 2), dtype=np.float32)
    for _ in range(n_steps):
        env.step(zero)
    env.new_snapshot()
    logger.info("Warmup 完成，已拍新快照")


def p1_identical_start(env: GoiEnv, n: int) -> dict:
    """P1：reset 后所有 agent 完整状态逐位一致。"""
    logger.info("--- P1：同起点（reset 后 N 复制体状态一致）---")
    obs = env.reset()  # (n, 33)
    # 以 agent0 为基准，比较其余 agent
    base = obs[0:1]
    max_diff = _max_abs_diff(obs, np.repeat(base, n, axis=0))
    idx, d = _worst_field(obs, np.repeat(base, n, axis=0))
    passed = max_diff < IDENTICAL_EPS
    logger.info("  N=%d 个 agent 相对 agent0 的最大状态差=%.3e（阈值 %.0e）", n, max_diff, IDENTICAL_EPS)
    logger.info("  差异最大字段：state[%d]，差值=%.3e", idx, d)
    logger.info("  P1 %s", "通过" if passed else "失败")
    return {"passed": passed, "max_diff": max_diff}


def p2_determinism_no_crosstalk(env: GoiEnv, n: int, horizon: int) -> dict:
    """P2：所有 agent 执行相同动作序列 → 轨迹逐位一致（确定性 + 零串扰）。"""
    logger.info("--- P2：确定性 + 零串扰（%d 步同一动作，跨 agent 比较）---", horizon)
    env.reset()
    seq = _seeded_action_seq(1, horizon, seed=42)[:, 0, :]  # (horizon, 2)，所有 agent 共用
    running_max = 0.0
    worst_step = -1
    trace = _trace_steps(horizon)
    curve = {}
    for t in range(1, horizon + 1):
        act = np.tile(seq[t - 1], (n, 1))  # 所有 agent 完全相同动作
        obs, _ = env.step(act)
        base = np.repeat(obs[0:1], n, axis=0)
        d = _max_abs_diff(obs, base)
        if d > running_max:
            running_max = d
            worst_step = t
        if t in trace:
            curve[t] = d
    logger.info("  跨 agent（同动作）分歧增长: %s",
                " ".join(f"s{t}={curve[t]:.2e}" for t in trace))
    idx, _ = _worst_field(obs, np.repeat(obs[0:1], n, axis=0))
    passed = running_max < IDENTICAL_EPS
    logger.info("  跨 agent 轨迹最大偏差=%.3e（阈值 %.0e），出现在 step %d",
                running_max, IDENTICAL_EPS, worst_step)
    logger.info("  末步差异最大字段：state[%d]", idx)
    logger.info("  P2 %s（若失败=存在非确定性或复制体互相干扰）", "通过" if passed else "失败")
    return {"passed": passed, "max_diff": running_max, "worst_step": worst_step}


def p3_isolation(env: GoiEnv, n: int, horizon: int) -> dict:
    """P3：agent0 轨迹不受其他 agent 行为影响。"""
    logger.info("--- P3：隔离性（agent0 序列固定，其他 agent 零 vs 随机）---")
    if n < 2:
        logger.info("  仅 1 agent，隔离性无意义，跳过")
        return {"passed": True, "skipped": True}

    seqX = _seeded_action_seq(1, horizon, seed=7)[:, 0, :]  # agent0 的固定序列
    other_rand = _seeded_action_seq(n, horizon, seed=999)   # 其他 agent 的随机序列

    # Run A：agent0=X，其他=0
    env.reset()
    trajA = []
    for t in range(horizon):
        act = np.zeros((n, 2), dtype=np.float32)
        act[0] = seqX[t]
        obs, _ = env.step(act)
        trajA.append(obs[0].copy())
    trajA = np.stack(trajA)

    # Run B：agent0=X，其他=随机
    env.reset()
    trajB = []
    diverged_others = 0.0
    for t in range(horizon):
        act = other_rand[t].copy()
        act[0] = seqX[t]  # agent0 用相同序列覆盖
        obs, _ = env.step(act)
        trajB.append(obs[0].copy())
        # 记录其他 agent 是否确实因随机动作而与 agent0 分叉（确认对照有效）
        diverged_others = max(diverged_others, _max_abs_diff(obs[1:2], obs[0:1]))
    trajB = np.stack(trajB)

    max_diff = _max_abs_diff(trajA, trajB)
    idx, _ = _worst_field(trajA, trajB)
    passed = max_diff < IDENTICAL_EPS
    control_ok = diverged_others > DIVERGE_EPS  # 其他 agent 确实动了（否则对照无效）
    logger.info("  agent0 轨迹 (其他=零) vs (其他=随机) 最大差=%.3e（阈值 %.0e）",
                max_diff, IDENTICAL_EPS)
    logger.info("  差异最大字段：state[%d]", idx)
    logger.info("  对照有效性：其他 agent 相对 agent0 分叉=%.3f（应 > %.1f）%s",
                diverged_others, DIVERGE_EPS, "OK" if control_ok else "WARN(对照无效)")
    logger.info("  P3 %s（若失败=agent 之间存在状态泄漏）", "通过" if passed else "失败")
    return {"passed": passed and control_ok, "max_diff": max_diff, "control_ok": control_ok}


def p4_cross_reset_reproducible(env: GoiEnv, n: int, horizon: int) -> dict:
    """P4：同起点 + 同序列跑两次，轨迹一致。"""
    logger.info("--- P4：跨 reset 可复现（同序列跑两次比较 agent0）---")
    seqX = _seeded_action_seq(1, horizon, seed=123)[:, 0, :]

    def run_once() -> np.ndarray:
        env.reset()
        traj = []
        for t in range(horizon):
            act = np.zeros((n, 2), dtype=np.float32)
            act[0] = seqX[t]
            obs, _ = env.step(act)
            traj.append(obs[0].copy())
        return np.stack(traj)

    t1 = run_once()
    t2 = run_once()
    max_diff = _max_abs_diff(t1, t2)
    idx, _ = _worst_field(t1, t2)
    passed = max_diff < IDENTICAL_EPS
    logger.info("  两次运行 agent0 轨迹最大差=%.3e（阈值 %.0e）", max_diff, IDENTICAL_EPS)
    logger.info("  差异最大字段：state[%d]", idx)
    logger.info("  P4 %s（若失败=reset 未完全恢复状态 或 物理非确定）", "通过" if passed else "失败")
    return {"passed": passed, "max_diff": max_diff}


def main() -> int:
    parser = argparse.ArgumentParser(description="方案 B 前提验证（游戏即并行世界模型）")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--agents", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--no-launch", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if STATE_DIM != 33:
        logger.error("goi_env.STATE_DIM=%d，期望 33。", STATE_DIM)
        return 2

    if not args.no_launch:
        # 多 agent：启动前必须写 numDuplicates，否则 C# 只建 1 个 agent
        from training.config import TrainConfig
        from training.rollout import _write_num_duplicates
        game_root = Path(TrainConfig().game_root)
        _write_num_duplicates(game_root, args.agents)
        logger.info("已写入 numDuplicates=%d 到 %s/GoiData/runtime_config.json", args.agents, game_root)

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

        r0 = p0_zero_action(env, args.agents, args.horizon)
        logger.info("")
        rc = p_chaos_regimes(env, args.agents, args.horizon)
        logger.info("")
        r1 = p1_identical_start(env, args.agents)
        logger.info("")
        r2 = p2_determinism_no_crosstalk(env, args.agents, args.horizon)
        logger.info("")
        r3 = p3_isolation(env, args.agents, args.horizon)
        logger.info("")
        r4 = p4_cross_reset_reproducible(env, args.agents, args.horizon)
        logger.info("=" * 64)

        results = {
            "P1 同起点": r1["passed"],
            "P2 确定性+零串扰": r2["passed"],
            "P3 隔离性": r3["passed"],
            "P4 跨reset可复现": r4["passed"],
        }

    logger.info("=== 方案 B 前提综合结论 ===")
    for k, v in results.items():
        logger.info("  %-20s %s", k, "通过" if v else "失败")
    all_pass = all(results.values())

    # 决定性诊断：交叉比对 P0b(零动作跨reset) vs P4(暴力动作跨reset) vs P0a/P1(跨agent)
    logger.info("--- 根因诊断 ---")
    p0b = r0["p0b_max"]          # 单 agent 跨 reset，零动作
    p0a = r0["p0a_max"]          # 跨 agent，零动作
    p4 = r4["max_diff"]          # 单 agent 跨 reset，暴力动作

    # (1) 物理引擎确定性：P0b 极小 → 刚体恢复 + Simulate 逐位可复现
    if p0b < IDENTICAL_EPS:
        logger.info("  ✓ 物理引擎确定（P0b=%.2e，零动作跨reset）：Simulate + 刚体快照可复现。", p0b)
    else:
        logger.info("  ✗ 物理引擎在零动作下已不可复现（P0b=%.2e）：更底层的非确定性。", p0b)

    # (2) 关键：动作相关的未恢复状态。P0b≈0 但 P4≫0 → PlayerControl 内部累积字段未快照
    if p0b < IDENTICAL_EPS and p4 >= IDENTICAL_EPS:
        logger.info("  ✗ 跨 reset 在暴力动作下不可复现（P4=%.2e），但零动作可复现（P0b=%.2e）："
                    "reset 未恢复 PlayerControl 的**动作相关内部状态**"
                    "（mouseVelocityAverage 自适应增益 / oldMouse 平滑项），跨 episode 残留 → 混沌放大。"
                    "【修复1】StepController 快照/恢复须纳入这些字段。", p4, p0b)

    # (3) 复制体起点对齐：P0a / P1
    if p0a >= IDENTICAL_EPS or not r1["passed"]:
        logger.info("  ✗ 复制体起点未对齐（P0a=%.2e，P1_max=%.2e）：各 agent 快照自身漂移态而非共享基准。"
                    "【修复2】广播单一权威快照到所有 agent，使并行 rollout 起点逐位一致。",
                    p0a, r1["max_diff"])
    else:
        logger.info("  ✓ 复制体起点对齐（P0a=%.2e）：零动作下 agent 间一致、无串扰。", p0a)

    # (4) 混沌：动作风格相关
    logger.info("  ⚠ 确定性混沌：violent/smooth-large 动作下分歧数步内 e 折爆炸；hold(稳定)动作可收敛。"
                "→ 方案 B 须短时域滚动重规划(MPC)，且优先采样连贯/稳定基元。")

    logger.info("  结论: %s", "方案 B 前提成立（游戏可作并行世界模型）" if all_pass
                else "方案 B 前提当前不成立；核心设施(确定性/隔离/reset刚体)可用，"
                     "但需【修复1】恢复控制内部态 +【修复2】广播权威快照，且规划须短时域滚动")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())

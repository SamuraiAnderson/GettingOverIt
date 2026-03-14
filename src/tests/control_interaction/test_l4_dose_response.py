"""
层次 4（重构版）：连续序列稳定性与区分性验证

验证 f(s0, a_list) = s_n 的两个关键属性，为 world model 训练奠定信任基础：

  A. 稳定性（Stability）：
     相同的 n_steps 步动作序列从同一 s0 出发，重复 n_trials 次，
     终态 s_n 几乎完全一致（确定性保证）。
     判据：max|s_n_i - s_n_j| < 1e-3  （排除 timestamp 维）
     意义：world model 有确定性训练目标

  B. 区分性（Separability）：
     5 种方向序列（零、右、左、上、下）从同一 s0 出发，
     终态 s_n 在状态空间中相互可区分。
     判据：任意两序列终态的欧式距离 > sep_threshold
     意义：动作对状态有实际区分能力，网络有信号可学

动作模式：
  constant — 恒定方向（原有），适合小幅验证
  random   — 固定种子随机序列，产生大范围运动，压力测试确定性

用法：
  python src/tests/control_interaction/test_l4_dose_response.py
  python src/tests/control_interaction/test_l4_dose_response.py --steps 500 --amplitude 300 --action-mode random
  python src/tests/control_interaction/test_l4_dose_response.py --no-launch --steps 500 --amplitude 300 --trials 5 --action-mode random
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
    plot_dim_max_diff,
    plot_dist_heatmap,
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "test_config.json"
_REPORT_DIR  = _REPO_ROOT / "src" / "Data" / "GameResults"

STATE_LABELS = [
    "player_x", "player_y", "vel_x", "vel_y", "ang_vel",
    "hub_x", "hub_y", "hub_vx", "hub_vy", "hub_angle",
    "slider_x", "slider_y", "slider_vx", "slider_vy", "slider_angle",
    "handle_x", "handle_y", "handle_vx", "handle_vy",
    "pole_x", "pole_y", "pole_vx", "pole_vy",
    "tip_x", "tip_y", "tip_vx", "tip_vy",
    "hammer_angle", "timestamp",
]

# 排除 timestamp（dim 28），它始终单调增加，不反映控制效果
EXCLUDE_DIMS  = {28}
PHYSICS_DIMS  = [d for d in range(STATE_DIM) if d not in EXCLUDE_DIMS]

STABILITY_PASS = 1e-3
STABILITY_WARN = 1e-2

# 5 种方向序列定义（单位向量，乘以 amplitude）
SEQ_DIRECTIONS = {
    "zero":  ( 0.0,  0.0),
    "right": (+1.0,  0.0),
    "left":  (-1.0,  0.0),
    "up":    ( 0.0, +1.0),
    "down":  ( 0.0, -1.0),
}


# ── 辅助 ────────────────────────────────────────────────────

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


def warmup_and_snapshot(env: GoiEnv, warmup_steps: int):
    """零动作等待动画结束，拍摄 Reset 基准快照。"""
    n = env.num_agents
    zero = np.zeros((n, 2), dtype=np.float32)
    logger.info("  Warmup：%d 步零动作（%d agents）等待动画结束...", warmup_steps, n)
    env.reset()
    for _ in range(warmup_steps):
        env.step(zero)
    env.new_snapshot()
    obs = env.reset()
    logger.info("  Warmup 完成，基准 player=(%.4f, %.4f)", obs[0, 0], obs[0, 1])


def _run_once(env: GoiEnv, action_2d: np.ndarray, n_steps: int) -> np.ndarray:
    """
    从 snapshot 出发执行固定动作 n_steps 步，返回 agent-1 的完整轨迹 (n_steps+1, STATE_DIM)。
    action_2d shape: (2,) — 单 agent 的动作向量，会被广播到所有 agents。
    """
    n = env.num_agents
    action = np.tile(action_2d, (n, 1)).astype(np.float32)
    obs = env.reset()
    traj = [obs[0].copy()]
    for _ in range(n_steps):
        obs, _ = env.step(action)
        traj.append(obs[0].copy())
    return np.stack(traj)


def _run_sequence(env: GoiEnv, actions: np.ndarray) -> np.ndarray:
    """
    从 snapshot 出发执行逐步动作序列，返回 agent-1 的完整轨迹 (n_steps+1, STATE_DIM)。
    actions shape: (n_steps, 2) — 单 agent 的动作序列，会被广播到所有 agents。
    """
    n = env.num_agents
    obs = env.reset()
    traj = [obs[0].copy()]
    for t in range(len(actions)):
        act = np.tile(actions[t], (n, 1)).astype(np.float32)
        obs, _ = env.step(act)
        traj.append(obs[0].copy())
    return np.stack(traj)


def _traj_summary(traj: np.ndarray) -> str:
    """返回轨迹的运动范围摘要（位移、速度、x/y 覆盖）。"""
    px, py = traj[:, 0], traj[:, 1]
    vx, vy = traj[:, 2], traj[:, 3]
    s0x, s0y = px[0], py[0]
    snx, sny = px[-1], py[-1]
    disp = float(np.sqrt((snx - s0x)**2 + (sny - s0y)**2))
    speed = np.sqrt(vx**2 + vy**2)
    return (
        f"disp={disp:.3f}  max_speed={float(speed.max()):.3f}  "
        f"x=[{float(px.min()):.2f},{float(px.max()):.2f}]  "
        f"y=[{float(py.min()):.2f},{float(py.max()):.2f}]"
    )


# ── 测试 A：稳定性 ─────────────────────────────────────────

def test_stability(
    env: GoiEnv,
    n_steps: int,
    amplitude: float,
    n_trials: int,
    action_mode: str = "random",
) -> dict:
    """
    同一动作序列重复 n_trials 次，比较终态 s_n 的一致性。

    action_mode:
      constant — 恒定下压 (0, -amplitude)
      random   — 固定种子随机序列 uniform(-amp, amp)，产生大范围运动
    """
    logger.info(
        "=== 测试 A：稳定性（%d 步 × %d 轮，mode=%s，amp=%.0f）===",
        n_steps, n_trials, action_mode, amplitude,
    )

    if action_mode == "random":
        rng = np.random.RandomState(42)
        actions = rng.uniform(-amplitude, amplitude, size=(n_steps, 2)).astype(np.float32)
    else:
        actions = None
        action_2d = np.array([0.0, -amplitude], dtype=np.float32)

    final_states = []
    trajs = []
    for trial in range(n_trials):
        if actions is not None:
            traj = _run_sequence(env, actions)
        else:
            traj = _run_once(env, action_2d, n_steps)
        trajs.append(traj)
        sn = traj[-1]
        final_states.append(sn.copy())
        logger.info(
            "  轮次 %d  player=(%.4f, %.4f)  hammer_angle=%.4f  %s",
            trial + 1, sn[0], sn[1], sn[27], _traj_summary(traj),
        )

    final_states = np.stack(final_states)   # (n_trials, STATE_DIM)

    per_dim_diff = np.zeros(STATE_DIM)
    for d in PHYSICS_DIMS:
        per_dim_diff[d] = float(
            np.abs(final_states[:, d] - final_states[0, d]).max()
        )

    max_diff  = float(per_dim_diff[PHYSICS_DIMS].max())
    worst_dim = int(np.argmax(per_dim_diff))
    worst_lbl = STATE_LABELS[worst_dim] if worst_dim < len(STATE_LABELS) else f"dim_{worst_dim}"

    if max_diff < STABILITY_PASS:
        verdict = "通过"
    elif max_diff < STABILITY_WARN:
        verdict = "警告"
    else:
        verdict = "失败"

    logger.info("--- 稳定性结果 ---")
    logger.info("  终态最大差异: %.2e  判断: %s", max_diff, verdict)
    logger.info("  最差维度: [%d] %s = %.2e", worst_dim, worst_lbl, per_dim_diff[worst_dim])

    bad = [(d, STATE_LABELS[d] if d < len(STATE_LABELS) else f"dim_{d}", float(per_dim_diff[d]))
           for d in PHYSICS_DIMS if per_dim_diff[d] > STABILITY_PASS]
    if bad:
        logger.info("  超过阈值的维度：")
        for idx, lbl, val in bad:
            logger.info("    [%2d] %-20s %.2e", idx, lbl, val)
    else:
        logger.info("  所有物理维度差异 < %.0e  [确定性✓]", STABILITY_PASS)

    return {
        "verdict":      verdict,
        "max_diff":     max_diff,
        "per_dim_diff": per_dim_diff,
        "final_states": final_states,
        "trajectories": trajs,
    }


# ── 测试 B：区分性 ─────────────────────────────────────────

def test_separability(
    env: GoiEnv,
    n_steps: int,
    amplitude: float,
    sep_threshold: float,
) -> dict:
    """
    5 种方向序列（零/右/左/上/下）从同一 s0 出发执行 n_steps 步，
    比较各终态 s_n 之间的欧式距离。
    """
    logger.info(
        "=== 测试 B：区分性（%d 步，5 种方向，amp=%.0f，阈值=%.3f）===",
        n_steps, amplitude, sep_threshold,
    )

    seq_names = list(SEQ_DIRECTIONS.keys())
    final_states = {}
    trajs = {}

    for name, (dx, dy) in SEQ_DIRECTIONS.items():
        action_2d = np.array([dx * amplitude, dy * amplitude], dtype=np.float32)
        traj = _run_once(env, action_2d, n_steps)
        trajs[name] = traj
        sn = traj[-1]
        final_states[name] = sn.copy()
        logger.info(
            "  序列 %-6s  player=(%.4f, %.4f)  vel=(%.4f, %.4f)  hammer=%.4f",
            name, sn[0], sn[1], sn[2], sn[3], sn[27],
        )

    # 计算配对欧式距离矩阵（仅物理维度）
    n = len(seq_names)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                si = final_states[seq_names[i]][PHYSICS_DIMS]
                sj = final_states[seq_names[j]][PHYSICS_DIMS]
                dist_matrix[i, j] = float(np.linalg.norm(si - sj))

    off_diag = dist_matrix[dist_matrix > 0]
    min_dist = float(off_diag.min()) if len(off_diag) > 0 else 0.0
    max_dist = float(off_diag.max()) if len(off_diag) > 0 else 0.0

    # 各活动序列与零序列的距离
    zero_dists = {}
    for name in seq_names:
        if name != "zero":
            si = final_states["zero"][PHYSICS_DIMS]
            sj = final_states[name][PHYSICS_DIMS]
            zero_dists[name] = float(np.linalg.norm(si - sj))

    separable = min_dist > sep_threshold
    verdict = "通过" if separable else "失败"

    logger.info("--- 区分性结果 ---")
    logger.info("  全局最小配对距离: %.4f  阈值: %.3f  判断: %s",
                min_dist, sep_threshold, verdict)
    logger.info("  全局最大配对距离: %.4f", max_dist)
    logger.info("  各活动序列与零序列距离:")
    for name, d in zero_dists.items():
        ok = "✓" if d > sep_threshold else "✗"
        logger.info("    %-6s: %.4f  %s", name, d, ok)

    logger.info("  距离矩阵（行/列：%s）:", seq_names)
    for i, row_name in enumerate(seq_names):
        row_str = "  ".join(f"{dist_matrix[i, j]:8.3f}" for j in range(n))
        logger.info("    %-6s: %s", row_name, row_str)

    return {
        "verdict":       verdict,
        "min_dist":      min_dist,
        "max_dist":      max_dist,
        "dist_matrix":   dist_matrix,
        "seq_names":     seq_names,
        "final_states":  final_states,
        "trajectories":  trajs,
        "zero_dists":    zero_dists,
    }


# ── 报告保存 ────────────────────────────────────────────────

def save_report(stab: dict, sep: dict, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "stability": {
            "verdict":      stab["verdict"],
            "max_diff":     stab["max_diff"],
            "per_dim_diff": stab["per_dim_diff"].tolist(),
        },
        "separability": {
            "verdict":      sep["verdict"],
            "min_dist":     sep["min_dist"],
            "max_dist":     sep["max_dist"],
            "dist_matrix":  sep["dist_matrix"].tolist(),
            "seq_names":    sep["seq_names"],
            "zero_dists":   sep["zero_dists"],
        },
    }
    path = report_dir / "l4_trajectory_stability_report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("报告已保存到 %s", path)
    return path


# ── 主函数 ──────────────────────────────────────────────────

def main():
    cfg             = _load_config()
    default_port    = cfg.get("port", 9000)
    default_timeout = cfg.get("connect_timeout", 60)
    l4_cfg          = cfg.get("l4", {})

    parser = argparse.ArgumentParser(description="L4 连续序列稳定性与区分性验证")
    parser.add_argument("--port",          type=int,   default=default_port)
    parser.add_argument("--timeout",       type=float, default=default_timeout)
    parser.add_argument("--steps",         type=int,   default=l4_cfg.get("n_steps", 500),
                        help="序列长度（默认500步=10秒@50Hz）")
    parser.add_argument("--amplitude",     type=float, default=l4_cfg.get("amplitude", 300.0),
                        help="动作幅值（默认300）")
    parser.add_argument("--trials",        type=int,   default=l4_cfg.get("n_trials", 5),
                        help="稳定性重复轮次（默认5）")
    parser.add_argument("--sep-threshold", type=float, default=l4_cfg.get("sep_threshold", 1.0),
                        help="区分性阈值：终态最小欧式距离（默认1.0）")
    parser.add_argument("--warmup-steps",  type=int,   default=l4_cfg.get("warmup_steps", 500))
    parser.add_argument("--action-mode",   type=str,   default="random",
                        choices=["constant", "random"],
                        help="动作模式: constant=恒定下压, random=固定种子随机序列（默认random）")
    parser.add_argument("--no-launch",     action="store_true")
    parser.add_argument("--no-plot",       action="store_true")
    parser.add_argument("--no-save",       action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.no_launch:
        launch_game()
        logger.info("等待游戏启动（最多 %.0fs）...", args.timeout)

    with GoiEnv(port=args.port, num_agents=2, timeout=args.timeout) as env:
        warmup_and_snapshot(env, args.warmup_steps)
        stab = test_stability(env, args.steps, args.amplitude, args.trials, args.action_mode)
        sep  = test_separability(env, args.steps, args.amplitude, args.sep_threshold)

    # ── 总结 ─────────────────────────────────────────────────
    logger.info("=== L4 总结 ===")
    logger.info("  A. 稳定性: %s  (max_diff=%.2e，阈值=%.0e)",
                stab["verdict"], stab["max_diff"], STABILITY_PASS)
    logger.info("  B. 区分性: %s  (min_dist=%.4f，阈值=%.3f)",
                sep["verdict"], sep["min_dist"], args.sep_threshold)

    overall = "通过" if stab["verdict"] == "通过" and sep["verdict"] == "通过" else "失败"
    logger.info("  综合判断: %s", overall)

    if not args.no_save:
        save_report(stab, sep, _REPORT_DIR)

    if not args.no_plot:
        # A - 稳定性：3条轨迹应完全重合
        plot_trajectories(
            stab["trajectories"],
            [f"trial_{i+1}" for i in range(len(stab["trajectories"]))],
            dims=[0, 1],
            title="L4-A Stability: player_x/y across trials (should overlap perfectly)",
        )
        plot_dim_max_diff(
            stab["per_dim_diff"],
            threshold=STABILITY_PASS,
            title="L4-A Stability: Per-Dimension Final-State Max Diff",
        )

        # B - 区分性：5条轨迹应明显分离
        plot_trajectories(
            [sep["trajectories"][n] for n in sep["seq_names"]],
            sep["seq_names"],
            dims=[0, 1],
            title="L4-B Separability: player_x/y for 5 action sequences (should diverge)",
        )
        plot_dist_heatmap(
            sep["dist_matrix"],
            sep["seq_names"],
            threshold=args.sep_threshold,
            title="L4-B Separability: Pairwise Final-State Distance Matrix",
        )


if __name__ == "__main__":
    main()

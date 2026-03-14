"""
层次 3：可重复性验证

验证相同动作序列在多次 reset() 后产生字节级一致的轨迹。
若轨迹高度一致（max_diff < 1e-4），说明物理是确定性的，控制测量可信。

判断标准：
  通过：max_diff < 1e-4
  警告：1e-4 ≤ max_diff < 1e-2
  失败：max_diff ≥ 1e-2

用法：
  python src/tests/control_interaction/test_l3_repeatability.py
  python src/tests/control_interaction/test_l3_repeatability.py --no-launch --trials 3
"""

import argparse
import csv
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
)

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent / "test_config.json"
_CSV_PATH = _REPO_ROOT / "src" / "Data" / "input_commands.csv"

STATE_LABELS = [
    "player_x", "player_y", "vel_x", "vel_y", "ang_vel",
    "hub_x", "hub_y", "hub_vx", "hub_vy", "hub_angle",
    "slider_x", "slider_y", "slider_vx", "slider_vy", "slider_angle",
    "handle_x", "handle_y", "handle_vx", "handle_vy",
    "pole_x", "pole_y", "pole_vx", "pole_vy",
    "tip_x", "tip_y", "tip_vx", "tip_vy",
    "hammer_angle", "timestamp",
]

PASS_THRESHOLD = 1e-4
WARN_THRESHOLD = 1e-2

# 排除非物理维度（timestamp 随时间累积，不能用于可重复性判断）
EXCLUDE_DIMS = {28}  # dim 28 = timestamp


def load_recorded_actions(repeats: int = 1) -> np.ndarray:
    """
    读取 input_commands.csv（人类实际操作记录），转换为动作序列 shape (N, 1, 2)。
    repeats：重复几遍（默认1遍=300步）。
    CSV 列：timestamp, mouseXdelta, mouseYdelta，delta 值已在 [-100,100] 范围内。
    """
    rows = []
    with open(_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dx = float(row["mouseXdelta"])
            dy = float(row["mouseYdelta"])
            rows.append([dx, dy])
    arr = np.array(rows, dtype=np.float32)   # (300, 2)
    arr = np.tile(arr, (repeats, 1))          # (300*repeats, 2)
    return arr[:, np.newaxis, :]             # (N, 1, 2)


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


def warmup_and_snapshot(env: GoiEnv, warmup_steps: int, csv_repeats: int = 2):
    """
    Warmup 分三阶段：
    1. 零动作步进（warmup_steps 步）：等待进入动画结束、物理稳定
    2. 回放人类录制动作（csv_repeats 遍 × 300 步）：把 player 推到有移动的位置
    3. 调用 new_snapshot()：以当前位置为新的 Reset 基准点
    """
    zero = np.array([[0.0, 0.0]], dtype=np.float32)

    logger.info("  Warmup 阶段1：%d 步零动作等待动画结束...", warmup_steps)
    env.reset()
    for _ in range(warmup_steps):
        env.step(zero)
    y0 = env.reset()[0, 1]
    x0 = env.reset()[0, 0]
    logger.info("  动画结束，player=(%.4f, %.4f)", x0, y0)

    recorded = load_recorded_actions(repeats=csv_repeats)
    total = len(recorded)
    logger.info("  Warmup 阶段2：回放人类录制动作 %d 步（%d 遍）...", total, csv_repeats)
    for a in recorded:
        env.step(a)

    obs = env.new_snapshot()
    xf, yf = obs[0, 0], obs[0, 1]
    logger.info(
        "  Warmup 完成，新基准 player=(%.4f, %.4f)  Δx=%+.4f  Δy=%+.4f",
        xf, yf, xf - x0, yf - y0,
    )


def collect_trajectory(env: GoiEnv, action_seq: np.ndarray) -> np.ndarray:
    """
    执行一轮固定动作序列，返回完整轨迹 shape (n_steps+1, STATE_DIM)。
    action_seq: (n_steps, 1, 2)
    """
    obs = env.reset()
    traj = [obs[0].copy()]
    for a in action_seq:
        obs, _ = env.step(a)
        traj.append(obs[0].copy())
    return np.stack(traj)


def run_repeatability(env: GoiEnv, n_trials: int) -> dict:
    """
    重复 n_trials 轮，每轮回放人类录制动作序列，比较轨迹差异。
    返回详细分析结果。
    """
    action_seq = load_recorded_actions(repeats=1)  # 300 步人类操作
    n_steps = len(action_seq)
    logger.info("=== 层次 3：可重复性验证（%d 轮 × %d 步，人类录制动作）===", n_trials, n_steps)

    trajectories = []
    for i in range(n_trials):
        traj = collect_trajectory(env, action_seq)
        dx = traj[-1, 0] - traj[0, 0]
        dy = traj[-1, 1] - traj[0, 1]
        logger.info("  轮次 %d 完成  player=(%.4f, %.4f)  Δx=%+.4f  Δy=%+.4f",
                    i + 1, traj[-1, 0], traj[-1, 1], dx, dy)
        trajectories.append(traj)

    trajectories = np.stack(trajectories)  # (n_trials, T, STATE_DIM)

    # 各维度跨所有轮次的最大差异（排除非物理维度）
    per_dim_diff = np.zeros(STATE_DIM)
    for d in range(STATE_DIM):
        if d in EXCLUDE_DIMS:
            per_dim_diff[d] = 0.0
        else:
            per_dim_diff[d] = float(np.abs(trajectories[:, :, d] - trajectories[0:1, :, d]).max())

    # 仅在物理维度上计算全局最大差异
    physics_dims = [d for d in range(STATE_DIM) if d not in EXCLUDE_DIMS]
    ref = trajectories[0]
    per_trial_max = []
    for i in range(1, n_trials):
        diff = np.abs(trajectories[i][:, physics_dims] - ref[:, physics_dims])
        per_trial_max.append(float(diff.max()))

    max_diff = float(max(per_trial_max)) if per_trial_max else 0.0

    worst_dim = int(np.argmax(per_dim_diff))
    worst_label = STATE_LABELS[worst_dim] if worst_dim < len(STATE_LABELS) else f"dim_{worst_dim}"

    if max_diff < PASS_THRESHOLD:
        verdict = "通过"
    elif max_diff < WARN_THRESHOLD:
        verdict = "警告"
    else:
        verdict = "失败"

    logger.info("--- 结果 ---")
    logger.info("  全局最大差异: %.2e  判断: %s", max_diff, verdict)
    logger.info("  最差维度: [%d] %s = %.2e", worst_dim, worst_label, per_dim_diff[worst_dim])

    bad_dims = [(i, STATE_LABELS[i] if i < len(STATE_LABELS) else f"dim_{i}", float(per_dim_diff[i]))
                for i in range(STATE_DIM) if per_dim_diff[i] > PASS_THRESHOLD]
    if bad_dims:
        logger.info("  超过阈值的维度：")
        for idx, lbl, val in bad_dims:
            logger.info("    [%2d] %-20s %.2e", idx, lbl, val)
    else:
        logger.info("  所有维度差异 < %.0e  [字节级一致]", PASS_THRESHOLD)

    if verdict == "失败":
        logger.warning("  [诊断] 检查 stepFrames 配置，确认 Physics2D.simulationMode=Script")

    return {
        "verdict": verdict,
        "max_diff": max_diff,
        "per_dim_diff": per_dim_diff.tolist(),
        "worst_dim": worst_dim,
        "worst_dim_label": worst_label,
        "trajectories": trajectories,
    }


def main():
    cfg = _load_config()
    default_port    = cfg.get("port", 9000)
    default_timeout = cfg.get("connect_timeout", 60)
    l3_cfg          = cfg.get("l3", {})

    parser = argparse.ArgumentParser(description="L3 可重复性验证（人类录制动作）")
    parser.add_argument("--port",          type=int,   default=default_port)
    parser.add_argument("--timeout",       type=float, default=default_timeout)
    parser.add_argument("--trials",        type=int,   default=l3_cfg.get("n_trials", 5))
    parser.add_argument("--warmup-steps",  type=int,   default=l3_cfg.get("warmup_steps", 500),
                        help="零动作等待动画结束的步数（默认500步≈4秒）")
    parser.add_argument("--csv-repeats",   type=int,   default=2,
                        help="回放人类录制动作的遍数（默认2遍=600步）")
    parser.add_argument("--no-launch",     action="store_true")
    parser.add_argument("--no-plot",       action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.no_launch:
        launch_game()
        logger.info("等待游戏启动（最多 %.0fs）...", args.timeout)

    # L3 可重复性保护：启用 RewiredMouseOverride，屏蔽真实鼠标
    rmo = l3_cfg.get("rewired_mouse_override", {})
    rmo_active = rmo.get("active", True)
    rmo_mouse_x = rmo.get("mouse_x_action_id", -1)
    rmo_mouse_y = rmo.get("mouse_y_action_id", -1)

    with GoiEnv(port=args.port, num_agents=1, timeout=args.timeout) as env:
        env.config_rewired_mouse(
            active=rmo_active,
            mouse_x_action_id=rmo_mouse_x,
            mouse_y_action_id=rmo_mouse_y,
        )
        logger.info(
            "RewiredMouseOverride 已配置: active=%s mouse_x=%d mouse_y=%d",
            rmo_active, rmo_mouse_x, rmo_mouse_y,
        )
        warmup_and_snapshot(env, args.warmup_steps, csv_repeats=args.csv_repeats)
        result = run_repeatability(env, args.trials)

    if not args.no_plot:
        trajs = result["trajectories"]
        traj_list   = [trajs[i] for i in range(len(trajs))]
        traj_labels = [f"trial_{i+1}" for i in range(len(trajs))]

        plot_trajectories(
            traj_list, traj_labels,
            dims=[0, 1],
            title="L3 Repeatability: player_x/y across trials (should overlap perfectly)",
        )
        plot_dim_max_diff(
            np.array(result["per_dim_diff"]),
            threshold=PASS_THRESHOLD,
            title="L3 Repeatability: Per-Dimension Max Diff",
        )


if __name__ == "__main__":
    main()

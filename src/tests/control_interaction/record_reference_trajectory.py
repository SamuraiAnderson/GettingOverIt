"""
参考轨迹录制 — 帧级步进 + 真实鼠标，人工示范一条可用于 BC 的轨迹。

与训练通道完全一致：GameRuntime 模式、物理 Script 模式、单 agent。
每步注入动作 [-100, 100] → STEP，记录 (state, action)。物理只在 STEP 时推进。

操控方式（delta 帧级相对位移）：
  每帧读鼠标相对屏幕中心的位移×灵敏度当动作注入，读后回中。为改善「飘」的手感，
  对动作做指数平滑(EMA)：action = smooth*上一动作 + (1-smooth)*本帧动作。
  smooth 越大越稳、越跟手滞后；fps 越低越像慢动作、越好精细操控（如开局过树）。

Windows 专用：ctypes 读取全局光标。
按 ESC 结束录制（也可 Ctrl+C）。

用法:
  python src/tests/control_interaction/record_reference_trajectory.py                      # 默认
  python src/tests/control_interaction/record_reference_trajectory.py --fps 20 --sensitivity 1.5 --smooth 0.5
  python src/tests/control_interaction/record_reference_trajectory.py --no-launch          # 游戏已在运行

输出（落在 src/Data/reference/，随仓库入库：人工示范无法重新生成，不能放进 gitignore 的目录）:
  src/Data/reference/reference_trajectory.json    可读轨迹(states/actions/报告)，便于检查
  src/Data/reference/reference_trajectory.pkl     BC 可直接加载(结构同 warmup_state.pkl)
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import pickle
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from training.config import TrainConfig

logger = logging.getLogger(__name__)

_PROJECT_CONFIG = _REPO_ROOT / "src" / "config" / "project.json"

# Win32 常量
_VK_ESCAPE = 0x1B
_SM_CXSCREEN = 0
_SM_CYSCREEN = 1


def _get_game_root() -> Path:
    with open(_PROJECT_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return Path(cfg["game"]["executable_path"]).parent


class MouseCapture:
    """相对鼠标捕获：每帧读取光标相对屏幕中心的位移，然后回中。"""

    def __init__(self) -> None:
        self._user32 = ctypes.windll.user32
        self.anchor_x = self._user32.GetSystemMetrics(_SM_CXSCREEN) // 2
        self.anchor_y = self._user32.GetSystemMetrics(_SM_CYSCREEN) // 2

    def recenter(self) -> None:
        self._user32.SetCursorPos(self.anchor_x, self.anchor_y)

    def read_delta(self) -> tuple[int, int]:
        pt = wintypes.POINT()
        self._user32.GetCursorPos(ctypes.byref(pt))
        dx = pt.x - self.anchor_x
        dy = pt.y - self.anchor_y
        self.recenter()
        return dx, dy

    def escape_pressed(self) -> bool:
        return bool(self._user32.GetAsyncKeyState(_VK_ESCAPE) & 0x8000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="录制人工参考轨迹（帧级步进 + 鼠标）")
    parser.add_argument("--steps", type=int, default=1000, help="最大录制步数（ESC 可提前结束）")
    parser.add_argument("--fps", type=float, default=15.0, help="录制帧率上限（越低越像慢动作，越好精细操控过树；默认 15，可 --fps 30 加快）")
    parser.add_argument(
        "--sensitivity", type=float, default=1.0,
        help="鼠标灵敏度：1 像素位移 → sensitivity 个动作单位（动作范围 ±100）",
    )
    parser.add_argument(
        "--smooth", type=float, default=0.4,
        help="动作指数平滑系数(0~0.95)：越大越稳但越滞后；0=不平滑",
    )
    parser.add_argument(
        "--invert-y", action="store_true", default=True,
        help="反转 Y 轴（鼠标上移=动作正，符合直觉；默认开启）",
    )
    parser.add_argument("--no-invert-y", dest="invert_y", action="store_false")
    parser.add_argument("--countdown", type=int, default=3, help="开始录制前的倒计时秒数")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--output", type=str, default="src/Data/reference/reference_trajectory.json",
        help="输出轨迹 JSON 路径（同名 .pkl 为 BC 可加载版本）",
    )
    parser.add_argument("--no-launch", action="store_true", help="不启动游戏（假设已运行）")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    config = TrainConfig()
    config.num_agents = 1
    scale = config.action_scale

    from env.goi_env import GoiEnv
    from start.game_launcher import GameLauncher
    from start.game_mode_controller import GameModeController
    from training.rollout import _write_num_duplicates

    game_root = _get_game_root()
    process = None

    if not args.no_launch:
        _write_num_duplicates(game_root, 1)
        GameModeController(str(game_root)).set_game_runtime_mode()
        process = GameLauncher().launch(wait=False)
        logger.info("游戏已启动")

    env = GoiEnv(port=config.port, num_agents=1)
    env.connect()
    env.reset()

    zero_actions = np.zeros((1, 2), dtype=np.float32)
    for _ in range(args.warmup_steps):
        env.step(zero_actions)
    env.new_snapshot()
    env.toggle_collider_visual(False)
    logger.info("Warmup 完成")

    obs = env.reset()
    all_states = [obs[0].copy()]
    all_actions: list[np.ndarray] = []

    start_y = float(obs[0, 1])
    max_y = start_y

    mouse = MouseCapture()
    sign_y = -1.0 if args.invert_y else 1.0
    frame_dt = 1.0 / args.fps if args.fps > 0 else 0.0
    smooth = float(np.clip(args.smooth, 0.0, 0.95))
    prev_ax, prev_ay = 0.0, 0.0

    logger.info("=" * 60)
    logger.info("准备录制[delta 相对位移]: 最多 %d 步, fps=%.0f, sensitivity=%.2f, smooth=%.2f, invert_y=%s",
                args.steps, args.fps, args.sensitivity, smooth, args.invert_y)
    logger.info("请把注意力放到游戏窗口，用鼠标操控锤子。按 ESC 结束。")
    for c in range(args.countdown, 0, -1):
        logger.info("  %d ...", c)
        time.sleep(1.0)
    logger.info("开始录制! 起始 y=%.2f", start_y)
    logger.info("=" * 60)

    mouse.recenter()

    try:
        for step in range(args.steps):
            loop_start = time.perf_counter()

            # delta：帧级相对位移 → 动作，再做指数平滑(EMA)改善手感，最后裁剪到 ±scale
            dx, dy = mouse.read_delta()
            raw_ax = dx * args.sensitivity
            raw_ay = sign_y * dy * args.sensitivity
            ax = smooth * prev_ax + (1.0 - smooth) * raw_ax
            ay = smooth * prev_ay + (1.0 - smooth) * raw_ay
            ax = float(np.clip(ax, -scale, scale))
            ay = float(np.clip(ay, -scale, scale))
            prev_ax, prev_ay = ax, ay
            action = np.array([ax, ay], dtype=np.float32)

            obs, _ = env.step(action.reshape(1, 2))

            all_states.append(obs[0].copy())
            all_actions.append(action.copy())

            cur_y = float(obs[0, 1])
            if cur_y > max_y:
                max_y = cur_y

            if (step + 1) % 20 == 0:
                logger.info(
                    "  step %4d/%d | pos=(%.2f, %.2f) | action=(%.1f, %.1f) | max_y=%.2f (Δ=%.2f)",
                    step + 1, args.steps,
                    float(obs[0, 0]), cur_y, ax, ay, max_y, max_y - start_y,
                )

            if mouse.escape_pressed():
                logger.info("检测到 ESC，结束录制。已录 %d 步。", len(all_actions))
                break

            if frame_dt > 0:
                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_dt:
                    time.sleep(frame_dt - elapsed)

    except KeyboardInterrupt:
        logger.info("用户中断(Ctrl+C)，已录 %d 步。", len(all_actions))

    finally:
        env.close()
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                pass

    if not all_actions:
        logger.warning("没有录到任何步，未保存。")
        return

    states_arr = np.array(all_states, dtype=np.float32)      # (T+1, 33)
    actions_arr = np.array(all_actions, dtype=np.float32)    # (T, 2)
    final_y = float(states_arr[-1, 1])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "source": "human_reference",
        "mode": "delta",
        "steps": len(all_actions),
        "start_y": round(start_y, 3),
        "max_y": round(max_y, 3),
        "final_y": round(final_y, 3),
        "sensitivity": args.sensitivity,
        "smooth": smooth,
        "invert_y": args.invert_y,
        "states": states_arr.tolist(),
        "actions": actions_arr.tolist(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    logger.info("轨迹 JSON 已保存: %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # BC 可加载版本（结构同 warmup_state.pkl：{"trajectories": [{raw_states, actions, score, iteration}]}）
    pkl_path = out_path.with_suffix(".pkl")
    bc_state = {
        "trajectories": [
            {
                "raw_states": states_arr,
                "actions": actions_arr,
                "score": 0.0,
                "iteration": 0,
            }
        ]
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(bc_state, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("BC 可加载版本已保存: %s", pkl_path)

    logger.info("=" * 60)
    logger.info("录制报告")
    logger.info("  步数: %d", len(all_actions))
    logger.info("  起始 y: %.2f", start_y)
    logger.info("  最高 y: %.2f  (Δ=%.2f)", max_y, max_y - start_y)
    logger.info("  最终 y: %.2f  (Δ=%.2f)", final_y, final_y - start_y)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

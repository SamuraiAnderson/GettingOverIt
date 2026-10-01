"""Replay stored rank1 then rank2 actions on virtual desktop 2, 8 players."""
import ctypes
import logging
from ctypes import wintypes
from pathlib import Path
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import start.game_launcher as game_launcher
from training.config import TrainConfig
from training.rollout import RolloutWorker
from training.run_logging import load_elites

ELITES = ROOT / "runs/sparse_geometry_20260926_S_s47/elites/elites_iter_0118.npz"
RANK1 = 17
RANK2 = 16
PLAYERS = 8
STEP_HZ = 30.0
HOLD_S = 8.0

_orig_launch = game_launcher.GameLauncher.launch
_pid_box: dict = {}


def _launch_on_interactive(self, wait=False, *, desktop=game_launcher._ISOLATED_DESKTOP):
    process = _orig_launch(self, wait=wait, desktop=None)
    _pid_box["pid"] = int(process.pid)
    return process


def _hwnds(pid: int) -> list[int]:
    user32 = ctypes.windll.user32
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        proc = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc))
        if proc.value == pid and user32.IsWindowVisible(hwnd):
            found.append(int(hwnd))
        return True

    user32.EnumWindows(callback, 0)
    return found


def _pin_desktop2(pid_box: dict, stop: threading.Event) -> None:
    from pyvda import AppView, VirtualDesktop

    target = VirtualDesktop(2)
    pinned: set[int] = set()
    while not stop.is_set():
        pid = pid_box.get("pid")
        if pid:
            for hwnd in _hwnds(pid):
                if hwnd in pinned:
                    continue
                try:
                    AppView(hwnd).move(target)
                    pinned.add(hwnd)
                    print(f"窗口 {hwnd} 已放到虚拟桌面 2", flush=True)
                except Exception as exc:
                    print(f"移动窗口失败: {exc}", flush=True)
        stop.wait(0.2)


def _play(worker: RolloutWorker, actions: np.ndarray, label: str) -> None:
    period = 1.0 / STEP_HZ
    zeros = np.zeros((PLAYERS, 2), np.float32)
    print(f"开始 {label}，{len(actions)} 步，8 个 player 同一动作", flush=True)
    for step, action in enumerate(actions):
        started = time.perf_counter()
        batch = np.repeat(action.reshape(1, 2), PLAYERS, axis=0)
        worker.env.step(batch)
        remain = period - (time.perf_counter() - started)
        if remain > 0:
            time.sleep(remain)
        if step % 120 == 0:
            print(f"{label} step {step}", flush=True)
    worker.env.step(zeros)
    print(f"{label} 结束", flush=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    game_launcher.GameLauncher.launch = _launch_on_interactive
    rows = load_elites(ELITES)
    rank1 = np.asarray(rows[RANK1]["actions"], np.float32)
    rank2 = np.asarray(rows[RANK2]["actions"], np.float32)
    cfg = TrainConfig()
    cfg.num_agents = PLAYERS
    cfg.random_deploy = False
    game_root = Path(cfg.game_root)
    runtime = game_root / "GoiData" / "runtime_config.json"
    original = runtime.read_bytes() if runtime.exists() else None
    worker = RolloutWorker(cfg)
    stop = threading.Event()
    pinner = threading.Thread(target=_pin_desktop2, args=(_pid_box, stop), daemon=True)
    pinner.start()
    print("虚拟桌面 2，8 players。先整组重放 rank1，再整组重放 rank2。", flush=True)
    try:
        worker.launch_game()
        worker.env.set_camera_free(False)
        _play(worker, rank1, "rank1")
        worker.env.reset()
        time.sleep(1.0)
        _play(worker, rank2, "rank2")
        print(f"停在最后一帧 {HOLD_S:.0f} 秒", flush=True)
        time.sleep(HOLD_S)
        print("replay finished", flush=True)
    finally:
        stop.set()
        worker.teardown()
        if original is None:
            if runtime.exists():
                runtime.unlink()
        else:
            runtime.write_bytes(original)


if __name__ == "__main__":
    main()

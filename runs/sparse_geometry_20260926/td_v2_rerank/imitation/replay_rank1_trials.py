"""Open-loop rank1 actions, 16 players, 10 resets. Score pot error against the demo."""
import ctypes
import json
import logging
from ctypes import wintypes
from pathlib import Path
import sys
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

import start.game_launcher as game_launcher
from training.config import TrainConfig
from training.reference_anchor import reference_position, scoring_context
from training.reference_corridor import replay_corridor_result
from training.reward import is_water
from training.rollout import RolloutWorker
from training.run_logging import load_elites

ELITES = ROOT / "runs/sparse_geometry_20260926_S_s47/elites/elites_iter_0118.npz"
ROUTE = ROOT / "src/Data/video_reference/37258923076-1-192/reference_trajectory_game_world.json"
OUT = Path(__file__).resolve().parent / "rank1_replay_16x10.json"
RANK1 = 17
PLAYERS = 16
TRIALS = 10
STAR = 800
MATCH_M = 1.5

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


def _pin_desktop2(stop: threading.Event) -> None:
    from pyvda import AppView, VirtualDesktop

    target = VirtualDesktop(2)
    pinned: set[int] = set()
    while not stop.is_set():
        pid = _pid_box.get("pid")
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


def _pots(route, states, schema, meta) -> np.ndarray:
    return reference_position(states, schema, meta)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    game_launcher.GameLauncher.launch = _launch_on_interactive
    row = load_elites(ELITES)[RANK1]
    actions = np.asarray(row["actions"], np.float32)
    demo_states = np.asarray(row["raw_states"], np.float64)
    if len(demo_states) != len(actions) + 1 or STAR >= len(demo_states):
        raise RuntimeError(f"rank1 shape states={demo_states.shape} actions={actions.shape}")

    cfg = TrainConfig()
    cfg.num_agents = PLAYERS
    cfg.random_deploy = False
    cfg.reference_trajectory = str(ROUTE)
    worker = RolloutWorker(cfg)
    route = worker.reference_corridor
    demo_schema, demo_meta = scoring_context(demo_states[0], cfg, None, route)
    demo = replay_corridor_result(route, demo_states, demo_schema, demo_meta)
    demo_pots = _pots(route, demo_states, demo_schema, demo_meta)
    star_pot = demo_pots[STAR]
    print(
        f"demo D={demo['distance']:.3f} cap={demo['cap']} star_pot=({star_pot[0]:.3f},{star_pot[1]:.3f})",
        flush=True,
    )

    game_root = Path(cfg.game_root)
    runtime = game_root / "GoiData" / "runtime_config.json"
    original = runtime.read_bytes() if runtime.exists() else None
    report = {
        "players": PLAYERS,
        "trials": TRIALS,
        "actions": int(len(actions)),
        "star_index": STAR,
        "match_m": MATCH_M,
        "success": "star-frame pot within match_m of the demo, valid, and not in water",
        "demo_D": demo["distance"],
        "demo_cap": demo["cap"],
        "demo_star_pot": [float(star_pot[0]), float(star_pot[1])],
        "runs": [],
    }
    stop = threading.Event()
    pinner = threading.Thread(target=_pin_desktop2, args=(stop,), daemon=True)
    pinner.start()
    try:
        worker.launch_game()
        worker.env.set_camera_free(False)
        for trial in range(TRIALS):
            obs = worker.env.reset()
            schema, meta = scoring_context(obs[0], cfg, worker.env, route)
            trackers = [
                route.tracker_from_state(obs[i], schema, meta) for i in range(PLAYERS)
            ]
            start_err = np.linalg.norm(_pots(route, obs, schema, meta) - demo_pots[0], axis=1)
            water = np.zeros(PLAYERS, dtype=bool)
            diverge = np.full(PLAYERS, -1, dtype=np.int32)
            star_err = np.full(PLAYERS, np.nan)
            for step, action in enumerate(actions):
                batch = np.repeat(action.reshape(1, 2), PLAYERS, axis=0)
                obs, _dones = worker.env.step(batch)
                pots = _pots(route, obs, schema, meta)
                err = np.linalg.norm(pots - demo_pots[step + 1], axis=1)
                for i in range(PLAYERS):
                    route.step_state(trackers[i], obs[i], schema, meta)
                    if is_water(float(obs[i, 1]), cfg):
                        water[i] = True
                    if diverge[i] < 0 and err[i] > MATCH_M:
                        diverge[i] = step + 1
                if step + 1 == STAR:
                    star_err = err.copy()
            end_err = np.linalg.norm(_pots(route, obs, schema, meta) - demo_pots[-1], axis=1)
            agents = []
            for i, tracker in enumerate(trackers):
                ok = bool(
                    np.isfinite(star_err[i]) and star_err[i] <= MATCH_M
                    and not tracker.invalid and not water[i]
                )
                agents.append({
                    "agent": i,
                    "D": float(tracker.confirmed_d),
                    "cap": int(tracker.last_credit_action),
                    "valid": not tracker.invalid,
                    "water": bool(water[i]),
                    "star_err_m": float(star_err[i]),
                    "end_err_m": float(end_err[i]),
                    "diverge_step": int(diverge[i]),
                    "success": ok,
                })
            n_ok = sum(a["success"] for a in agents)
            ds = [a["D"] for a in agents]
            line = {
                "trial": trial,
                "start_err_median_m": float(np.median(start_err)),
                "success": n_ok,
                "D_median": float(np.median(ds)),
                "D_max": float(np.max(ds)),
                "star_err_median_m": float(np.nanmedian(star_err)),
                "star_err_min_m": float(np.nanmin(star_err)),
                "agents": agents,
            }
            report["runs"].append(line)
            OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(
                f"trial {trial}: success {n_ok}/{PLAYERS}  "
                f"D med {line['D_median']:.2f} max {line['D_max']:.2f}  "
                f"star err med {line['star_err_median_m']:.2f} min {line['star_err_min_m']:.2f}",
                flush=True,
            )
        hits = sum(a["success"] for run in report["runs"] for a in run["agents"])
        print(f"replay finished  success {hits}/{PLAYERS * TRIALS}", flush=True)
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

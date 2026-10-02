"""Visible cold-start policy test using eight independent native physics scenes."""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/config/project.json").is_file())
sys.path.insert(0, str(ROOT / "src/tests/control_interaction"))
from test_reset_state import ensure_idle, runtime_settings, save_report, sha256
from start.game_launcher import GameLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / ("policy_0269_isolated_8players_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--hidden", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.fps < 0:
        raise ValueError("Positive step count and nonnegative FPS required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8-sig"))
    game = ROOT / "artifacts/reset_game"
    production = Path(project["game"]["executable_path"]).parent
    checkpoint = ROOT / "runs/optimizer_momentum_fix_20261001/checkpoints/ppo_iter_0269.pt"
    experiment_dll = ROOT / "bin/WorldIsolationExperiment/IsolatedWorldExperiment.dll"
    for required in (checkpoint, experiment_dll, game / "BepInEx/plugins/GameRuntime_v2.dll",
                     Path(__file__).with_name("IsolatedWorldExperiment.cs"),
                     production / "GoiData/Colliders/player_contour.json", production / "GoiData/anchor_calibration.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    ensure_idle(int(project["tcp_port"]))
    args.out.mkdir(parents=True, exist_ok=False)
    for relative in ("GoiData/Colliders/player_contour.json", "GoiData/anchor_calibration.json"):
        shutil.copyfile(production / relative, game / relative)
    request_path = game / "GoiData/isolation_experiment.json"
    plugin_path = game / "BepInEx/plugins/IsolatedWorldExperiment.dll"
    if request_path.exists() or plugin_path.exists():
        raise RuntimeError("Existing isolation experiment deployment must be handled before this run")
    shutil.copyfile(experiment_dll, plugin_path)
    request = dict(output=str(args.out.resolve()), worlds=8, resets=0, livePolicy=True, liveSteps=args.steps)
    request_path.write_text(json.dumps(request), encoding="utf-8")
    save_report(args.out / "request.json", request)
    for filename in ("IsolatedWorldExperiment.cs", "run_policy_isolated_8players.py"):
        shutil.copyfile(Path(__file__).with_name(filename), args.out / filename)
    shutil.copyfile(plugin_path, args.out / plugin_path.name)
    overrides = dict(suppressHubDepthWrite=True, experimentalControlledUpdates=True, experimentalSceneReset=True,
                     experimentalNativeNewGame=True, experimentalNativeInputSmoothing=True,
                     experimentalResetDiagnostics=True, stepFrames=1, dynamicsProbeMode="", dynamicsProbeFile="")
    report = dict(status="STARTING", checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), agents=8,
                  physics_layout="8 independent native Mian scenes; one original player per world",
                  initialization="live cold start; no reset, warmup, snapshot or teleport before policy",
                  known_boundary="Extra world initialization synchronization changes trajectory relative to old single-world reference",
                  steps=args.steps, runtime_overrides=overrides, plugin_sha256=sha256(game / "BepInEx/plugins/GameRuntime_v2.dll"),
                  experiment_sha256=sha256(plugin_path), fps_limit=args.fps)
    proc = None
    started = time.monotonic()
    try:
        with runtime_settings(game, 1, int(project["tcp_port"]), overrides):
            launcher = GameLauncher(str(game / "GettingOverIt.exe"))
            proc = launcher.launch() if args.hidden else launcher.launch(desktop=None)
            report.update(game_pid=proc.pid, status="LOADING_WORLDS_AND_POLICY")
            save_report(args.out / "run_status.json", report)
            import numpy as np
            import torch
            from training import local_evaluation as evaluation
            from env import make_goi_env
            torch.set_num_threads(4)

            class ColdWorldWorker(evaluation.RolloutWorker):
                def __init__(self, config):
                    config.game_root = str(game)
                    config.warmup_steps = 0
                    super().__init__(config)
                    self.cold_used = False

                def launch_game(self):
                    ready_path = args.out / "worlds_ready.json"
                    while not ready_path.exists():
                        if proc.poll() is not None:
                            raise RuntimeError("Game exited during world initialization")
                        if (args.out / "finished.json").exists():
                            raise RuntimeError((args.out / "finished.json").read_text(encoding="utf-8-sig"))
                        if time.monotonic() - started > 150:
                            raise TimeoutError("Eight worlds failed to initialize")
                        time.sleep(.25)
                    self.cold = np.float32(json.loads(ready_path.read_text(encoding="utf-8-sig"))["states"])
                    assert self.cold.shape == (8, 35) and np.isfinite(self.cold).all()
                    worlds = json.loads((args.out / "worlds_epoch_0.json").read_text(encoding="utf-8-sig"))
                    assert len(worlds) == 8 and len({w["sceneHandle"] for w in worlds}) == 8
                    assert all(w["valid"] and not w["isDefault"] for w in worlds)
                    self.env = make_goi_env(self.config, timeout=120, startup_settle_s=0)
                    self.env.connect()
                    self.env._sock.settimeout(120)
                    assert self.env.state_dim == 35
                    self.env.set_camera_free(True)
                    report.update(status="RUNNING", policy_start_seconds=time.monotonic() - started, completed_steps=0,
                                  scene_handles=[w["sceneHandle"] for w in worlds])
                    save_report(args.out / "run_status.json", report)
                    real_step = self.env.step
                    def step(actions):
                        step_started = time.monotonic()
                        obs, done = real_step(actions)
                        assert obs.shape == (8, 35) and np.isfinite(obs).all()
                        report["completed_steps"] += 1
                        if report["completed_steps"] == 1 or report["completed_steps"] % 120 == 0:
                            report["player_y"] = obs[:, 1].tolist()
                            save_report(args.out / "run_status.json", report)
                            logging.info("POLICY 0269: %d/%d, 8 independent worlds, max y=%.3f", report["completed_steps"], args.steps, obs[:, 1].max())
                        if args.fps > 0:
                            time.sleep(max(0, 1 / args.fps - (time.monotonic() - step_started)))
                        return obs, done
                    self.env.step = step

                def _reset_and_deploy(self):
                    if self.cold_used:
                        raise RuntimeError("Only one live cold rollout is supported")
                    self.cold_used = True
                    self._last_deploy_positions = {}
                    return self.cold.copy()

                def teardown(self):
                    if self.env is not None:
                        self.env.close()

            evaluation.RolloutWorker = ColdWorldWorker
            sys.argv = ["local_evaluation", "--checkpoint", str(checkpoint), "--output", str(args.out),
                        "--steps", str(args.steps), "--rounds", "1", "--num-agents", "8", "--jitter", "0",
                        "--modes", "deterministic", "--seed", "47"]
            evaluation.main()
            closed_path = args.out / "closed.json"
            close_deadline = time.monotonic() + 15
            while not closed_path.exists() and time.monotonic() < close_deadline:
                time.sleep(.05)
            finished = json.loads((args.out / "finished.json").read_text(encoding="utf-8-sig"))
            closed = json.loads(closed_path.read_text(encoding="utf-8-sig"))
            if any(x["status"] != "COMPLETE" or x["steps"] != args.steps for x in (finished, closed)):
                raise RuntimeError(f"Incomplete runtime completion/close: {finished}, {closed}")
            report["runtime_close_verified"] = True
            report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
        raise
    finally:
        if args.hidden and proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
        request_path.unlink(missing_ok=True)
        try:
            plugin_path.unlink(missing_ok=True)
        except PermissionError:
            report["inactive_plugin_file_pending_cleanup"] = str(plugin_path)
        report.update(checkpoint_unchanged=sha256(checkpoint) == report["checkpoint_sha256"],
                      game_left_open=proc is not None and proc.poll() is None, wall_seconds=time.monotonic() - started)
        save_report(args.out / "run_status.json", report)
        log = game / "BepInEx/LogOutput.log"
        if log.exists():
            shutil.copyfile(log, args.out / "game.log")
        manifest_path = args.out / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest.update(initialization=report["initialization"], physics_layout=report["physics_layout"],
                            runtime_overrides=overrides, plugin_sha256=report["plugin_sha256"], experiment_sha256=report["experiment_sha256"])
            save_report(manifest_path, manifest)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Visible cold-start policy run using the standard multi-player runtime."""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/tests/control_interaction"))
from test_reset_state import ensure_idle, runtime_settings, save_report, sha256
from start.game_launcher import GameLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / ("policy_isolated_8players_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/optimizer_momentum_fix_20261001/checkpoints/ppo_iter_0269.pt")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--hidden", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.fps < 0: raise ValueError("Invalid step count or FPS")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8-sig"))
    game = ROOT / "artifacts/reset_game"
    production = Path(project["game"]["executable_path"]).parent
    ensure_idle(project["tcp_port"])
    args.out.mkdir(parents=True, exist_ok=False)
    for relative in ("GoiData/Colliders/player_contour.json", "GoiData/anchor_calibration.json"):
        shutil.copyfile(production / relative, game / relative)
    report = dict(status="STARTING", checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                  agents=8, physics_layout="isolated_worlds", initialization="cold; no warmup, reset or teleport",
                  steps=args.steps, completed_steps=0, plugin_sha256=sha256(game / "BepInEx/plugins/GameRuntime_v2.dll"))
    proc = None
    started = time.monotonic()
    try:
        with runtime_settings(game, 8, project["tcp_port"], dict(experimentalResetDiagnostics=True, dynamicsProbeMode="")):
            launcher = GameLauncher(str(game / "GettingOverIt.exe"))
            proc = launcher.launch() if args.hidden else launcher.launch(desktop=None)
            import torch
            from training import local_evaluation as evaluation
            from env import make_goi_env
            torch.set_num_threads(4)
            original_worker = evaluation.RolloutWorker

            class ColdWorker(original_worker):
                def launch_game(self):
                    self.config.game_root = str(game)
                    self.env = make_goi_env(self.config, timeout=180)
                    self.env.connect()
                    self.env._sock.settimeout(180)
                    assert self.env.hello["physics_backend"] == "isolated_worlds"
                    assert self.env.hello["physics_world_count"] == 8
                    self.cold = self.env.new_snapshot() # Read/save boundary; advances no physics.
                    self.env.set_camera_free(True)
                    report.update(status="RUNNING", game_pid=proc.pid, ready_seconds=time.monotonic()-started)
                    save_report(args.out / "diagnostics_0_0.json", self.env.reset_diagnostics())
                    real_step = self.env.step
                    def step(actions):
                        begin = time.monotonic()
                        states, done = real_step(actions)
                        report["completed_steps"] += 1
                        count = report["completed_steps"]
                        if count == 31 or count == args.steps:
                            save_report(args.out / f"diagnostics_0_{count}.json", self.env.reset_diagnostics())
                        if count % 120 == 0 or count == args.steps: save_report(args.out / "run_status.json", report)
                        if args.fps: time.sleep(max(0, 1 / args.fps - (time.monotonic()-begin)))
                        return states, done
                    self.env.step = step

                def _reset_and_deploy(self):
                    self._last_deploy_positions = {}
                    result, self.cold = self.cold, None
                    if result is None: raise RuntimeError("Cold runner supports one rollout")
                    return result.copy()

                def teardown(self):
                    if self.env is not None: self.env.close()

            previous_argv = sys.argv
            try:
                evaluation.RolloutWorker = ColdWorker
                sys.argv = ["local_evaluation", "--checkpoint", str(args.checkpoint), "--output", str(args.out),
                            "--steps", str(args.steps), "--rounds", "1", "--num-agents", "8", "--jitter", "0",
                            "--modes", "deterministic", "--seed", "47"]
                evaluation.main()
            finally:
                evaluation.RolloutWorker = original_worker
                sys.argv = previous_argv
            if report["completed_steps"] != args.steps: raise RuntimeError("Incomplete rollout")
            report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="ERROR", error=str(exc))
        raise
    finally:
        if (args.hidden or report["status"] != "COMPLETE") and proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
        report.update(wall_seconds=time.monotonic()-started, game_left_open=proc is not None and proc.poll() is None,
                      checkpoint_unchanged=sha256(args.checkpoint) == report["checkpoint_sha256"])
        save_report(args.out / "run_status.json", report)
        shutil.copyfile(game / "BepInEx/LogOutput.log", args.out / "game.log")


if __name__ == "__main__": main()

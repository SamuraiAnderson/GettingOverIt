"""File-driven independent-scene experiment; uses existing native startup and stepping."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/config/project.json").is_file())
sys.path.insert(0, str(ROOT / "src/tests/control_interaction"))
from test_reset_state import ensure_idle, runtime_settings, save_report, sha256
from start.game_launcher import GameLauncher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--worlds", type=int, default=2)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--resets", type=int, default=1)
    p.add_argument("--other-zero", action="store_true")
    p.add_argument("--startup-sync", action="store_true")
    p.add_argument("--scene-sync-only", action="store_true")
    args = p.parse_args()
    game = ROOT / "artifacts/reset_game"
    port = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8-sig"))["tcp_port"]
    ensure_idle(port)
    args.out.mkdir(parents=True, exist_ok=False)
    episode = ROOT / "runs/policy_0269_cold_8players_20261002_visual_fix/deterministic_000_agent1.json"
    actions = [component for row in json.loads(episode.read_text())["trace"][:args.steps] for component in row["action"]]
    assert len(actions) == 2 * args.steps
    request = dict(output=str(args.out.resolve()), worlds=args.worlds, resets=args.resets,
                   burnSteps=200, actions=actions, otherZero=args.other_zero,
                   startupSync=args.startup_sync, sceneSyncOnly=args.scene_sync_only)
    request_path = game / "GoiData/isolation_experiment.json"
    previous = request_path.read_bytes() if request_path.exists() else None
    overrides = dict(suppressHubDepthWrite=True, experimentalControlledUpdates=True,
                     experimentalSceneReset=True, experimentalNativeNewGame=True,
                     experimentalNativeInputSmoothing=True, experimentalResetDiagnostics=True,
                     stepFrames=1, dynamicsProbeMode="", dynamicsProbeFile="")
    report = dict(status="RUNNING", worlds=args.worlds, steps=args.steps, resets=args.resets,
                  other_zero=args.other_zero, startup_sync=args.startup_sync,
                  scene_sync_only=args.scene_sync_only, source=str(episode), source_sha256=sha256(episode),
                  game_runtime_sha256=sha256(game / "BepInEx/plugins/GameRuntime_v2.dll"),
                  experiment_sha256=sha256(game / "BepInEx/plugins/IsolatedWorldExperiment.dll"),
                  runtime_overrides=overrides)
    proc = None
    started = time.monotonic()
    try:
        request_path.write_text(json.dumps(request), encoding="utf-8")
        save_report(args.out / "request.json", request)
        with runtime_settings(game, 1, port, overrides):
            proc = GameLauncher(str(game / "GettingOverIt.exe")).launch(wait=False)
            report["pid"] = proc.pid
            save_report(args.out / "result.json", report)
            deadline = started + 180
            while not (args.out / "finished.json").exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"Game exited {proc.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError("World experiment exceeded 180s")
                time.sleep(0.25)
            report["experiment"] = json.loads((args.out / "finished.json").read_text(encoding="utf-8-sig"))
            if report["experiment"]["status"] != "COMPLETE":
                raise RuntimeError(report["experiment"].get("error", "Experiment error"))
            report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
        if previous is None:
            request_path.unlink(missing_ok=True)
        else:
            request_path.write_bytes(previous)
        report["wall_seconds"] = time.monotonic() - started
        save_report(args.out / "result.json", report)
        log = game / "BepInEx/LogOutput.log"
        if log.exists():
            shutil.copyfile(log, args.out / "game.log")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

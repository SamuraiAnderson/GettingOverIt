"""Replay archived actions through the production multi-player TCP interface."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/tests/control_interaction"))
from test_reset_state import ensure_idle, runtime_settings, save_report, sha256
from start.game_launcher import GameLauncher
from env import GoiEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--resets", type=int, default=1)
    parser.add_argument("--other-zero", action="store_true")
    parser.add_argument("--actions", type=Path, default=ROOT / "runs/policy_0269_cold_8players_20261002_visual_fix/deterministic_000_agent1.json")
    args = parser.parse_args()
    game = ROOT / "artifacts/reset_game"
    port = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8-sig"))["tcp_port"]
    ensure_idle(port)
    args.out.mkdir(parents=True, exist_ok=False)
    actions = np.float32([row["action"] for row in json.loads(args.actions.read_text())["trace"][:args.steps]])
    if actions.shape != (args.steps, 2): raise ValueError("Action source is too short")
    report = dict(status="RUNNING", worlds=args.worlds, steps=args.steps, resets=args.resets,
                  other_zero=args.other_zero, source=str(args.actions), source_sha256=sha256(args.actions),
                  backend="isolated_worlds", game_runtime_sha256=sha256(game / "BepInEx/plugins/GameRuntime_v2.dll"))
    proc = env = None
    started = time.monotonic()
    try:
        with runtime_settings(game, args.worlds, port, dict(experimentalResetDiagnostics=True, dynamicsProbeMode="")):
            proc = GameLauncher(str(game / "GettingOverIt.exe")).launch()
            env = GoiEnv(port=port, num_agents=args.worlds, timeout=180, protocol="anchor35")
            env.connect()
            env._sock.settimeout(180)
            assert env.hello["physics_backend"] == "isolated_worlds"
            for epoch in range(args.resets + 1):
                initial = env.new_snapshot() if epoch == 0 else env.reset()
                save_report(args.out / f"diagnostics_{epoch}_0.json", env.reset_diagnostics())
                with (args.out / f"trace_epoch_{epoch:02d}.jsonl").open("w") as output:
                    output.write(json.dumps(dict(step=0, states=initial.tolist())) + "\n")
                    for t, action in enumerate(actions, 1):
                        batch = np.tile(action, (args.worlds, 1))
                        if args.other_zero: batch[1:] = 0
                        state, _ = env.step(batch)
                        output.write(json.dumps(dict(step=t, states=state.tolist())) + "\n")
                        if t in {30, 31, 32, 40, args.steps}:
                            save_report(args.out / f"diagnostics_{epoch}_{t}.json", env.reset_diagnostics())
            report["status"] = "COMPLETE"
    except BaseException as exc:
        report.update(status="ERROR", error=str(exc))
        raise
    finally:
        if env is not None: env.close()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
        report["wall_seconds"] = time.monotonic() - started
        save_report(args.out / "result.json", report)
        shutil.copyfile(game / "BepInEx/LogOutput.log", args.out / "game.log")


if __name__ == "__main__": main()

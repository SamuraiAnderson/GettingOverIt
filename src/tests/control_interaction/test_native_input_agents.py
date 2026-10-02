"""Online input isolation check against recorded single-player native sequences.

Uses the isolated controlled runtime with the dynamics recorder disabled. Each
agent receives a different fixture, then the mapping is rotated after reset.
Compares native mouseInput, oldMouse and gain at every step, without requiring
the agents' body trajectories to agree. Run in getting-over-it-analysis.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess

import numpy as np

from test_native_dynamics import actions_for, read_trace
from test_reset_state import ROOT, ensure_idle, runtime_settings, save_report, sha256, validate_obs

KINDS = ("pulse", "hold", "reverse", "random", "zero")


def input_values(control):
    if not control["valid"]:
        raise ValueError("Invalid controller diagnostics")
    values = np.float32([control[name][axis] for name in ("mouseInput", "oldMouse") for axis in ("x", "y")]
                       + [control["mouseVelocityAverage"]])
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite controller input state")
    return values


def run(args):
    from env.goi_env import GoiEnv
    from start.game_launcher import GameLauncher

    reference = json.loads(args.reference.read_text(encoding="utf-8-sig"))
    if reference["status"] != "COMPLETE" or not reference["analysis"]["native_input"]["passed"]:
        raise ValueError("Native input reference did not pass")
    steps, seed = (reference["parameters"][key] for key in ("steps", "seed"))
    actions = {kind: actions_for(kind, steps, seed) for kind in KINDS}
    traces = {}
    for kind in KINDS:
        case = next(c for c in reference["cases"] if c["mode"] == "native" and c["actions"] == kind)
        traces[kind] = read_trace(args.reference.parent / case["raw"], actions[kind], "native")
    project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8-sig"))
    production = Path(project["game"]["executable_path"]).resolve().parent
    port = int(project["tcp_port"])
    launcher = GameLauncher(str(args.game_exe.resolve()))
    if launcher.game_root == production:
        raise ValueError("Use an isolated game copy")
    plugin = sha256(launcher.game_root / "BepInEx/plugins/GameRuntime_v2.dll")
    if plugin != reference["fingerprints"]["plugin"]:
        raise ValueError("Plugin differs from the accepted input reference")
    ensure_idle(port)
    args.out.mkdir(parents=True, exist_ok=False)
    production_files = [production / "BepInEx/plugins/GameRuntime_v2.dll", production / "GoiData/runtime_config.json"]
    report = {"status": "RUNNING", "agents": args.agents, "steps_per_round": steps,
              "burn_steps": args.burn_steps, "reference": str(args.reference.resolve()), "plugin": plugin,
              "production_before": {str(p): sha256(p) for p in production_files}, "rounds": [], "errors": []}
    overrides = dict(experimentalControlledUpdates=True, experimentalSceneReset=True,
                     experimentalNativeNewGame=True, experimentalNativeInputSmoothing=True,
                     experimentalResetDiagnostics=True, stepFrames=1, dynamicsProbeMode="", dynamicsProbeFile="")
    proc, env = None, None
    try:
        with runtime_settings(launcher.game_root, args.agents, port, overrides):
            proc = launcher.launch(wait=False)
            env = GoiEnv(port=port, num_agents=args.agents, timeout=120, startup_settle_s=2)
            env.connect()
            env._sock.settimeout(120)
            with (args.out / "diagnostics.jsonl").open("w", encoding="utf-8") as raw:
                for round_index in range(2):
                    if round_index:
                        for action in np.random.default_rng(seed + 1).uniform(-100, 100, (args.burn_steps, args.agents, 2)):
                            obs, _ = env.step(action.astype(np.float32))
                            validate_obs(obs, args.agents)
                        validate_obs(env.reset(), args.agents)
                    kinds = [KINDS[(agent + round_index) % len(KINDS)] for agent in range(args.agents)]
                    previous = env.reset_diagnostics()["current"]
                    mismatches = []
                    round_actions = np.stack([actions[k] for k in kinds], axis=1)
                    np.save(args.out / f"actions_{round_index}.npy", round_actions)
                    for step, action in enumerate(round_actions):
                        obs, done = env.step(action)
                        validate_obs(obs, args.agents)
                        if np.any(done):
                            raise ValueError("Unexpected done")
                        packet = env.reset_diagnostics()
                        raw.write(json.dumps({"round": round_index, "step": step, "packet": packet}) + "\n")
                        current = packet["current"]
                        if (current["manualCalls"] - previous["manualCalls"] != args.agents
                                or current["simulationCalls"] - previous["simulationCalls"] != args.agents
                                or current["timeScale"] != 0 or len(current["controls"]) != args.agents):
                            raise ValueError("Controller/physics schedule mismatch")
                        for agent, kind in enumerate(kinds):
                            actual = input_values(current["controls"][agent])
                            expected = np.float32(traces[kind][step * 3 + 1]["controls"][:5])
                            if not np.array_equal(actual.view(np.uint32), expected.view(np.uint32)):
                                mismatches.append({"step": step, "agent": agent, "fixture": kind,
                                                   "actual": actual.tolist(), "expected": expected.tolist()})
                        previous = current
                    report["rounds"].append({"round": round_index, "agent_fixtures": kinds,
                                             "compared_agent_steps": steps * args.agents,
                                             "passed": not mismatches, "mismatches": mismatches})
                    save_report(args.out / "result.json", report)
        report["status"] = "PASS" if all(r["passed"] for r in report["rounds"]) else "FAIL"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "ERROR"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        logging.exception("Input isolation check failed")
    finally:
        if env is not None:
            env.close()
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=15)
        log = launcher.game_root / "BepInEx/LogOutput.log"
        if log.exists():
            shutil.copyfile(log, args.out / "game.log")
        report["production_after"] = {str(p): sha256(p) for p in production_files}
        report["production_unchanged"] = report["production_before"] == report["production_after"]
        if not report["production_unchanged"]:
            report["status"] = "ERROR"
            report["errors"].append("Production files changed")
        save_report(args.out / "result.json", report)
    print(report["status"], args.out / "result.json")
    return {"PASS": 0, "FAIL": 1, "ERROR": 2}[report["status"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-exe", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--burn-steps", type=int, default=1000)
    args = parser.parse_args()
    if not 2 <= args.agents <= 64 or args.burn_steps < 1:
        parser.error("Requires 2..64 agents and a positive burn length")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(run(args))

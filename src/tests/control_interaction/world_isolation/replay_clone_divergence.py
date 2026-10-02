"""Replay archived actions from a fresh process; no policy inference or reset."""
import argparse
import json
import logging
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src/tests/control_interaction"))
from test_reset_state import ensure_idle, runtime_settings, save_report, sha256
from env.goi_env import GoiEnv
from start.game_launcher import GameLauncher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--original-actions", action="store_true")
    parser.add_argument("--probe-mode", default="observe")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    source = ROOT / "runs/policy_0269_cold_8players_20261002_visual_fix"
    game = ROOT / "artifacts/reset_game"
    port = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8-sig"))["tcp_port"]
    ensure_idle(port)
    args.out.mkdir(parents=True, exist_ok=False)
    episodes = [json.loads((source / f"deterministic_000_agent{i}.json").read_text()) for i in range(8)]
    shared = np.float32([x["action"] for x in episodes[1]["trace"][:args.steps]])
    actions = np.repeat(shared[:, None, :], args.agents, axis=1)
    if args.original_actions:
        actions = np.float32([[episodes[i]["trace"][t]["action"] for i in range(args.agents)] for t in range(args.steps)])
    probe_config = game / "GoiData/clone_probe.json"
    previous_probe = probe_config.read_bytes() if probe_config.exists() else None
    probe_config.write_text(json.dumps({"output": str((args.out / "phases.jsonl").resolve()), "mode": args.probe_mode}))
    overrides = dict(suppressHubDepthWrite=True, experimentalControlledUpdates=True,
                     experimentalSceneReset=True, experimentalNativeNewGame=True,
                     experimentalNativeInputSmoothing=True, experimentalResetDiagnostics=True,
                     stepFrames=1, dynamicsProbeMode="", dynamicsProbeFile="")
    report = {"status": "RUNNING", "agents": args.agents, "steps": args.steps,
              "action_source": str(source), "action_mode": "original" if args.original_actions else "shared_agent1",
              "probe_mode": args.probe_mode, "plugin_sha256": sha256(game / "BepInEx/plugins/GameRuntime_v2.dll")}
    proc, env, states, packets = None, None, [], []
    started = time.monotonic()
    try:
        with runtime_settings(game, args.agents, port, overrides):
            proc = GameLauncher(str(game / "GettingOverIt.exe")).launch(wait=False)
            env = GoiEnv(port=port, num_agents=args.agents, timeout=90, startup_settle_s=0)
            env.connect()
            env._sock.settimeout(90)
            cold = env.reset_diagnostics()
            assert cold["resetNumber"] == 0
            states.append(np.float32(cold["current"]["states"]).reshape(args.agents, 33))
            packets.append({"step": 0, "packet": cold})
            report["startup_seconds"] = time.monotonic() - started
            for step, action in enumerate(actions, 1):
                obs, done = env.step(action)
                assert not np.any(done)
                states.append(obs.copy())
                if 27 <= step <= 34 or step == args.steps:
                    packets.append({"step": step, "packet": env.reset_diagnostics()})
            values = np.array(states, dtype=np.float32)
            cols = [i for i in range(33) if i != 28]
            differences = []
            if args.agents > 2:
                for i in range(2, args.agents):
                    ds = np.any(values[:, i, cols] != values[:, 1, cols], axis=1)
                    first = np.flatnonzero(ds)
                    differences.append({"agent": i, "first_divergence_from_agent1": int(first[0]) if len(first) else None})
            report.update(status="COMPLETE", comparisons=differences)
            if args.original_actions:
                expected = np.float32([x["states"][:args.steps+1] for x in episodes[:args.agents]]).transpose(1, 0, 2)
                report["matches_archived_states_without_timestamp"] = bool(np.array_equal(values[:, :, cols].view(np.uint32), expected[:, :, cols].view(np.uint32)))
    except BaseException as exc:
        report.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if env is not None:
            env.close()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
        if previous_probe is None:
            probe_config.unlink(missing_ok=True)
        else:
            probe_config.write_bytes(previous_probe)
        np.savez_compressed(args.out / "trace.npz", states=np.array(states), actions=actions)
        save_report(args.out / "diagnostics.json", packets)
        save_report(args.out / "result.json", report)
        log = game / "BepInEx/LogOutput.log"
        if log.exists():
            shutil.copyfile(log, args.out / "game.log")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

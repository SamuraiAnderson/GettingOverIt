"""Online acceptance of the production multi-player interface (no probe DLL)."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from test_reset_state import ensure_idle, runtime_settings, save_report
from start.game_launcher import GameLauncher
from env import GoiEnv


def comparable(states):
    values = np.asarray(states, dtype=np.float32)
    assert np.isfinite(values).all(), "Nonfinite observation"
    return np.delete(values, 28, axis=-1).copy().view(np.uint32)


def body_values(packet):
    return [{key: value for key, value in b.items() if key != "agent"} for b in packet["current"]["bodies"]]


def run_case(game, port, agents, steps, out):
    proc = env = None
    result = dict(agents=agents, status="RUNNING")
    started = time.monotonic()
    try:
        with runtime_settings(game, agents, port, {"experimentalResetDiagnostics": True,
                                                 "dynamicsProbeMode": "", "dynamicsProbeFile": ""}):
            proc = GameLauncher(str(game / "GettingOverIt.exe")).launch()
            env = GoiEnv(port=port, num_agents=agents, timeout=180, protocol="anchor35")
            env.connect()
            env._sock.settimeout(180)
            assert env.hello["physics_backend"] == "isolated_worlds"
            assert env.hello["physics_world_count"] == agents
            cold = env.new_snapshot()
            before = env.reset_diagnostics()
            assert len(set(before["sceneHandles"])) == agents
            result["ready_seconds"] = time.monotonic() - started
            time.sleep(.15)
            assert body_values(before) == body_values(env.reset_diagnostics()), "Idle changes physical bodies"
            rng = np.random.default_rng(47)
            action_sequence = rng.uniform(-100, 100, (steps, 2)).astype(np.float32)
            action_sequence[:40] = [40, 30]
            batch = lambda action: np.tile(action, (agents, 1))
            actions = batch(np.float32([40, 30]))
            trace = [cold]
            env.observation_diagnostics()
            for action in action_sequence:
                trace.append(env.step(batch(action))[0])
            trace = np.asarray(trace)
            for i in range(1, agents):
                assert np.array_equal(comparable(trace[:, 0]), comparable(trace[:, i])), "Identical actions diverge"
            observed = env.observation_diagnostics()
            assert len(observed["agents"]) == agents and observed["aligned"]
            after = env.reset_diagnostics()
            assert after["current"]["simulationCalls"] - before["current"]["simulationCalls"] == steps * agents
            result["same_actions_equal"] = True
            for repeat in range(2):
                reset = env.reset()
                assert np.array_equal(comparable(reset), comparable(cold)), "Reset differs from cold start"
                assert body_values(before) == body_values(env.reset_diagnostics()), "Reset body/controller boundary differs"
                assert before["current"]["worldBodyCount"] == env.reset_diagnostics()["current"]["worldBodyCount"], "Reset leaked native bodies"
                for t in range(steps):
                    got = env.step(batch(action_sequence[t]))[0]
                    assert np.array_equal(comparable(got), comparable(trace[t+1])), f"Reset replay diverges at {t+1}"
            result["cold_reset_repeats_equal"] = True
            if agents > 1:
                env.reset()
                for t in range(steps):
                    mixed = batch(action_sequence[t])
                    mixed[1:] = rng.uniform(-100, 100, (agents-1, 2))
                    got = env.step(mixed)[0]
                    assert np.array_equal(comparable(got[0]), comparable(trace[t+1, 0])), "Other players affect world 0"
                result["different_actions_isolated"] = True
            env.reset()
            # Training contract: zero warmup + NEW_SNAPSHOT + subsequent rollouts.
            for _ in range(100): env.step(np.zeros((agents, 2), np.float32))
            warm = env.new_snapshot()
            warm_diag = env.reset_diagnostics()
            warm_trace = [env.step(actions)[0] for _ in range(40)]
            assert np.array_equal(comparable(env.reset()), comparable(warm)), "Warm snapshot reset differs"
            assert body_values(env.reset_diagnostics()) == body_values(warm_diag)
            for expected in warm_trace:
                assert np.array_equal(comparable(env.step(actions)[0]), comparable(expected))
            result["warm_snapshot_equal"] = True
            env.reset()
            before_teleport = env.reset_diagnostics()
            teleported = env.teleport(0.0, 4.0, agent_index=agents - 1)
            after_teleport = env.reset_diagnostics()
            if agents > 1:
                assert np.array_equal(comparable(teleported[:-1]), comparable(warm[:-1])), "Teleport advanced another world"
                assert [b for b in before_teleport["current"]["bodies"] if b["agent"] != agents-1] == [
                    b for b in after_teleport["current"]["bodies"] if b["agent"] != agents-1]
            tele_snapshot = env.new_snapshot()
            for _ in range(40): env.step(actions)
            assert np.array_equal(comparable(env.reset()), comparable(tele_snapshot)), "Teleport snapshot replay differs"
            result["teleport_isolated_and_snapshot_equal"] = True
            env.export_colliders()
            contour = json.loads((game / "GoiData/Colliders/player_contour.json").read_text(encoding="utf-8-sig"))
            assert set(contour["parts"]) == {"body", "tip", "pot", "pot_sides"}, "Exporter lost flattened parts"
            result.update(status="PASS", hello=env.hello, initial_bodies=body_values(before))
            np.save(out / f"trace_{agents}.npy", trace)
    except BaseException as exc:
        result.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if env is not None: env.close()
        if proc is not None and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=15)
        result["wall_seconds"] = time.monotonic() - started
        save_report(out / f"case_{agents}.json", result)
        log = game / "BepInEx/LogOutput.log"
        if log.exists(): (out / f"game_{agents}.log").write_bytes(log.read_bytes())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=Path, default=ROOT / "artifacts/reset_game")
    parser.add_argument("--agents", type=int, nargs="+", default=[1, 2, 8])
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    port = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8-sig"))["tcp_port"]
    ensure_idle(port)
    args.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for agents in args.agents:
        cases.append(run_case(args.game, port, agents, args.steps, args.out))
        print(f"PASS {agents} worlds", flush=True)
    reference = np.load(args.out / f"trace_{args.agents[0]}.npy")[:, 0]
    for agents in args.agents[1:]:
        assert np.array_equal(comparable(reference), comparable(np.load(args.out / f"trace_{agents}.npy")[:, 0])), "World count changes trajectory"
        assert cases[0]["initial_bodies"] == cases[args.agents.index(agents)]["initial_bodies"][:len(cases[0]["initial_bodies"])], "World count changes starting bodies"
    save_report(args.out / "result.json", dict(status="PASS", cases=cases, cross_count_equal=True))


if __name__ == "__main__":
    main()

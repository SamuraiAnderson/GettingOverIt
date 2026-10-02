"""Reset-state acceptance, without PPO, warmup, or replacing the startup snapshot.

Run with conda run -n getting-over-it-analysis python -u <this file>.
Each fresh process supplies a first-reset baseline, followed by histories of
different lengths and waits after reset. A zero-action probe exposes effects
not visible until physics advances. PASS certifies only the observed contract,
not a complete solver snapshot. See README.md for thresholds and limitations.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
LOG = logging.getLogger("reset_state")

# Authoritative layout: Core/Interfaces/IPlayerServices.cs, NOT old skill prose.
FIELDS = (
    "player_x", "player_y", "player_vx", "player_vy", "player_w",
    "hub_x", "hub_y", "hub_vx", "hub_vy", "hub_angle",
    "slider_x", "slider_y", "slider_vx", "slider_vy", "slider_angle",
    "handle_x", "handle_y", "handle_vx", "handle_vy",
    "pole_x", "pole_y", "pole_vx", "pole_vy",
    "tip_x", "tip_y", "tip_vx", "tip_vy", "hammer_angle", "timestamp",
    "cursor_x", "cursor_y", "cursor_vx", "cursor_vy", "anchor_x", "anchor_y",
)
GROUPS = {
    "position": ((0, 1, 5, 6, 10, 11, 15, 16, 19, 20, 23, 24), "m"),
    "linear_velocity": ((2, 3, 7, 8, 12, 13, 17, 18, 21, 22, 25, 26), "m/s"),
    "angular_velocity": ((4,), "deg/s"),
    "angle": ((9, 14, 27), "deg"),
    "cursor_position": ((29, 30), "m"),
    "cursor_velocity": ((31, 32), "m/s"),
}


@dataclass(frozen=True)
class Limits:
    position: float = 1e-3
    linear_velocity: float = 1e-3
    angular_velocity: float = 0.1
    angle: float = 0.05
    cursor_position: float = 1e-3
    cursor_velocity: float = 1e-3
    rest_linear_speed: float = 0.05
    rest_angular_speed: float = 1.0


def validate_obs(obs, agents):
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim != 2 or obs.shape[0] != agents or obs.shape[1] not in (33, 35):
        raise ValueError(f"Unexpected observation shape: {obs.shape}, agents={agents}")
    if not np.isfinite(obs).all():
        raise ValueError("Nonfinite observation; cannot certify reset")
    if np.any(np.all(obs[:, :28] == 0, axis=1)):
        raise ValueError("All-zero agent packet (possible server error fallback)")
    return obs.copy()


def compare(actual, expected, limits):
    """Compare corresponding agents; angles wrap, angular velocities do not."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or actual.ndim != 2:
        raise ValueError(f"Comparison shapes differ: {actual.shape}, {expected.shape}")
    if actual.shape[1] not in (33, 35) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Unsupported or nonfinite comparison")
    groups = dict(GROUPS)
    if actual.shape[1] == 35:
        groups["anchor_position"] = ((33, 34), "m")
    result = {}
    for name, (columns, unit) in groups.items():
        delta = actual[:, columns] - expected[:, columns]
        if name == "angle":
            delta = (delta + 180.0) % 360.0 - 180.0
        error = np.abs(delta)
        agent, local = np.unravel_index(error.argmax(), error.shape)
        field = columns[local]
        tolerance = limits.position if name == "anchor_position" else getattr(limits, name)
        result[name] = {
            "max_error": float(error[agent, local]), "unit": unit, "limit": tolerance,
            "passed": bool(error[agent, local] <= tolerance),
            "agent": int(agent), "field": FIELDS[field], "index": field,
            "actual": float(actual[agent, field]), "expected": float(expected[agent, field]),
        }
    columns = [i for i in range(actual.shape[1]) if i != 28]
    return {
        "passed": all(item["passed"] for item in result.values()),
        "groups": result,
        "bitwise_equal_without_timestamp": bool(
            np.array_equal(np.asarray(actual[:, columns], dtype=np.float32).view(np.uint32),
                           np.asarray(expected[:, columns], dtype=np.float32).view(np.uint32))),
    }


def at_rest(obs, limits):
    pairs = [(name, cols) for name, cols in (
        ("player", (2, 3)), ("hub", (7, 8)), ("slider", (12, 13)),
        ("handle", (17, 18)), ("pole", (21, 22)), ("tip", (25, 26)), ("cursor", (31, 32)))]
    speeds = {name: np.linalg.norm(obs[:, cols].astype(np.float64), axis=1).tolist()
              for name, cols in pairs}
    angular = np.abs(obs[:, 4].astype(np.float64)).tolist()
    return {
        "passed": max(max(v) for v in speeds.values()) <= limits.rest_linear_speed
                  and max(angular) <= limits.rest_angular_speed,
        "linear_speeds_m_s": speeds, "player_angular_speeds_deg_s": angular,
        "linear_limit": limits.rest_linear_speed, "angular_limit": limits.rest_angular_speed,
    }


def assess(trace, reference, limits, require_rest=True, rest_only_at_reset=False,
           require_alignment=False):
    if trace.shape != reference.shape:
        raise ValueError("Probe/reference shapes differ")
    frames = []
    for step, obs in enumerate(trace):
        frames.append({
            "step_after_reset": step,
            "vs_cold_reference": compare(obs, reference[step], limits),
            "cross_agent": compare(obs, np.repeat(obs[0:1], len(obs), axis=0), limits),
            "stationarity": at_rest(obs, limits),
        })
    matched = all(f["vs_cold_reference"]["passed"] for f in frames)
    aligned = all(f["cross_agent"]["passed"] for f in frames)
    resting = all(f["stationarity"]["passed"] for f in (frames[:1] if rest_only_at_reset else frames))
    return {
        "passed": matched and (aligned or not require_alignment) and (resting or not require_rest),
        "alignment_required": require_alignment,
        "matches_reference": matched, "agents_aligned": aligned, "stationary": resting,
        "frames": frames,
    }


def history_actions(kind, steps, agents, seed):
    if kind == "none":
        return np.zeros((0, agents, 2), dtype=np.float32)
    if kind == "hold":
        return np.tile(np.array([40, 30], dtype=np.float32), (steps, agents, 1))
    # Different histories per agent catch residuals hidden by shared actions.
    return np.random.default_rng(seed).uniform(-100, 100, (steps, agents, 2)).astype(np.float32)


def diagnostic_bodies(packet, agents):
    frame = packet["current"]
    rows = frame["bodies"]
    if len(rows) != agents * 7 or len(frame["controls"]) != agents:
        raise ValueError("Expected six rigidbodies and one cursor per agent")
    values = []
    for agent in range(agents):
        bodies = [b for b in rows if b["agent"] == agent]
        if len(bodies) != 7 or len({b["name"] for b in bodies}) != 7:
            raise ValueError("Missing or duplicate diagnostic body")
        values.append([[b["position"]["x"], b["position"]["y"], b["rotation"],
                        b["velocity"]["x"], b["velocity"]["y"], b["angularVelocity"]] for b in bodies])
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all() or not all(c["valid"] for c in frame["controls"]):
        raise ValueError("Invalid body/controller diagnostics")
    return result


def assess_diagnostics(data, agents, limits, probe_kind, probe_steps, reference, require_rest=True):
    packets = [data[k] for k in ("after_reset", "after_wait", "after_probe")]
    matrices = [diagnostic_bodies(p, agents) for p in packets]
    start, idle, end = [p["current"] for p in packets]
    delta = np.abs(matrices[1] - matrices[0])
    delta[:, :, 2] = np.abs((matrices[1][:, :, 2] - matrices[0][:, :, 2] + 180) % 360 - 180)
    idle_ok = (delta[:, :, :2].max() <= 1e-5 and delta[:, :, 2].max() <= 1e-4
               and delta[:, :, 3:].max() <= 1e-6 and start["controls"] == idle["controls"]
               and start["manualCalls"] == idle["manualCalls"]
               and start["simulationCalls"] == idle["simulationCalls"])
    advancement_ok = (end["manualCalls"] - idle["manualCalls"] == probe_steps * agents
                      and end["simulationCalls"] - idle["simulationCalls"] == probe_steps)
    rest_matrices = matrices if probe_kind == "zero" else matrices[:2]
    resting = all(np.linalg.norm(m[:, :, 3:5], axis=2).max() <= limits.rest_linear_speed
                  and np.abs(m[:, :, 5]).max() <= limits.rest_angular_speed for m in rest_matrices)
    reference_matrix = diagnostic_bodies(reference["after_reset"], agents)
    error = np.abs(matrices[0] - reference_matrix)
    error[:, :, 2] = np.abs((matrices[0][:, :, 2] - reference_matrix[:, :, 2] + 180) % 360 - 180)
    errors = {"position": float(error[:, :, :2].max()), "angle": float(error[:, :, 2].max()),
              "linear_velocity": float(error[:, :, 3:5].max()), "angular_velocity": float(error[:, :, 5].max())}
    body_reference_ok = all(v <= getattr(limits, k) for k, v in errors.items())
    lifecycle_ok = (len({p["epoch"] for p in packets}) == 1
                    and all(p["controlledUpdates"] for p in packets)
                    and all(p["current"]["worldBodyCount"] == reference["after_reset"]["current"]["worldBodyCount"]
                            for p in packets))
    clock_paused = all(p["current"].get("timeScale") == 0 for p in packets)
    return {"passed": bool(idle_ok and advancement_ok and (resting or not require_rest)
                           and body_reference_ok and lifecycle_ok and clock_paused),
            "automatic_clock_paused": clock_paused,
            "idle_stable": bool(idle_ok), "step_counts_correct": advancement_ok,
            "all_bodies_at_rest": bool(resting), "body_reference_errors": errors,
            "body_reference_matched": body_reference_ok, "lifecycle_valid": lifecycle_ok,
            "world_body_count": start["worldBodyCount"], "epoch": packets[0]["epoch"]}


def measure_case(env, last_obs, actions, idle_s, probe_steps, agents, path, diagnostics=False,
                 probe_kind="zero", probe_seed=0, cold_live=False):
    """Always preserve received data, including partial/invalid failed cases."""
    history = [] if last_obs is None else [last_obs.copy()]
    trace, executed, probe_actions = [], [], []
    reset_seconds = actual_wait = None
    diagnostic_data = {}
    try:
        for action in actions:
            obs, done = env.step(action)
            history.append(np.array(obs, copy=True))
            executed.append(action.copy())
            validate_obs(obs, agents)
            if np.any(done):
                raise ValueError("Unexpected done during history")
        started = time.perf_counter()
        if cold_live:
            if not diagnostics:
                raise ValueError("Live cold reference requires diagnostics")
            packet = env.reset_diagnostics()
            obs = np.asarray(packet["current"]["states"], dtype=np.float32).reshape(agents, 33)
        else:
            obs = env.reset()
        reset_seconds = time.perf_counter() - started
        trace.append(np.array(obs, copy=True))
        validate_obs(obs, agents)
        if diagnostics:
            diagnostic_data["after_reset"] = env.reset_diagnostics()
        started = time.perf_counter()
        time.sleep(idle_s)  # No protocol calls: D is cached and N would change the target snapshot.
        actual_wait = time.perf_counter() - started
        if diagnostics:
            diagnostic_data["after_wait"] = env.reset_diagnostics()
        probe = np.zeros((probe_steps, agents, 2), dtype=np.float32)
        if probe_kind != "zero":
            shared = history_actions(probe_kind, probe_steps, 1, probe_seed)
            probe = np.repeat(shared, agents, axis=1)
        for action in probe:
            obs, done = env.step(action)
            trace.append(np.array(obs, copy=True))
            probe_actions.append(action.copy())
            validate_obs(obs, agents)
            if np.any(done):
                raise ValueError("Unexpected done after reset")
        if diagnostics:
            diagnostic_data["after_probe"] = env.reset_diagnostics()
        return np.stack(trace), reset_seconds, actual_wait
    finally:
        np.savez_compressed(path, history_states=np.asarray(history),
                            history_actions=np.asarray(executed), reset_and_probe=np.asarray(trace),
                            probe_actions=np.asarray(probe_actions))
        if diagnostic_data:
            save_report(path.with_suffix(".diagnostics.json"), diagnostic_data)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def ensure_idle(port):
    # Do not kill or accidentally connect to a user's existing game.
    if sys.platform == "win32":
        result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq GettingOverIt.exe", "/FO", "CSV", "/NH"],
                                capture_output=True, check=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        # tasklist uses the Windows code page, which can differ from Python's UTF-8
        # mode. Only the ASCII executable name matters; localized messages do not.
        if any(row and row[0].lower() == "gettingoverit.exe"
               for row in csv.reader(result.stdout.decode("ascii", errors="replace").splitlines())):
            raise RuntimeError("An existing GettingOverIt.exe is running; stop it before this test")
    with socket.socket() as sock:
        sock.settimeout(0.5)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} already has a listener")


@contextmanager
def runtime_settings(game_root, agents, port, overrides=None):
    from start.game_mode_controller import GameModeController, GameMode
    config_path = game_root / "GoiData" / "runtime_config.json"
    signals = game_root / "GoiData" / "ControlSignals"
    paths = [config_path] + [signals / f"mode_{mode.value}.signal" for mode in GameMode]
    original = {p: p.read_bytes() if p.exists() else None for p in paths}
    try:
        config = json.loads(original[config_path].decode("utf-8-sig")) if original[config_path] else {}
        config.update(numDuplicates=agents, tcpPort=port)
        config.update(overrides or {})
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        GameModeController(str(game_root)).set_game_runtime_mode()
        yield config
    finally:
        for path, contents in original.items():
            if contents is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(contents)


def save_report(path, report):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for attempt in range(6):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))  # Windows indexer/AV may briefly hold the old file.


def overall_status(report):
    if report["errors"] or len(report["cases"]) != report["expected_cases"]:
        return "ERROR"
    return "PASS" if all(row.get("assessment", {}).get("passed", False) and row.get("history_exercised", True)
                         and row.get("diagnostic_assessment", {}).get("passed", True)
                         for row in report["cases"]) else "FAIL"


def run(args):
    from env.goi_env import GoiEnv
    from start.game_launcher import GameLauncher
    from start.game_mode_controller import GameModeController
    launcher = GameLauncher(str(args.game_exe.resolve())) if args.game_exe else GameLauncher()
    ensure_idle(args.port)
    args.out.mkdir(parents=True, exist_ok=False)
    limits = Limits(**{key: getattr(args, key) for key in asdict(Limits())})
    histories = [("none", 0), ("hold", 40)] + [("random", n) for n in args.burn_steps]
    report = {
        "version": 1, "status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "same-agent reset versus reference and probe; cross-agent divergence is informational",
        "raw_state_schema": {"name": "legacy33", "dimension": 33},
        "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "thresholds": asdict(limits), "fields": list(FIELDS), "timestamp_excluded": 28,
        "limitations": [
            "Scene reset with diagnostics uses live state before any RESET as the cold reference; legacy mode uses first RESET.",
            "Q diagnostics expose all body velocities and selected control fields, not the full solver state.",
            "Without Q diagnostics, wait effects are observed only at the next STEP.",
            "Histories run sequentially; differences are not uniquely attributed to waiting versus prior history.",
            "Stationarity is checked separately, including common motion shared by all agents.",
            "Thresholds are provisional acceptance criteria, not a measured numerical noise floor.",
        ],
        "expected_cases": args.restarts * (1 + args.cycles * len(histories) * len(args.idle_seconds)),
        "cases": [], "sessions": [], "errors": [],
        "fingerprints": {
            "script": sha256(Path(__file__)),
            "deployed_plugin": sha256(launcher.game_root / "BepInEx/plugins/GameRuntime_v2.dll"),
            "game_assembly": sha256(launcher.game_root / "GettingOverIt_Data/Managed/Assembly-CSharp.dll"),
            "project_config": sha256(ROOT / "src/config/project.json"),
        },
    }
    report_path = args.out / "result.json"
    save_report(report_path, report)
    reference = None
    diagnostic_reference = None
    try:
        overrides = {}
        if args.controlled_updates:
            overrides["experimentalControlledUpdates"] = True
        if args.reset_diagnostics:
            overrides["experimentalResetDiagnostics"] = True
        if args.scene_reset:
            overrides["experimentalSceneReset"] = True
            overrides["experimentalControlledUpdates"] = True
        with runtime_settings(launcher.game_root, args.agents, args.port, overrides) as config:
            report["runtime_config_requested"] = config
            for restart in range(args.restarts):
                proc, env = None, None
                try:
                    ensure_idle(args.port)
                    # Mode signals are consumed by the game at startup.
                    GameModeController(str(launcher.game_root)).set_game_runtime_mode()
                    proc = launcher.launch(wait=False)
                    session = {"restart": restart, "pid": proc.pid}
                    report["sessions"].append(session)
                    save_report(report_path, report)
                    env = GoiEnv(port=args.port, num_agents=args.agents, timeout=args.timeout,
                                 startup_settle_s=args.startup_settle)
                    env.connect()
                    env._sock.settimeout(args.timeout)
                    effective = launcher.game_root / "GoiData/runtime_config.json"
                    session["runtime_config_effective"] = json.loads(effective.read_text(encoding="utf-8-sig"))
                    jobs = [("cold", 0, 0.0, -1)]
                    for cycle in range(args.cycles):
                        conditions = [(kind, steps, idle, cycle) for kind, steps in histories
                                      for idle in args.idle_seconds]
                        if not args.fixed_order:
                            np.random.default_rng(args.seed + restart * 100 + cycle).shuffle(conditions)
                        jobs.extend(conditions)
                    last_obs = None
                    for case_index, (kind, steps, idle, cycle) in enumerate(jobs):
                        case_id = f"restart_{restart:02d}_case_{case_index:03d}"
                        record = {"id": case_id, "restart": restart, "history": kind,
                                  "history_steps": steps, "cycle": cycle, "idle_requested_s": idle,
                                  "raw_file": case_id + ".npz"}
                        seed = args.seed + max(cycle, 0)  # same commanded history across restarts/waits
                        record.update(history_seed=seed, status="RUNNING")
                        report["cases"].append(record)
                        save_report(report_path, report)
                        actions = history_actions(kind, steps, args.agents, seed)
                        trace, elapsed, waited = measure_case(
                            env, last_obs, actions, idle, args.probe_steps, args.agents,
                            args.out / record["raw_file"], args.reset_diagnostics, args.probe_kind, args.seed,
                            cold_live=(kind == "cold" and args.scene_reset and args.reset_diagnostics))
                        record["reference_origin"] = ("live_before_first_reset" if args.scene_reset and
                                                       args.reset_diagnostics else "first_reset")
                        if reference is None:
                            reference = trace.copy()
                            report["reference_case"] = case_id
                        record.update(reset_seconds=elapsed, idle_actual_s=waited, history_seed=seed,
                                      assessment=assess(trace, reference, limits, not args.allow_moving_start,
                                                        args.probe_kind != "zero"))
                        if args.reset_diagnostics:
                            diagnostic = json.loads((args.out / record["raw_file"]).with_suffix(
                                ".diagnostics.json").read_text(encoding="utf-8"))
                            if diagnostic_reference is None:
                                diagnostic_reference = diagnostic
                            record["diagnostic_assessment"] = assess_diagnostics(
                                diagnostic, args.agents, limits, args.probe_kind, args.probe_steps,
                                diagnostic_reference, not args.allow_moving_start)
                        if steps:
                            with np.load(args.out / record["raw_file"]) as raw:
                                positions = raw["history_states"][:, :, [0, 1, 23, 24, 29, 30]]
                                movement = np.max(np.abs(positions - positions[0:1]), axis=(0, 2))
                            record["history_max_position_change_per_agent_m"] = movement.tolist()
                            record["history_exercised"] = bool(np.all(movement > 1e-3))
                        last_obs = trace[-1].copy()
                        record["status"] = "COMPLETE"
                        save_report(report_path, report)
                        LOG.info("%s %s(%d) wait=%.3fs reset=%.4fs %s", case_id, kind, steps, idle,
                                 elapsed, "PASS" if record["assessment"]["passed"] and
                                 record.get("history_exercised", True) and
                                 record.get("diagnostic_assessment", {}).get("passed", True) else "FAIL")
                finally:
                    try:
                        if env is not None:
                            env.close()
                    finally:
                        if proc is not None:
                            if proc.poll() is None:
                                proc.terminate()
                            try:
                                proc.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait(timeout=15)
    except (Exception, KeyboardInterrupt) as exc:
        LOG.exception("Reset acceptance did not complete")
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        if report["cases"] and report["cases"][-1]["status"] == "RUNNING":
            report["cases"][-1].update(status="ERROR", error=report["errors"][-1])
    report["status"] = overall_status(report)
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_report(report_path, report)
    LOG.info("%s: %d/%d cases; result: %s", report["status"], len(report["cases"]),
             report["expected_cases"], report_path)
    return {"PASS": 0, "FAIL": 1, "ERROR": 2}[report["status"]]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--game-exe", type=Path, help="Use an isolated game copy")
    parser.add_argument("--controlled-updates", action="store_true", help="Opt into experimental FixedUpdate gate")
    parser.add_argument("--reset-diagnostics", action="store_true", help="Save live idle and reset-stage diagnostics")
    parser.add_argument("--scene-reset", action="store_true", help="Experimental fixed-start scene reconstruction")
    parser.add_argument("--fixed-order", action="store_true", help="Repeat identical history order across processes")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--burn-steps", type=int, nargs="+", default=[200, 1000])
    parser.add_argument("--idle-seconds", type=float, nargs="+", default=[0.0, 0.25, 1.0])
    parser.add_argument("--probe-steps", type=int, default=1)
    parser.add_argument("--probe-kind", choices=["zero", "hold", "random"], default="zero",
                        help="Shared deterministic actions across agents; moving probes require rest only at reset")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--startup-settle", type=float, default=8.0,
                        help="Wall-clock startup readiness wait; no warmup STEPs")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--port", type=int, help="Default: src/config/project.json tcp_port")
    parser.add_argument("--out", type=Path, help="New output directory; refuses to overwrite")
    parser.add_argument("--allow-moving-start", action="store_true",
                        help="Diagnostic mode: report stationarity but do not require it for PASS")
    for key, value in asdict(Limits()).items():
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=value)
    args = parser.parse_args(argv)
    if not 2 <= args.agents <= 64:
        parser.error("--agents must be 2..64 so cross-agent alignment is tested")
    if min(args.restarts, args.cycles, args.probe_steps, *args.burn_steps) < 1:
        parser.error("restarts, cycles, probe-steps and burn-steps must be positive")
    if any(not np.isfinite(v) or v < 0 for v in args.idle_seconds + [args.startup_settle]):
        parser.error("wait times must be finite and nonnegative")
    if not np.isfinite(args.timeout) or args.timeout <= 0 or args.seed < 0:
        parser.error("timeout must be finite and positive; seed must be nonnegative")
    if any(not np.isfinite(getattr(args, key)) or getattr(args, key) < 0 for key in asdict(Limits())):
        parser.error("thresholds must be finite and nonnegative")
    if args.port is None:
        project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8-sig"))
        args.port = int(project["tcp_port"])
    if not 1 <= args.port <= 65535:
        parser.error("port must be 1..65535")
    if args.out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        args.out = ROOT / "runs" / ("reset_state_" + stamp)
    return args


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        return run(parse_args(argv))
    except (Exception, KeyboardInterrupt):
        LOG.exception("ERROR: test could not start (no acceptance verdict)")
        return 2


if __name__ == "__main__":
    sys.exit(main())

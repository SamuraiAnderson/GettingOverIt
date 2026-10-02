"""Opt-in native/candidate dynamics experiment; requires an isolated game copy.

Native preserves automatic physics, lifecycle, hierarchy and controller state;
the Hub depth fix is disabled for that reference unless explicitly requested.
native_dual changes only the pre-controller mouseInput write during the probe.
Candidate is the fixed-start reset runtime. Initial-state mismatches prevent an
equivalence verdict; repeatability alone is never interpreted as equivalence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from test_reset_state import ROOT, ensure_idle, runtime_settings, save_report, sha256, validate_obs

LOG = logging.getLogger("native_dynamics")
PHASES = ("before_control", "after_control", "after_physics")
LIMITS = {"position_m": 1e-3, "angle_deg": 0.05, "velocity_m_s": 1e-3, "angular_velocity_deg_s": 0.1}
MOTOR_PARAMETERS = ("hinge_cap", "slider_cap", "hub_cap", "controller_hinge_cap", "controller_slider_cap")


def actions_for(kind, steps, seed):
    actions = np.zeros((steps, 2), dtype=np.float32)
    if kind == "hold":
        actions[:] = (40, 30)
    elif kind == "pulse":
        actions[0] = (40, 30)
        if steps > 20:
            actions[20] = (-40, -30)
    elif kind == "random":
        actions[:] = np.random.default_rng(seed).uniform(-100, 100, actions.shape)
    elif kind == "reverse":
        actions[:] = (40, 30)
        actions[steps // 2:] *= -1
    return actions


def read_trace(path, actions, mode):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()]
    if len(rows) != len(actions) * 3:
        raise ValueError(f"Incomplete trace: {len(rows)} rows, expected {len(actions) * 3}")
    if int(Path(str(path) + ".done").read_text()) != len(actions):
        raise ValueError("Invalid completion marker")
    for index, row in enumerate(rows):
        step, phase = divmod(index, 3)
        if (row["step"], row["phase"], row["mode"]) != (step, PHASES[phase], mode):
            raise ValueError("Wrong phase order, step or mode")
        if row["controllerCalls"] != step + 1 or row["physicsCalls"] != step + (phase == 2):
            raise ValueError("Controller/physics advancement mismatch")
        for field, size in (("bodies", 42), ("bodyConfig", 63), ("controls", 11), ("motors", 6), ("axes", 4)):
            values = np.asarray(row[field], dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {field}")
        if not np.isfinite(row["contacts"]).all() or len(set(row["bodyNames"])) != 7:
            raise ValueError("Invalid contacts or body identity")
        if row["fixedDeltaTime"] <= 0:
            raise ValueError("Invalid physics step duration")
        expected_scale = 0 if mode == "candidate" else 1
        if row["timeScale"] != expected_scale:
            raise ValueError("Unexpected simulation clock")
        if phase:
            if row["axisReads"] != [1, 1, 2, 2]:
                raise ValueError(f"Unexpected axis reads: {row['axisReads']}")
            np.testing.assert_array_equal(np.float32(row["axes"]), np.r_[actions[step], [0, 0]].astype(np.float32))
    return rows


def body_error(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1, 7, 6), np.asarray(b, dtype=float).reshape(-1, 7, 6)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid body comparison")
    delta = np.abs(a - b)
    delta[..., 2] = np.abs((a[..., 2] - b[..., 2] + 180) % 360 - 180)
    values = {"position_m": float(delta[..., :2].max()), "angle_deg": float(delta[..., 2].max()),
              "velocity_m_s": float(delta[..., 3:5].max()), "angular_velocity_deg_s": float(delta[..., 5].max())}
    return {"errors": values, "within_limits": all(values[k] <= LIMITS[k] for k in LIMITS),
            "bitwise_equal": bool(np.array_equal(a.astype(np.float32).view(np.uint32), b.astype(np.float32).view(np.uint32)))}


def comparison(left, right):
    if left[0]["bodyNames"] != right[0]["bodyNames"]:
        raise ValueError("Body ordering differs")
    first = body_error(left[0]["bodies"], right[0]["bodies"])
    config_delta = float(np.max(np.abs(np.array(left[0]["bodyConfig"]) - right[0]["bodyConfig"])))
    phases = {}
    for phase in PHASES:
        a, b = ([row for row in trace if row["phase"] == phase] for trace in (left, right))
        phases[phase] = body_error([r["bodies"] for r in a], [r["bodies"] for r in b])
        phases[phase]["controls_max_abs_delta"] = float(np.max(np.abs(np.array([r["controls"] for r in a]) - np.array([r["controls"] for r in b]))))
        phases[phase]["motors_max_abs_delta"] = float(np.max(np.abs(np.array([r["motors"][:4] for r in a]) - np.array([r["motors"][:4] for r in b]))))
    return {"initial_bodies": first, "body_config_max_abs_delta": config_delta,
            "fixed_dt_equal": left[0]["fixedDeltaTime"] == right[0]["fixedDeltaTime"], "phases": phases,
            "trajectory_attribution": "INITIAL_STATE_MISMATCH" if not first["within_limits"] else "INITIAL_BODIES_WITHIN_LIMITS_NOT_FULL_SOLVER_STATE"}


def summarize(cases, traces):
    pairs = []
    for a, b in itertools.combinations(cases, 2):
        if a["actions"] != b["actions"]:
            continue
        if a["mode"] != b["mode"] and a["repeat"] != b["repeat"]:
            continue
        pairs.append({"left": a["id"], "right": b["id"], "kind": "self_repeat" if a["mode"] == b["mode"] else "cross_mode",
                      **comparison(traces[a["id"]], traces[b["id"]])})
    return {"equivalence": "NOT_CERTIFIED", "comparisons": pairs,
            "repeatability": assess_repeatability(cases, traces),
            "native_parameters": assess_native_parameters(cases, traces),
            "native_start": assess_native_start(cases, traces),
            "native_input": assess_native_input(cases, traces),
            "first_step": {case["id"]: {phase: {key: row[key] for key in ("controls", "motors", "axes", "inputBeforeInjection")}
                                         for phase, row in zip(PHASES, traces[case["id"]][:3])} for case in cases}}


def assess_repeatability(cases, traces):
    """Require repeated starts for every fixture; compare recorded float32 state."""
    groups = {}
    fields = ("bodies", "controls", "motors", "bodyConfig", "motorParameters", "axes", "inputBeforeInjection")
    for case in cases:
        groups.setdefault((case["mode"], case["actions"]), []).append(case)
    results = []
    for key, group in groups.items():
        mismatches = []
        for a, b in itertools.combinations(group, 2):
            left, right = traces[a["id"]], traces[b["id"]]
            if len(left) != len(right) or not left:
                mismatches.append({"left": a["id"], "right": b["id"], "field": "trace_length"})
                continue
            identity = ("step", "phase", "bodyNames", "fixedDeltaTime")
            if any(any(ar[f] != br[f] for f in identity) for ar, br in zip(left, right)):
                mismatches.append({"left": a["id"], "right": b["id"], "field": "identity_or_timestep"})
            for field in fields:
                av, bv = (np.asarray([[r[field][axis] for axis in ("x", "y")]
                                      if field == "inputBeforeInjection" else r[field] for r in trace],
                                     dtype=np.float32) for trace in (left, right))
                if (not np.isfinite(av).all() or not np.isfinite(bv).all()
                        or not np.array_equal(av.view(np.uint32), bv.view(np.uint32))):
                    mismatches.append({"left": a["id"], "right": b["id"], "field": field})
        results.append({"mode": key[0], "actions": key[1], "repeats": len(group),
                        "passed": len(group) >= 2 and not mismatches, "mismatches": mismatches})
    return {"passed": bool(results) and all(r["passed"] for r in results), "fields": fields, "groups": results,
            "scope": "Recorded body/controller/parameter state across starts; not all Unity hidden state"}


def assess_native_input(cases, traces):
    """Compare observed native input state, independently of body trajectory error.

    The reference runs the game's automatic lifecycle. The candidate must match
    every measured stage, including pre-injection history and release tails.
    This does not certify spatial cursor targets or complete game dynamics.
    """
    indexed = {(c["mode"], c["actions"], c["repeat"]): c for c in cases}
    candidates = [c for c in cases if c["mode"] == "candidate"]
    columns = [0, 1, 2, 3, 4, 8, 9, 10]  # input, old input, gain, pause, sensitivity, trackpad
    results = []
    for case in candidates:
        reference = indexed.get(("native", case["actions"], case["repeat"]))
        if reference is None:
            results.append({"id": case["id"], "passed": False, "reason": "Missing paired native reference"})
            continue
        left, right = traces[reference["id"]], traces[case["id"]]
        if len(left) != len(right) or not left:
            results.append({"id": case["id"], "passed": False, "reason": "Incomplete paired traces"})
            continue
        errors = []
        max_delta = 0.
        for a, b in zip(left, right):
            av, bv = (np.asarray(r["controls"], dtype=np.float32)[columns] for r in (a, b))
            valid = np.isfinite(av).all() and np.isfinite(bv).all()
            if valid:
                max_delta = max(max_delta, float(np.max(np.abs(av.astype(float) - bv))))
            if (not valid or not np.array_equal(av.view(np.uint32), bv.view(np.uint32))
                    or (a["step"], a["phase"]) != (b["step"], b["phase"])):
                errors.append({"step": b["step"], "phase": b["phase"]})
        results.append({"id": case["id"], "reference": reference["id"], "passed": not errors,
                        "rows": len(right), "max_abs_delta": max_delta, "mismatches": errors})
    required = {"hold", "pulse", "reverse", "random"}
    covered = {c["actions"] for c in candidates}
    return {"passed": bool(results) and all(c["passed"] for c in results) and required <= covered,
            "missing_fixtures": sorted(required - covered), "control_columns": columns,
            "scope": "Bitwise input/history/gain/pause/settings at all recorded stages; not body/cursor dynamics",
            "cases": results}


def assess_native_start(cases, traces):
    """Accept the observed starting state within declared tolerances, not bitwise equality."""
    indexed = {(c["mode"], c["actions"], c["repeat"]): c for c in cases}
    candidates = [c for c in cases if c["mode"] == "candidate"]
    if not candidates:
        return {"passed": False, "reason": "Missing candidate"}
    results = []
    # mouseInput/oldMouse, adaptive gain, mw.xy, oldAngle, pause timer,
    # sensitivity and trackpad. Exact for input/gain/settings, spatial/angle
    # limits for the state inherited from the physical trajectory.
    control_limits = np.array([0, 0, 0, 0, 0, 1e-3, 1e-3, .05, 0, 0, 0])
    for case in candidates:
        reference = indexed.get(("native", case["actions"], case["repeat"]))
        if reference is None:
            results.append({"id": case["id"], "passed": False, "reason": "Missing paired native reference"})
            continue
        a, b = traces[reference["id"]][0], traces[case["id"]][0]
        if a["bodyNames"] != b["bodyNames"]:
            raise ValueError("Starting body ordering differs")
        bodies = body_error(a["bodies"], b["bodies"])
        controls = np.abs(np.asarray(a["controls"]) - np.asarray(b["controls"]))
        controls_ok = bool(controls.shape == (11,) and np.isfinite(controls).all()
                           and np.all(controls <= control_limits))
        results.append({"id": case["id"], "reference": reference["id"],
                        "passed": bodies["within_limits"] and controls_ok,
                        "bodies": bodies, "controller_errors": controls.tolist(),
                        "controller_within_limits": controls_ok})
    return {"passed": all(c["passed"] for c in results), "cases": results,
            "body_limits": LIMITS, "controller_limits": control_limits.tolist(),
            "scope": "Initial seven-body state and recorded controller values; not full solver or trajectory equivalence"}


def assess_native_parameters(cases, traces):
    """Parameter-only acceptance against a measured native reference, at every phase.

    Deliberately excludes trajectory/state equality and live motor target speeds.
    Controller caches must also match, or FixedUpdate can undo a property-only fix.
    """
    native = [c for c in cases if c["mode"] == "native"]
    candidate = [c for c in cases if c["mode"] == "candidate"]
    if not native or not candidate:
        return {"passed": False, "reason": "Both native and candidate traces are required"}
    reference = traces[native[0]["id"]][0]
    expected = np.asarray(reference.get("motorParameters", []), dtype=float)
    if expected.shape != (5,) or not np.isfinite(expected).all() or np.any(expected <= 0):
        return {"passed": False, "reason": "Missing or invalid native motor parameters"}
    if expected[0] != expected[3] or expected[1] != expected[4]:
        return {"passed": False, "reason": "Native motor caches disagree with joint properties"}
    results = []
    for case in cases:
        mismatches = []
        for row in traces[case["id"]]:
            actual = np.asarray(row.get("motorParameters", []), dtype=float)
            errors = []
            if actual.shape != (5,) or not np.isfinite(actual).all():
                errors.append("missing_or_invalid_motor_parameters")
            else:
                errors += [name for name, a, b in zip(MOTOR_PARAMETERS, actual, expected) if a != b]
            if not np.array_equal(row["bodyConfig"], reference["bodyConfig"]):
                errors.append("body_config")
            if row["fixedDeltaTime"] != reference["fixedDeltaTime"]:
                errors.append("fixed_dt")
            if errors:
                mismatches.append({"step": row["step"], "phase": row["phase"], "fields": errors})
        results.append({"id": case["id"], "passed": not mismatches,
                        "rows": len(traces[case["id"]]), "mismatches": mismatches})
    return {"passed": all(c["passed"] for c in results), "reference": native[0]["id"],
            "expected_motor_parameters": dict(zip(MOTOR_PARAMETERS, expected.tolist())),
            "scope": "Recorded motor caps/caches, body configuration and fixed dt only; not full dynamics equivalence",
            "cases": results}


def run_case(args, launcher, mode, kind, repeat):
    from env.goi_env import GoiEnv
    case_id = f"{kind}_{mode}_{repeat:02d}"
    path = args.out / (case_id + ".jsonl")
    fixture = args.out / (case_id + ".fixture.json")
    actions = actions_for(kind, args.steps, args.seed)
    fixture.write_text(json.dumps({"actions": actions.flatten().tolist(), "warmupFrames": args.native_warmup,
                                   "output": str(path.resolve())}), encoding="utf-8")
    overrides = {"dynamicsProbeMode": mode, "dynamicsProbeFile": str(fixture.resolve()),
                 "suppressHubDepthWrite": (mode == "candidate" if getattr(args, "hub_depth_fix", None) is None else args.hub_depth_fix),
                 "experimentalControlledUpdates": mode == "candidate", "experimentalSceneReset": mode == "candidate",
                 "experimentalResetDiagnostics": False, "stepFrames": 1,
                 "experimentalNativeInputSmoothing": args.native_input_smoothing and mode == "candidate",
                 "experimentalNativeNewGame": args.native_new_game and mode == "candidate"}
    ensure_idle(args.port)
    proc, env = None, None
    record = {"id": case_id, "mode": mode, "actions": kind, "repeat": repeat, "raw": path.name,
              "hub_depth_fix": overrides["suppressHubDepthWrite"]}
    started = time.monotonic()
    try:
        with runtime_settings(launcher.game_root, 1, args.port, overrides) as config:
            record["config"] = config
            proc = launcher.launch(wait=False)
            record["pid"] = proc.pid
            if mode == "candidate":
                env = GoiEnv(port=args.port, num_agents=1, timeout=args.timeout, startup_settle_s=2)
                env.connect()
                env._sock.settimeout(args.timeout)
                env._send_all(b"Y")
                obs, _ = env._recv_response()
                validate_obs(obs, 1)
                observations = [obs]
                for action in actions:
                    obs, done = env.step(action)
                    validate_obs(obs, 1)
                    if np.any(done):
                        raise ValueError("Unexpected done")
                    observations.append(obs)
                np.savez_compressed(args.out / (case_id + ".npz"), observations=observations, actions=actions)
            deadline = started + args.timeout
            while not Path(str(path) + ".done").exists():
                if proc.poll() is not None:
                    raise RuntimeError(f"Game exited: {proc.returncode}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Missing trace completion: {case_id}")
                time.sleep(0.2)
            trace = read_trace(path, actions, mode)
            record.update(status="COMPLETE", seconds=time.monotonic() - started, rows=len(trace))
            return record, trace
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
            shutil.copyfile(log, args.out / (case_id + ".game.log"))


def run(args):
    from start.game_launcher import GameLauncher
    launcher = GameLauncher(str(args.game_exe.resolve()))
    project = json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8-sig"))
    production = Path(project["game"]["executable_path"]).resolve().parent
    if launcher.game_root == production:
        raise ValueError("Use an independent game copy, not the production installation")
    ensure_idle(args.port)
    args.out.mkdir(parents=True, exist_ok=False)
    production_files = [production / "BepInEx/plugins/GameRuntime_v2.dll", production / "GoiData/runtime_config.json"]
    production_before = {str(p): sha256(p) for p in production_files}
    report = {"status": "RUNNING", "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "initial_body_limits": LIMITS, "cases": [], "errors": [],
              "fingerprints": {"script": sha256(Path(__file__)), "plugin": sha256(launcher.game_root / "BepInEx/plugins/GameRuntime_v2.dll"),
                               "game_assembly": sha256(launcher.game_root / "GettingOverIt_Data/Managed/Assembly-CSharp.dll")},
              "limitations": ["Native reference still uses BepInEx and input interception; not an uninstrumented executable.",
                              "Native readiness plus zero-input warmup differs from candidate canonical initialization.",
                              "Contact and joint solver hidden caches are not observed.",
                              "Single-agent, short-horizon dynamics; does not certify task interfaces or full gameplay.",
                              "Native autosave-to-disk is suppressed. Other PlayerPrefs changes are not isolated.",
                              "Body tolerances are predefined screening thresholds, not a proof of matching solver states."]}
    traces = {}
    try:
        for kind in args.actions:
            for repeat in range(args.repeats):
                for mode in args.modes:
                    LOG.info("Running %s %s repeat=%d", kind, mode, repeat)
                    save_report(args.out / "result.json", report)
                    case, trace = run_case(args, launcher, mode, kind, repeat)
                    report["cases"].append(case)
                    traces[case["id"]] = trace
                    save_report(args.out / "result.json", report)
                    LOG.info("Recorded %s: %d rows, %.1fs", case["id"], len(trace), case["seconds"])
        report.update(status="COMPLETE", analysis=summarize(report["cases"], traces))
        if args.require_native_parameters and not report["analysis"]["native_parameters"]["passed"]:
            report["status"] = "FAIL"
        if args.require_native_start and not report["analysis"]["native_start"]["passed"]:
            report["status"] = "FAIL"
        if args.require_native_input and not report["analysis"]["native_input"]["passed"]:
            report["status"] = "FAIL"
        if args.require_repeatable and not report["analysis"]["repeatability"]["passed"]:
            report["status"] = "FAIL"
    except (Exception, KeyboardInterrupt) as exc:
        LOG.exception("Dynamics experiment failed")
        report["status"] = "ERROR"
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        report["production_before"] = production_before
        report["production_after"] = {str(p): sha256(p) for p in production_files}
        report["production_unchanged"] = report["production_before"] == report["production_after"]
        if not report["production_unchanged"]:
            report["status"] = "ERROR"
            report["errors"].append("Production files changed")
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_report(args.out / "result.json", report)
    LOG.info("%s: %s", report["status"], args.out / "result.json")
    return {"COMPLETE": 0, "FAIL": 1, "ERROR": 2}[report["status"]]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-exe", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", choices=["native", "native_dual", "candidate"], default=["native", "native_dual", "candidate"])
    parser.add_argument("--actions", nargs="+", choices=["zero", "hold", "pulse", "reverse", "random"], default=["zero", "hold", "pulse"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--native-warmup", type=int, default=240)
    parser.add_argument("--native-new-game", action="store_true", help="Use the experimental native new-game initialization")
    parser.add_argument("--native-input-smoothing", action="store_true", help="Inject axes only; preserve native input history")
    parser.add_argument("--hub-depth-fix", action=argparse.BooleanOptionalAction, default=None,
                        help="Override Hub depth fix; default off for native references, on for candidate")
    parser.add_argument("--require-repeatable", action="store_true", help="Require bitwise repeated-start body/controller traces")
    parser.add_argument("--require-native-input", action="store_true", help="Require exact native input state across hold/pulse/reverse/random")
    parser.add_argument("--require-native-start", action="store_true", help="Require initial body/controller state within declared limits")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--port", type=int)
    parser.add_argument("--require-native-parameters", action="store_true",
                        help="Exit 1 if measured joint caps/caches, body config or fixed dt differ from native")
    args = parser.parse_args(argv)
    if args.require_repeatable and args.repeats < 2:
        parser.error("--require-repeatable requires at least two repeats")
    if args.require_native_input and (not args.native_input_smoothing
            or not {"native", "candidate"}.issubset(args.modes)
            or not {"hold", "pulse", "reverse", "random"}.issubset(args.actions) or args.steps < 4):
        parser.error("--require-native-input requires smoothing, native/candidate, hold/pulse/reverse/random and >=4 steps")
    if args.require_native_start and (args.native_warmup != 0 or not {"native", "candidate"}.issubset(args.modes)):
        parser.error("--require-native-start requires native/candidate and --native-warmup 0")
    if min(args.repeats, args.steps) < 1 or args.native_warmup < 0 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Invalid count or timeout")
    if len(set(args.modes)) != len(args.modes) or len(set(args.actions)) != len(args.actions):
        parser.error("Duplicate modes or action fixtures")
    if args.port is None:
        args.port = int(json.loads((ROOT / "src/config/project.json").read_text(encoding="utf-8-sig"))["tcp_port"])
    args.out = args.out.resolve()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(run(parse_args()))

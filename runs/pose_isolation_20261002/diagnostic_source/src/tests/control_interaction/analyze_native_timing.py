"""Analyze native event audits by physics boundaries, independently of render frames.

This is a diagnostic, not a native-equivalence acceptance test. Each input is a
test_native_dynamics output directory created with --timing-audit.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from test_native_dynamics import body_error, read_trace, actions_for
from test_reset_state import save_report


def distance(a, b):
    a, b = (np.asarray(x, dtype=np.float32).reshape(-1, 7, 6) for x in (a, b))
    result = body_error(a, b)
    result["position_distance_m"] = float(np.linalg.norm(a.astype(float)[..., :2] - b[..., :2], axis=-1).max())
    return result


def brief(event):
    return {k: event[k] for k in ("sequence", "phase", "renderFrame", "controllers", "physics", "fixedTime",
                                 "time", "deltaTime", "pause", "loadFinished", "targetFrameRate", "captureFramerate")}


def read_audit(path, trace):
    events = [json.loads(line) for line in Path(str(path) + ".timeline.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    if int(Path(str(path) + ".timeline.done").read_text()) != len(events) or not events:
        raise ValueError("Incomplete timing audit")
    for i, event in enumerate(events):
        if event["sequence"] != i:
            raise ValueError("Broken event sequence")
        for field in ("bodies", "transforms"):
            if len(event[field]) != 42 or not np.isfinite(event[field]).all():
                raise ValueError("Invalid audit state")
    for phase in ("before_controller", "after_controller", "before_physics", "after_physics"):
        rows = [e for e in events if e["phase"] == phase]
        if not rows or len(set(e["fixedTime"] for e in rows)) != len(rows):
            raise ValueError("Missing or repeated physics boundary")
        counter = "controllers" if "controller" in phase else "physics"
        if any(b[counter] != a[counter] + 1 for a, b in zip(rows, rows[1:])):
            raise ValueError("Skipped physics/controller count")
    lookup = {(e["phase"], e["fixedTime"]): e for e in events}
    mapping = {"before_control": "before_controller", "after_control": "after_controller", "after_physics": "after_physics"}
    for row in trace:
        event = lookup[(mapping[row["phase"]], row["fixedTime"])]
        if event["renderFrame"] != row["renderFrame"] or not distance(event["bodies"], row["bodies"])["bitwise_equal"]:
            raise ValueError("Probe/audit sampling boundary mismatch")
    return events


def describe(events, trace):
    first = trace[0]
    ready = next(e for e in events if e["phase"] == "before_controller" and e["fixedTime"] == first["fixedTime"])
    opening = [brief(e) for e in events if ".Start" in e["phase"] or "OpeningAnimation" in e["phase"]]
    loaded = next((brief(e) for e in events if e["loadFinished"]), None)
    # Pair actual method calls (including nested opening callbacks), never rows
    # from separate renders. Record rigidbody changes and rendered XY changes.
    stacks, changes = {}, {}
    for event in events:
        phase = event["phase"]
        if "." not in phase:
            continue
        name = phase.split("_", 1)[1]
        if phase.startswith("before_"):
            stacks.setdefault(name, []).append(event)
        elif stacks.get(name):
            before = stacks[name].pop()
            error = distance(before["bodies"], event["bodies"])
            a, b = (np.asarray(e["transforms"]).reshape(7, 6) for e in (before, event))
            xy = float(np.abs(a[:, :2] - b[:, :2]).max())
            item = changes.setdefault(name, {"calls": 0, "body_changed_calls": 0, "max_body_component_delta": 0.,
                                              "max_transform_xy_delta": 0., "first_body_change": None})
            item["calls"] += 1
            delta = float(np.abs(np.asarray(before["bodies"]) - event["bodies"]).max())
            item["max_body_component_delta"] = max(item["max_body_component_delta"], delta)
            item["max_transform_xy_delta"] = max(item["max_transform_xy_delta"], xy)
            if not error["bitwise_equal"]:
                item["body_changed_calls"] += 1
                if item["first_body_change"] is None:
                    delta_values = np.asarray(event["bodies"]) - before["bodies"]
                    item["first_body_change"] = {"before": brief(before), "after": brief(event), "error": error,
                        "target": event.get("operationTarget"), "value": event.get("operationValue"),
                        "changed_components": [{"index": int(i), "before": before["bodies"][i], "after": event["bodies"][i]}
                                               for i in np.flatnonzero(delta_values)]}
    return {"events": len(events), "scene_bound": brief(events[0]), "opening_events": opening,
            "first_load_finished": loaded, "first_probe": brief(ready),
            "interpolation_at_probe": ready["interpolation"], "auto_sync_at_probe": ready["autoSync"],
            "simulation_mode_at_probe": ready["simulationMode"], "callback_changes": changes,
            "probe_render_frames": [r["renderFrame"] for r in trace if r["phase"] == "before_control"],
            "physics_dt_values": sorted(set(e["fixedDeltaTime"] for e in events)),
            "render_delta_values": sorted(set(e["deltaTime"] for e in events if e["phase"] == "before_PlayerControl.Update"))}


def validate_pose_sites(sites, expected_skips, events=None):
    """Reject silent/no-op ablations and missing fine-grained audit records."""
    if not isinstance(sites, list) or len(sites) != 22 or [s["index"] for s in sites] != list(range(22)):
        raise ValueError("Unexpected or incomplete pose call-site manifest")
    executed = set()
    for site in sites:
        calls, skipped = site["calls"], site["skippedCalls"]
        operation = site.get("operation")
        if calls < 0 or skipped != (calls if operation in expected_skips else 0):
            raise ValueError("Pose ablation did not skip exactly the selected operations")
        if calls:
            executed.add(operation)
        if events is not None:
            for boundary in ("before", "after"):
                phase = f"{boundary}_PoseWrite.{site['index']}.{operation}"
                rows = [e for e in events if e["phase"] == phase]
                if len(rows) != calls or any(e["skipped"] != (operation in expected_skips) for e in rows):
                    raise ValueError("Pose call-site audit is incomplete or contradicts the manifest")
    if not set(expected_skips) <= executed:
        raise ValueError("Selected pose ablation never executed")
    return sites


def compare(left, right):
    lt, le = left
    rt, re = right
    first_physics_difference = None
    lp, rp = ([e for e in events if e["phase"] == "after_physics"] for events in (le, re))
    for a, b in zip(lp, rp):
        if a["physics"] != b["physics"]:
            raise ValueError("Bad physics alignment")
        error = distance(a["bodies"], b["bodies"])
        if not error["bitwise_equal"]:
            first_physics_difference = {"left": brief(a), "right": brief(b), "error": error}
            break
    controls = [0, 1, 2, 3, 4, 8, 9, 10]
    return {"scene_bodies": distance(le[0]["bodies"], re[0]["bodies"]),
            "initial_bodies": distance(lt[0]["bodies"], rt[0]["bodies"]),
            "trajectory": distance([r["bodies"] for r in lt], [r["bodies"] for r in rt]),
            "input_state_equal": bool(np.array_equal(np.asarray([r["controls"] for r in lt], dtype=np.float32)[:, controls],
                                                     np.asarray([r["controls"] for r in rt], dtype=np.float32)[:, controls])),
            "fixed_times_equal": [r["fixedTime"] for r in lt] == [r["fixedTime"] for r in rt],
            "physics_event_counts": [len(lp), len(rp)], "first_physics_difference": first_physics_difference}


def analyze(directories):
    details, traces, pairs, sources = {}, {}, [], {}
    for directory in directories:
        report = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        if report["status"] != "COMPLETE" or not report["production_unchanged"]:
            raise ValueError(f"Incomplete experiment: {directory}")
        args = report["parameters"]
        sources[directory.name] = {"directory": str(directory.resolve()), "parameters": args,
                                   "fingerprints": report["fingerprints"],
                                   "production_unchanged": report["production_unchanged"]}
        for case in report["cases"]:
            key = directory.name + "/" + case["id"]
            path = directory / case["raw"]
            trace = read_trace(path, actions_for(case["actions"], args["steps"], args["seed"]), "native")
            events = read_audit(path, trace)
            traces[key] = trace, events
            details[key] = describe(events, trace)
            if args.get("pose_write_audit") or args.get("skip_pose_operations"):
                sites = json.loads(Path(str(path) + ".pose_sites.json").read_text(encoding="utf-8-sig"))
                details[key]["pose_sites"] = validate_pose_sites(sites, args.get("skip_pose_operations", []),
                                                                events if args.get("pose_write_audit") else None)
    for a, b in itertools.combinations(traces, 2):
        pairs.append({"left": a, "right": b, **compare(traces[a], traces[b])})
    groups = {}
    for name in sources:
        own = [p for p in pairs if all(p[k].startswith(name + "/") for k in ("left", "right"))]
        groups[name] = {"launches": sum(k.startswith(name + "/") for k in traces),
                        "pairs": len(own),
                        "all_initial_bodies_bitwise_equal": all(p["initial_bodies"]["bitwise_equal"] for p in own) if own else None,
                        "all_trajectories_bitwise_equal": all(p["trajectory"]["bitwise_equal"] for p in own) if own else None,
                        "all_input_states_equal": all(p["input_state_equal"] for p in own) if own else None,
                        "max_initial_position_distance_m": max((p["initial_bodies"]["position_distance_m"] for p in own), default=None),
                        "max_trajectory_position_distance_m": max((p["trajectory"]["position_distance_m"] for p in own), default=None)}
    return {"status": "DIAGNOSTIC_COMPLETE", "scope": "Instrumented native reference; ablations are diagnostic changes, not fixes",
            "source_runs": sources, "groups": groups, "cases": details, "comparisons": pairs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.directories)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_report(args.out, result)
    print(f"Analyzed {len(result['cases'])} launches, {len(result['comparisons'])} pairs: {args.out}")

"""Compare zero-input native/candidate poses, independent of global rigid motion.

Consumes complete test_native_dynamics runs. Never starts or changes the game.
Time zero means each mode's own readiness boundary; candidate already completed
its canonical initialization. Additional steps are counted from that boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from test_native_dynamics import read_trace


BODY_NAMES = ["Player", "Hub", "Slider", "Handle", "PoleMiddle", "Tip", "Cursor"]


def wrap(angle):
    return (np.asarray(angle) + 180) % 360 - 180


def bodies(row):
    if row["bodyNames"] != BODY_NAMES:
        raise ValueError("Unexpected body identity/order")
    result = np.asarray(row["bodies"], dtype=float).reshape(7, 6)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite body state")
    return result


def player_frame(state):
    angle = np.deg2rad(state[0, 2])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return (state[:, :2] - state[0, :2]) @ rotation


def describe(row):
    state = bodies(row)
    return {
        "player_xy_m": state[0, :2].tolist(), "player_angle_deg": float(wrap(state[0, 2])),
        "tip_to_hub_distance_m": float(np.linalg.norm(state[5, :2] - state[1, :2])),
        "cursor_to_tip_distance_m": float(np.linalg.norm(state[6, :2] - state[5, :2])),
        "hinge_angle_deg": row["motors"][4], "slider_translation_m": row["motors"][5],
        "max_linear_speed_m_s": float(np.linalg.norm(state[:, 3:5], axis=1).max()),
        "max_angular_speed_deg_s": float(np.abs(state[:, 5]).max()),
        "controller_values": row["controls"],
    }


def difference(native, candidate):
    a, b = bodies(native), bodies(candidate)
    world = b[:, :2] - a[:, :2]
    local = player_frame(b) - player_frame(a)
    angle = wrap(b[:, 2] - a[:, 2])
    relative_angle = wrap((b[:, 2] - b[0, 2]) - (a[:, 2] - a[0, 2]))
    return {
        "max_world_position_distance_m": float(np.linalg.norm(world, axis=1).max()),
        "max_world_position_component_m": float(np.abs(world).max()),
        "max_world_angle_difference_deg": float(np.abs(angle).max()),
        "max_position_distance_after_aligning_player_m": float(np.linalg.norm(local, axis=1).max()),
        "tip_distance_after_aligning_player_m": float(np.linalg.norm(local[5])),
        "slider_translation_difference_m": float(candidate["motors"][5] - native["motors"][5]),
        "bodies": [{"name": name, "world_delta_xy_m": world[i].tolist(),
                    "angle_delta_deg": float(angle[i]), "player_frame_delta_xy_m": local[i].tolist(),
                    "relative_angle_delta_deg": float(relative_angle[i]),
                    "velocity_delta_xy_m_s": (b[i, 3:5] - a[i, 3:5]).tolist(),
                    "angular_velocity_delta_deg_s": float(b[i, 5] - a[i, 5])}
                   for i, name in enumerate(native["bodyNames"])],
    }


def checkpoint(rows, count):
    # Initial sample is before the first controller call; later samples are after
    # exactly count physics updates. Never compare a native render timestamp.
    return rows[0] if count == 0 else rows[3 * (count - 1) + 2]


def analyze(folder, checkpoints):
    report = json.loads((folder / "result.json").read_text(encoding="utf-8-sig"))
    if report["status"] != "COMPLETE" or not report["analysis"]["native_parameters"]["passed"]:
        raise ValueError("Require a complete run with matching native parameters")
    steps = report["parameters"]["steps"]
    if min(checkpoints) < 0 or max(checkpoints) > steps:
        raise ValueError("Checkpoint outside recorded steps")
    traces, cases = {}, {}
    hashes = {}
    for case in report["cases"]:
        if case["actions"] != "zero" or case["mode"] not in ("native", "candidate"):
            raise ValueError("Pose isolation requires zero actions in native/candidate only")
        fixture = json.loads((folder / (case["id"] + ".fixture.json")).read_text())
        actions = np.asarray(fixture["actions"], dtype=np.float32).reshape(-1, 2)
        if len(actions) != steps or np.any(actions != 0):
            raise ValueError("Not a zero-input trajectory")
        path = folder / case["raw"]
        rows = read_trace(path, actions, case["mode"])
        # Zero command alone is insufficient: pre-existing smoothed input could
        # still move the cursor. Verify both current and previous controller input.
        if any(np.any(np.asarray(row["controls"][:4]) != 0) for row in rows):
            raise ValueError("Controller input history is not zero")
        key = case["mode"], case["repeat"]
        if key in traces:
            raise ValueError("Duplicate mode/repeat")
        traces[key], cases[key] = rows, case["id"]
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    repeats = sorted({repeat for mode, repeat in traces})
    if len(repeats) < 2 or any((mode, repeat) not in traces for mode in ("native", "candidate") for repeat in repeats):
        raise ValueError("Require paired independent repeats in both modes")
    measured = []
    for count in checkpoints:
        pairs = []
        for repeat in repeats:
            a, b = (checkpoint(traces[mode, repeat], count) for mode in ("native", "candidate"))
            pairs.append({"repeat": repeat, "native": describe(a), "candidate": describe(b),
                          "difference": difference(a, b)})
        scalars = [key for key, value in pairs[0]["difference"].items() if isinstance(value, float)]
        ranges = {key: [min(p["difference"][key] for p in pairs), max(p["difference"][key] for p in pairs)] for key in scalars}
        repeatability = {}
        for mode in ("native", "candidate"):
            diffs = [difference(checkpoint(traces[mode, a], count), checkpoint(traces[mode, b], count))
                     for a, b in itertools.combinations(repeats, 2)]
            repeatability[mode] = {key: max(abs(d[key]) for d in diffs) for key in scalars}
        measured.append({"additional_physics_steps": count, "additional_simulated_seconds": count * traces["native", repeats[0]][0]["fixedDeltaTime"],
                         "cross_mode_ranges": ranges, "self_repeat_max_difference": repeatability, "pairs": pairs})
    return {"status": "COMPLETE", "source_report": str((folder / "result.json").resolve()),
            "source_plugin_sha256": report["fingerprints"]["plugin"], "repeats_per_mode": len(repeats),
            "native_extra_warmup_before_recording": report["parameters"]["native_warmup"],
            "zero_actions_and_zero_input_history_verified": True, "native_parameters_matched": True,
            "measurements": measured, "raw_sha256": hashes,
            "limits": ["Native and candidate have different initialization recipes before time zero.",
                       "Player-frame alignment removes global translation and rotation, not joint articulation.",
                       "Body coordinates are Rigidbody2D reference points, not pot geometric center or collision outlines.",
                       "Small initial velocities and zero input do not guarantee identical hidden solver states.",
                       "This test does not separate pose template, animation, hierarchy and update-scheduling effects."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[0, 40, 120, 240, 600, 1200])
    args = parser.parse_args()
    output = analyze(args.run.resolve(), args.checkpoints)
    path = args.run / "start_pose_analysis.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for item in output["measurements"]:
        print(item["additional_physics_steps"], item["cross_mode_ranges"])
    print(path)

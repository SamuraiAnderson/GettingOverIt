"""Recompute the independent-world experiment's checks from raw traces."""
import copy
import json
from pathlib import Path
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/config/project.json").is_file())
BASE = ROOT / "runs/isolated_worlds_20261002"
COLS33 = [i for i in range(33) if i != 28]
COLS35 = [i for i in range(35) if i != 28]
POSITIONS = [(0, 1), (5, 6), (10, 11), (15, 16), (19, 20), (23, 24), (29, 30)]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def trace(case, epoch=0):
    report = read(BASE / case / "result.json")
    assert report["status"] == "COMPLETE", case
    path = BASE / case / f"trace_epoch_{epoch:02d}.jsonl"
    values = np.array([json.loads(x)["states"] for x in path.read_text().splitlines()], dtype=np.float32)
    assert values.shape[0] == report["steps"] + 1
    assert np.isfinite(values).all()
    return values


def comparison(a, b):
    assert a.shape[0] == b.shape[0]
    cols = COLS33 if min(a.shape[-1], b.shape[-1]) == 33 else COLS35
    unequal = a[..., cols].view(np.uint32) != b[..., cols].view(np.uint32)
    per_step = unequal.reshape(len(a), -1).any(axis=1)
    first = np.flatnonzero(per_step)
    position_error = max(float(np.linalg.norm(a[..., xy].astype(float) - b[..., xy], axis=-1).max()) for xy in POSITIONS)
    return {"bitwise_equal_excluding_timestamp": not bool(unequal.any()),
            "first_difference_step": int(first[0]) if len(first) else None,
            "compared_steps_including_start": len(a),
            "compared_columns": cols,
            "max_observed_body_or_cursor_position_error_m": position_error}


def normalized_shapes(case, slot=0, epoch=0):
    world = read(BASE / case / f"worlds_epoch_{epoch}.json")[slot]
    shapes = copy.deepcopy(world["colliders"])
    for row in shapes:
        row["path"] = row["path"].replace("ExperimentPlayer_1", "ExperimentPlayer_0")
    return world, shapes


def main():
    cold = trace("two_1200_reset")
    repeat = trace("two_1200_reset_repeat")
    single = trace("one_1200_reset")
    tcp = np.load(BASE / "reference_single_tcp/trace.npz")["states"]
    checks = {}
    checks["single_driver_vs_unchanged_tcp_reference"] = comparison(single[:, 0], tcp[:, 0])
    for case in ("two_1200_reset", "two_1200_reset_repeat"):
        for epoch in range(read(BASE / case / "result.json")["resets"] + 1):
            current = trace(case, epoch)
            checks[f"{case}_epoch{epoch}_worlds_match"] = comparison(current[:, 0], current[:, 1])
            checks[f"{case}_epoch{epoch}_vs_original_cold"] = comparison(current, cold)
    for epoch in (1, 2):
        checks[f"single_reset{epoch}_vs_original_tcp"] = comparison(trace("one_1200_reset", epoch)[:, 0], tcp[:, 0])
    checks["world0_unchanged_when_world1_actions_zero"] = comparison(trace("two_1200_otherzero")[:, 0], cold[:, 0])
    checks["single_extra_startup_sync_vs_two_worlds"] = comparison(trace("one_startupsync_1200")[:, 0], cold[:, 0])
    checks["pilot_reproduces_first40_steps"] = comparison(trace("pilot_two_40"), cold[:41])
    assert all(v["bitwise_equal_excluding_timestamp"] for v in checks.values())

    boundaries = {
        "two_worlds_vs_unchanged_single_reference": comparison(cold[:, 0], tcp[:, 0]),
        "removing_explicit_sync_does_not_fix_two_world_reference_difference": comparison(trace("two_scenesync_40")[:, 0], tcp[:41, 0]),
        "single_without_explicit_sync_vs_reference": comparison(trace("one_scenesync_40")[:, 0], tcp[:41, 0]),
    }
    assert not boundaries["two_worlds_vs_unchanged_single_reference"]["bitwise_equal_excluding_timestamp"]
    a_world, a_shapes = normalized_shapes("two_1200_reset_repeat")
    b_world, b_shapes = normalized_shapes("two_1200_reset_repeat", 1)
    _, one_shapes = normalized_shapes("pilot_one_40")
    assert a_world["valid"] and b_world["valid"]
    assert not a_world["isDefault"] and not b_world["isDefault"]
    assert a_world["sceneHandle"] != b_world["sceneHandle"]
    assert a_shapes == b_shapes
    shape_differences = [{"path": x["path"], "field": key, "single": y[key], "two": value}
                         for x, y in zip(a_shapes, one_shapes) for key, value in x.items() if value != y[key]]
    assert all(x["path"] == "Props/tower+crane" and x["field"] == "position" for x in shape_differences)
    assert all(x["detection"] == "Continuous" for x in a_shapes if x["path"].endswith("/Tip"))
    cold_two = read(BASE / "two_1200_reset_repeat/diagnostics_0_0.json")[0]["current"]
    cold_one = read(BASE / "reference_single_tcp/diagnostics.json")[0]["packet"]["current"]
    body_differences = [{"body": a["name"], "field": k, "single": b[k], "two": value}
                        for a, b in zip(cold_two["bodies"], cold_one["bodies"])
                        for k, value in a.items() if k != "name" and value != b[k]]
    pos_delta = max(np.linalg.norm([a["position"][k] - b["position"][k] for k in ("x", "y")])
                    for a, b in zip(cold_two["bodies"], cold_one["bodies"]))
    angle_delta = max(abs(a["rotation"] - b["rotation"]) for a, b in zip(cold_two["bodies"], cold_one["bodies"]))
    detailed = []
    for path in sorted((BASE / "two_1200_reset_repeat").glob("diagnostics_*.json")):
        packets = read(path)
        normalize = lambda obj: json.dumps(obj, sort_keys=True).replace("ExperimentPlayer_1", "ExperimentPlayer_0")
        entry = {"file": path.name, **{key: normalize(packets[0]["current"][key]) == normalize(packets[1]["current"][key])
                                     for key in ("bodies", "controls", "joints")}}
        if not path.name.endswith("_0.json"):
            assert entry["bodies"] and entry["controls"] and entry["joints"]
        detailed.append(entry)
    result = {
        "status": "ISOLATION_PASS_REFERENCE_EQUIVALENCE_NOT_YET_PASS",
        "scope": "One archived 1200-action trace; two independent native Mian scenes in one process; CPU physics stepped sequentially.",
        "readiness": "Experimental only; not a production implementation or a guarantee for arbitrary inputs/terrain.",
        "checks": checks, "reference_boundaries": boundaries,
        "sampled_full_diagnostics_between_worlds": detailed,
        "identical_action_full_length_cases": {"cold_processes": 2, "reset_epochs": 3, "steps_per_epoch": 1200, "worlds_per_epoch": 2},
        "world_geometry": {"bodies_including_inactive_per_world": a_world["bodyCount"],
                           "colliders_including_inactive_per_world": a_world["colliderCount"],
                           "same_shapes_materials_ccd_layers_between_worlds": a_shapes == b_shapes,
                           "single_vs_two_metadata_differences": shape_differences},
        "initial_native_body_difference": {"max_position_m": float(pos_delta), "max_rotation_deg": angle_delta,
                                           "fields": body_differences},
        "interpretation": [
            "Independent worlds removed the observed step31 cross-agent divergence for this trace.",
            "World0 was bitwise unaffected when world1 received zero instead of archived actions.",
            "Single-world file driver and resets reproduce the existing TCP reference for all1200 steps.",
            "One explicit SyncTransforms before the first action makes the single-world trace identical to two-world output for all1200 steps.",
            "Removing explicit SyncTransforms from the simulation helper alone does not remove the extra initialization synchronization effect.",
            "At step0, equal observations do not imply equal rigidbody state: before first action, the two scenes are at different transform synchronization boundaries.",
            "Native solver internals remain unobserved; no claim identifies the exact internal CCD/TOI ordering algorithm."
        ],
        "production_changed": False,
    }
    (BASE / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed_checks": len(checks),
                      "initial_position_delta_m": float(pos_delta), "initial_rotation_delta_deg": angle_delta,
                      "summary": str(BASE / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()

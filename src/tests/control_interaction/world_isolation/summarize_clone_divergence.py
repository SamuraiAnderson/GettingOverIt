"""Recompute evidence for the step-31 clone contact divergence experiment."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "runs/clone_divergence_step31_20261002"
COLS = [i for i in range(33) if i != 28]


def trace(name):
    with np.load(RUN / name / "trace.npz") as source:
        return source["states"].copy(), source["actions"].copy()


def equal(a, b):
    return bool(np.array_equal(np.float32(a).view(np.uint32), np.float32(b).view(np.uint32)))


def frames(name):
    return [json.loads(line) for line in (RUN / name / "phases.jsonl").read_text().splitlines()]


def bodies(frame, agent):
    return [{k: v for k, v in body.items() if k not in ("agent", "name")}
            for body in frame["bodies"] if body["agent"] == agent]


def joints(frame, agent):
    return [{k: v for k, v in joint.items() if k not in ("agent", "name")}
            for raw in frame["joints"] for joint in [json.loads(raw)] if joint["agent"] == agent]


def main():
    base, actions = trace("baseline_shared8")
    observed, _ = trace("probe_shared8")
    assert equal(base[:, :, COLS], observed[:, :, COLS])
    assert all(equal(actions[:, 0], actions[:, i]) for i in range(1, 8))
    original = json.loads((RUN / "probe_original8/result.json").read_text())
    assert original["matches_archived_states_without_timestamp"]
    rows = frames("probe_shared8")
    assert len(rows[0]["pairs"]) == 1008
    assert all(p["ignored"] for p in rows[0]["pairs"])
    assert not any(p["owner"] >= 0 and p["otherOwner"] >= 0 and p["owner"] != p["otherOwner"]
                   for row in rows for col in row["colliders"] for p in col["contacts"])
    phase_summary = []
    for row in rows:
        if row["step"] != 31:
            continue
        f = row["frame"]
        phase_summary.append({"phase": row["phase"],
                              "all_clone_bodies_equal": all(bodies(f, 1) == bodies(f, i) for i in range(2, 8)),
                              "all_clone_controls_equal": all(f["controls"][1] == f["controls"][i] for i in range(2, 8)),
                              "all_clone_joints_equal": all(joints(f, 1) == joints(f, i) for i in range(2, 8))})
    assert all(s["all_clone_bodies_equal"] and s["all_clone_controls_equal"] and s["all_clone_joints_equal"] for s in phase_summary[:2])
    assert not phase_summary[2]["all_clone_bodies_equal"]
    interventions = []
    for name in ("ignore_rock2_shared8", "ignore_rock3_shared8", "discrete_shared8"):
        s, a = trace(name)
        pre_equal = equal(s[:31, :, COLS], base[:31, :, COLS])
        aligned = all(equal(s[:, 1, COLS], s[:, i, COLS]) for i in range(2, 8))
        assert pre_equal and aligned and equal(actions, a)
        interventions.append({"case": name, "same_actions": True, "states_through_step30_unchanged": pre_equal,
                              "seven_clones_bitwise_equal_through_step40": aligned})
    rocks = [r for r in rows[0]["terrain"] if len(r["worldPoints"]) == 26]
    p, q = [np.float32(r["worldPoints"]).reshape(-1, 2) for r in rocks]
    tip_deltas = np.linalg.norm(base[31, 1:, 23:25] - base[31, 1, 23:25], axis=1)
    counts = []
    for name in ("probe_shared3", "probe_shared4", "probe_shared8"):
        counts.append(json.loads((RUN / name / "result.json").read_text()))
    summary = {
        "status": "COMPLETE", "steps": 40,
        "finding": "The first clone split occurs inside step-31 physics, at continuous hammer-tip contacts with overlapping terrain contours.",
        "scope": "Seven duplicates compared with one another; original agent0 has a separate cold-state discrepancy.",
        "shared_actions_identical": True,
        "read_only_probe_preserves_baseline_bitwise": True,
        "original_action_replay_matches_archived_run_bitwise": True,
        "cross_agent_collision_pairs_ignored": 1008,
        "cross_agent_contacts_observed": 0,
        "phases_at_step31": phase_summary,
        "max_tip_distance_between_clone1_and_other_clones_at_step31_m": float(tip_deltas.max()),
        "overlapping_terrain": {"names": [r["name"] for r in rocks], "vertices_each": 13,
                                "exactly_matching_corresponding_vertices": int(np.all(p == q, axis=1).sum()),
                                "remaining_vertices_differ": True},
        "interventions": interventions, "agent_count_comparisons": counts,
        "limitations": [
            "Contact ordering and impulses are observed; internal native CCD/TOI scheduling and tie-breaking are not exposed.",
            "Disabling a terrain contact or switching to discrete collision changes dynamics and is diagnostic only.",
            "Passing intervention alignment covers these 40 steps, not all gameplay.",
            "No production dynamics or terrain was changed. Temporary probe is removed after the experiment."
        ],
        "artifacts": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).with_name("replay_clone_divergence.py"),
            Path(__file__).with_name("CloneDivergenceProbe.cs"))},
    }
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("agent_count_comparisons", "artifacts")}, indent=2))


if __name__ == "__main__":
    main()

"""Check archived per-agent policy actions, motion, and independent scene IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    read = lambda name: json.loads((args.run / name).read_text(encoding="utf-8-sig"))
    status = read("run_status.json")
    assert status["status"] == "COMPLETE" and status["completed_steps"] == status["steps"]
    epochs = read("worlds_epoch_0.json")
    assert len(epochs) == 8 and len({e["sceneHandle"] for e in epochs}) == 8
    assert all(e["valid"] and not e["isDefault"] for e in epochs)
    states, actions = [], []
    for i in range(8):
        episode = read(f"deterministic_000_agent{i}.json")
        states.append(episode["states"])
        actions.append([row["action"] for row in episode["trace"]])
    states, actions = np.float32(states), np.float32(actions)
    assert states.shape == (8, status["steps"] + 1, 35)
    assert actions.shape == (8, status["steps"], 2)
    cols = [i for i in range(35) if i != 28]
    observed = states[:, :, cols].copy().view(np.uint32)
    action_bits = actions.copy().view(np.uint32)
    same_states = [np.array_equal(observed[0], x) for x in observed]
    same_actions = [np.array_equal(action_bits[0], x) for x in action_bits]
    native = {}
    norm = lambda obj: re.sub(r"ExperimentPlayer_\d+", "Player", json.dumps(obj, sort_keys=True))
    for step in (31, status["steps"]):
        packets = read(f"diagnostics_0_{step}.json")
        native[str(step)] = {key: all(norm(x["current"][key]) == norm(packets[0]["current"][key]) for x in packets)
                             for key in ("bodies", "controls", "joints")}
    displacement = np.linalg.norm(states[:, :, :2] - states[:, :1, :2], axis=-1).max(axis=1)
    first_differences = []
    for value in observed:
        found = np.flatnonzero(np.any(value != observed[0], axis=1))
        first_differences.append(int(found[0]) if len(found) else None)
    result = {"completed_steps": status["steps"], "independent_world_count": len(epochs),
              "checkpoint_unchanged": status["checkpoint_unchanged"],
              "observation_columns_compared": cols,
              "all_agent_actions_bitwise_equal": all(same_actions),
              "all_agent_observations_bitwise_equal_without_timestamp": all(same_states),
              "first_difference_from_agent0_by_agent": first_differences,
              "sampled_native_state_equality": native,
              "max_body_displacement_m_by_agent": displacement.tolist(),
              "max_body_y_by_agent": states[:, :, 1].max(axis=1).tolist(),
              "end_body_xy_by_agent": states[:, -1, :2].tolist(),
              "scope": "This checkpoint, deterministic cold rollout, 1200 steps; original single-world initialization equivalence remains a separate unresolved check."}
    assert status["checkpoint_unchanged"] and np.isfinite(states).all() and np.isfinite(actions).all()
    assert all(same_actions) and all(same_states) and all(displacement > 1)
    assert all(all(check.values()) for check in native.values())
    result["status"] = "PASS"
    (args.run / "consistency_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

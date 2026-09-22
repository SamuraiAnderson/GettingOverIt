"""Summarize aligned native-contact audits and measure proxy pose calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from training.deploy_sampling import rod_angle_deg, player_hub_angle_deg
from training.contact_features import ContactConfig, TerrainContactIndex, compute_contact_sequence, split_body_pot_local
from training.deploy_sampling import extract_tip_local
from training.config import TrainConfig
from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons


def diagnose(directory):
    directory = Path(directory)
    result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    counts = {}
    for episode in result["episodes"]:
        for name, values in episode["native_comparison"].items():
            count = counts.setdefault(name, dict.fromkeys(
                ("known_frames", "native_positive", "proxy_positive", "false_positive", "false_negative"), 0))
            for key in count:
                count[key] += values[key]
    for values in counts.values():
        n = values["known_frames"]
        values["mismatch_fraction"] = (values["false_positive"] + values["false_negative"]) / n if n else None
    offsets = {"tip": [], "body": []}
    controls, valid, poses = [], [], []
    env_data, player_data = load_colliders(Path(TrainConfig().game_root))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    geometry_hash = hashlib.sha256(json.dumps([env_data, player_data], sort_keys=True).encode()).hexdigest()
    if manifest["geometry_sha256"] != geometry_hash:
        raise ValueError("Geometry changed since collection; refuse to recompute with different colliders")
    body, pot = split_body_pot_local(player_data)
    geometry = dict(tip_local=extract_tip_local(player_data), body_local=body, pot_local=pot,
                    env_polygons=TerrainContactIndex(extract_polygons(env_data)))
    calibration = ContactConfig(tip_angle_offset_deg=-180., body_angle_offset_deg=-84.49833)
    calibrated_counts = {}
    for path in sorted(directory.glob("*_native.json")):
        packets = json.loads(path.read_text(encoding="utf-8"))
        stem = path.stem.removesuffix("_native")
        if not all(p.get("aligned") for p in packets):
            raise ValueError(f"Unaligned native packet in {path}")
        for agent in range(len(packets[0]["agents"])):
            states = json.loads((directory / f"{stem}_agent{agent}.json").read_text(encoding="utf-8"))["states"]
            corrected = compute_contact_sequence(np.asarray(states, dtype=np.float32), cfg=calibration, **geometry)
            physics = np.array([p["agents"][agent]["physicsContact"] for p in packets[:len(states)]])
            known = np.array([p["agents"][agent]["physicsKnown"] for p in packets[:len(states)]], dtype=bool)
            for name, column, part in (("tip_contact", 0, 0), ("body_contact", 2, 1), ("pot_contact", 3, 2)):
                proxy, native = corrected[known[:, part], column], physics[known[:, part], part]
                count = calibrated_counts.setdefault(name, {"frames": 0, "false_positive": 0, "false_negative": 0})
                count["frames"] += len(proxy)
                count["false_positive"] += int(((proxy > .5) & (native < .5)).sum())
                count["false_negative"] += int(((proxy < .5) & (native > .5)).sum())
            for state, packet in zip(states, packets):
                data = packet["agents"][agent]
                controls.append(data["control"])
                valid.append(data["controlKnown"])
                pose = data["partPose"]
                poses.append(pose)
                for name, native, estimated in (
                    ("tip", pose[2], rod_angle_deg(np.array(state), "pole")),
                    ("body", pose[7], player_hub_angle_deg(np.array(state))),
                ):
                    offsets[name].append((native - estimated + 180) % 360 - 180)
    controls, valid, poses = np.array(controls), np.array(valid, dtype=bool), np.array(poses)
    return {
        "source": str(directory), "contact_comparison": counts,
        "control_names": ["oldMouse_x", "oldMouse_y", "mouseVelocityAverage", "mouseSnap", "inputsToSkip", "pauseInputTimer"],
        "control_known_fraction": valid.mean(axis=0).tolist(),
        "control_min": [float(controls[valid[:, i], i].min()) if valid[:, i].any() else None for i in range(6)],
        "control_max": [float(controls[valid[:, i], i].max()) if valid[:, i].any() else None for i in range(6)],
        "pose_offset_percentiles": {name: dict(zip(("p0", "p10", "p50", "p90", "p100"),
                                   np.percentile(values, [0, 10, 50, 90, 100]).tolist()))
                                   for name, values in offsets.items()},
        "pose_offset_circular_mean": {name: float(np.degrees(np.angle(np.exp(1j * np.radians(values)).mean())))
                                      for name, values in offsets.items()},
        "calibrated_pose_only_comparison": calibrated_counts,
        "calibrated_offsets": {"tip_angle_offset": -180., "body_angle_offset": -84.49833},
        "pot_to_body_position_max_m": float(np.linalg.norm(poses[:, 10:12] - poses[:, 5:7], axis=1).max()),
        "coverage_note": "Frame agreement on these episodes only; no body-positive examples is not body-contact validation.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.directory)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

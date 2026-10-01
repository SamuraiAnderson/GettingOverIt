"""Rebuild real archived features and summarize representation smoke evidence offline."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from training.config import TrainConfig
from training.dataset import build_dynamics
from training.decision_buffer import DecisionRolloutBuffer
from training.experimental_rollout import segment_action
from training.policy_features import FeatureBuilder, SurfaceIndex, feature_names, scale_report
from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_evaluation(directory):
    manifest, results = (read_json(directory / name) for name in ("manifest.json", "results.json"))
    assert results["failure"] is None, results["failure"]
    cfg = TrainConfig()
    for field in fields(cfg):
        if field.name in manifest["config"]:
            setattr(cfg, field.name, manifest["config"][field.name])
    env_data, player_data = load_colliders(Path(cfg.game_root))
    geometry_hash = hashlib.sha256(json.dumps([env_data, player_data], sort_keys=True).encode()).hexdigest()
    assert geometry_hash == manifest["geometry_sha256"], "Geometry changed since collection"
    surfaces = SurfaceIndex(extract_polygons(env_data))
    names = feature_names(cfg)
    runs = []
    for rollout in results["rollouts"]:
        stem = f'{rollout["mode"]}_{rollout["trial"]:03d}'
        buffer = DecisionRolloutBuffer.load(directory / f"{stem}.npz")
        packets = read_json(directory / f"{stem}_native.json")
        scales = read_json(directory / f"{stem}_features.json")["scales"]
        rebuilt, feature_error, action_error = [], 0., 0.
        for agent, episode in enumerate(buffer.episodes):
            trajectory = read_json(directory / f"{stem}_agent{agent}.json")
            states = np.asarray(trajectory["states"], np.float32)
            contacts = np.asarray(trajectory["contacts"], np.float32)
            native = [p["agents"][agent] for p in packets[:len(states)]]
            extra = FeatureBuilder(cfg, surfaces).sequence(states,
                [p["physicsContact"][:3] for p in native], [p["physicsKnown"][:3] for p in native])
            expected = np.concatenate([np.stack([build_dynamics(s, c) for s, c in zip(states, contacts)]), extra], axis=1)
            feature_error = max(feature_error, float(np.abs(expected - episode["dynamics"]).max()))
            rebuilt.append(extra)
            for start, duration, parameters in zip(episode["decision_steps"], episode["durations"], episode["actions"]):
                for tick in range(int(duration)):
                    expected_action = segment_action(parameters, tick, cfg.action_segment_steps)
                    actual = episode["act_history"][int(start)+tick+1]
                    action_error = max(action_error, float(np.abs(expected_action - actual).max()))
        assert feature_error == 0., feature_error
        assert action_error == 0., action_error
        extra = np.concatenate(rebuilt)
        recomputed = scale_report(extra, cfg)
        # Early diagnostic files used unqualified names: events.tip_known was
        # overwritten by geometry.tip_known. Keep that historical evidence and
        # rebuild the complete, unambiguous report from the archived inputs.
        legacy_names = not all("." in name for name in scales)
        expected = {k.split(".", 1)[1]: v for k, v in recomputed.items()} if legacy_names else recomputed
        assert expected == scales, "Online/offline scale report differs"
        scales = recomputed
        runs.append({"mode": rollout["mode"], "frames_including_reset": len(extra),
            "max_feature_error": feature_error, "max_segment_action_error": action_error,
            "scale_fields": len(scales), "legacy_report_rebuilt": legacy_names,
            "clipped_features": {k: v for k, v in scales.items() if v["at_clip_fraction"]},
            "largest_abs_p99": sorted(scales.items(), key=lambda kv: kv[1]["abs_p99"], reverse=True)[:5],
            "known_fraction": {name: float(extra[:, i].mean()) for i, name in enumerate(names) if name.endswith("known")},
            "rollout": rollout})
    return {"geometry_hash_verified": geometry_hash, "summary": results["summary"], "runs": runs,
            "updates": results["updates"]}


def checkpoint_deltas(before_path, after_path):
    before = torch.load(before_path, map_location="cpu", weights_only=False)["model"]
    after = torch.load(after_path, map_location="cpu", weights_only=False)["model"]
    groups = {}
    for name, value in before.items():
        delta = (after[name].double() - value.double()).square().sum().item()
        if delta:
            group = name.split(".")[0]
            groups[group] = groups.get(group, 0.) + delta
    return {name: value**.5 for name, value in groups.items()}


def verify_training(directory):
    checkpoints = directory / "checkpoints"
    rollouts = checkpoints / "rollouts"
    warmup = checkpoint_deltas(rollouts / "behavior_0001.pt", rollouts / "behavior_0002.pt")
    assert set(warmup) == {"critic_head"}, warmup
    resume = checkpoint_deltas(checkpoints / "ppo_iter_0002.pt", rollouts / "behavior_0003.pt")
    assert not resume, resume
    update = checkpoint_deltas(rollouts / "behavior_0003.pt", checkpoints / "ppo_iter_0003.pt")
    assert {"spatial_encoder", "gru", "actor_mean", "event_encoder", "event_to_heads",
            "event_to_backbone", "feature_projections"} <= set(update), update
    return {"warmup_changed_group_l2": warmup, "resume_changed_group_l2": resume,
            "resumed_ppo_changed_group_l2": update,
            "replay": read_json(checkpoints / "replay_0003/replay.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, default=Path("runs/p2_representation_acceptance"))
    parser.add_argument("--training", type=Path, default=Path("runs/p2_representation_training_smoke"))
    args = parser.parse_args()
    result = {"scope": "Engineering verification only; no multi-seed learning comparison",
        "evaluations": {name: verify_evaluation(args.acceptance / "checkpoints" / name)
                        for name in ("gru_segments_v1", "transformer_concat")},
        "training": verify_training(args.training),
        "benchmark": read_json(args.acceptance / "backbone_benchmark.json"),
        "initial_rejected_run": read_json(args.acceptance / "checkpoints/gru_segments/results.json")["failure"]}
    output = args.acceptance / "verification_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Verified feature/action reconstruction, scale reports, critic warmup, resume and PPO gradients: {output}")
    for name, candidate in result["evaluations"].items():
        for run in candidate["runs"]:
            print(name, run["mode"], "frames", run["frames_including_reset"],
                  "clipped", run["clipped_features"], "known", run["known_fraction"])
    print(json.dumps(result["training"], indent=2))

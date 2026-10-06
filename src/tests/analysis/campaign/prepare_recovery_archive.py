"""Import an explicitly selected offline screen as recovery demonstrations; no game."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/"src"))
from training.config import TrainConfig
from training.dataset import Trajectory
from training.recovery_archive import RecoveryArchive, RecoverySegment, episode_identity
from training.reference_anchor import scoring_context
from training.reference_corridor import ReferenceCorridor, replay_corridor_result
from training.reference_fingerprint import reference_fingerprint, assert_fingerprint_compatible, load_fingerprint_sidecar
from training.run_logging import load_elites


def identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def import_screen(report_path, cfg, output, iteration):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    expected = reference_fingerprint(cfg, route_meta=route.metadata, schema_id="anchor35", raw_dim=35)
    source_cfg = TrainConfig(**report["config"])
    source_fp = reference_fingerprint(source_cfg, route_meta=report["reference"], schema_id="anchor35", raw_dim=35)
    assert_fingerprint_compatible(expected, source_fp, context="offline recovery screen")
    if cfg.feature_step_seconds != source_cfg.feature_step_seconds:
        raise ValueError("Screen and learner use different physical time steps")
    archive = RecoveryArchive(cfg, output, expected, route)
    episodes = {row["index"]:row for row in report["episodes"]}
    loaded, inputs, segments = {}, [identity(report_path)], []
    for selected in report["selected"]:
        ep = episodes[selected["episode_index"]]
        if ep["identity"] not in loaded:
            source = ep["sources"][0]
            path = Path(source["path"])
            inputs.append(identity(path))
            if source["kind"] == "full_episode_archive":
                assert_fingerprint_compatible(expected, load_fingerprint_sidecar(path), context=str(path))
                traj = Trajectory(**load_elites(path)[source["pool_index"]])
            elif source["kind"] == "frozen_evaluation":
                raw = json.loads(path.read_text(encoding="utf-8"))
                states = np.asarray(raw["states"], np.float32)
                actions = np.asarray([r["action"] for r in raw["trace"]], np.float32)
                schema, meta = scoring_context(states, config=cfg, route=route)
                played = replay_corridor_result(route, states, schema, meta)
                stem, agent = path.stem.rsplit("_agent", 1)
                rollout_path = path.parent/f"{stem}.npz"
                inputs.append(identity(rollout_path))
                with np.load(rollout_path, allow_pickle=False) as saved:
                    metadata = json.loads(str(saved["metadata"]))
                    index = metadata["collection_metadata"]["episode_agent_ids"].index(int(agent))
                    durations = saved[f"episode_{index}_durations"]
                    if len(durations) != len(actions) or not np.all(durations == 1):
                        raise ValueError("Recovery context requires one physical action per decision")
                    np.testing.assert_array_equal(actions, saved[f"episode_{index}_actions"])
                    terminal = saved[f"episode_{index}_dones"]
                np.testing.assert_allclose(played["rewards"], [r["reward"] for r in raw["trace"]], atol=1e-5, rtol=0)
                traj = Trajectory(states, actions, contact=np.asarray(raw["contacts"], np.float32),
                    corridor_progress=played["legacy_credit"], guidance_index=played["guidance_index"],
                    corridor_credit_events=played["credit_events"], corridor_confirmed_progress=played["confirmed_progress"],
                    step_rewards=played["rewards"], terminated=terminal, termination_known=True,
                    trajectory_valid=played["valid"], progress_mode=played["mode"], iteration=-1)
            else:
                raise ValueError(f"Unsupported demonstration source: {source['kind']}")
            if episode_identity(traj) != ep["identity"]:
                raise ValueError("Screen source episode contents changed")
            loaded[ep["identity"]] = traj
        traj = loaded[ep["identity"]]
        candidates = archive.candidates(traj)
        matches = [r for r in candidates if (r["start_step"],r["end_step"]) ==
                   (selected["start_step"],selected["end_step"])]
        if len(matches) != 1:
            raise ValueError("Screened recovery segment cannot be reproduced")
        row = matches[0]
        for key in ("content_sha256", "delta_s_m", "s_start_m", "T_steps",
                    "departure_relative_y_m", "return_relative_y_m", "return_projection_speed_mps"):
            if row[key] != selected[key]:
                raise ValueError(f"Screened descriptor changed: {key}")
        row.update(source=ep["sources"][0], screened_candidate_id=selected["candidate_id"])
        segments.append(RecoverySegment(traj, row))
    archive._select(segments)
    if len(archive.segments) != len(segments):
        raise ValueError("Screen selections violate current recovery admission/cell rules")
    path = archive.save(iteration)
    restored = RecoveryArchive(cfg, output, expected, route)
    assert restored.restore(path) == len(segments)
    result = dict(mode="offline_demonstration_import", source_inputs=inputs, archive=identity(path),
                  segments=len(segments), episodes=len(loaded), config=asdict(cfg),
                  note="Frozen probe episodes explicitly imported as demonstrations are no longer held-out evaluation data",
                  rule=archive.rule, descriptors=[s.descriptor for s in archive.segments])
    (Path(output)/"import.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**checkpoint["config"])
    cfg.recovery_archive_enabled = True
    cfg.gru_sil_recovery_fraction = .25
    path, archive = import_screen(args.screen, cfg, args.output, checkpoint["iteration"])
    print(json.dumps(dict(path=str(path), segments=len(archive.segments), rule=archive.rule), ensure_ascii=False))


if __name__ == "__main__":
    main()

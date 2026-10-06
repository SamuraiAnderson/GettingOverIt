"""Offline departure/return screening; no elite admission or optimizer is involved."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from training.config import TrainConfig
from training.reference_corridor import ReferenceCorridor
from training.return_segments import (closed_excursions, boundary_metrics, admission_rejections, select_cells, replay)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def apply_selection(report, minimum, start_bin, delta_bin, reject_upper, reject_negative_return):
    rows = report["candidates"]
    prior_ranks = {r["candidate_id"]:r["display_rank"] for r in report.get("selected", [])}
    for row in rows:
        row.pop("display_rank", None)
        if row["candidate_id"] in prior_ranks:
            row["previous_display_rank"] = prior_ranks[row["candidate_id"]]
    kept, decisions = select_cells(rows, minimum, start_bin, delta_bin,
                                  reject_upper=reject_upper, reject_negative_return=reject_negative_return)
    for row in rows:
        row["decision"] = decisions[row["candidate_id"]]
    kept.sort(key=lambda r: (-r["delta_s_m"], r["s_start_m"], r["T_seconds"]))
    for rank, row in enumerate(kept, 1):
        row["display_rank"] = rank
    forward = [r for r in rows if r["delta_s_m"] >= minimum]
    lower = [r for r in forward if not admission_rejections(r, minimum, reject_upper, False)]
    admitted = [r for r in lower if not admission_rejections(r, minimum, reject_upper, reject_negative_return)]
    report["selected"] = kept
    report["counts"].update(forward_eligible=len(forward), after_boundary_side_filter=len(lower),
                            after_return_speed_filter=len(admitted), selected=len(kept),
                            forward_eligible_unique_contents=len({r["content_sha256"] for r in forward}),
                            after_filters_unique_contents=len({r["content_sha256"] for r in admitted}))
    report["rule"].update(min_forward_m=minimum, start_bin_m=start_bin, delta_bin_m=delta_bin,
                          reject_upper_boundary=reject_upper, reject_negative_return_speed=reject_negative_return,
                          upper_definition="world-y(anchor) - world-y(ordered reference projection) > 0 at either endpoint",
                          departure_endpoint="last inside state before departure",
                          return_endpoint="first inside state after departure",
                          return_speed_definition="(s[end_step] - s[end_step-1]) / feature_step_seconds; zero is allowed",
                          filter_order="admission filters first, then cell winners; all original candidates reconsidered")
    report["sensitivity"] = []
    for threshold in [1., 2., 3., 5.]:
        for width in [.5, 1., 2.]:
            selected, _ = select_cells(rows, threshold, width, width, reject_upper=reject_upper,
                                       reject_negative_return=reject_negative_return)
            report["sensitivity"].append(dict(min_forward_m=threshold, start_bin_m=width, delta_bin_m=width,
                eligible=sum(not admission_rejections(r, threshold, reject_upper, reject_negative_return) for r in rows),
                selected=len(selected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", nargs="+", type=Path)
    source.add_argument("--rescreen", type=Path, help="重新筛选已保存的全部候选，不只筛选上次赢家")
    parser.add_argument("--episodes", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-forward", type=float, default=2.)
    parser.add_argument("--start-bin", type=float, default=1.)
    parser.add_argument("--delta-bin", type=float, default=1.)
    parser.add_argument("--reject-upper-boundary", action="store_true")
    parser.add_argument("--reject-negative-return-speed", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.rescreen:
        report = json.loads(args.rescreen.read_text(encoding="utf-8"))
        cfg = TrainConfig(**report["config"])
        route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
        if route.metadata["sha256"] != report["reference"]["sha256"]:
            raise ValueError("Reference changed since the original screen")
        archive_path = (args.rescreen.parent / report["data_archive"]).resolve()
        with np.load(archive_path, allow_pickle=False) as data:
            for row in report["candidates"]:
                i = row["episode_index"]
                row.update(boundary_metrics(row, data[f"e{i}_states"], data[f"e{i}_progress"],
                                            route, cfg.feature_step_seconds))
        report["version"] = 2
        report["data_archive"] = str(archive_path)
        report["rescreened_from"] = dict(path=str(args.rescreen.resolve()),
            sha256=hashlib.sha256(args.rescreen.read_bytes()).hexdigest(),
            episode_archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest())
        apply_selection(report, args.min_forward, args.start_bin, args.delta_bin,
                        args.reject_upper_boundary, args.reject_negative_return_speed)
        write_json(args.output / "screening.json", report)
        print(json.dumps(dict(counts=report["counts"], selected=report["selected"]), ensure_ascii=False, indent=2))
        return
    base = json.loads((args.runs[-1] / "run_meta.json").read_text(encoding="utf-8"))
    cfg = TrainConfig(**base["config"])
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    if route.metadata["sha256"] != base["reference_corridor"]["sha256"]:
        raise ValueError("Reference changed since collection")
    compatibility = ["reference_corridor_radius", "reference_projection_window", "reference_max_step",
                     "reference_simplify_tolerance", "reference_anchor_id", "reference_anchor_verified",
                     "reference_anchor_calibration_sha256", "feature_step_seconds"]
    episodes, inputs, counts = {}, [], Counter()

    def add(states, actions, source, terminated=None):
        counts["episode_references"] += 1
        states, actions = np.asarray(states, np.float32), np.asarray(actions, np.float32)
        if states.shape != (len(actions) + 1, 35) or actions.shape != (len(actions), 2):
            raise ValueError(f"Incomplete physical states in {source}")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite episode in {source}")
        identity = hashlib.sha256(states.tobytes() + actions.tobytes()).hexdigest()
        if identity in episodes:
            episodes[identity]["sources"].append(source)
            return
        episodes[identity] = dict(states=states, actions=actions, sources=[source],
                                  terminated=terminated, identity=identity)

    for run in args.runs:
        meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
        if (meta["reference_corridor"]["sha256"] != route.metadata["sha256"]
                or any(meta["config"].get(k) != base["config"].get(k) for k in compatibility)):
            raise ValueError(f"Incompatible collection configuration: {run}")
        paths = sorted((run / "elites/full_episodes").glob("elites_iter_*.npz"))
        if not paths:
            raise ValueError(f"No full episode archive: {run}")
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                archive = json.loads(str(data["archive_metadata_json"]))
                if archive.get("origin") != "episode_reset" or archive.get("truncation") != "none":
                    raise ValueError(f"Not a complete episode archive: {path}")
                fp = json.loads(path.with_suffix(".npz.fingerprint.json").read_text(encoding="utf-8"))
                if fp["fingerprint"] != base["reference_fingerprint"]["fingerprint"]:
                    raise ValueError(f"Incompatible archived reference: {path}")
                for i in range(int(data["n_trajectories"])):
                    if not bool(data[f"t{i}_termination_known"]):
                        raise ValueError("Full episode termination flags unavailable")
                    add(data[f"t{i}_states"], data[f"t{i}_actions"],
                        dict(kind="full_episode_archive", path=str(path.resolve()), pool_index=i,
                             iteration=int(data["iterations"][i])), data[f"t{i}_terminated"])
            inputs.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f"Loaded {run.name}: {len(paths)} snapshots, {len(episodes)} unique episodes so far", flush=True)
    for directory in args.episodes:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (manifest["reference"]["sha256"] != route.metadata["sha256"]
                or any(manifest["config"].get(k) != base["config"].get(k) for k in compatibility)):
            raise ValueError(f"Incompatible evaluation: {directory}")
        for path in sorted(directory.glob("*_agent[0-9]*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            prefix, agent_text = path.stem.rsplit("_agent", 1)
            rollout = directory / f"{prefix}.npz"
            terminated = None
            if rollout.exists():
                with np.load(rollout, allow_pickle=False) as saved:
                    metadata = json.loads(str(saved["metadata"]))
                    agent_ids = metadata["collection_metadata"]["episode_agent_ids"]
                    episode = agent_ids.index(int(agent_text))
                    durations = saved[f"episode_{episode}_durations"]
                    if len(durations) != len(data["trace"]) or not np.all(durations == 1):
                        raise ValueError("Evaluation terminal flags need one physical step per decision")
                    terminated = saved[f"episode_{episode}_dones"]
            add(data["states"], [r["action"] for r in data["trace"]],
                dict(kind="frozen_evaluation", path=str(path.resolve()),
                     condition=data["metrics"].get("noise_intervention")), terminated)
            inputs.append(dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    candidates, episode_index, arrays = [], [], {}
    segment_hashes = Counter()
    for index, ep in enumerate(episodes.values()):
        states, actions = ep["states"], ep["actions"]
        inside, progress, segments, gaps, invalid = replay(states, cfg, route)
        terminated = ep["terminated"]
        counts["episodes_without_termination_flags"] += terminated is None
        internal_done = bool(terminated is not None and np.asarray(terminated[:-1]).any())
        counts["episodes_with_internal_done"] += internal_done
        counts["episodes_with_invalid_displacement"] += bool(invalid)
        counts["episodes_with_reference_gap"] += bool(gaps.any())
        if internal_done or invalid:
            raise ValueError(f"Unexpected discontinuous episode {ep['sources'][0]}")
        # Cross-check existing evaluation trace instead of silently changing membership.
        for source in ep["sources"]:
            if source["kind"] == "frozen_evaluation":
                saved = json.loads(Path(source["path"]).read_text(encoding="utf-8"))["trace"]
                assert np.array_equal(inside[1:], [r["inside"] for r in saved])
                # Saved projection is episode-relative and nonnegative on the starting segment.
                assert np.allclose(np.maximum(0., progress[1:] - progress[0]),
                                   [r["projection_m"] for r in saved], atol=1e-9, rtol=0.)
        rows, unfinished = closed_excursions(inside, progress, segments, gaps)
        counts["episodes_ending_outside_after_departure"] += unfinished
        arrays[f"e{index}_states"], arrays[f"e{index}_actions"] = states, actions
        arrays[f"e{index}_inside"], arrays[f"e{index}_progress"] = inside, progress
        for row in rows:
            a, b = row["start_step"], row["end_step"]
            content = hashlib.sha256(states[a:b + 1].tobytes() + actions[a:b].tobytes()).hexdigest()
            segment_hashes[content] += 1
            cid = f"e{index:03d}_{a:04d}_{b:04d}"
            row.update(candidate_id=cid, episode_index=index, T_seconds=row["T_steps"] * cfg.feature_step_seconds,
                       content_sha256=content)
            row.update(boundary_metrics(row, states, progress, route, cfg.feature_step_seconds))
            candidates.append(row)
        episode_index.append(dict(index=index, identity=ep["identity"], sources=ep["sources"],
                                  steps=len(actions), termination_flags_available=terminated is not None,
                                  gap_frames=int(gaps.sum()), invalid_frames=invalid))
        if index % 20 == 0:
            print(f"Replayed {index + 1}/{len(episodes)} unique complete episodes", flush=True)
    for row in candidates:
        row["identical_segment_occurrences"] = segment_hashes[row["content_sha256"]]
    counts.update(unique_episodes=len(episodes), completed_excursions=len(candidates),
                  unique_completed_segment_contents=len(segment_hashes))
    report = dict(version=2, mode="offline_screen_only", counts=dict(counts),
                  rule=dict(min_forward_m=args.min_forward, start_bin_m=args.start_bin, delta_bin_m=args.delta_bin,
                            cell="(reference segment, floor(s_start/start_bin), floor(delta_s/delta_bin))",
                            winner="lowest T_seconds; ties: largest delta_s, lowest s_start, stable candidate_id",
                            intervals="half-open bins; last inside state to first return; no merging across returns",
                            projection="absolute ordered reference arclength; same continuous segment only",
                            membership="existing shared tracker with radius; no extra reentry hold condition",
                            hidden_state_criteria=False, global_capacity_pruning=False),
                  reference=route.metadata, config=asdict(cfg), inputs=inputs,
                  limitations=["Historical full episodes were retained by the old elite admission policy; this is not all training rollout data.",
                               "PPO input archives do not contain full world states; excluded, not reconstructed approximately.",
                               "Selected segments are offline candidates, not inserted into training or the elite pool."],
                  episodes=episode_index, candidates=candidates,
                  data_archive="episodes.npz")
    apply_selection(report, args.min_forward, args.start_bin, args.delta_bin,
                    args.reject_upper_boundary, args.reject_negative_return_speed)
    np.savez_compressed(args.output / "episodes.npz", **arrays)
    write_json(args.output / "screening.json", report)
    print(json.dumps(dict(counts=report["counts"], selected=report["selected"]), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

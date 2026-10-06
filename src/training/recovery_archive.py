"""Independent departure/return archive, retaining original reset-to-end context.

Admission and replacement depend only on the documented geometric descriptors.
Neither global elite rank nor a model advantage is an archive admission test.
"""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import Trajectory
from .elite_archive import EliteTrajectoryArchive
from .reference_fingerprint import assert_fingerprint_compatible, load_fingerprint_sidecar
from .return_segments import replay, closed_excursions, boundary_metrics, select_cells
from .run_logging import load_elites, save_elites


def episode_identity(traj):
    return hashlib.sha256(np.asarray(traj.raw_states, np.float32).tobytes()
                          + np.asarray(traj.actions, np.float32).tobytes()).hexdigest()


@dataclass
class RecoverySegment:
    trajectory: Trajectory
    descriptor: dict

    @property
    def start(self):
        # Action at last inside state caused the departure: do not imitate it.
        return int(self.descriptor["first_outside_step"])

    @property
    def end(self):
        # Actions [start, end) include the action landing at first return state.
        return int(self.descriptor["end_step"])


class RecoveryArchive:
    def __init__(self, config, directory, fingerprint, route):
        self.config, self.directory = config, Path(directory)
        self.fingerprint, self.route = fingerprint, route
        if config.reference_progress_mode != "confirmed_endpoint_v2" or config.action_segment_steps != 1:
            raise ValueError("Recovery archive needs reference v2 and one physical step per action")
        widths = (config.recovery_min_forward_m, config.recovery_start_bin_m, config.recovery_delta_bin_m)
        if not np.isfinite(widths).all() or min(widths) <= 0:
            raise ValueError("Recovery distance thresholds and bin widths must be positive")
        self.segments = []

    @property
    def rule(self):
        return dict(version=1, min_forward_m=self.config.recovery_min_forward_m,
                    start_bin_m=self.config.recovery_start_bin_m, delta_bin_m=self.config.recovery_delta_bin_m,
                    reject_upper_boundary=True, reject_negative_return_speed=True,
                    cell="(segment_index, floor(s_start/start_bin), floor(delta_s/delta_bin))",
                    winner="min T_seconds; ties: max delta_s, min s_start, stable content ID",
                    learning_interval="first outside state through action causing first return; end exclusive",
                    reward_target="original full episode discounted return; no recovery bonus",
                    step_seconds=self.config.feature_step_seconds)

    @property
    def trajectories(self):
        unique = {}
        for segment in self.segments:
            unique.setdefault(segment.descriptor["episode_identity"], segment.trajectory)
        return list(unique.values())

    def candidates(self, traj):
        if (not EliteTrajectoryArchive._complete(traj) or not traj.trajectory_valid
                or traj.raw_states.shape[1] != 35):
            return []
        inside, progress, segments, gaps, invalid = replay(traj.raw_states, self.config, self.route)
        if invalid:
            return []
        rows, _ = closed_excursions(inside, progress, segments, gaps)
        identity = episode_identity(traj)
        for row in rows:
            a, b = row["start_step"], row["end_step"]
            content = hashlib.sha256(traj.raw_states[a:b+1].tobytes()+traj.actions[a:b].tobytes()).hexdigest()
            row.update(candidate_id=f"{identity}:{a}:{b}", episode_identity=identity,
                       content_sha256=content, T_seconds=row["T_steps"]*self.config.feature_step_seconds)
            row.update(boundary_metrics(row, traj.raw_states, progress, self.route, self.config.feature_step_seconds))
        return rows

    def _select(self, segments):
        # Repeated archives/overlapping episodes must not multiply sampling mass.
        unique = {}
        for s in segments:
            unique.setdefault(s.descriptor["candidate_id"], s)
        rows = [s.descriptor for s in unique.values()]
        winners, _ = select_cells(rows, self.config.recovery_min_forward_m,
                                   self.config.recovery_start_bin_m, self.config.recovery_delta_bin_m,
                                   reject_upper=True, reject_negative_return=True)
        self.segments = [unique[row["candidate_id"]] for row in winners]

    def ingest(self, trajectories, iteration, *, source="online_rollout"):
        before = {s.descriptor["candidate_id"] for s in self.segments}
        incoming, incomplete = [], 0
        for original in trajectories:
            if not EliteTrajectoryArchive._complete(original):
                incomplete += 1
                continue
            traj = replace(original, iteration=iteration,
                           raw_states=np.asarray(original.raw_states, np.float32),
                           actions=np.asarray(original.actions, np.float32))
            for row in self.candidates(traj):
                row["source"] = source
                incoming.append(RecoverySegment(traj, row))
        self._select([*self.segments, *incoming])
        after = {s.descriptor["candidate_id"] for s in self.segments}
        return dict(recovery_archive_size=len(after), recovery_archive_episodes=len(self.trajectories),
                    recovery_archive_added=len(after-before), recovery_archive_evicted=len(before-after),
                    recovery_archive_closed_candidates=len(incoming), recovery_archive_incomplete=incomplete)

    def save(self, iteration):
        trajectories = self.trajectories
        indices = {episode_identity(t): i for i, t in enumerate(trajectories)}
        path = self.directory / f"recovery_iter_{iteration:04d}.npz"
        save_elites(path, trajectories, self.fingerprint, archive_metadata={
            "schema":"recovery_segments_v1", "origin":"episode_reset", "truncation":"none",
            "snapshot_iteration":iteration, "rule":self.rule,
            "segments":[dict(s.descriptor, episode_index=indices[s.descriptor["episode_identity"]])
                        for s in self.segments]})
        return path

    def restore(self, path):
        assert_fingerprint_compatible(self.fingerprint, load_fingerprint_sidecar(path), context=str(path))
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["archive_metadata_json"]))
        if (meta.get("schema") != "recovery_segments_v1" or meta.get("origin") != "episode_reset"
                or meta.get("truncation") != "none" or meta.get("rule") != self.rule):
            raise ValueError("Recovery snapshot has incompatible schema, context or selection rule")
        trajectories = [Trajectory(**r) for r in load_elites(path)]
        candidates = {i: {r["candidate_id"]:r for r in self.candidates(t)} for i,t in enumerate(trajectories)}
        restored = []
        for saved in meta["segments"]:
            row = dict(saved)
            index = row.pop("episode_index")
            measured = candidates.get(index, {}).get(row["candidate_id"])
            if measured is None or any(row.get(k) != v for k,v in measured.items()):
                raise ValueError("Recovery descriptor differs from original physical trajectory")
            restored.append(RecoverySegment(trajectories[index], row))
        self._select(restored)
        if len(self.segments) != len(restored):
            self.segments = []
            raise ValueError("Recovery snapshot has duplicate cells or inadmissible segments")
        return len(self.segments)

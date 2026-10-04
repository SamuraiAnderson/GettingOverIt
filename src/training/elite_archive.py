"""Bounded full-episode archive, independent of the optimizer and SIL sampler."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .dataset import Trajectory
from .reference_corridor import PROGRESS_MODE_V2, trajectory_endpoint
from .reference_fingerprint import assert_fingerprint_compatible, load_fingerprint_sidecar
from .reward import corridor_score_of, is_water_trajectory
from .run_logging import load_elites, save_elites
from .sil_archive import ingest_td_pareto, td_point


class EliteTrajectoryArchive:
    """Keep full reset-to-end episodes; (t, D) selects, but never truncates them.

    Reuses the established Pareto/capacity policy, including its no-water rule
    and spatial diversity slots. This object has no model, sampler or RNG.
    """

    def __init__(self, config, directory, fingerprint):
        if not config.reference_trajectory or config.reference_progress_mode != PROGRESS_MODE_V2:
            raise ValueError("Full-episode elite archive requires confirmed_endpoint_v2 reference scoring")
        if config.elite_archive_capacity < 1:
            raise ValueError("elite_archive_capacity must be positive")
        self.config = replace(config, sil_traj_cap=config.elite_archive_capacity)
        self.directory = Path(directory)
        self.fingerprint = fingerprint
        self.trajectories = []

    @staticmethod
    def _complete(traj):
        n = len(traj.actions)
        if (n < 1 or traj.raw_states.ndim != 2 or traj.raw_states.shape[1] not in (33, 35)
                or traj.actions.shape != (n, 2) or len(traj.raw_states) != n + 1
                or traj.progress_mode != PROGRESS_MODE_V2 or not traj.termination_known):
            return False
        for name, length in (("contact", n + 1), ("guidance_index", n + 1),
                             ("corridor_confirmed_progress", n + 1),
                             ("corridor_credit_events", n), ("corridor_progress", n),
                             ("step_rewards", n), ("terminated", n)):
            data = getattr(traj, name)
            if data is None or len(data) != length or not np.isfinite(data).all():
                return False
        return bool(np.isfinite(traj.raw_states).all() and np.isfinite(traj.actions).all()
                    and not np.asarray(traj.terminated[:-1]).any())

    def ingest(self, trajectories, iteration):
        # Admission writes score/iteration/role. Never mutate the live PPO trajectories.
        incoming = [replace(t) for t in trajectories if self._complete(t)]
        scores = [corridor_score_of(t, self.config) for t in incoming]
        self.trajectories, stats = ingest_td_pareto(
            self.trajectories, incoming, scores, iteration, self.config)
        result = {k.replace("sil_archive_", "elite_archive_"): v for k, v in stats.items()
                  if k.startswith("sil_archive_")}
        result["elite_archive_rejected_incomplete"] = float(len(trajectories) - len(incoming))
        result["elite_archive_max_distance_m"] = max(
            (trajectory_endpoint(t).distance for t in self.trajectories), default=0.0)
        return result

    def save(self, iteration):
        return save_elites(self.directory / f"elites_iter_{iteration:04d}.npz",
                           self.trajectories, self.fingerprint, archive_metadata={
                               "schema": "full_episode_elites_v1", "origin": "episode_reset",
                               "truncation": "none", "selection": "td_pareto",
                               "snapshot_iteration": iteration,
                               "iteration_meaning": "collection/update iteration; behavior weights precede its update",
                           })

    def restore(self, path):
        assert_fingerprint_compatible(self.fingerprint, load_fingerprint_sidecar(path), context=str(path))
        with np.load(path) as data:
            meta = json.loads(str(data["archive_metadata_json"])) if "archive_metadata_json" in data else {}
        if (meta.get("schema") != "full_episode_elites_v1" or meta.get("truncation") != "none"
                or meta.get("origin") != "episode_reset"):
            raise ValueError("Expected an unpruned full-episode elite archive, not a legacy SIL pool")
        restored = [Trajectory(**row) for row in load_elites(path)]
        if any(not self._complete(t) for t in restored):
            raise ValueError("Elite archive is missing full episode context")
        if any(td_point(t) is None or is_water_trajectory(t.raw_states, self.config) for t in restored):
            raise ValueError("Elite archive contains an invalid or water trajectory")
        # Re-ingesting a capped pool can reorder its farthest-point selection.
        # SIL samples by index: preserve the exact snapshot order/roles on resume.
        self.trajectories = restored
        if len(restored) > self.config.sil_traj_cap:
            self.trajectories, _ = ingest_td_pareto(restored, [], [], 0, self.config)
        return len(self.trajectories)

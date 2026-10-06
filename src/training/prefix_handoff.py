"""Frozen evaluation intervention; never a training action source.

The collector still evaluates the policy at every frame. Its decision buffer
contains policy probes; trajectories and previous-action inputs contain the
actions actually sent to the game. Prefix length k executes demo [0, k), then
hands action k to the policy without resetting its recurrent state.
"""
import hashlib

import numpy as np


class PrefixHandoff:
    def __init__(self, trajectory, cuts, source=None):
        self.trajectory = trajectory
        self.cuts = np.asarray(cuts)
        self.source = source or {}

    def validate(self, config, steps, deterministic, offsets=None):
        if (not deterministic or config.policy_backbone != "gru"
                or config.action_segment_steps != 1 or config.random_deploy
                or (offsets is not None and np.any(offsets))):
            raise ValueError("Prefix handoff requires deterministic GRU, single-step actions and exact spawn")
        if (self.cuts.shape != (config.num_agents,) or self.cuts.dtype.kind not in "iu"
                or np.any(self.cuts < 0) or np.any(self.cuts > steps)):
            raise ValueError("One integer prefix length in [0, steps] per agent is required")
        t = self.trajectory
        if (t.actions.ndim != 2 or t.actions.shape[1] != 2
                or len(t.raw_states) != len(t.actions) + 1 or len(t.actions) < self.cuts.max()
                or not np.isfinite(t.actions).all() or not np.isfinite(t.raw_states).all()
                or np.any(np.abs(t.actions) > config.action_scale)):
            raise ValueError("Prefix requires complete, finite, bounded demonstration actions and states")

    def override(self, step, active, actions):
        result = actions.copy()
        for agent in active:
            if step < self.cuts[agent]:
                result[agent] = self.trajectory.actions[step]
        return result

    def metadata(self):
        t = self.trajectory
        return {"evaluation_only": True, "prefix_steps_by_agent": self.cuts.tolist(),
                "source": self.source,
                "trajectory_sha256": hashlib.sha256(t.raw_states.tobytes() + t.actions.tobytes()).hexdigest(),
                "decision_actions": "Policy probes, NOT executed demonstration-prefix actions",
                "executed_actions": "Episode JSON trace; act_history contains actual previous actions",
                "memory": "Continuous GRU on actual states and executed actions from reset"}

    def compare(self, trajectory, agent, policy_probes):
        """Report observable replay errors, not an assertion of hidden physics equality."""
        cut = int(self.cuts[agent])
        n = min(cut, len(trajectory.actions))
        if not np.array_equal(trajectory.actions[:n], self.trajectory.actions[:n]):
            raise ValueError("Executed prefix differs from demonstration")
        suffix_error = float(np.max(np.abs(trajectory.actions[n:] - policy_probes[n:]))) if n < len(trajectory.actions) else None
        if suffix_error is not None and suffix_error != 0:
            raise ValueError("Executed suffix differs from policy probes")
        actual, demo = trajectory.raw_states[:n+1], self.trajectory.raw_states[:n+1]
        if actual.shape != demo.shape or actual.shape[1] != 35:
            raise ValueError("State comparison requires matching calibrated 35D states")
        errors = {}
        for name, pairs in (("body_position_m", (0, 5, 10, 15, 19, 23, 33)),
                            ("body_velocity_m_s", (2, 7, 12, 17, 21, 25)),
                            ("cursor_position_m", (29,)), ("cursor_velocity_m_s", (31,))):
            delta = np.max(np.stack([np.linalg.norm(actual[:, k:k+2]-demo[:, k:k+2], axis=1)
                                    for k in pairs]), axis=0)
            errors[name] = {"initial": float(delta[0]), "at_handoff": float(delta[-1]),
                            "prefix_max": float(delta.max())}
        angle = np.abs((actual[:, [9, 14, 27]] - demo[:, [9, 14, 27]] + 180) % 360 - 180)
        errors["angle_deg"] = {"at_handoff": float(angle[-1].max()), "prefix_max": float(angle.max())}
        errors["player_angular_velocity_deg_s"] = {
            "at_handoff": float(abs(actual[-1, 4]-demo[-1, 4])),
            "prefix_max": float(np.abs(actual[:, 4]-demo[:, 4]).max())}
        return {"prefix_steps": cut, "completed_prefix_steps": n,
                "handoff_reached": len(trajectory.actions) > cut,
                "prefix_actions_exact": True, "suffix_action_max_error": suffix_error,
                "observable_errors": errors, "hidden_physics_state_equality": "not observable"}

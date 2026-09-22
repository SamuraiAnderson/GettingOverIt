"""Keep archived actions and physical agent/episode ordering intact during replay."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.tests.control_interaction.replay_evaluation_actions import load_batch, batches_through, select_episode


def episode(agent, steps):
    return {"states": np.zeros((steps+1, 33)).tolist(),
            "metrics": {"agent": agent, "mode": "exploration", "trial": 1},
            "trace": [{"step": i+1, "action": [agent+1., -i]} for i in range(steps)]}


class ActionReplayTests(unittest.TestCase):
    def test_agent_order_and_zero_after_termination(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for agent, steps in [(0, 3), (1, 1)]:
                (base/f"exploration_001_agent{agent}.json").write_text(json.dumps(episode(agent, steps)))
            actions, _ = load_batch(base, "exploration", 1, 2)
            np.testing.assert_array_equal(actions[:, 0], [[1, 0], [1, -1], [1, -2]])
            np.testing.assert_array_equal(actions[:, 1], [[2, 0], [0, 0], [0, 0]])

    def test_prefix_preserves_evaluation_order(self):
        self.assertEqual(batches_through("exploration", 1, 2),
                         [("deterministic", 0), ("deterministic", 1), ("exploration", 0), ("exploration", 1)])
        self.assertEqual(batches_through("deterministic", 0, 2), [("deterministic", 0)])

    def test_corrupt_step_sequence_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            data = episode(0, 2)
            data["trace"][1]["step"] = 4
            (base/"exploration_001_agent0.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_batch(base, "exploration", 1, 1)

    def test_selection_uses_progress_not_dwell_success(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = Path(folder)/"campaign"
            campaign.mkdir()
            (campaign/"manifest.json").write_text(json.dumps({"order": [["C", 1]]}))
            target = Path(folder)/"campaign_C_s1/checkpoints/eval_test"
            target.mkdir(parents=True)
            for agent, progress in [(0, 8), (1, 2)]:
                data = episode(agent, 2)
                data["metrics"].update(credited_progress_m=progress, outside_fraction=.2,
                                       steps=2, success=(agent==1))
                data["trace"][-1]["progress_m"] = progress
                (target/f"exploration_001_agent{agent}.json").write_text(json.dumps(data))
            self.assertEqual(select_episode(campaign)[0]["agent"], 0)


if __name__ == "__main__":
    unittest.main()

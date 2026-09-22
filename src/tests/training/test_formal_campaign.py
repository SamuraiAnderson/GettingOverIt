"""Budget and command-line safeguards for the bounded formal campaign."""
import unittest

from src.tests.analysis.run_corridor_formal import choose_endpoint, BudgetStop
from src.training.main_ppo import parse_args


class FormalCampaignTests(unittest.TestCase):
    def test_budget_preserves_all_six_runs_and_holdout_reserve(self):
        rows = [dict(physical_samples=96000, recent_samples_per_iteration=9000,
                     recent_iteration_seconds=70) for _ in range(6)]
        target, estimates = choose_endpoint(rows, 16000, 540)
        self.assertEqual(target, 240000)
        self.assertGreaterEqual(estimates[-2]["estimated_seconds"], 16000)
        self.assertLess(estimates[-1]["estimated_seconds"], 16000)

    def test_budget_can_finish_at_midpoint_without_extra_training(self):
        rows = [dict(physical_samples=97000, recent_samples_per_iteration=9000,
                     recent_iteration_seconds=70) for _ in range(6)]
        target, _ = choose_endpoint(rows, 3500, 540)
        self.assertEqual(target, 96000)
        with self.assertRaises(BudgetStop):
            choose_endpoint(rows, 3400, 540)

    def test_explicit_batch_threads_and_checkpoint_interval(self):
        args = parse_args(["--num-agents", "8", "--ppo-batch-size", "128",
                           "--torch-threads", "4", "--save-interval", "1"])
        self.assertEqual((args.num_agents, args.ppo_batch_size, args.torch_threads, args.save_interval),
                         (8, 128, 4, 1))
        for option in ("--ppo-batch-size", "--torch-threads", "--save-interval"):
            with self.assertRaises(SystemExit):
                parse_args([option, "0"])


if __name__ == "__main__":
    unittest.main()

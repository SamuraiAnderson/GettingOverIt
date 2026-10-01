"""CLI bounds for the current PPO entry."""
import unittest

from src.training.main_ppo import parse_args


class PPOLaunchBoundsTests(unittest.TestCase):
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

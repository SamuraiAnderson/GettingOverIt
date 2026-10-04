"""(t, D) envelope viewer uses the training Pareto helper and writes plots."""
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest

import numpy as np

from src.tests.evaluation.present import plot_sil_td_envelope as plot
from src.tests.training.test_sil_archive import _traj
from src.training.run_logging import save_elites
from src.training.sil_archive import pareto_front


class SILTdEnvelopePlotTests(unittest.TestCase):
    def test_catalog_marks_upper_left_front(self):
        trajs = [_traj(100, 4.0), _traj(377, 6.0), _traj(200, 3.0)]
        catalog = plot.catalog_pool(trajs, source="memory")
        front = {(row["t"], round(row["D"], 3)) for row in catalog["envelope"]}
        self.assertEqual(front, {(100, 4.0), (377, 6.0)})
        self.assertEqual(catalog["n_front"], 2)
        dominated = [row for row in catalog["points"] if row["comparable"] and not row["on_front"]]
        self.assertEqual([(row["t"], round(row["D"], 3)) for row in dominated], [(200, 3.0)])
        self.assertEqual(
            {plot.td_point(t) for t in pareto_front(trajs)},
            {(100, 4.0), (377, 6.0)},
        )

    def test_writes_snapshot_and_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            elites = root / "elites"
            out = root / "td_envelope"
            early = [_traj(200, 3.0, iteration=1), _traj(400, 4.0, iteration=1)]
            late = [_traj(100, 4.0, iteration=2), _traj(377, 6.0, iteration=2),
                    _traj(200, 3.0, iteration=2)]
            save_elites(elites / "elites_iter_0001.npz", early)
            save_elites(elites / "elites_iter_0002.npz", late)
            self.assertEqual(
                [p.name for p in plot.discover_pools(elites)],
                ["elites_iter_0001.npz", "elites_iter_0002.npz"],
            )
            plot.main(["--elites-dir", str(elites), "--output", str(out)])
            self.assertTrue((out / "envelope.png").is_file())
            self.assertTrue((out / "envelope_history.png").is_file())
            payload = (out / "envelope.json").read_text(encoding="utf-8")
            self.assertIn('"n_front": 2', payload)
            self.assertIn('"t": 100', payload)

    def test_single_pool_skips_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pool = root / "elites_iter_0003.npz"
            save_elites(pool, [_traj(90, 5.0)])
            out = root / "one"
            plot.main(["--pool", str(pool), "--output", str(out)])
            self.assertTrue((out / "envelope.png").is_file())
            self.assertFalse((out / "envelope_history.png").exists())
            self.assertGreater((out / "envelope.png").stat().st_size, 1000)

    def test_sampling_frame_mass_uses_actual_chunk_length_and_writes_both_views(self):
        from src.tests.training.test_gru_sil import GRUSILSamplingTests
        trajs = GRUSILSamplingTests().trajectories()
        cfg = replace(plot.TrainConfig(), recurrent_seq_len=32)
        data = plot.sampling_catalog(trajs, cfg, draws=3000, seed=7)
        from src.tests.evaluation.present.render_episode_motion import sampling_path_probabilities
        aligned = sampling_path_probabilities(data, trajs)
        for mode, arrays in aligned.items():
            for row in data["modes"][mode]["chunks"]:
                np.testing.assert_array_equal(arrays[row["trajectory"]][row["start"]:row["end"]], row["probability"])
                np.testing.assert_array_equal(arrays[row["trajectory"]][row["cap"]:], 0.)
        import copy
        missing = copy.deepcopy(data)
        missing["modes"][data["comparison_modes"][0]]["chunks"].pop()
        with self.assertRaises(ValueError): sampling_path_probabilities(missing, trajs)
        for mode in data["modes"].values():
            self.assertAlmostEqual(sum(mode["expected_phase_frame_share"].values()), 1.)
            self.assertAlmostEqual(sum(mode["sampled_phase_frame_share"].values()), 1.)
            self.assertEqual(sum(r["observed_draws"] for r in mode["chunks"]), 3000)
            # Includes a final 31-step chunk: expected frame mass is not a chunk histogram.
            denominator = sum(r["probability"]*(r["end"]-r["start"]) for r in mode["chunks"])
            numerator = sum(r["probability"]*r["phase_counts"].get("tree_support",0) for r in mode["chunks"])
            self.assertAlmostEqual(mode["expected_phase_frame_share"]["tree_support"], numerator/denominator)
        data["source_pool"] = {"path": "synthetic_test_pool.npz"}
        with tempfile.TemporaryDirectory() as folder:
            for path in plot.plot_sampling(data, Path(folder)):
                self.assertGreater(path.stat().st_size, 1000)

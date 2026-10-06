"""Boundary and selection semantics for the offline departure/return screen."""
import unittest
from types import SimpleNamespace
import numpy as np

from src.tests.analysis.campaign.screen_return_segments import (
    admission_rejections, boundary_metrics, closed_excursions, select_cells,
)


class ReturnSegmentScreenTest(unittest.TestCase):
    def test_net_progress_not_highwater_or_sum_of_positive_steps(self):
        rows, unfinished = closed_excursions([True, False, False, True], [2., 9., 1., 3.], [0]*4, [False]*4)
        self.assertEqual(unfinished, 0)
        self.assertEqual((rows[0]["start_step"], rows[0]["end_step"], rows[0]["T_steps"]), (0, 3, 3))
        self.assertEqual(rows[0]["delta_s_m"], 1.)

    def test_first_return_closes_segment_and_open_tail_is_not_accepted(self):
        rows, unfinished = closed_excursions([True, False, True, False, True, False],
                                             [0., 1., 2., 3., 4., 8.], [0]*6, [False]*6)
        self.assertEqual([(r["start_step"], r["end_step"]) for r in rows], [(0, 2), (2, 4)])
        self.assertEqual(unfinished, 1)

    def test_gap_and_segment_change_are_not_continuous_excursions(self):
        self.assertEqual(closed_excursions([True, False, True], [0., 3., 5.], [0, 0, 1], [False]*3)[0], [])
        self.assertEqual(closed_excursions([True, False, True], [0., 3., 5.], [0]*3, [False, True, False])[0], [])

    def test_only_declared_cell_and_duration_determine_replacement(self):
        def row(cid, start, delta, duration):
            return dict(candidate_id=cid, segment_index=0, s_start_m=start, delta_s_m=delta, T_seconds=duration)
        rows = [row("slow_long", 2.1, 2.9, 27), row("fast_short", 2.3, 2.1, 26),
                row("other_start", 3., 2.2, 100), row("other_delta", 2.2, 3., 100),
                row("below", 1., 1.999, 1)]
        selected, decisions = select_cells(rows, 2., 1., 1.)
        self.assertEqual({r["candidate_id"] for r in selected}, {"fast_short", "other_start", "other_delta"})
        self.assertEqual(decisions["slow_long"]["winner_id"], "fast_short")
        self.assertEqual(decisions["below"]["reason"], "below_min_delta")
        finer, _ = select_cells(rows, 2., 1., .5)
        self.assertIn("slow_long", {r["candidate_id"] for r in finer})

    def test_equal_duration_tie_uses_delta_and_is_order_independent(self):
        rows = [dict(candidate_id=str(i), segment_index=0, s_start_m=1.2, delta_s_m=d, T_seconds=1.)
                for i, d in enumerate([2.1, 2.4])]
        self.assertEqual(select_cells(rows, 2., 1., 1.)[0], select_cells(list(reversed(rows)), 2., 1., 1.)[0])
        self.assertEqual(select_cells(rows, 2., 1., 1.)[0][0]["delta_s_m"], 2.4)

    def test_boundary_metrics_use_world_y_and_final_projection_increment(self):
        segment = SimpleNamespace(points=np.array([[0., 0.], [10., 0.]]), starts=np.array([0.]),
                                  vectors=np.array([[10., 0.]]), lengths=np.array([10.]), length=10.)
        states = np.zeros((4, 35))
        states[:, 34] = [-1., -2., -2., -1.]
        states[-1, 3] = 50.  # Rising vertically does not mean returning forward or from above.
        row = dict(segment_index=0, start_step=0, end_step=3)
        measured = boundary_metrics(row, states, np.array([1., 4., 3.2, 3.]),
                                    SimpleNamespace(segments=[segment]), .1)
        self.assertEqual(measured["departure_relative_y_m"], -1.)
        self.assertEqual(measured["return_relative_y_m"], -1.)
        self.assertAlmostEqual(measured["return_projection_speed_mps"], -2.)
        self.assertEqual(measured["return_projection_xy_m"], [3., 0.])

    def test_upper_either_endpoint_and_negative_speed_but_not_zero_are_rejected(self):
        base = dict(delta_s_m=2., departure_relative_y_m=-1., return_relative_y_m=-1.,
                    return_projection_speed_mps=0.)
        self.assertEqual(admission_rejections(base, 2., True, True), [])
        self.assertEqual(admission_rejections(dict(base, departure_relative_y_m=1.), 2., True, True),
                         ["upper_departure"])
        self.assertEqual(admission_rejections(dict(base, return_relative_y_m=1.), 2., True, True),
                         ["upper_return"])
        self.assertEqual(admission_rejections(dict(base, return_projection_speed_mps=-.1), 2., True, True),
                         ["negative_return_projection_speed"])

    def test_filters_run_before_cell_replacement_so_runner_up_can_survive(self):
        base = dict(segment_index=0, s_start_m=2.1, delta_s_m=2.1, departure_relative_y_m=-1.,
                    return_relative_y_m=-1., return_projection_speed_mps=1.)
        rows = [dict(base, candidate_id="fast_upper", T_seconds=1., return_relative_y_m=1.),
                dict(base, candidate_id="fast_backward", T_seconds=2., return_projection_speed_mps=-1.),
                dict(base, candidate_id="lower_forward", T_seconds=3.)]
        kept, _ = select_cells(rows, 2., 1., 1., reject_upper=True, reject_negative_return=True)
        self.assertEqual([r["candidate_id"] for r in kept], ["lower_forward"])


if __name__ == "__main__":
    unittest.main()

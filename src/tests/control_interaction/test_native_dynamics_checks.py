"""Offline checks for rejecting misleading native-dynamics measurements."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_native_dynamics import PHASES, assess_native_input, assess_native_parameters, assess_native_start, body_error, comparison, read_trace


def fixture():
    rows = []
    for phase_index, phase in enumerate(PHASES):
        rows.append({"step": 0, "phase": phase, "mode": "native", "controllerCalls": 1,
                     "physicsCalls": int(phase_index == 2), "bodies": [0.] * 42,
                     "bodyConfig": [1.] * 63, "bodyNames": [str(n) for n in range(7)],
                     "controls": [0.] * 11, "motors": [0.] * 6, "axes": [40., 30., 0., 0.],
                     "motorParameters": [10000., 10000., 100., 10000., 10000.],
                     "axisReads": [1, 1, 2, 2], "contacts": [], "fixedDeltaTime": 1 / 120,
                     "timeScale": 1})
    return rows


class DynamicsChecks(unittest.TestCase):
    def input_fixtures(self):
        cases = [{"id": f"{kind}_{mode}", "mode": mode, "actions": kind, "repeat": 0}
                 for kind in ("hold", "pulse", "reverse", "random") for mode in ("native", "candidate")]
        return cases, {c["id"]: copy.deepcopy(fixture()) for c in cases}

    def test_native_input_checks_every_phase_and_history_gain(self):
        cases, traces = self.input_fixtures()
        self.assertTrue(assess_native_input(cases, traces)["passed"])
        for phase in range(3):
            for column in (0, 2, 4):
                changed = copy.deepcopy(traces)
                changed["pulse_candidate"][phase]["controls"][column] = .001
                self.assertFalse(assess_native_input(cases, changed)["passed"])

    def test_native_input_requires_reference_coverage_and_finite_values(self):
        cases, traces = self.input_fixtures()
        for excluded in ("pulse_candidate", "random_native"):
            self.assertFalse(assess_native_input([c for c in cases if c["id"] != excluded], traces)["passed"])
        traces["hold_candidate"][1]["controls"][0] = float("nan")
        self.assertFalse(assess_native_input(cases, traces)["passed"])

    def test_native_input_does_not_certify_body_dynamics(self):
        cases, traces = self.input_fixtures()
        traces["hold_candidate"][2]["bodies"][0] = 1.
        self.assertTrue(assess_native_input(cases, traces)["passed"])

    def read(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            Path(str(path) + ".done").write_text("1")
            return read_trace(path, np.float32([[40, 30]]), "native")

    def test_complete_measurement(self):
        self.assertEqual(len(self.read(fixture())), 3)

    def test_missing_physics_rejected(self):
        with self.assertRaises(ValueError):
            self.read(fixture()[:2])

    def test_wrong_actual_axis_rejected(self):
        rows = fixture()
        rows[1]["axes"][0] = 0
        with self.assertRaises(AssertionError):
            self.read(rows)

    def test_extra_controller_call_rejected(self):
        rows = fixture()
        rows[2]["controllerCalls"] = 2
        with self.assertRaises(ValueError):
            self.read(rows)

    def test_wrong_clock_rejected(self):
        rows = fixture()
        rows[0]["timeScale"] = 0
        with self.assertRaises(ValueError):
            self.read(rows)

    def test_initial_mismatch_prevents_attribution(self):
        left, right = fixture(), fixture()
        right[0]["bodies"][0] = 0.02
        self.assertEqual(comparison(left, right)["trajectory_attribution"], "INITIAL_STATE_MISMATCH")

    def test_angle_wrap_does_not_hide_velocity_difference(self):
        a, b = np.zeros(42), np.zeros(42)
        a[2], b[2] = 359.999, -0.001
        a[5] = 360
        errors = body_error(a, b)["errors"]
        self.assertAlmostEqual(errors["angle_deg"], 0)
        self.assertEqual(errors["angular_velocity_deg_s"], 360)

    def test_nonfinite_rejected(self):
        rows = fixture()
        rows[2]["bodies"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.read(rows)

    def test_body_identity_mismatch_rejected(self):
        left, right = fixture(), copy.deepcopy(fixture())
        right[0]["bodyNames"] = right[0]["bodyNames"][::-1]
        with self.assertRaises(ValueError):
            comparison(left, right)

    def test_native_parameters_check_all_phases_and_controller_caches(self):
        cases = [{"id": "native", "mode": "native"}, {"id": "candidate", "mode": "candidate"}]
        original = {"native": fixture(), "candidate": fixture()}
        self.assertTrue(assess_native_parameters(cases, original)["passed"])
        for phase in range(3):
            for field in (1, 4):  # live slider or the cache that can overwrite it
                changed = copy.deepcopy(original)
                changed["candidate"][phase]["motorParameters"][field] = 4300.
                self.assertFalse(assess_native_parameters(cases, changed)["passed"])

    def test_parameters_do_not_certify_trajectories(self):
        cases = [{"id": "native", "mode": "native"}, {"id": "candidate", "mode": "candidate"}]
        traces = {"native": fixture(), "candidate": fixture()}
        traces["candidate"][0]["bodies"][0] = 0.2
        self.assertTrue(assess_native_parameters(cases, traces)["passed"])
        self.assertEqual(comparison(traces["native"], traces["candidate"])["trajectory_attribution"], "INITIAL_STATE_MISMATCH")

    def test_missing_reference_and_old_trace_do_not_pass_parameters(self):
        self.assertFalse(assess_native_parameters([], {})["passed"])
        cases = [{"id": "native", "mode": "native"}, {"id": "candidate", "mode": "candidate"}]
        traces = {"native": fixture(), "candidate": fixture()}
        del traces["candidate"][1]["motorParameters"]
        self.assertFalse(assess_native_parameters(cases, traces)["passed"])

    def test_start_requires_paired_native_and_checks_cursor_and_input_history(self):
        cases = [{"id": mode, "mode": mode, "actions": "zero", "repeat": 0}
                 for mode in ("native", "candidate")]
        traces = {"native": fixture(), "candidate": fixture()}
        self.assertTrue(assess_native_start(cases, traces)["passed"])
        self.assertFalse(assess_native_start(cases[1:], traces)["passed"])
        for field, index in (("bodies", 36), ("controls", 2), ("controls", 8)):
            changed = copy.deepcopy(traces)
            changed["candidate"][0][field][index] = .02
            self.assertFalse(assess_native_start(cases, changed)["passed"])

    def test_start_tolerance_does_not_mean_bitwise_or_trajectory_match(self):
        cases = [{"id": mode, "mode": mode, "actions": "zero", "repeat": 0}
                 for mode in ("native", "candidate")]
        traces = {"native": fixture(), "candidate": fixture()}
        traces["candidate"][0]["bodies"][0] = .0005
        traces["candidate"][2]["bodies"][0] = 1.
        result = assess_native_start(cases, traces)
        self.assertTrue(result["passed"])
        self.assertFalse(result["cases"][0]["bodies"]["bitwise_equal"])


if __name__ == "__main__":
    unittest.main()

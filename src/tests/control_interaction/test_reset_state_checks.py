"""Offline regression checks for reset acceptance; never starts the game."""
import json
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.control_interaction.test_reset_state import (
    Limits, assess, compare, history_actions, measure_case, overall_status,
    ensure_idle, main, parse_args, run, runtime_settings, validate_obs, assess_diagnostics,
    assess_native_motor_limits, assess_native_start_bodies,
)


def state(width=33):
    obs = np.zeros((2, width), dtype=np.float32)
    obs[:, [0, 5, 10, 15, 19, 23, 29]] = -44
    obs[:, [1, 6, 11, 16, 20, 24, 30]] = -2
    obs[:, 23] = -42
    return obs


class ResetChecks(unittest.TestCase):
    def test_native_start_rejects_repeatable_clone_cursor_offset(self):
        names = ['Player', 'Hub', 'Slider', 'Handle', 'PoleMiddle', 'Tip', 'Cursor']
        bodies = [{"agent": a, "name": name, "position": {"x": 1., "y": 2.},
                   "rotation": 0., "velocity": {"x": 0., "y": 0.}, "angularVelocity": 0.}
                  for a in range(2) for name in names]
        packet = {"current": {"bodies": bodies, "controls": [{"valid": True}] * 2}}
        data = {phase: copy.deepcopy(packet) for phase in ('after_reset', 'after_wait')}
        reference = {'bodyNames': names, 'bodies': [1., 2., 0., 0., 0., 0.] * 7}
        self.assertTrue(assess_native_start_bodies(data, 2, reference, Limits())['passed'])
        for phase in data:
            data[phase]['current']['bodies'][-1]['position']['y'] += .035
        self.assertFalse(assess_native_start_bodies(data, 2, reference, Limits())['passed'])
        self.assertEqual(data['after_reset'], data['after_wait'])

    def test_native_caps_checked_for_every_clone_and_after_probe(self):
        expected = {"hinge_cap": 10000., "slider_cap": 10000., "hub_cap": 100.}
        joints = []
        for agent in range(2):
            for name, kind, connected, cap in (("Player", "HingeJoint2D", "Hub", 100.),
                    ("Hub", "HingeJoint2D", "Slider", 10000.), ("Slider", "SliderJoint2D", "Handle", 10000.)):
                joints.append(json.dumps(dict(agent=agent, name=name, type=kind, connected=connected, maxMotorTorque=cap)))
        data = {phase: {"current": {"joints": list(joints)}} for phase in ("after_reset", "after_wait", "after_probe")}
        self.assertTrue(assess_native_motor_limits(data, 2, expected)["passed"])
        changed = copy.deepcopy(data)
        joint = json.loads(changed["after_probe"]["current"]["joints"][-1])
        joint["maxMotorTorque"] = 4300.
        changed["after_probe"]["current"]["joints"][-1] = json.dumps(joint)
        self.assertFalse(assess_native_motor_limits(changed, 2, expected)["passed"])
        changed["after_probe"]["current"]["joints"].pop()
        self.assertFalse(assess_native_motor_limits(changed, 2, expected)["passed"])

    def test_diagnostics_reject_hidden_motion_extra_steps_and_unpaused_clock(self):
        bodies = [{"agent": a, "name": str(i), "position": {"x": 1, "y": 2},
                   "rotation": 0, "velocity": {"x": 0, "y": 0}, "angularVelocity": 0}
                  for a in range(2) for i in range(7)]
        packet = {"epoch": 1, "controlledUpdates": True, "current": {
            "bodies": bodies, "controls": [{"valid": True}] * 2,
            "manualCalls": 0, "simulationCalls": 0, "worldBodyCount": 14, "timeScale": 0}}
        reference = {key: copy.deepcopy(packet) for key in ("after_reset", "after_wait", "after_probe")}
        reference["after_probe"]["current"].update(manualCalls=2, simulationCalls=1)
        def check(data):
            return assess_diagnostics(data, 2, Limits(), "zero", 1, reference)["passed"]
        self.assertTrue(check(reference))
        changed = copy.deepcopy(reference)
        changed["after_probe"]["current"]["controls"][0]["mouseInput"] = {"x": 1, "y": 0}
        self.assertFalse(check(changed))  # Body state alone cannot certify input-history reset.
        for field, value in [("timeScale", 1), ("simulationCalls", 2), ("worldBodyCount", 15)]:
            changed = copy.deepcopy(reference)
            changed["after_probe"]["current"][field] = value
            self.assertFalse(check(changed), field)
        changed = copy.deepcopy(reference)
        changed["after_probe"]["current"]["bodies"][5]["angularVelocity"] = 65
        self.assertFalse(check(changed))
        changed = copy.deepcopy(reference)
        changed["after_wait"]["current"]["controls"][0]["valid"] = False
        with self.assertRaises(ValueError):
            check(changed)

    def test_cold_cross_agent_divergence_is_informational(self):
        cold = np.stack([state(), state()])
        cold[:, 1, 0] += 0.1
        result = assess(cold, cold, Limits())
        self.assertFalse(result["agents_aligned"])
        self.assertTrue(result["passed"])
        self.assertFalse(assess(cold, cold, Limits(), require_alignment=True)["passed"])
        changed = cold.copy()
        changed[1, 1, 0] += 0.01
        self.assertFalse(assess(changed, cold, Limits())["passed"])

    def test_live_cold_reference_never_calls_reset(self):
        class Env:
            def reset(self):
                raise AssertionError("Cold reference must precede RESET")

            def reset_diagnostics(self):
                return {"current": {"states": state().flatten().tolist()}}

            def step(self, action):
                return state(), np.zeros(2, dtype=bool)

        with tempfile.TemporaryDirectory() as directory:
            trace, _, _ = measure_case(Env(), None, [], 0, 1, 2,
                Path(directory) / "cold.npz", diagnostics=True, cold_live=True)
            np.testing.assert_array_equal(trace[0], state())

    def test_angles_wrap_but_angular_velocity_does_not(self):
        before, after = state(), state()
        before[:, [9, 14, 27]] = 359.99
        after[:, [9, 14, 27]] = 0.01
        result = compare(after, before, Limits())
        self.assertTrue(result["passed"])
        self.assertFalse(result["bitwise_equal_without_timestamp"])
        before[:, 4], after[:, 4] = 359.99, 0.01
        self.assertFalse(compare(after, before, Limits())["groups"]["angular_velocity"]["passed"])

    def test_timestamp_is_not_a_physics_error(self):
        a, b = state(), state()
        b[:, 28] = 9876
        result = compare(a, b, Limits())
        self.assertTrue(result["passed"])
        self.assertTrue(result["bitwise_equal_without_timestamp"])

    def test_common_kick_cannot_pass_by_cross_agent_agreement(self):
        reference = np.stack([state(), state()])
        kicked = reference.copy()
        kicked[:, :, 4] = -65
        result = assess(kicked, kicked, Limits())
        self.assertTrue(result["agents_aligned"])
        self.assertTrue(result["matches_reference"])
        self.assertFalse(result["stationary"])
        self.assertFalse(result["passed"])
        self.assertFalse(assess(kicked, reference, Limits(), False)["passed"])

    def test_first_step_impulse_fails_even_when_reset_packet_matches(self):
        reference = np.stack([state(), state()])
        kicked = reference.copy()
        kicked[1, :, 2] = 0.5
        result = assess(kicked, reference, Limits())
        self.assertTrue(result["frames"][0]["vs_cold_reference"]["passed"])
        self.assertFalse(result["passed"])

    def test_tip_cursor_and_anchor_are_checked(self):
        for column, group in [(25, "linear_velocity"), (31, "cursor_velocity"), (33, "anchor_position")]:
            a, b = state(35), state(35)
            a[1, column] += 1
            result = compare(a, b, Limits())
            self.assertFalse(result["passed"])
            self.assertEqual(result["groups"][group]["agent"], 1)
            self.assertEqual(result["groups"][group]["index"], column)

    def test_invalid_packets_fail(self):
        for obs in [np.zeros((2, 33)), np.full((2, 33), np.nan), state()[:1], state()[:, :29]]:
            with self.assertRaises(ValueError):
                validate_obs(obs, 2)

    def test_one_failed_or_missing_case_cannot_pass(self):
        record = {"assessment": {"passed": True}}
        report = {"errors": [], "expected_cases": 2, "cases": [record]}
        self.assertEqual(overall_status(report), "ERROR")
        report["cases"].append({"assessment": {"passed": False}})
        self.assertEqual(overall_status(report), "FAIL")
        report["cases"][1] = {"assessment": {"passed": True}, "history_exercised": False}
        self.assertEqual(overall_status(report), "FAIL")
        report["cases"][1] = record
        self.assertEqual(overall_status(report), "PASS")
        report["errors"].append("timeout")
        self.assertEqual(overall_status(report), "ERROR")

    def test_histories_reproducible_and_independent_per_agent(self):
        a = history_actions("random", 200, 2, 10)
        self.assertTrue(np.array_equal(a, history_actions("random", 200, 2, 10)))
        self.assertFalse(np.array_equal(a[:, 0], a[:, 1]))

    def test_measurement_does_not_take_new_snapshot_or_read_cached_diagnostics(self):
        class Env:
            def __init__(self):
                self.commands = []

            def reset(self):
                self.commands.append("R")
                return state()

            def step(self, action):
                self.commands.append("S")
                return state(), np.zeros(2, dtype=bool)

        env = Env()
        with tempfile.TemporaryDirectory() as directory, patch("time.sleep") as sleep:
            path = Path(directory) / "case.npz"
            trace, _, _ = measure_case(env, state(), history_actions("hold", 2, 2, 0), 0.25, 1, 2, path)
            self.assertEqual(env.commands, ["S", "S", "R", "S"])
            sleep.assert_called_once_with(0.25)
            self.assertEqual(trace.shape, (2, 2, 33))
            with np.load(path) as data:
                self.assertEqual(data["history_states"].shape, (3, 2, 33))
                self.assertEqual(data["history_actions"].shape, (2, 2, 2))
                self.assertEqual(data["reset_and_probe"].shape, (2, 2, 33))

    def test_partial_invalid_data_is_preserved(self):
        class Env:
            def reset(self):
                return np.full((2, 33), np.nan, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failed.npz"
            with self.assertRaises(ValueError):
                measure_case(Env(), None, [], 0, 1, 2, path)
            with np.load(path) as data:
                self.assertTrue(np.isnan(data["reset_and_probe"]).all())

    def test_runtime_configuration_and_mode_signals_restored_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "GoiData/runtime_config.json"
            signal = root / "GoiData/ControlSignals/mode_game_testing.signal"
            signal.parent.mkdir(parents=True)
            config.write_bytes(b'{"numDuplicates":8,"tcpPort":9012}')
            signal.write_bytes(b"original")
            original = config.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with runtime_settings(root, 2, 9123):
                    self.assertEqual(json.loads(config.read_text())["tcpPort"], 9123)
                    raise RuntimeError("fixture failure")
            self.assertEqual(config.read_bytes(), original)
            self.assertEqual(signal.read_bytes(), b"original")
            self.assertFalse((signal.parent / "mode_game_runtime.signal").exists())

    def test_cli_uses_project_port(self):
        args = parse_args([])
        project = Path(__file__).resolve().parents[2] / "config/project.json"
        self.assertEqual(args.port, json.loads(project.read_text())["tcp_port"])
        self.assertEqual(args.probe_steps, 1)

    def test_each_restart_recreates_consumed_runtime_signal(self):
        class Env:
            def __init__(self, **kwargs):
                self._sock = SimpleNamespace(settimeout=lambda timeout: None)

            def connect(self):
                pass

            def reset(self):
                return state()

            def step(self, action):
                return state(), np.zeros(2, dtype=bool)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal = root / "GoiData/ControlSignals/mode_game_runtime.signal"
            launches = []

            def launch(wait):
                self.assertTrue(signal.exists(), "Every process needs its own mode signal")
                signal.unlink()  # Same consumption as the game at startup.
                launches.append(len(launches) + 1)
                return SimpleNamespace(pid=launches[-1], poll=lambda: 0, wait=lambda timeout: 0)

            args = parse_args(["--out", str(root / "results"), "--restarts", "3",
                               "--burn-steps", "1", "--idle-seconds", "0"])
            with patch("start.game_launcher.GameLauncher", return_value=SimpleNamespace(
                    game_root=root, launch=launch)), \
                 patch("env.goi_env.GoiEnv", Env), \
                 patch("tests.control_interaction.test_reset_state.ensure_idle"), \
                 patch("time.sleep"):
                self.assertEqual(run(args), 1)  # Stationary fake histories do not exercise physics.
            report = json.loads((args.out / "result.json").read_text())
            self.assertEqual(len(launches), 3)
            self.assertEqual(len(report["cases"]), report["expected_cases"])
            self.assertEqual(report["errors"], [])
            self.assertFalse(signal.exists())

    def test_preflight_error_is_not_a_numerical_failure(self):
        with patch("tests.control_interaction.test_reset_state.run", side_effect=RuntimeError("port busy")):
            with self.assertLogs("reset_state", level="ERROR"):
                self.assertEqual(main([]), 2)

    def test_tasklist_localized_bytes_and_existing_process(self):
        prefix = "tests.control_interaction.test_reset_state."
        with patch(prefix + "sys.platform", "win32"), \
             patch(prefix + "subprocess.CREATE_NO_WINDOW", 0, create=True), \
             patch(prefix + "subprocess.run") as tasklist, \
             patch(prefix + "socket.socket") as socket:
            socket.return_value.__enter__.return_value.connect_ex.return_value = 1
            tasklist.return_value = SimpleNamespace(stdout=b"INFO: \xd0\xc5\xcf\xa2\r\n")
            ensure_idle(9001)
            tasklist.return_value = SimpleNamespace(stdout=b'"GettingOverIt.exe","123","\xd0\xc5\xcf\xa2"\r\n')
            with self.assertRaisesRegex(RuntimeError, "existing GettingOverIt"):
                ensure_idle(9001)


if __name__ == "__main__":
    unittest.main()

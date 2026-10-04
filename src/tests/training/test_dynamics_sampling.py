"""Momentum-based SIL weighting: physics null cases and allocation contracts."""
import copy
from collections import Counter
import unittest

import numpy as np

from src.tests.training.test_elite_archive import episode
from training.climb_diagnostics import DYNAMICS_SPEC, contact_dynamics_importance, PhaseRules
from training.gru_sil import GRUSILSamplingPlan, DYNAMICS_SAMPLING_MODE


def physics_episode(n=128, force=None, internal=False):
    traj = episode(t=n, tail=0)
    traj.raw_states[:] = 0
    traj.contact[:] = 0
    masses = np.asarray(DYNAMICS_SPEC["masses"])
    weights = masses/masses.sum()
    drag = np.asarray(DYNAMICS_SPEC["linear_drag"])
    dt = 1/120
    velocity = np.zeros(2)
    for t in range(n+1):
        relative = np.zeros((6,2))
        if internal:
            relative[0,0] = np.sin(t*.6)*3
            relative[-1,0] = -relative[0,0]*masses[0]/masses[-1]
        if t:
            extra = np.zeros(2) if force is None else np.asarray(force(t-1))
            drag_relative = (relative*(weights*drag)[:,None]).sum(axis=0)
            velocity = (velocity+dt*(np.array([0.,-30.])+extra-drag_relative))/(1+dt*(weights*drag).sum())
        for j, column in enumerate(DYNAMICS_SPEC["velocity_indices"]):
            traj.raw_states[t,column:column+2] = velocity+relative[j]
    return traj


class DynamicsSamplingTests(unittest.TestCase):
    def diagnose(self, traj, cap=None):
        return contact_dynamics_importance(traj.raw_states, traj.contact, cap or len(traj.actions))

    def test_ballistic_gravity_and_airborne_internal_motion_are_not_events(self):
        for internal in (False, True):
            traj = physics_episode(internal=internal)
            result = self.diagnose(traj)
            self.assertLess(np.abs(result["non_gravity_acceleration"]).max(), .002)
            self.assertEqual(result["events"], [])
            self.assertTrue(np.all(result["priority"] == 0))
            # Contact mislabelling alone must not promote a gravity-only change.
            traj.contact[:,0] = 1
            self.assertFalse(self.diagnose(traj)["core"].any())

    def test_static_support_cancels_gravity_without_receiving_priority(self):
        traj = physics_episode()
        traj.raw_states[:] = 0
        traj.contact[:,3] = 1
        result = self.diagnose(traj)
        np.testing.assert_allclose(result["non_gravity_acceleration"][:,1], 30)
        self.assertFalse(result["core"].any())
        self.assertEqual(result["priority"].sum(), 0)

    def test_sustained_contact_acceleration_and_short_context_stop_before_airborne_peak(self):
        traj = physics_episode(force=lambda t: [30,150] if 32 <= t < 64 else [0,0])
        traj.contact[32:65,0] = 1
        result = self.diagnose(traj)
        self.assertTrue(result["core"][32:64].all())
        self.assertGreater(result["priority"][20], 0)
        self.assertGreater(result["priority"][69], 0)
        self.assertTrue(np.all(result["priority"][70:] == 0))
        self.assertFalse(result["core"][:32].any())

    def test_redirecting_velocity_is_kept_but_dissipative_pot_impact_is_not(self):
        traj = physics_episode()
        traj.raw_states[:] = 0
        for column in DYNAMICS_SPEC["velocity_indices"]:
            traj.raw_states[:40,column:column+2] = [5,0]
            traj.raw_states[40:,column:column+2] = [0,5]
        traj.contact[39:41,0] = 1
        turn = self.diagnose(traj)
        self.assertTrue(turn["core"][39])
        self.assertGreater(turn["turning_acceleration"][39], 500)
        self.assertTrue(any(e["kind"] == "turning" for e in turn["events"]))
        traj.contact[:] = 0
        traj.contact[40:,3] = 1
        for column in DYNAMICS_SPEC["velocity_indices"]:
            traj.raw_states[:40,column:column+2] = [5,-10]
            traj.raw_states[40:,column:column+2] = [0,0]
        landing = self.diagnose(traj)
        self.assertTrue(landing["passive_impact"][39])
        self.assertFalse(landing["core"].any())

    def test_events_beyond_cap_do_not_reach_back_into_the_training_prefix(self):
        traj = physics_episode(force=lambda t: [40,150] if t >= 90 else [0,0])
        traj.contact[90:,0] = 1
        self.assertEqual(self.diagnose(traj, 85)["priority"].sum(), 0)

    def test_farthest_trajectories_get_larger_mass_ties_equal_and_length_does_not_dilute_mass(self):
        trajs = [episode(t=64, distance=4), episode(t=160, distance=8), episode(t=96, distance=8)]
        plan = GRUSILSamplingPlan(trajs, [0,1,2], 32, DYNAMICS_SAMPLING_MODE)
        p = plan.trajectory_probabilities
        self.assertGreater(p[1], p[0])
        self.assertAlmostEqual(p[1], p[2])
        for i in range(3):
            self.assertAlmostEqual(sum(v for k,v in plan.probabilities().items() if k[0] == i), p[i])
        self.assertAlmostEqual(sum(p.values()), 1)
        changed = GRUSILSamplingPlan(trajs, [0,1,2], 32, DYNAMICS_SAMPLING_MODE,
                                      PhaseRules(tree_x_min=100, tree_x_max=200))
        self.assertEqual(plan.probabilities(), changed.probabilities())

    def test_actual_draws_match_joint_probabilities_and_maintain_positive_air_floor(self):
        traj = physics_episode(force=lambda t: [50,150] if 32 <= t < 64 else [0,0])
        traj.contact[32:65,0] = 1
        plan = GRUSILSamplingPlan([traj], [0], 32, DYNAMICS_SAMPLING_MODE)
        p = plan.probabilities()
        self.assertGreater(p[(0,32,64,128)], 10*p[(0,96,128,128)])
        self.assertGreater(p[(0,96,128,128)], 0)
        rng, again = np.random.default_rng(7), np.random.default_rng(7)
        counts = Counter()
        for _ in range(10000):
            draw = plan.draw(rng)
            self.assertEqual(draw, plan.draw(again))
            counts[draw[:4]] += 1
        for key in p: self.assertAlmostEqual(counts[key]/10000, p[key], delta=.015)

    def test_invalid_physics_inputs_rejected_and_arrays_not_modified(self):
        traj = physics_episode()
        before = copy.deepcopy((traj.raw_states, traj.contact))
        self.diagnose(traj)
        np.testing.assert_array_equal(traj.raw_states, before[0])
        np.testing.assert_array_equal(traj.contact, before[1])
        with self.assertRaises(ValueError):
            contact_dynamics_importance(traj.raw_states, traj.contact, 128, 0)
        traj.raw_states[0,2] = np.nan
        with self.assertRaises(ValueError): self.diagnose(traj)


if __name__ == "__main__": unittest.main()

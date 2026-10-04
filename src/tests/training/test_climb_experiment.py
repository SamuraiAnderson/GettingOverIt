"""Motion alignment, reward invariance and strict offline experiment contracts."""
import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"src"))
from training.climb_diagnostics import PhaseRules, classify_motion, discount_summary, runs
from training.climb_experiment import ARMS, identity, load_recipe, resolve_config, reset_experiment_critic
from training.config import TrainConfig
from training.experimental_policy import create_actor_critic
from training.gru_sil import phase_chunk_groups
from training.main_ppo import main, reference_reward_settings
from training.reference_corridor import CorridorSegment, ReferenceCorridor, replay_corridor_result
from training.reference_fingerprint import reference_fingerprint
from src.tests.training.test_elite_archive import episode


class ClimbExperimentTests(unittest.TestCase):
    def baseline(self):
        return replace(TrainConfig.ppo_mainline(), num_agents=8, steps_per_rollout=1200, random_deploy=False,
                       gru_sil_enabled=True, elite_archive_enabled=True,
                       reference_progress_mode="confirmed_endpoint_v2", reference_trajectory="test-route.json")

    def recipe(self, root, arm="control"):
        cfg = self.baseline()
        checkpoint = root/"source/checkpoints/model.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(cfg), "model": {}}, checkpoint)
        pool = root/"source/pool.npz"; pool.write_bytes(b"pool")
        inputs = {"checkpoint": identity(checkpoint)}
        for name in ("elite_pool", "pool_fingerprint", "route", "runtime_dll"):
            inputs[name] = identity(pool)
        resolved = resolve_config(asdict(cfg), arm, 7, 3)
        return {"schema": "tree_climb_experiment_v1", "arm": arm, "seed": 7, "iterations": 3,
                "baseline_config": asdict(cfg), "resolved_config": asdict(resolved), "inputs": inputs,
                "run_dir": str(root/"experiment/run"), "source_files": [], "phase_rules": asdict(PhaseRules()),
                "initialization": "same_actor_and_std_fresh_optimizer_reset_critic_two_warmup_rounds",
                "budget": {"iterations": 3, "maximum_executed_world_steps": 28800,
                           "warmup_included": True, "unit": "executed_world_steps_not_active_transitions"}}

    def test_one_intervention_and_identical_common_initialization_config(self):
        base = asdict(resolve_config(asdict(self.baseline()), "control", 7, 3))
        for arm, overrides in ARMS.items():
            actual = asdict(resolve_config(asdict(self.baseline()), arm, 7, 3))
            self.assertEqual({k:v for k,v in actual.items() if v != base[k]}, overrides)
        with self.assertRaises(ValueError):
            resolve_config(asdict(replace(self.baseline(), gru_sil_enabled=False)), "control", 7, 3)
        self.assertLessEqual(base["sil_start_iter"],base["value_warmup_iters"])

    def test_added_guidance_field_does_not_restart_legacy_resume_warmup(self):
        old=asdict(self.baseline()); old.pop("reference_guidance_sigma")
        current=dict(old,reference_guidance_sigma=None)
        self.assertEqual(reference_reward_settings(old),reference_reward_settings(current))
        current["reference_corridor_radius"]=3
        self.assertNotEqual(reference_reward_settings(old),reference_reward_settings(current))

    def test_movement_labels_are_pre_action_and_do_not_use_D(self):
        s = np.zeros((25,35), np.float32); c = np.ones((25,5), np.float32)
        s[:,33] = -30; s[:,23] = -30; s[:,24] = -2
        c[8:20] = 0; s[8:20,3] = 2; s[8:20,34] = np.linspace(0,1,12)
        labels, result = classify_motion(s,c,PhaseRules(preparation_steps=4))
        self.assertEqual(len(labels),24)
        self.assertEqual(labels[3],"tree_support")
        self.assertEqual(labels[4],"takeoff_preparation")
        self.assertEqual(labels[8],"tree_air_ascent")
        self.assertEqual(labels[20],"tree_support")
        self.assertTrue(result["flights"][0]["takeoff_candidate"])
        self.assertEqual(runs([True,True,False,True]),[(0,2),(3,4)])
        c[:8] = 0
        self.assertFalse(classify_motion(s,c)[1]["flights"][0]["takeoff_candidate"])

    def test_future_rewards_and_boundary_starts(self):
        report = discount_summary([0,0,4],.5,.1,[0,2,3])
        self.assertEqual([r["action"] for r in report["starts"]],[0,2])
        self.assertEqual(report["starts"][0]["finite_tail_return"],1)
        self.assertEqual(report["starts"][0]["next_positive_delay_s"],.2)
        self.assertEqual(report["bootstrap"],0)
        with self.assertRaises(ValueError):
            PhaseRules(minimum_flight_steps=1.5)

    def test_guidance_sigma_leaves_rewards_and_old_maps_unchanged(self):
        cfg = TrainConfig(reference_corridor_radius=1.5, reference_progress_mode="confirmed_endpoint_v2")
        segment = CorridorSegment(np.array([[0.,0.],[10.,0.]]),np.ones(1))
        old = ReferenceCorridor([segment],cfg)
        same = ReferenceCorridor([segment],replace(cfg,reference_guidance_sigma=1.5))
        wide = ReferenceCorridor([segment],replace(cfg,reference_guidance_sigma=3.))
        m0,m1,m2 = [r.guidance_maps((20,20),-5,-5,1)[0] for r in (old,same,wide)]
        np.testing.assert_array_equal(m0,m1)
        self.assertGreater(np.count_nonzero(m2),np.count_nonzero(m0))
        # Legacy 33D origin scoring keeps this test independent of calibration files.
        states = np.zeros((4,33)); states[:,0]=[0,1,2,3]; states[:,1]=[0,0,3,3]
        for key in ("rewards","confirmed_progress","credit_events"):
            np.testing.assert_array_equal(replay_corridor_result(old,states)[key],replay_corridor_result(wide,states)[key])
        self.assertEqual(reference_fingerprint(cfg),reference_fingerprint(wide.config))
        with self.assertRaises(ValueError):
            ReferenceCorridor([segment],replace(cfg,reference_guidance_sigma=float("nan")))

    def test_phase_sampler_never_trains_the_uncredited_tail(self):
        t=episode(t=9,tail=7)
        t.raw_states[:,33]=-30; t.raw_states[:,24]=-2
        groups=phase_chunk_groups([t],[0],4,PhaseRules())
        chunks=[v for values in groups.values() for v in values]
        self.assertEqual(sorted((a,b) for ti,a,b,cap in chunks),[(0,4),(4,8),(8,9)])
        self.assertTrue(all(cap==9 for ti,a,b,cap in chunks))

    def test_hash_lock_and_unregistered_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); recipe=self.recipe(root); path=root/"recipe.json"
            path.write_text(json.dumps(recipe))
            load_recipe(path)
            recipe["resolved_config"]["lr"] *= 2
            path.write_text(json.dumps(recipe))
            with self.assertRaises(ValueError): load_recipe(path)
            recipe=self.recipe(root); path.write_text(json.dumps(recipe))
            Path(recipe["inputs"]["elite_pool"]["path"]).write_bytes(b"changed")
            with self.assertRaises(ValueError): load_recipe(path)

    def test_production_dry_run_inherits_config_without_game_or_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); recipe=self.recipe(root,"long_discount"); path=root/"recipe.json"
            path.write_text(json.dumps(recipe))
            stdout=io.StringIO()
            with patch.object(sys,"argv",["main_ppo","--experiment-config",str(path),"--dry-run"]), \
                 patch("training.main_ppo.RolloutWorker",side_effect=AssertionError("No game")), \
                 patch("training.main_ppo.create_actor_critic",side_effect=AssertionError("No model")), \
                 contextlib.redirect_stdout(stdout):
                main()
            resolved=json.loads(stdout.getvalue())["resolved_config"]
            self.assertEqual(resolved,recipe["resolved_config"])
            self.assertFalse(Path(recipe["run_dir"]).exists())
            with patch.object(sys,"argv",["main_ppo","--experiment-config",str(path),"--dry-run","--gamma",".98"]):
                with self.assertRaises(ValueError): main()

    def test_critic_reset_preserves_every_policy_tensor(self):
        cfg=replace(self.baseline(),d_model=16,nhead=2,num_layers=1)
        model=create_actor_critic(cfg)
        before={k:v.clone() for k,v in model.state_dict().items()}
        reset_experiment_critic(model)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items() if not k.startswith("critic_head.")))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items() if k.startswith("critic_head.")))

    def test_recipe_regeneration_uses_mixed_arm_and_current_production_parser(self):
        from src.tests.analysis.campaign.prepare_tree_climb import write_recipes
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); recipe=self.recipe(root)
            saved=torch.load(recipe["inputs"]["checkpoint"]["path"],weights_only=False)
            with patch("training.main_ppo.create_actor_critic",side_effect=AssertionError("Configuration only")):
                plans=write_recipes(root/"new_configs",saved,recipe["inputs"],[],PhaseRules())
            self.assertEqual(len(plans),24)
            self.assertEqual({p["arm"] for p in plans},{"control","guidance_wide","sil_mixed","long_discount"})
            for plan in plans:
                loaded,cfg=load_recipe(plan["path"])
                self.assertEqual(cfg.gru_sil_sampling_mode,
                                 "quality_dynamics_v3" if plan["arm"]=="sil_mixed" else "trajectory_uniform")
            self.assertFalse((root/"new_configs/training").exists())


if __name__=="__main__": unittest.main()

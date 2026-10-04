"""Isolated real-data SIL budget comparison; no PPO or environment interaction."""
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from tests.analysis.campaign.prepare_tree_climb import make_encoder
from training.config import TrainConfig
from training.decision_buffer import DecisionRolloutBuffer
from training.elite_archive import EliteTrajectoryArchive
from training.experimental_policy import create_actor_critic
from training.gru_sil import (GRUSILSamplingPlan, eligible_sil_trajectories, gru_sil_update,
    sequence_predictions, sequence_features, discounted_returns, rollout_distributions, conditional_kl)
from training.ppo_trainer import PPOTrainer
from training.random_state import restore_random_state
from training.reference_corridor import ReferenceCorridor, trajectory_endpoint
from training.sequence_cache import SILObservationCache


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def ident(traj):
    return hashlib.sha256(traj.raw_states.tobytes() + traj.actions.tobytes()).hexdigest()


def finalize(report):
    """Audit completed arms; allow tiny CUDA reduction rounding in scalar KL."""
    old, new = report["arms"]["old_4"], report["arms"]["new_16"]
    differences = []
    def compare(a, b, path):
        assert type(a) is type(b), path
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for key in a:
                compare(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for i, (x,y) in enumerate(zip(a,b)):
                compare(x,y,f"{path}/{i}")
        elif isinstance(a, float):
            assert abs(a-b) <= 1e-8, (path,a,b)
            if a != b:
                differences.append({"field": path, "absolute_difference": abs(a-b)})
        else:
            assert a == b, (path,a,b)
    compare(old["sampling"]["chunks"], new["sampling"]["chunks"][:len(old["sampling"]["chunks"])], "chunks")
    compare(old["step_trace"], new["step_trace"][:len(old["step_trace"])], "step_trace")
    assert all(sha(p) == digest for p,digest in report["input_hashes"].items())
    assert all(sha(ROOT/p) == digest for p,digest in report["source_hashes"].items())
    source = torch.load(next(p for p in report["input_hashes"] if p.endswith("ppo_iter_0209.pt")), map_location="cpu", weights_only=False)
    optimizer_steps = {}
    for name, arm in report["arms"].items():
        after = torch.load(OUT/"checkpoints"/f"{name}_after.pt", map_location="cpu", weights_only=False)
        counts = Counter(int(state["step"] - source["optimizer"]["state"][pid]["step"])
                         for pid,state in after["optimizer"]["state"].items())
        assert max(counts) == arm["metrics"]["gru_sil_updates"]
        assert set(counts) <= {0, arm["metrics"]["gru_sil_updates"]}
        assert torch.equal(after["model"]["actor_log_std"], source["model"]["actor_log_std"])
        assert all(torch.isfinite(v).all() for v in after["model"].values())
        assert all(torch.isfinite(v).all() for state in after["optimizer"]["state"].values()
                   for v in state.values() if torch.is_tensor(v))
        assert abs(arm["audited_kl_mean"]-arm["metrics"]["gru_sil_kl"]) <= 1e-8
        assert abs(arm["audited_kl_max_frame"]-arm["metrics"]["gru_sil_frame_kl_max"]) <= 1e-8
        optimizer_steps[name] = dict(counts)
    report["checks"].update(same_first_four_sampling_choices_and_step_results=True,
                             prefix_scalar_absolute_tolerance=1e-8, prefix_scalar_differences=differences,
                             source_inputs_unchanged=True, source_code_unchanged=True,
                             standard_deviation_unchanged=True, finite_model_and_optimizer=True,
                             optimizer_step_deltas=optimizer_steps)
    report["status"] = "completed"
    write("report.json", report)
    print("Completed isolated SIL budget test and final audit", flush=True)


@torch.no_grad()
def predict(model, frames):
    return tuple(a.cpu().numpy() for a in sequence_predictions(model, frames))


def fit(prediction, before, traj, cfg, start, end):
    mean, std, value = prediction
    old_mean, old_std, old_value = before
    raw = np.clip(np.arctanh(np.clip(traj.actions / cfg.action_scale, -1 + 1e-6, 1 - 1e-6)),
                  -cfg.sil_target_clamp, cfg.sil_target_clamp)
    returns = discounted_returns(traj.step_rewards, traj.terminated,
        0. if traj.terminated[-1] else float(old_value[-1]), cfg.gamma)
    advantage = np.clip(returns - old_value[:-1], 0, cfg.sil_advantage_cap)
    sl = slice(start, end)
    loss = .5 * np.square((mean[:-1] - raw) / old_std[:-1]).sum(-1)
    error = cfg.action_scale * np.tanh(mean[:-1]) - traj.actions
    return {"start": start, "end": end, "frames": end - start,
            "action_rmse": float(np.sqrt(np.mean(error[sl] ** 2))),
            "fixed_advantage_actor_loss": float(cfg.gru_sil_loss_coef * np.mean(advantage[sl] * loss[sl])),
            "baseline_positive_frames": int((advantage[sl] > 0).sum()),
            "fixed_return_value_huber": float(np.mean((advantage[sl] > 0) * np.where(
                np.abs(value[:-1][sl]-returns[sl]) < 1, .5 * (value[:-1][sl]-returns[sl])**2,
                np.abs(value[:-1][sl]-returns[sl]) - .5)))}


def main():
    torch.set_num_threads(4)
    started = time.perf_counter()
    run = ROOT / "runs/gru_sil_dynamics_1h2_20261004"
    ckpt = run / "checkpoints/ppo_iter_0209.pt"
    pool = run / "elites/full_episodes/elites_iter_0209.npz"
    evaluation = run / "post_training_review/checkpoint_0209"
    inputs = [ckpt, pool, evaluation / "exploration_000.npz", evaluation / "behavior.pt",
              evaluation / "manifest.json"]
    input_hashes = {str(p): sha(p) for p in inputs}
    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = replace(TrainConfig(**saved["config"]), gru_sil_max_updates=16, gru_sil_budget_ratio=1.)
    assert cfg.gru_sil_sampling_mode == "quality_dynamics_v3" and cfg.gru_sil_seed == 7
    assert cfg.gru_sil_cache_mb == 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_actor_critic(cfg).to(device).eval()
    model.load_state_dict(saved["model"])
    model.exploration_boost = saved["loop_state"]["boost"]
    assert model.exploration_boost == 0
    behavior = torch.load(evaluation / "behavior.pt", map_location="cpu", weights_only=False)
    assert behavior["model"].keys() == saved["model"].keys()
    assert all(torch.equal(v, behavior["model"][k]) for k, v in saved["model"].items())
    buffer = DecisionRolloutBuffer.load(evaluation / "exploration_000.npz")
    assert buffer.size == 9600 and not buffer.policy_context.get("deterministic")
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))
    archive = EliteTrajectoryArchive(cfg, OUT / "unused_elite_output", meta["reference_fingerprint"])
    assert archive.restore(pool) == 17
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    assert route.metadata["sha256"] == saved["reference_corridor"]["sha256"]
    encoder, _ = make_encoder(cfg, route)
    cache = encoder._gru_sil_observation_cache = SILObservationCache()
    iteration = int(saved["iteration"]) + 1
    plan = GRUSILSamplingPlan(archive.trajectories, eligible_sil_trajectories(archive.trajectories),
                             cfg.recurrent_seq_len, cfg.gru_sil_sampling_mode,
                             step_seconds=cfg.feature_step_seconds)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.gru_sil_seed, iteration, 714]))
    draws = [plan.draw(rng) for _ in range(16)]
    ranked = sorted(range(17), key=lambda i: trajectory_endpoint(archive.trajectories[i]).distance, reverse=True)
    selected = sorted({d[0] for d in draws} | set(ranked[:2]))
    report = {"status": "preparing", "scope": "SIL-only isolated budget probe, no PPO update and no game",
              "device": device, "actual_ppo_updates": 0, "budget_reference_updates": 16,
              "budget_reference_note": "Explicit offline allowance for testing the 16-candidate ceiling. Production still requires actual PPO updates; zero PPO updates normally gives zero SIL budget.",
              "sampling_iteration": iteration, "sampling_seed": cfg.gru_sil_seed,
              "input_hashes": input_hashes, "source_hashes": {str(p.relative_to(ROOT)): sha(p)
                  for p in sorted((ROOT / "src/training").glob("*.py"))},
              "config": asdict(cfg), "checks": {"behavior_matches_checkpoint": True,
                  "exploration_rollout_decisions": buffer.size, "pool_size": 17},
              "planned_draws": [list(d) for d in draws], "arms": {},
              "limitations": ["Fixed elite histories use recorded previous actions; fitting gains do not establish closed-loop jumping.",
                  "The KL guard uses the saved exploration evaluation histories, not a newly sampled post-PPO training rollout.",
                  "This is one preselected sampler stream (next iteration 210), not a multi-seed effectiveness estimate.",
                  "Advantages and returns are frozen at the starting policy for before/after fit comparisons, so critic changes cannot by themselves create apparent actor improvement."]}
    write("report.json", report)
    print(f"Verified checkpoint, behavior, route, {buffer.size} exploration decisions and 17 complete elites", flush=True)
    cache.begin_round(encoder, archive.trajectories, cfg.gru_sil_cache_mb * 1024**2)
    frames, baseline_predictions = {}, {}
    for i in selected:
        t0 = time.perf_counter()
        frames[i] = cache.get(encoder, archive.trajectories[i])
        baseline_predictions[i] = predict(model, frames[i])
        print(f"Prepared elite {i} ({selected.index(i)+1}/{len(selected)}) in {time.perf_counter()-t0:.2f}s", flush=True)
    report["preparation_cache"] = cache.metrics()
    cache.end_round()
    baseline_guard = rollout_distributions(model, buffer)
    initial_named = {k: p.detach().clone() for k, p in model.named_parameters()}
    windows = sorted({(d[0], d[1], d[2]) for d in draws} |
                     {(i, a, b) for i in ranked[:2] for a, b in ((160,192),(192,224),(224,256))})
    report["fixed_windows"] = [{"trajectory": i, "rank": ranked.index(i)+1, "start": a, "end": b,
                                "source_id": ident(archive.trajectories[i]),
                                "before": fit(baseline_predictions[i], baseline_predictions[i],
                                    archive.trajectories[i], cfg, a, b)} for i,a,b in windows]
    report["status"] = "updating"
    write("report.json", report)
    for name, maximum, ratio in (("old_4", 4, .25), ("new_16", 16, 1.)):
        arm_cfg = replace(cfg, gru_sil_max_updates=maximum, gru_sil_budget_ratio=ratio)
        model.config = arm_cfg
        model.load_state_dict(saved["model"])
        trainer = PPOTrainer(arm_cfg)
        trainer.restore_checkpoint_state(copy.deepcopy(saved["trainer_state"]), model)
        trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=arm_cfg.lr, weight_decay=arm_cfg.weight_decay)
        trainer._restore_optimizer(trainer.optimizer, copy.deepcopy(saved["optimizer"]))
        restore_random_state(saved["rng_state"])
        trace = []
        step = trainer.optimizer.step

        def measured_step(*args, **kwargs):
            row = trainer.last_gru_sil_sampling["chunks"][-1]
            ti, a, b = row["trajectory"], row["start"], row["end"]
            traj = archive.trajectories[ti]
            with torch.no_grad():
                dist, value = model.distribution_from_features(sequence_features(model, frames[ti], a, b))
                returns = discounted_returns(traj.step_rewards, traj.terminated, row["bootstrap"], cfg.gamma)
                advantage = (torch.as_tensor(returns[a:b], device=device) - value).clamp(0, cfg.sil_advantage_cap)
                assert int((advantage > 0).sum()) == row["positive_frames"]
                target = torch.atanh(torch.as_tensor(traj.actions[a:b], device=device).div(cfg.action_scale).clamp(-1+1e-6,1-1e-6)).clamp(-cfg.sil_target_clamp,cfg.sil_target_clamp)
                fixed_std = dist.stddev.detach().clone()
                loss_before = float((advantage*.5*((dist.mean-target)/fixed_std).square().sum(-1)).mean())
                positive_steps = (torch.where(advantage>0)[0]+a).cpu().tolist()
            result = step(*args, **kwargs)
            with torch.no_grad():
                dist, _ = model.distribution_from_features(sequence_features(model, frames[ti], a, b))
                loss_after = float((advantage*.5*((dist.mean-target)/fixed_std).square().sum(-1)).mean())
            trace.append({"draw": len(trainer.last_gru_sil_sampling["chunks"]), "trajectory": ti,
                "start": a, "end": b, "positive_steps": positive_steps,
                "actor_loss_before_step_fixed_advantage": loss_before,
                "actor_loss_after_step_fixed_advantage": loss_after})
            print(f"{name}: attempted update {len(trace)}, elite {ti} [{a},{b}), positive={len(positive_steps)}, actor {loss_before:.6f}->{loss_after:.6f}", flush=True)
            return result

        t0 = time.perf_counter()
        with patch.object(trainer.optimizer, "step", side_effect=measured_step), \
             patch.object(PPOTrainer, "update", side_effect=AssertionError("PPO must not run")), \
             patch.object(PPOTrainer, "update_critic_only", side_effect=AssertionError("PPO must not run")):
            metrics = gru_sil_update(trainer, model, archive, encoder, buffer,
                                     budget_updates=16, iteration=iteration)
        elapsed = time.perf_counter()-t0
        record = copy.deepcopy(trainer.last_gru_sil_sampling)
        for item in trace:
            item["accepted"] = record["chunks"][item["draw"]-1]["accepted"]
        for chunk in record["chunks"]:
            chunk["rank"] = ranked.index(chunk["trajectory"])+1
            chunk["source_id"] = ident(archive.trajectories[chunk["trajectory"]])
        kl = conditional_kl(baseline_guard, rollout_distributions(model, buffer))
        assert float(kl.mean()) <= cfg.gru_sil_target_kl and float(kl.max()) <= cfg.gru_sil_max_frame_kl
        assert torch.equal(model.actor_log_std.detach(), saved["model"]["actor_log_std"].to(device))
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert all(torch.isfinite(v).all() for state in trainer.optimizer.state.values()
                   for v in state.values() if torch.is_tensor(v))
        predictions = {i: predict(model, frames[i]) for i in selected}
        fitting = [{"trajectory": i, "start": a, "end": b,
                    **fit(predictions[i], baseline_predictions[i], archive.trajectories[i], cfg, a, b)}
                   for i,a,b in windows]
        accepted = [t for t in trace if t["accepted"]]
        positives = Counter((t["trajectory"], s) for t in accepted for s in t["positive_steps"])
        delta = {k: float((p.detach()-initial_named[k]).abs().max()) for k,p in model.named_parameters()}
        arm = {"metrics": metrics, "warm_update_seconds_including_instrumentation": elapsed,
               "sampling": record, "step_trace": trace, "fit_after": fitting,
               "accepted_positive_presentations": sum(positives.values()),
               "unique_accepted_positive_frames": len(positives),
               "repeated_accepted_positive_presentations": sum(positives.values())-len(positives),
               "immediate_actor_loss_improved_steps": sum(t["actor_loss_after_step_fixed_advantage"] < t["actor_loss_before_step_fixed_advantage"] for t in accepted),
               "key_window_coverage": [{"rank": ranked.index(i)+1, "start": a, "end": b,
                    "unique_positive_frames": sum((i,s) in positives for s in range(a,b)),
                    "positive_presentations": sum(positives[(i,s)] for s in range(a,b))}
                    for i in ranked[:2] for a,b in ((160,192),(192,224),(224,256))],
               "parameter_max_absolute_delta": delta,
               "audited_kl_mean": float(kl.mean()), "audited_kl_max_frame": float(kl.max())}
        (OUT / "checkpoints").mkdir(exist_ok=True)
        torch.save({"probe_only": True, "source_checkpoint": str(ckpt), "source_iteration": 209,
                    "config": asdict(arm_cfg), "model": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "optimizer": trainer.optimizer.state_dict(), "metrics": metrics},
                   OUT / "checkpoints" / f"{name}_after.pt")
        report["arms"][name] = arm
        write("report.json", report)
        print(f"{name}: accepted {metrics['gru_sil_updates']}/{metrics['gru_sil_budget']}, KL={float(kl.mean()):.6f}, max-frame={float(kl.max()):.6f}", flush=True)
    report["total_seconds"] = time.perf_counter()-started
    finalize(report)


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize(json.loads((OUT/"report.json").read_text(encoding="utf-8")))
    else:
        main()

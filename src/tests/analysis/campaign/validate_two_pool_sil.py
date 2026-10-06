"""Offline two-pool sampling/positive-advantage/update preflight on model copies."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/"src"))
from tests.analysis.campaign.prepare_tree_climb import make_encoder
from training.config import TrainConfig
from training.elite_archive import EliteTrajectoryArchive
from training.experimental_policy import create_actor_critic
from training.exploration_windows import configure_precision
from training.gru_sil import (GRUSILSamplingPlan, RecoverySamplingPlan, discounted_returns, eligible_sil_trajectories,
                              gru_sil_update, sequence_predictions)
from training.ppo_buffer import PPORolloutBuffer
from training.ppo_trainer import PPOTrainer
from training.recovery_archive import RecoveryArchive
from training.reference_corridor import ReferenceCorridor
from training.reference_fingerprint import reference_fingerprint
from training.sequence_cache import SILObservationCache
from training.sil_pool_schedule import pool_for_slot


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--elite-pool", type=Path, required=True)
    parser.add_argument("--recovery-pool", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    started = time.perf_counter()
    inputs = {str(p.resolve()):sha(p) for p in (args.checkpoint,args.elite_pool,args.recovery_pool,args.rollout)}
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = TrainConfig(**saved["config"])
    cfg.recovery_archive_enabled = True
    cfg.gru_sil_recovery_fraction = .25
    configure_precision(cfg)
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    fp = reference_fingerprint(cfg, route_meta=route.metadata, schema_id="anchor35", raw_dim=35)
    elite = EliteTrajectoryArchive(cfg, args.output/"unused", fp)
    elite.restore(args.elite_pool)
    recovery = RecoveryArchive(cfg, args.output/"unused", fp, route)
    recovery.restore(args.recovery_pool)
    plan = GRUSILSamplingPlan(elite.trajectories, eligible_sil_trajectories(elite.trajectories),
                             cfg.recurrent_seq_len, cfg.gru_sil_sampling_mode, step_seconds=cfg.feature_step_seconds)
    rplan = RecoverySamplingPlan(recovery, cfg.recurrent_seq_len, cfg.feature_step_seconds)
    assert abs(sum(plan.probabilities().values())-1) < 1e-12
    assert abs(sum(rplan.probabilities().values())-1) < 1e-12
    probability = dict(elite_trajectory_probabilities=plan.trajectory_probabilities,
        recovery_segments=[dict(segment_id=s.descriptor["candidate_id"], start=s.start, end=s.end,
                               conditional_mass=sum(p for (j,*_),p in rplan.probabilities().items() if j==i))
                           for i,s in enumerate(recovery.segments)],
        pool_sequence_16=[pool_for_slot(i,.25) for i in range(16)])
    assert all(abs(s["conditional_mass"]-1/len(recovery.segments))<1e-12 for s in probability["recovery_segments"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_actor_critic(cfg).to(device).eval()
    model.load_state_dict(saved["model"])
    model.exploration_boost = saved.get("loop_state",{}).get("boost",0.)
    trainer = PPOTrainer(cfg)
    trainer.restore_checkpoint_state(copy.deepcopy(saved["trainer_state"]), model)
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    trainer._restore_optimizer(trainer.optimizer, copy.deepcopy(saved["optimizer"]))
    buffer = PPORolloutBuffer.load(args.rollout)
    if getattr(buffer,"collection_metadata",{}).get("evaluation_only"):
        raise ValueError("KL guard must use an archived training rollout, not a probe")
    encoder, _ = make_encoder(cfg, route)
    cache = encoder._gru_sil_observation_cache = SILObservationCache()
    cache.begin_round(encoder, [*elite.trajectories,*recovery.trajectories], cfg.gru_sil_cache_mb*1024**2)
    diagnostics = []
    for i,segment in enumerate(recovery.segments):
        traj = segment.trajectory
        _,_,v = sequence_predictions(model,cache.get(encoder,traj))
        bootstrap = 0. if traj.terminated[-1] else float(v[-1])
        returns = discounted_returns(traj.step_rewards,traj.terminated,bootstrap,cfg.gamma)
        advantage = np.maximum(returns-v[:-1].cpu().numpy(),0)
        weights = np.minimum(advantage,cfg.sil_advantage_cap)
        a,b = segment.start,segment.end
        diagnostics.append(dict(index=i,segment_id=segment.descriptor["candidate_id"],start=a,end=b,
            positive_frames=int((advantage[a:b]>0).sum()),frames=b-a,advantage_sum=float(weights[a:b].sum()),
            bootstrap=bootstrap,chunks=[dict(start=s,end=e,positive_frames=int((advantage[s:e]>0).sum()),probability=p)
                                       for (j,s,e,_),p in rplan.probabilities().items() if j==i]))
        print(f"Recovery {i+1}: positive {diagnostics[-1]['positive_frames']}/{b-a} frames",flush=True)
    cache.end_round()
    report = dict(status="running",scope="Two isolated SIL rounds on copies of the final model/optimizer; no game or PPO update",
        config=asdict(cfg),inputs=inputs,probabilities=probability,recovery_initial_advantage=diagnostics,rounds=[],
        budget_note="16 candidate capacity probe per round, not a claim of new accepted PPO minibatches",
        limitations=["Acceptance is not proof of better closed-loop recovery.",
                     "Original episode rewards, bootstrap and positive-advantage gate are unchanged.",
                     "Imported frozen recovery probe episodes are now demonstrations, not held-out evaluation data."])
    def write():
        (args.output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    write()
    for offset in (1,2):
        before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        metrics = gru_sil_update(trainer,model,elite,encoder,buffer,budget_updates=16,
                                iteration=int(saved["iteration"])+offset,recovery_archive=recovery)
        record = copy.deepcopy(trainer.last_gru_sil_sampling)
        diffs = {k:float((v.detach().cpu().to(torch.float64)-before[k].to(torch.float64)).abs().max())
                 for k,v in model.state_dict().items()}
        item = dict(metrics=metrics,sampling=record,changed_tensors=sum(v>0 for v in diffs.values()),
                    max_parameter_change=max(diffs.values()))
        report["rounds"].append(item)
        print(f"Round {offset}: elite accepted {metrics['gru_sil_elite_updates']}, recovery accepted {metrics['gru_sil_recovery_updates']}; candidates E={metrics['gru_sil_elite_candidates']} R={metrics['gru_sil_recovery_candidates']}",flush=True)
        assert finite(model.state_dict()) and finite(trainer.optimizer.state_dict())
        assert metrics["gru_sil_elite_candidates"]+metrics["gru_sil_recovery_candidates"] == len(record["chunks"])
        assert len(record["chunks"]) <= metrics["gru_sil_budget"] <= 16
        state = copy.deepcopy(trainer.checkpoint_state())
        resumed = PPOTrainer(cfg)
        resumed.restore_checkpoint_state(state,model)
        assert resumed.gru_sil_pool_cursor==trainer.gru_sil_pool_cursor
        write()
    assert all(sha(p)==value for p,value in inputs.items())
    report.update(status="completed",finite_model_optimizer=True,formal_inputs_unchanged=True,
                  elapsed_seconds=time.perf_counter()-started,quota_cursor=trainer.gru_sil_pool_cursor)
    write()


if __name__ == "__main__":
    main()

"""Replay archived inputs to isolate first-step drift and learning-rate sensitivity."""
from __future__ import annotations

import copy
from dataclasses import asdict
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from training.config import TrainConfig
from training.decision_buffer import DecisionRolloutBuffer, evaluated_batches
from training.experimental_policy import create_actor_critic
from training.policy_audit import audit_policy, policy_replay_metrics
from training.ppo_trainer import PPOTrainer
from tests.analysis.benchmark_player_throughput import CHECKPOINT, write_json

OUTPUT = ROOT / "runs/pipeline_20260920"


def main():
    torch.set_num_threads(4)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    buffer = DecisionRolloutBuffer.load(ROOT / "runs/player_throughput_20260919/n02_start/checkpoints/rollout_00.npz")
    rows = []
    def model_for(lr):
        cfg = TrainConfig(**saved["config"])
        cfg.lr, cfg.ppo_batch_size = lr, 128
        model = create_actor_critic(cfg).cuda().eval()
        model.load_state_dict(saved["model"])
        return cfg, model

    # Same first minibatch and fresh AdamW for each controlled loss ablation.
    for component in ("joint", "policy", "value_shared", "value_head"):
        cfg, model = model_for(1e-4)
        audit = audit_policy(model, buffer)
        if component == "value_head":
            for name, p in model.named_parameters():
                p.requires_grad_(name.startswith("critic_head."))
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=cfg.lr, weight_decay=cfg.weight_decay)
        np.random.seed(77)
        batch, lp, entropy, value = next(evaluated_batches(model, buffer, 128))
        ratio = (lp-batch.old_log_probs).exp()
        policy_loss = -torch.min(ratio*batch.advantages, ratio.clamp(1-cfg.clip_epsilon, 1+cfg.clip_epsilon)*batch.advantages).mean()
        clipped = batch.old_values+(value-batch.old_values).clamp(-cfg.clip_epsilon, cfg.clip_epsilon)
        value_loss = .5*torch.maximum((value-batch.returns).square(), (clipped-batch.returns).square()).mean()
        loss = (policy_loss-cfg.ent_coef*entropy.mean() if component in ("joint", "policy") else 0)
        if component in ("joint", "value_shared", "value_head"):
            loss = loss+cfg.vf_coef*value_loss
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        np.random.seed(99)
        drift = policy_replay_metrics(model, buffer, max_samples=512)
        rows.append({"kind":"single_step", "component":component, "lr":cfg.lr,
                     "policy_loss":float(policy_loss), "value_loss":float(value_loss),
                     "gradient_norm":float(grad), "std":model.actor_log_std.detach().exp().cpu().tolist(),
                     "conditional_std":(model.actor_log_std.detach().clamp(cfg.ppo_log_std_min,cfg.ppo_log_std_max).exp()*np.sqrt(1-cfg.ou_phi**2)).cpu().tolist(),
                     "audit":audit, "after_step":drift})
        print(json.dumps(rows[-1]), flush=True)
        del loss, lp, value, entropy, ratio, batch, policy_loss, value_loss, clipped, model, optimizer
        gc.collect(); torch.cuda.empty_cache()
    for lr in (1e-4, 3e-5, 1e-5):
        for seed in (77, 99, 123):
            cfg, model = model_for(lr)
            np.random.seed(seed)
            trainer = PPOTrainer(cfg)
            start = time.perf_counter()
            metrics = trainer.update(model, buffer)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            np.random.seed(901)
            drift = policy_replay_metrics(model, buffer, max_samples=512)
            rows.append({"kind":"full_update", "lr":lr,"seed":seed,"seconds":seconds,
                         "metrics":metrics,"after_step":drift,"trace":trainer.last_update_trace})
            write_json(OUTPUT / "kl_diagnosis.json", {"checkpoint":str(CHECKPOINT), "rows":rows,
                       "note":"fresh optimizer, fixed archived rollout; diagnostic updates discarded"})
            print(json.dumps({k:rows[-1][k] for k in ("lr","seed","seconds","metrics","after_step")}), flush=True)
            del model, trainer
            gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Conservative recurrent self-imitation over complete elite episodes.

This is positive-return-weighted *mean imitation*, not an off-policy PPO
likelihood: elite actions include OU noise, whose latent history is not saved.
We freeze std, clamp inverse-tanh targets, and constrain cumulative drift from
the post-PPO policy on every frame of the current rollout.
"""
import copy
import math

import numpy as np
import torch
from torch.nn import functional as F

from .decision_buffer import DecisionRolloutBuffer
from .elite_archive import EliteTrajectoryArchive
from .policy_features import feature_layout
from .reference_corridor import PROGRESS_MODE_V2, trajectory_endpoint

_INPUTS = ("dynamics", "patches", "act_history")
_SPARSE = ("contour_tokens", "contour_mask", "relation_tokens", "relation_mask")


def validate_gru_sil(config):
    if (config.policy_backbone != "gru" or config.action_segment_steps != 1
            or feature_layout(config)[1] or config.contact_heads_enabled):
        raise ValueError("GRU SIL requires the base GRU, single head and one action per frame")
    if not config.reference_trajectory or config.reference_progress_mode != PROGRESS_MODE_V2:
        raise ValueError("GRU SIL requires confirmed_endpoint_v2 reference scoring")
    if config.sil_enabled:
        raise ValueError("GRU SIL cannot run alongside legacy window SIL")
    values = [config.gru_sil_budget_ratio, config.gru_sil_loss_coef, config.gru_sil_value_coef,
              config.gru_sil_target_kl, config.gru_sil_max_frame_kl, config.sil_advantage_cap,
              config.sil_target_clamp]
    if (not np.isfinite(values).all() or min(values) <= 0
            or config.gru_sil_budget_ratio > 1 or config.gru_sil_max_updates < 1
            or config.recurrent_seq_len < 1 or config.gru_sil_seed < 0):
        raise ValueError("Invalid GRU SIL budget, loss, KL or sampling configuration")


def discounted_returns(rewards, terminated, bootstrap, gamma):
    """Use the whole tail; a time limit bootstraps, an actual terminal does not."""
    result = np.empty(len(rewards), np.float32)
    following = float(bootstrap)
    for t in range(len(rewards) - 1, -1, -1):
        following = float(rewards[t]) + gamma * following * (1 - bool(terminated[t]))
        result[t] = following
    return result


def _step(model, frames, start, end, hidden):
    device = next(model.parameters()).device
    def tensor(name):
        return torch.as_tensor(frames[name][start:end], dtype=torch.float32, device=device)[None]
    return model.recurrent_features(
        *(tensor(name) for name in _INPUTS), hidden=hidden,
        **{name: tensor(name) for name in _SPARSE if name in frames})


def sequence_features(model, frames, start, end):
    """Fresh current-weight burn-in from reset, detached before the training chunk."""
    hidden = None
    with torch.no_grad():
        for first in range(0, start, 128):
            _, hidden = _step(model, frames, first, min(first + 128, start), hidden)
    output, _ = _step(model, frames, start, end, hidden)
    return output[0]


@torch.no_grad()
def sequence_predictions(model, frames):
    """Memory-bounded full recurrent replay, including the bootstrap state."""
    hidden, means, stds, values = None, [], [], []
    for start in range(0, len(frames["dynamics"]), 128):
        output, hidden = _step(model, frames, start, start + 128, hidden)
        dist, value = model.distribution_from_features(output[0])
        means.append(dist.mean)
        stds.append(dist.stddev)
        values.append(value)
    return torch.cat(means), torch.cat(stds), torch.cat(values)


@torch.no_grad()
def rollout_distributions(model, buffer):
    """OU-conditional distributions at fixed observed histories/noise histories.

    All current rollout decisions are covered, including low-noise phases.
    This is a sampled-history trust region, not a guarantee on unseen states.
    """
    means, stds = [], []
    phi = float(buffer.policy_context["noise_phi"])
    for episode in buffer.episodes:
        mean, std, _ = sequence_predictions(model, episode)
        steps = torch.as_tensor(episode["decision_steps"], device=mean.device)
        mean, std = mean[steps], std[steps]
        scale = torch.as_tensor(episode["noise_scale"], device=mean.device)[:, None]
        previous = torch.as_tensor(episode["noise_prev"], device=mean.device)
        std = model._scaled_std(std, scale)
        conditional_mean, conditional_std = model._ou_conditional(mean, std, previous, phi)
        means.append(conditional_mean)
        stds.append(conditional_std)
    return torch.cat(means), torch.cat(stds)


def conditional_kl(baseline, current):
    mean0, std0 = baseline
    mean1, std1 = current
    return (torch.log(std1 / std0) + (std0.square() + (mean0 - mean1).square())
            / (2 * std1.square()) - 0.5).sum(-1).clamp_min(0)


def gru_sil_update(trainer, model, archive, encoder, buffer, *, budget_updates, iteration):
    cfg = trainer.config
    validate_gru_sil(cfg)
    if not isinstance(buffer, DecisionRolloutBuffer) or not buffer.size:
        raise ValueError("GRU SIL needs a nonempty current decision rollout for its KL guard")
    if buffer.policy_context.get("deterministic"):
        raise ValueError("GRU SIL cannot follow a deterministic evaluation rollout")
    limit = min(cfg.gru_sil_max_updates, int(math.floor(max(0, budget_updates) * cfg.gru_sil_budget_ratio)))
    metrics = {"gru_sil_updates": 0, "gru_sil_budget": limit, "gru_sil_attempts": 0,
               "gru_sil_positive_frames": 0, "gru_sil_sampled_frames": 0,
               "gru_sil_rejected_steps": 0, "gru_sil_kl": 0., "gru_sil_frame_kl_max": 0.,
               "gru_sil_attempted_kl_max": 0., "gru_sil_loss": 0.}
    record = {"version": 1, "mode": "gru_full_prefix_positive_advantage", "iteration": iteration,
              "seed": cfg.gru_sil_seed, "budget": limit, "chunks": []}
    trainer.last_gru_sil_sampling = record
    candidates = [i for i, t in enumerate(archive.trajectories)
                  if EliteTrajectoryArchive._complete(t) and t.trajectory_valid
                  and trajectory_endpoint(t).valid and trajectory_endpoint(t).cap >= 0]
    if not limit or not candidates:
        return metrics
    if trainer.optimizer is None:
        raise ValueError("GRU SIL must run after PPO initializes/restores its optimizer")
    model.eval()
    optimizer = trainer.optimizer
    baseline = rollout_distributions(model, buffer)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.gru_sil_seed, iteration, 714]))
    cache = {}  # Only this call's selected trajectories; no stale neural embeddings/values.
    for _ in range(limit):
        ti = int(rng.choice(candidates))
        traj = archive.trajectories[ti]
        cap = min(trajectory_endpoint(traj).cap + 1, len(traj.actions))
        length = cfg.recurrent_seq_len
        start = int(rng.integers(math.ceil(cap / length))) * length
        end = min(start + length, cap)
        if ti not in cache:
            cache[ti] = encoder.encode_sequence(traj)
        frames = cache[ti]
        # Bootstrap is rebuilt with current weights, just like the prefix.
        _, _, all_values = sequence_predictions(model, frames)
        bootstrap = 0. if traj.terminated[-1] else float(all_values[-1])
        returns = discounted_returns(traj.step_rewards, traj.terminated, bootstrap, cfg.gamma)
        features = sequence_features(model, frames, start, end)
        dist, values = model.distribution_from_features(features)
        target = torch.as_tensor(returns[start:end], device=values.device)
        advantage = (target - values.detach()).clamp(min=0, max=cfg.sil_advantage_cap)
        positive = advantage > 0
        count = int(positive.sum())
        row = {"trajectory": ti, "source_iteration": traj.iteration, "start": start, "end": end,
               "cap": cap, "bootstrap": bootstrap, "positive_frames": count, "accepted": False}
        record["chunks"].append(row)
        metrics["gru_sil_sampled_frames"] += end - start
        metrics["gru_sil_positive_frames"] += count
        if not count:
            continue
        action = torch.as_tensor(traj.actions[start:end], device=values.device)
        raw = torch.atanh((action / cfg.action_scale).clamp(-1 + 1e-6, 1 - 1e-6))
        raw = raw.clamp(-cfg.sil_target_clamp, cfg.sil_target_clamp)
        per_frame = .5 * ((dist.mean - raw) / dist.stddev.detach()).square().sum(-1)
        actor_loss = (advantage * per_frame).mean()
        # Positive-only value correction; no forced imitation when the critic says no.
        value_loss = (F.smooth_l1_loss(values, target, reduction="none") * positive).mean()
        loss = cfg.gru_sil_loss_coef * actor_loss + cfg.gru_sil_value_coef * value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite GRU SIL loss")
        weights_before = copy.deepcopy(model.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), trainer._grad_norm_adapter.max_grad_norm)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite GRU SIL gradient")
        if not metrics["gru_sil_updates"]:
            # Preserve second moments, but do not let PPO's momentum steer SIL.
            for p in model.parameters():
                if p.grad is not None and "exp_avg" in optimizer.state.get(p, {}):
                    optimizer.state[p]["exp_avg"].zero_()
        try:
            optimizer.step()
            metrics["gru_sil_attempts"] += 1
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite parameters after GRU SIL step")
            with torch.no_grad():
                kl = conditional_kl(baseline, rollout_distributions(model, buffer))
            if not torch.isfinite(kl).all():
                raise FloatingPointError("Non-finite KL after GRU SIL step")
        except BaseException:
            # A failed audit or interruption must not leave an unvalidated update.
            model.load_state_dict(weights_before)
            optimizer.load_state_dict(optimizer_before)
            optimizer.zero_grad(set_to_none=True)
            raise
        mean_kl, max_kl = float(kl.mean()), float(kl.max())
        row.update(kl=mean_kl, frame_kl_max=max_kl, loss=float(loss.detach()))
        metrics["gru_sil_attempted_kl_max"] = max(metrics["gru_sil_attempted_kl_max"], mean_kl)
        if mean_kl > cfg.gru_sil_target_kl or max_kl > cfg.gru_sil_max_frame_kl:
            model.load_state_dict(weights_before)
            optimizer.load_state_dict(optimizer_before)
            optimizer.zero_grad(set_to_none=True)
            metrics["gru_sil_rejected_steps"] += 1
            break
        row["accepted"] = True
        metrics["gru_sil_updates"] += 1
        metrics["gru_sil_loss"] += float(loss.detach())
        metrics["gru_sil_kl"], metrics["gru_sil_frame_kl_max"] = mean_kl, max_kl
        optimizer.zero_grad(set_to_none=True)
    metrics["gru_sil_loss"] /= max(1, metrics["gru_sil_updates"])
    return metrics

"""Behavior-policy contract and frozen-weight replay diagnostics for PPO."""
from __future__ import annotations

import numpy as np
import torch


def policy_context(model, noise_phi: float, deterministic: bool = False) -> dict:
    context = {
        "log_prob_version": 2, "noise_phi": float(noise_phi),
        "exploration_boost": float(model.exploration_boost),
        "action_scale": float(model.config.action_scale),
        "log_std_min": float(model.config.ppo_log_std_min),
        "log_std_max": float(model.config.ppo_log_std_max),
        "deterministic": deterministic,
    }
    if getattr(model.config, "delayed_exploration_enabled", False):
        from .delayed_exploration import OPEN_STEPS
        context["delayed_exploration"] = {
            "floor": float(model.config.delayed_exploration_floor),
            "ramp_steps": int(model.config.delayed_exploration_ramp_steps),
            "open_steps": list(OPEN_STEPS),
        }
    from .policy_features import experimental_enabled
    if experimental_enabled(model.config):
        context["experiment"] = {"version": 1, **{name: getattr(model.config, name) for name in (
            "policy_backbone", "relation_features", "event_features", "event_branch", "geometry_features",
            "trend_features", "event_source", "normalize_action_history", "feature_position_scale",
            "feature_velocity_scale", "feature_angular_scale", "feature_clip", "feature_step_seconds",
            "geometry_query_radius", "action_segment_steps", "segment_interrupt_contacts")}}
    return context


@torch.no_grad()
def policy_replay_metrics(model, buffer, batch_size=64, max_samples=None):
    """Measure drift on recorded actions without treating a new policy as behavior.

    Sampling settings and raw/action consistency are still checked. This function
    alone does NOT establish that the archived behavior probabilities are correct.
    """
    if not buffer.size or not buffer.policy_context:
        raise ValueError("PPO audit requires samples and a behavior-policy context")
    context = buffer.policy_context
    if context != policy_context(model, context["noise_phi"], context["deterministic"]):
        raise ValueError("Behavior-policy settings changed after collection")
    model.eval()
    buffer.device = next(model.parameters()).device
    errors, kls, saturated = [], [], []
    count = 0
    from .decision_buffer import evaluated_batches
    # Diagnostics must neither draw from nor reseed the training shuffle/OU RNG.
    for batch, lp, _, _ in evaluated_batches(model, buffer, batch_size, normalize_advantages=False,
                                            rng=np.random.default_rng(0)):
        log_ratio = lp - batch.old_log_probs
        errors.extend(torch.expm1(log_ratio).abs().cpu().tolist())
        kls.extend((torch.expm1(log_ratio) - log_ratio).cpu().tolist())
        saturated.extend((batch.actions.abs() >= .999 * model.config.action_scale)
                         .any(dim=-1).float().cpu().tolist())
        if not torch.allclose(model._squash_action(batch.raw_actions), batch.actions,
                              atol=1e-5, rtol=1e-6):
            raise ValueError("Stored actions/segment parameters do not match stored raw samples")
        count += len(lp)
        if max_samples is not None and count >= max_samples:
            break
    metrics = {
        "preupdate_ratio_max_error": float(np.max(errors)),
        "preupdate_kl": float(np.mean(kls)),
        "action_saturation_fraction": float(np.mean(saturated)),
        "audit_samples": float(count),
    }
    if not np.isfinite(errors).all() or not np.isfinite(kls).all():
        raise ValueError(f"Non-finite policy replay: {metrics}")
    return metrics


def audit_policy(model, buffer, batch_size=64, max_samples=None, tolerance=1e-3):
    """Replay with the frozen behavior weights; reject mismatched probabilities.

    The 0.1% tolerance covers float32 CUDA kernel rounding, not policy lag.
    Acceptance tests use all samples; ordinary synchronous updates check 128.
    """
    metrics = policy_replay_metrics(model, buffer, batch_size, max_samples)
    if metrics["preupdate_ratio_max_error"] > tolerance:
        raise ValueError(f"Frozen behavior-policy replay failed: {metrics}")
    return metrics

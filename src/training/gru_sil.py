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
from .climb_diagnostics import (PhaseRules, classify_motion, phase_counts, launch_event_windows,
                                contact_dynamics_importance, DYNAMICS_SPEC)
from .sequence_cache import SILObservationCache
from .sil_pool_schedule import pool_for_slot

_INPUTS = ("dynamics", "patches", "act_history")
_SPARSE = ("contour_tokens", "contour_mask", "relation_tokens", "relation_mask")
EVENT_SAMPLING_MODE = "phase_event_mixture_v2"
DYNAMICS_SAMPLING_MODE = "quality_dynamics_v3"
SAMPLING_PHASE_PROBABILITY = {"trajectory_uniform": 0., "phase_balanced_v1": 1.,
                              "phase_mixture_v1": .5, EVENT_SAMPLING_MODE: .5, DYNAMICS_SAMPLING_MODE: 0.}
DYNAMICS_SAMPLING_SPEC = {"version": DYNAMICS_SAMPLING_MODE, "trajectory_uniform_fraction": .1,
                         "distance_rank_temperature_fraction": .25, "chunk_uniform_fraction": .1,
                         "no_dynamics_fallback": "uniform_chunks_in_selected_trajectory"}
EVENT_SAMPLING_SPEC = {"version": EVENT_SAMPLING_MODE, "base_probability": .5,
                       "target_soft_phase_fraction": .5, "target_event_fraction": .5,
                       "max_probability_ratio": 4., "support_lookback_states": 64,
                       "event_normalization": "uniform_approach_or_tree_group_then_event_then_overlap",
                       "no_event_fallback": "trajectory_uniform"}


def bounded_target_distribution(target, base):
    """Cap target at 7*base, so the final 50/50 mixture lies in [.5, 4]*base.

    Redistribute removed mass proportionally to remaining base-weighted capacity.
    This is a probability cap, not an additional multiplier on the SIL loss.
    """
    ceiling = (2 * EVENT_SAMPLING_SPEC["max_probability_ratio"] - 1) * base
    result = np.minimum(target, ceiling)
    missing = 1. - result.sum()
    if missing > 0:
        slack = ceiling - result
        result += missing * slack / slack.sum()
    return result


def validate_gru_sil(config):
    if config.recovery_archive_enabled:
        pool_for_slot(0, config.gru_sil_recovery_fraction)
        if config.gru_sil_sampling_mode != DYNAMICS_SAMPLING_MODE:
            raise ValueError("Two-pool SIL requires quality_dynamics_v3")
    if (not isinstance(config.gru_sil_max_updates, int) or isinstance(config.gru_sil_max_updates, bool)
            or config.gru_sil_max_updates < 1):
        raise ValueError("GRU SIL maximum updates must be a positive integer")
    if (not isinstance(config.gru_sil_cache_mb, int) or isinstance(config.gru_sil_cache_mb, bool)
            or config.gru_sil_cache_mb < 0):
        raise ValueError("GRU SIL cache size must be a nonnegative integer MiB value")
    if config.gru_sil_sampling_mode not in SAMPLING_PHASE_PROBABILITY:
        raise ValueError("Unknown GRU SIL sampling mode")
    if config.gru_sil_phase_rules is not None:
        PhaseRules(**config.gru_sil_phase_rules)
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
        data = frames[name][start:end]
        # Cached NumPy arrays are read-only. Copy instead of exposing a writable
        # CPU tensor alias into the cache (also avoids PyTorch's read-only warning).
        convert = torch.tensor if not data.flags.writeable else torch.as_tensor
        return convert(data, dtype=torch.float32, device=device)[None]
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
        from .exploration_windows import zero_noise_enabled
        std = model._scaled_std(std, scale, allow_zero=zero_noise_enabled(model.config))
        conditional_mean, conditional_std = model._ou_conditional(mean, std, previous, phi)
        means.append(conditional_mean)
        stds.append(conditional_std)
    return torch.cat(means), torch.cat(stds)


def conditional_kl(baseline, current):
    mean0, std0 = baseline
    mean1, std1 = current
    # Dirac actions have no finite Gaussian KL. Their drift is diagnostic only.
    active = (std0 > 0).all(-1)
    if not torch.equal(active, (std1 > 0).all(-1)):
        raise ValueError("Noise regime changed during policy update")
    mean0, std0, mean1, std1 = mean0[active], std0[active], mean1[active], std1[active]
    return (torch.log(std1 / std0) + (std0.square() + (mean0 - mean1).square())
            / (2 * std1.square()) - 0.5).sum(-1).clamp_min(0)


@torch.no_grad()
def deterministic_action_drift(model, buffer, current=None):
    """Actual mouse-action drift on archived deterministic histories, not a KL."""
    from .decision_buffer import has_zero_noise
    if not has_zero_noise(buffer):
        return {"deterministic_action_max_delta": 0., "deterministic_action_rms_delta": 0.}
    means, _ = rollout_distributions(model, buffer) if current is None else current
    mask = torch.as_tensor(np.concatenate([e["noise_scale"] == 0 for e in buffer.episodes]), device=means.device)
    targets = torch.as_tensor(np.concatenate([e["actions"] for e in buffer.episodes]), device=means.device)
    delta = (model._squash_action(means[mask]) - targets[mask]).norm(dim=-1)
    if not torch.isfinite(delta).all():
        raise FloatingPointError("Non-finite deterministic action drift")
    return {"deterministic_action_max_delta": float(delta.max()),
            "deterministic_action_rms_delta": float(delta.square().mean().sqrt())}


def phase_chunk_groups(trajectories, candidates, length, rules):
    """Uniform phase, then uniform chunk. Labels use raw context; no cached neural states."""
    groups = {}
    for ti in candidates:
        traj = trajectories[ti]
        labels, _ = classify_motion(traj.raw_states, traj.contact, rules)
        cap = min(trajectory_endpoint(traj).cap + 1, len(traj.actions))
        for start in range(0, cap, length):
            end = min(start + length, cap)
            counts = phase_counts(labels[start:end])
            phase = min(counts, key=lambda name: (-counts[name], name))
            groups.setdefault(phase, []).append((ti, start, end, cap))
    return groups


def eligible_sil_trajectories(trajectories):
    return [i for i, t in enumerate(trajectories)
            if EliteTrajectoryArchive._complete(t) and t.trajectory_valid
            and trajectory_endpoint(t).valid and trajectory_endpoint(t).cap >= 0]


class GRUSILSamplingPlan:
    """One sampling implementation for training, exact probabilities and offline plots.

    The v1/v2 mixture selects its branch with probability 1/2, not a per-round quota.
    The v3 distribution separates trajectory quality from local dynamics priority.
    Chunks remain contiguous and never cross the confirmed cap. The legacy mode
    deliberately keeps the same two RNG draws as before this sampler was added.
    """
    def __init__(self, trajectories, candidates, length, mode, rules=None, step_seconds=1/120):
        if mode not in SAMPLING_PHASE_PROBABILITY or type(length) is not int or length < 1:
            raise ValueError("Invalid GRU SIL sampling plan")
        self.mode, self.length = mode, length
        self.phase_probability = SAMPLING_PHASE_PROBABILITY[mode]
        self.candidates = list(candidates)
        if not self.candidates or len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Sampling plan requires unique eligible trajectories")
        self.caps = {i: min(trajectory_endpoint(trajectories[i]).cap + 1, len(trajectories[i].actions))
                     for i in self.candidates}
        if min(self.caps.values()) < 1:
            raise ValueError("Sampling plan contains an empty prefix")
        rules = rules or PhaseRules()
        self.groups = (phase_chunk_groups(trajectories, candidates, length, rules)
                       if self.phase_probability and mode != EVENT_SAMPLING_MODE else {})
        self.phase_names = sorted(self.groups)
        self.events, self.chunk_details = {}, {}
        self.dynamics = {}
        if mode == EVENT_SAMPLING_MODE:
            self._build_event_distribution(trajectories, rules)
        elif mode == DYNAMICS_SAMPLING_MODE:
            self._build_dynamics_distribution(trajectories, step_seconds)

    def _build_dynamics_distribution(self, trajectories, step_seconds):
        spec = DYNAMICS_SAMPLING_SPEC
        distances = np.maximum([trajectory_endpoint(trajectories[i]).distance for i in self.candidates], 0.)
        # Average ranks preserve equal treatment of tied distances. Soft ranking
        # emphasizes far-reaching demonstrations without letting one D outlier
        # consume most of the pool, or rewarding a short time-to-failure.
        rank = np.array([(distances > d).sum()+.5*((distances == d).sum()-1) for d in distances])
        quality = np.exp(-rank/max(1., len(rank)*spec["distance_rank_temperature_fraction"]))
        quality /= quality.sum()
        traj_p = spec["trajectory_uniform_fraction"]/len(quality) + (1-spec["trajectory_uniform_fraction"])*quality
        self.trajectory_probabilities = dict(zip(self.candidates, traj_p.tolist()))
        self.chunks, marginal = [], []
        for i, cap in self.caps.items():
            traj = trajectories[i]
            diag = contact_dynamics_importance(traj.raw_states, traj.contact, cap, step_seconds)
            self.dynamics[i] = diag
            chunks = [(i, a, min(a+self.length, cap), cap) for a in range(0, cap, self.length)]
            salience = np.array([diag["priority"][a:b].sum() for _,a,b,_ in chunks])
            targeted = salience/salience.sum() if salience.sum() else np.full(len(chunks), 1/len(chunks))
            conditional = spec["chunk_uniform_fraction"]/len(chunks) + (1-spec["chunk_uniform_fraction"])*targeted
            for chunk, local_p, value in zip(chunks, conditional, salience):
                _, a, b, _ = chunk
                p = self.trajectory_probabilities[i] * local_p
                self.chunks.append(chunk); marginal.append(p)
                self.chunk_details[chunk] = {
                    "trajectory_probability": self.trajectory_probabilities[i],
                    "conditional_chunk_probability": float(local_p), "probability": float(p),
                    "priority_mass": float(value), "core_frames": int(diag["core"][a:b].sum()),
                    "free_flight_frames": int(diag["free_flight"][a:b].sum()),
                    "context_frames": int(((diag["priority"][a:b]>0)&~diag["core"][a:b]).sum())}
        self.marginal = np.asarray(marginal)
        self.distribution_summary = dict(spec, dynamics_spec=DYNAMICS_SPEC, step_seconds=step_seconds,
                                         trajectory_probabilities=self.trajectory_probabilities)

    def _build_event_distribution(self, trajectories, rules):
        self.chunks = [(i, a, min(a+self.length, cap), cap)
                       for i, cap in self.caps.items() for a in range(0, cap, self.length)]
        base = np.array([1. / len(self.candidates) / math.ceil(cap/self.length)
                         for i, a, b, cap in self.chunks])
        labels = {i: classify_motion(trajectories[i].raw_states, trajectories[i].contact, rules)[0]
                  for i in self.candidates}
        memberships = [phase_counts(labels[i][a:b]) for i, a, b, cap in self.chunks]
        names = sorted({k for row in memberships for k in row})
        # Every phase present in a chunk contributes; no majority label or tie breaker.
        soft = np.array([[row.get(k, 0)/(b-a) for k in names]
                         for row, (_, a, b, _) in zip(memberships, self.chunks)])
        phase = (soft / soft.sum(axis=0)).mean(axis=1)
        self.events = {i: launch_event_windows(trajectories[i].raw_states, trajectories[i].contact, cap, rules)
                       for i, cap in self.caps.items()}
        with_events = [i for i in self.candidates if self.events[i]]
        n_events = sum(map(len, self.events.values()))
        event_groups = {}
        for windows in self.events.values():
            for window in windows:
                name = window["event_group"]
                event_groups[name] = event_groups.get(name, 0) + 1
        event = np.zeros(len(self.chunks))
        if with_events:
            for j, (i, a, b, cap) in enumerate(self.chunks):
                for window in self.events[i]:
                    overlap = max(0, min(b, window["end"]) - max(a, window["start"]))
                    event[j] += (overlap / (window["end"]-window["start"])
                                 / len(event_groups) / event_groups[window["event_group"]])
        else:
            event = base.copy()
        raw_target = .5 * phase + .5 * event
        self.target = bounded_target_distribution(raw_target, base)
        self.marginal = .5 * base + .5 * self.target
        self.distribution_summary = dict(EVENT_SAMPLING_SPEC,
            event_count=n_events, event_trajectories=len(with_events), event_groups=event_groups,
            clipped_chunks=int(np.count_nonzero(raw_target > 7*base)),
            redistributed_target_mass=float(np.maximum(raw_target-7*base, 0).sum()))
        for j, chunk in enumerate(self.chunks):
            i, a, b, cap = chunk
            self.chunk_details[chunk] = {
                "base_probability": float(base[j]), "soft_phase_probability": float(phase[j]),
                "event_probability": float(event[j]), "raw_target_probability": float(raw_target[j]),
                "target_probability": float(self.target[j]), "probability": float(self.marginal[j]),
                "probability_ratio": float(self.marginal[j]/base[j]),
                "event_overlaps": [dict(window, overlap_steps=max(0, min(b, window["end"])-max(a, window["start"])))
                                   for window in self.events[i] if a < window["end"] and b > window["start"]]}

    def draw(self, rng):
        if self.mode == DYNAMICS_SAMPLING_MODE:
            chunk = self.chunks[int(rng.choice(len(self.chunks), p=self.marginal))]
            return (*chunk, None, "quality_dynamics")
        choose_phase = (self.phase_probability == 1 or
                        (self.phase_probability > 0 and rng.random() < self.phase_probability))
        if choose_phase:
            if self.mode == EVENT_SAMPLING_MODE:
                chunk = self.chunks[int(rng.choice(len(self.chunks), p=self.target))]
                return (*chunk, None, "phase_event_weighted")
            phase = str(rng.choice(self.phase_names))
            group = self.groups[phase]
            chunk = group[int(rng.integers(len(group)))]
            return (*chunk, phase, "phase_balanced")
        ti = int(rng.choice(self.candidates))
        cap = self.caps[ti]
        start = int(rng.integers(math.ceil(cap / self.length))) * self.length
        return ti, start, min(start + self.length, cap), cap, None, "trajectory_uniform"

    def probabilities(self):
        """Marginal probability of each chunk before advantage/KL filtering."""
        if self.mode in (EVENT_SAMPLING_MODE, DYNAMICS_SAMPLING_MODE):
            return dict(zip(self.chunks, self.marginal.tolist()))
        result = {}
        for ti, cap in self.caps.items():
            chunks = math.ceil(cap / self.length)
            for start in range(0, cap, self.length):
                result[(ti, start, min(start + self.length, cap), cap)] = (
                    (1 - self.phase_probability) / len(self.candidates) / chunks)
        for group in self.groups.values():
            for chunk in group:
                result[chunk] += self.phase_probability / len(self.groups) / len(group)
        return result


class RecoverySamplingPlan:
    """Uniform total mass per retained segment, dynamics weighting within it."""
    def __init__(self, archive, length, step_seconds):
        if type(length) is not int or length < 1 or not archive.segments:
            raise ValueError("Recovery sampling requires nonempty segments and a positive chunk length")
        self.chunks, marginal, self.chunk_details = [], [], {}
        for i, segment in enumerate(archive.segments):
            traj, start, end = segment.trajectory, segment.start, segment.end
            if not 0 <= start < end <= len(traj.actions):
                raise ValueError("Invalid recovery learning interval")
            diag = contact_dynamics_importance(traj.raw_states, traj.contact, end, step_seconds)
            chunks = [(i, a, min(a+length, end), end) for a in range(start, end, length)]
            salience = np.array([diag["priority"][a:b].sum() for _,a,b,_ in chunks])
            targeted = salience/salience.sum() if salience.sum() else np.full(len(chunks), 1/len(chunks))
            uniform = DYNAMICS_SAMPLING_SPEC["chunk_uniform_fraction"]
            conditional = uniform/len(chunks)+(1-uniform)*targeted
            for chunk, local_p in zip(chunks, conditional):
                p = float(local_p)/len(archive.segments)
                self.chunks.append(chunk)
                marginal.append(p)
                self.chunk_details[chunk] = dict(segment_probability=1/len(archive.segments),
                    conditional_chunk_probability=float(local_p), probability=p,
                    segment_id=segment.descriptor["candidate_id"], learning_start=start, learning_end=end)
        self.marginal = np.asarray(marginal)

    def draw(self, rng):
        return (*self.chunks[int(rng.choice(len(self.chunks), p=self.marginal))], None, "recovery_dynamics")

    def probabilities(self):
        return dict(zip(self.chunks, self.marginal.tolist()))


def gru_sil_update(trainer, model, archive, encoder, buffer, *, budget_updates, iteration, recovery_archive=None):
    if getattr(buffer, "collection_metadata", {}).get("evaluation_only"):
        raise ValueError("Evaluation-only rollouts cannot be used for GRU SIL training")
    cache = getattr(encoder, "_gru_sil_observation_cache", None)
    if cache is None:
        cache = encoder._gru_sil_observation_cache = SILObservationCache()
    try:
        return _gru_sil_update(trainer, model, archive, encoder, buffer,
                               budget_updates=budget_updates, iteration=iteration, cache=cache,
                               recovery_archive=recovery_archive)
    finally:
        cache.end_round()


def _gru_sil_update(trainer, model, archive, encoder, buffer, *, budget_updates, iteration, cache, recovery_archive=None):
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
    rules = PhaseRules(**(cfg.gru_sil_phase_rules or {}))
    record = {"version": 1, "mode": "gru_full_prefix_positive_advantage", "iteration": iteration,
              "seed": cfg.gru_sil_seed, "budget": limit, "chunks": [],
              "sampling_mode": cfg.gru_sil_sampling_mode, "phase_rules": vars(rules),
              "phase_branch_probability": SAMPLING_PHASE_PROBABILITY[cfg.gru_sil_sampling_mode]}
    trainer.last_gru_sil_sampling = record
    candidates = eligible_sil_trajectories(archive.trajectories)
    if cfg.recovery_archive_enabled and recovery_archive is None:
        raise ValueError("Recovery sampling enabled but no independent recovery archive supplied")
    recovery = recovery_archive if cfg.recovery_archive_enabled else None
    have_recovery = recovery is not None and bool(recovery.segments)
    for pool in ("elite", "recovery"):
        for name in ("budget", "candidates", "sampled_frames", "positive_frames", "attempts", "updates",
                     "rejected_steps", "accepted_positive_frames", "zero_advantage_chunks",
                     "advantage_sum", "accepted_advantage_sum"):
            metrics[f"gru_sil_{pool}_{name}"] = 0
    both = bool(candidates) and have_recovery
    cursor = trainer.gru_sil_pool_cursor
    planned = [pool_for_slot(cursor+i, cfg.gru_sil_recovery_fraction) if both
               else "elite" if candidates else "recovery" for i in range(limit)] if candidates or have_recovery else []
    for pool in planned:
        metrics[f"gru_sil_{pool}_budget"] += 1
    record.update(pool_quota=dict(enabled=cfg.recovery_archive_enabled,
                  recovery_fraction=cfg.gru_sil_recovery_fraction, cursor_before=cursor, cursor_after=cursor,
                  planned=planned, both_available=both,
                  note="quota is consumed on candidate draw, even if advantage/KL rejects; empty pools lend their share"))
    active = [*archive.trajectories, *(recovery.trajectories if have_recovery else [])]
    cache.begin_round(encoder, active, cfg.gru_sil_cache_mb * 1024**2)
    if not limit or not planned:
        metrics.update(cache.metrics())
        return metrics
    if trainer.optimizer is None:
        raise ValueError("GRU SIL must run after PPO initializes/restores its optimizer")
    model.eval()
    optimizer = trainer.optimizer
    baseline = rollout_distributions(model, buffer)
    from .decision_buffer import has_zero_noise
    hybrid = has_zero_noise(buffer)
    if hybrid:
        # Keep the retired rejection counter at zero for historical log compatibility.
        metrics.update(gru_sil_deterministic_rejections=0., gru_sil_deterministic_action_max_delta=0.,
                       gru_sil_attempted_action_delta_max=0.)
    rng = np.random.default_rng(np.random.SeedSequence([cfg.gru_sil_seed, iteration, 714]))
    plan = GRUSILSamplingPlan(archive.trajectories, candidates, cfg.recurrent_seq_len,
                              cfg.gru_sil_sampling_mode, rules, step_seconds=cfg.feature_step_seconds) if candidates else None
    recovery_plan = RecoverySamplingPlan(recovery, cfg.recurrent_seq_len, cfg.feature_step_seconds) if have_recovery else None
    if plan is not None and cfg.gru_sil_sampling_mode in (EVENT_SAMPLING_MODE, DYNAMICS_SAMPLING_MODE):
        record["weighting"] = plan.distribution_summary
    if have_recovery:
        record["recovery_weighting"] = dict(segment_count=len(recovery.segments),
            segment_probability=1/len(recovery.segments), chunk_uniform_fraction=DYNAMICS_SAMPLING_SPEC["chunk_uniform_fraction"],
            segments=[s.descriptor for s in recovery.segments])
    phase_cache = {}
    for pool in planned:
        selected_plan = plan if pool == "elite" else recovery_plan
        ti, start, end, cap, selected_phase, branch = selected_plan.draw(rng)
        if both:
            trainer.gru_sil_pool_cursor += 1
            record["pool_quota"]["cursor_after"] = trainer.gru_sil_pool_cursor
        traj = archive.trajectories[ti] if pool == "elite" else recovery.segments[ti].trajectory
        prefix = f"gru_sil_{pool}_"
        metrics[prefix+"candidates"] += 1
        frames = cache.get(encoder, traj)
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
               "cap": cap, "bootstrap": bootstrap, "positive_frames": count, "accepted": False,
               "sampling_branch": branch, "pool":pool,
               "cap_kind":"confirmed_endpoint" if pool == "elite" else "first_return",
               "advantage_sum":float(advantage.sum())}
        if selected_plan.chunk_details:
            row["sampling_weight"] = dict(selected_plan.chunk_details[(ti, start, end, cap)])
            share = (1-cfg.gru_sil_recovery_fraction if pool == "elite" else cfg.gru_sil_recovery_fraction) if both else 1.
            row["sampling_weight"].update(pool_share_target=share,
                target_marginal_probability=share*row["sampling_weight"]["probability"])
        if pool == "recovery":
            row["segment_id"] = recovery.segments[ti].descriptor["candidate_id"]
        # Older/synthetic 33D pools still support legacy sampling; phase ablation requires 35D.
        if traj.raw_states.shape[1] == 35:
            key = id(traj)
            if key not in phase_cache:
                phase_cache[key] = classify_motion(traj.raw_states, traj.contact, rules)[0]
            labels = phase_cache[key][start:end]
            pos = positive.detach().cpu().numpy()
            weights = advantage.detach().cpu().numpy()
            row.update(selected_phase=selected_phase, phase_counts=phase_counts(labels),
                       positive_by_phase=phase_counts(labels[pos]),
                       advantage_sum_by_phase={name: float(weights[labels == name].sum()) for name in np.unique(labels)})
        record["chunks"].append(row)
        metrics["gru_sil_sampled_frames"] += end - start
        metrics["gru_sil_positive_frames"] += count
        metrics[prefix+"sampled_frames"] += end-start
        metrics[prefix+"positive_frames"] += count
        metrics[prefix+"advantage_sum"] += row["advantage_sum"]
        if not count:
            metrics[prefix+"zero_advantage_chunks"] += 1
            row["skip_reason"] = "no_positive_advantage"
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
            metrics[prefix+"attempts"] += 1
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite parameters after GRU SIL step")
            with torch.no_grad():
                current = rollout_distributions(model, buffer)
                kl = conditional_kl(baseline, current)
                drift = deterministic_action_drift(model, buffer, current) if hybrid else {}
            if not torch.isfinite(kl).all():
                raise FloatingPointError("Non-finite KL after GRU SIL step")
        except BaseException:
            # A failed audit or interruption must not leave an unvalidated update.
            model.load_state_dict(weights_before)
            optimizer.load_state_dict(optimizer_before)
            optimizer.zero_grad(set_to_none=True)
            raise
        mean_kl, max_kl = (float(kl.mean()), float(kl.max())) if kl.numel() else (0., 0.)
        row.update(kl=mean_kl, frame_kl_max=max_kl, loss=float(loss.detach()))
        row.update(drift)
        if hybrid:
            metrics["gru_sil_attempted_action_delta_max"] = max(
                metrics["gru_sil_attempted_action_delta_max"], drift["deterministic_action_max_delta"])
        metrics["gru_sil_attempted_kl_max"] = max(metrics["gru_sil_attempted_kl_max"], mean_kl)
        if mean_kl > cfg.gru_sil_target_kl or max_kl > cfg.gru_sil_max_frame_kl:
            model.load_state_dict(weights_before)
            optimizer.load_state_dict(optimizer_before)
            optimizer.zero_grad(set_to_none=True)
            metrics["gru_sil_rejected_steps"] += 1
            metrics[prefix+"rejected_steps"] += 1
            row["rejection_reason"] = "stochastic_kl"
            break
        row["accepted"] = True
        metrics["gru_sil_updates"] += 1
        metrics[prefix+"updates"] += 1
        metrics[prefix+"accepted_positive_frames"] += count
        metrics[prefix+"accepted_advantage_sum"] += row["advantage_sum"]
        metrics["gru_sil_loss"] += float(loss.detach())
        metrics["gru_sil_kl"], metrics["gru_sil_frame_kl_max"] = mean_kl, max_kl
        if hybrid:
            metrics["gru_sil_deterministic_action_max_delta"] = drift["deterministic_action_max_delta"]
        optimizer.zero_grad(set_to_none=True)
    metrics["gru_sil_loss"] /= max(1, metrics["gru_sil_updates"])
    metrics.update(cache.metrics())
    return metrics

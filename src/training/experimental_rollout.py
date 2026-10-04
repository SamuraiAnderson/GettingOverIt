"""Physical-step collection for independent representation and action-segment experiments."""
import time
import numpy as np
import torch

from .dataset import Trajectory, build_dynamics, left_pad_sequence
from .decision_buffer import DecisionRolloutBuffer
from .policy_audit import policy_context
from .policy_features import FeatureBuilder, SurfaceIndex, scale_report
from .delayed_exploration import scales_for_agents
from .reward import NewHighConfirmer, is_water, step_reward


def segment_action(parameters, elapsed, horizon):
    """Two sampled bounded endpoints; interpolation itself is deterministic."""
    parameters = np.asarray(parameters)
    if horizon == 1:
        return parameters.copy()
    alpha = elapsed / (horizon - 1)
    return ((1-alpha)*parameters[:2] + alpha*parameters[2:]).astype(np.float32)


def collect_decisions(worker, model, n_steps, eff_map=None, normalizer=None, *,
                      deterministic=False, initial_offsets=None, native_diagnostics=False):
    cfg, env = worker.config, worker.env
    if n_steps < 1 or env is None:
        raise ValueError("Positive rollout length and connected environment required")
    started = time.perf_counter()
    initial_encoded_frames = model.encoded_frame_count
    timings = {name: 0. for name in ("patch", "contact", "features", "inference", "env", "bootstrap", "diagnostics")}
    device = next(model.parameters()).device
    native_required = cfg.event_source == "native" and (
        cfg.event_features or cfg.event_branch or (cfg.action_segment_steps > 1 and cfg.segment_interrupt_contacts))
    read_native = native_required or native_diagnostics
    worker.last_native_diagnostics = []
    sequence = None

    def diagnostic():
        nonlocal sequence
        if not read_native:
            return None
        clock = time.perf_counter()
        packet = env.observation_diagnostics()
        if packet.get("version", 0) < 2 or not packet.get("aligned"):
            raise ValueError("Experiments require aligned native physics diagnostics v2+")
        if sequence is not None and packet["sequence"] <= sequence:
            raise ValueError("Native state did not advance")
        sequence = packet["sequence"]
        seconds = packet["fixedDeltaTime"] * packet["stepFrames"]
        if not np.isclose(seconds, cfg.feature_step_seconds, atol=1e-6, rtol=0):
            raise ValueError(f"feature_step_seconds={cfg.feature_step_seconds} differs from native step {seconds}; "
                             "select --feature-step-seconds explicitly")
        worker.last_native_diagnostics.append(packet)
        timings["diagnostics"] += time.perf_counter() - clock
        return packet

    def contact_signal(agent, contacts, packet):
        if cfg.event_source == "native" and packet is not None:
            row = packet["agents"][agent]
            return np.asarray(row["physicsContact"][:3]), np.asarray(row["physicsKnown"][:3], dtype=bool)
        if cfg.event_source == "geometry":
            known = np.array([worker._tip_local is not None, worker._body_only_local is not None,
                              worker._pot_only_local is not None]) & bool(worker._solid_polygons)
            return contacts[agent, [0, 2, 3]], known
        return np.zeros(3), np.zeros(3, dtype=bool)

    model.eval()
    if read_native:
        env.observation_diagnostics()  # Enable cache before RESET.
    obs = worker._reset_and_deploy()
    if initial_offsets is not None:
        if cfg.random_deploy or np.shape(initial_offsets) != (cfg.num_agents, 2):
            raise ValueError("Fixed offsets require fixed deployment and (N, 2) offsets")
        positions = obs[:, :2].copy() + initial_offsets
        for i, position in enumerate(positions):
            if np.any(initial_offsets[i]):
                obs = env.teleport(float(position[0]), float(position[1]), agent_index=i)
    agents = sorted(worker._get_stable_agents(obs))
    contacts, contact_previous = worker._compute_contact_batch(obs, {})
    packet = diagnostic()
    if cfg.geometry_features and worker._feature_surface_index is None:
        worker._feature_surface_index = SurfaceIndex(worker._solid_polygons or [])
    surfaces = worker._feature_surface_index if cfg.geometry_features else None
    builders = {i: FeatureBuilder(cfg, surfaces) for i in agents}
    score_schema = score_meta = None
    if worker.reference_corridor and agents:
        from .reference_anchor import scoring_context
        score_schema, score_meta = scoring_context(obs[agents[0]], cfg, env, worker.reference_corridor)
    trackers = {
        i: worker.reference_corridor.tracker_from_state(obs[i], score_schema, score_meta)
        for i in agents
    } if worker.reference_corridor else {}
    confirmers = {i: NewHighConfirmer(float(obs[i, 1]), cfg.dwell_steps, cfg.dwell_drop_tol) for i in agents}
    peaks = {i: float(obs[i, 1]) for i in agents}
    phi = cfg.ou_phi if cfg.ou_enabled and not deterministic else 0.
    buffer = DecisionRolloutBuffer(policy_context(model, phi, deterministic))
    noises = {i: (np.random.randn(model.parameter_dim).astype(np.float32) if phi
                  else np.zeros(model.parameter_dim, np.float32)) for i in agents}
    hidden = {i: None for i in agents}
    sparse_names = ("contour_tokens", "contour_mask", "relation_tokens", "relation_mask")
    use_sparse = bool(cfg.sparse_geometry)
    frames = {i: {name: [] for name in ("dynamics", "patches", "act_history", "step_rewards")} for i in agents}
    if use_sparse:
        for i in agents:
            for name in sparse_names:
                frames[i][name] = []
    records = {i: [] for i in agents}
    raw_states = {i: [obs[i].copy()] for i in agents}
    raw_actions = {i: [] for i in agents}
    contact_history = {i: [contacts[i].copy()] for i in agents}
    extras = {i: builders[i].step(obs[i], *contact_signal(i, contacts, packet))[0] for i in agents}
    extra_history = {i: [extras[i]] for i in agents}
    all_segments, active_segments, finished = [], {}, set()
    corridor_count = corridor_inside = corridor_gap = 0
    corridor_progress = corridor_penalty = 0.
    # 走廊模式逐帧字段（供 SIL 剪枝/排序/MAP-Elites bd/P3.2 优势加权；非走廊模式保持空 dict）：
    # corridor_credit[i] 与 raw_actions[i] 对齐（动作 t 获得的 credited 进度）；
    # corridor_guidance[i] 与 raw_states[i] 对齐（处理完 state[t] 后的引导段索引，-1 = 无引导）。
    # 首帧为 tracker 初始状态。与 rollout.py collect_ppo 的口径一致。
    from .reference_corridor import PROGRESS_MODE_V2, progress_mode_of
    distance_name = "delta_d" if trackers and progress_mode_of(cfg) == PROGRESS_MODE_V2 else "progress"
    corridor_credit = {i: [] for i in agents} if trackers else {}
    corridor_events = {i: [] for i in agents} if trackers else {}
    corridor_confirmed = {i: [0.0] for i in agents} if trackers else {}
    corridor_rewards = {i: [] for i in agents} if trackers else {}
    corridor_terminated = {i: [] for i in agents} if trackers else {}
    invalid_agents = set()
    corridor_guidance = ({i: [trackers[i].guidance_index if trackers[i].guidance_index is not None else -1]
                          for i in agents} if trackers else {})

    def frame_inputs(agent, patch, sparse_row=None):
        features = np.concatenate([build_dynamics(obs[agent], contacts[agent]), extras[agent]])
        frames[agent]["dynamics"].append(features)
        frames[agent]["patches"].append(patch.copy())
        frames[agent]["act_history"].append(raw_actions[agent][-1].copy() if raw_actions[agent] else np.zeros(2, np.float32))
        if use_sparse:
            for name, value in zip(sparse_names, sparse_row):
                frames[agent][name].append(value.copy())

    def tensor(values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)

    def sparse_kwargs(ids):
        if not use_sparse:
            return {}
        stacked = [tensor(np.stack([
            left_pad_sequence(np.asarray(frames[i][name][-cfg.context_len:]), cfg.context_len)[0]
            for i in ids
        ])) for name in sparse_names]
        return dict(zip(sparse_names, stacked))

    def last_sparse_kwargs(ids):
        if not use_sparse:
            return {}
        return dict(zip(sparse_names, [
            tensor([frames[i][name][-1] for i in ids])[:, None] for name in sparse_names
        ]))

    def windows(ids):
        arrays, masks = [], []
        for i in ids:
            data = [left_pad_sequence(np.asarray(frames[i][name][-cfg.context_len:]), cfg.context_len)
                    for name in ("dynamics", "patches", "act_history")]
            arrays.append([a for a, m in data])
            masks.append(data[0][1])
        return (*(tensor(np.stack([a[k] for a in arrays])) for k in range(3)), tensor(masks))

    for step in range(n_steps):
        active = [i for i in agents if i not in finished]
        clock = time.perf_counter()
        patches = worker._ppo_patches(obs, eff_map, trackers)
        sparse = worker._ppo_sparse(obs) if use_sparse else None
        timings["patch"] += time.perf_counter() - clock
        for i in active:
            frame_inputs(i, patches[i], None if sparse is None else tuple(plane[i] for plane in sparse))
        clock = time.perf_counter()
        # The GRU consumes EVERY physical observation, even while an option is active.
        recurrent = {}
        with torch.no_grad():
            if cfg.policy_backbone == "gru":
                inputs = [tensor([frames[i][name][-1] for i in active])[:, None]
                          for name in ("dynamics", "patches", "act_history")]
                h = torch.cat([hidden[i] if hidden[i] is not None else inputs[0].new_zeros(1, 1, cfg.d_model)
                               for i in active], dim=1)
                output, h = model.recurrent_features(*inputs, hidden=h, **last_sparse_kwargs(active))
                for k, i in enumerate(active):
                    recurrent[i], hidden[i] = output[k, 0], h[:, k:k+1].detach()
            ready = [i for i in active if i not in active_segments]
            batch_size = cfg.ppo_inference_batch_size or max(1, len(ready))
            for start in range(0, len(ready), batch_size):
                ids = ready[start:start+batch_size]
                previous_noise = tensor([noises[i] for i in ids])
                scale_np = scales_for_agents(cfg, ids, step, deterministic)
                scale_t = torch.from_numpy(scale_np).to(device)
                if cfg.policy_backbone == "gru":
                    av = model.sample_features(torch.stack([recurrent[i] for i in ids]), previous_noise, phi, deterministic, noise_scale=scale_t)
                else:
                    dynamics, pats, acts, mask = windows(ids)
                    av = model.get_action_and_value(
                        dynamics, pats, acts, mask, previous_noise, phi, deterministic,
                        noise_scale=scale_t,
                        **sparse_kwargs(ids),
                    )
                output = [a.detach().cpu().numpy() for a in (av.action, av.raw_action, av.log_prob, av.value, av.noise)]
                if not all(np.isfinite(a).all() for a in output):
                    raise FloatingPointError("Non-finite experimental policy output")
                for j, i in enumerate(ids):
                    parameters, raw, lp, value, noise = [a[j] for a in output]
                    active_segments[i] = {"decision_steps": step, "durations": 0, "actions": parameters.copy(),
                        "raw_actions": raw.copy(), "noise_prev": noises[i].copy(), "log_probs": float(lp),
                        "values": float(value), "rewards": 0., "dones": False,
                        "noise_scale": float(scale_np[j])}
                    noises[i] = noise.copy()
        timings["inference"] += time.perf_counter() - clock
        executed = np.zeros((cfg.num_agents, 2), dtype=np.float32)
        for i in active:
            option = active_segments[i]
            executed[i] = segment_action(option["actions"], option["durations"], cfg.action_segment_steps)
        clock = time.perf_counter()
        new_obs, dones = env.step(executed)
        timings["env"] += time.perf_counter() - clock
        clock = time.perf_counter()
        new_contacts, contact_previous = worker._compute_contact_batch(new_obs, contact_previous)
        timings["contact"] += time.perf_counter() - clock
        packet = diagnostic()
        for i in active:
            clock = time.perf_counter()
            extras[i], edge = builders[i].step(new_obs[i], *contact_signal(i, new_contacts, packet))
            extra_history[i].append(extras[i])
            timings["features"] += time.perf_counter() - clock
            reward, peaks[i] = step_reward(obs[i], new_obs[i], eff_map, cfg, running_max_y=peaks[i],
                normalizer=normalizer, confirmer=confirmers[i], corridor=trackers.get(i),
                state_schema=score_schema, reference_metadata=score_meta)
            option = active_segments[i]
            option["rewards"] += cfg.gamma ** option["durations"] * reward
            option["durations"] += 1
            done = bool(dones[i]) or is_water(new_obs[i, 1], cfg)
            option["dones"] = done
            interrupted = cfg.action_segment_steps > 1 and cfg.segment_interrupt_contacts and edge
            if done or interrupted or option["durations"] >= cfg.action_segment_steps or step == n_steps - 1:
                records[i].append(option)
                all_segments.append({"agent": i, "start": option["decision_steps"], "duration": option["durations"],
                    "reason": "terminal" if done else "contact" if interrupted else
                              "horizon" if option["durations"] == cfg.action_segment_steps else "rollout_end"})
                del active_segments[i]
            if done:
                finished.add(i)
            frames[i]["step_rewards"].append(reward)
            raw_actions[i].append(executed[i].copy())
            raw_states[i].append(new_obs[i].copy())
            contact_history[i].append(new_contacts[i].copy())
            if trackers:
                # 此时 tracker 已在 step_reward 内完成本步 step：last_step.progress
                # 是本动作获得的 credited；guidance_index 是 new_obs 帧的引导段。
                result = trackers[i].last_step
                corridor_count += 1
                corridor_inside += int(result.inside)
                corridor_gap += int(result.gap)
                if result.invalid:
                    invalid_agents.add(i)
                else:
                    corridor_progress += getattr(result, distance_name)
                    corridor_penalty += result.deviation_penalty
                corridor_credit[i].append(float(result.legacy_credit))
                corridor_events[i].append(bool(result.credit_event))
                corridor_confirmed[i].append(float(result.confirmed_progress))
                corridor_rewards[i].append(float(reward))
                corridor_terminated[i].append(bool(done))
                gid = trackers[i].guidance_index
                corridor_guidance[i].append(gid if gid is not None else -1)
        obs, contacts = new_obs, new_contacts
        if len(finished) == len(agents):
            break

    clock = time.perf_counter()
    patches = worker._ppo_patches(obs, eff_map, trackers)
    sparse = worker._ppo_sparse(obs) if use_sparse else None
    trajectories = []
    for i in agents:
        # Finished agents' final frame can precede the global loop's last frame.
        if i in finished:
            final_dynamics = np.concatenate([build_dynamics(raw_states[i][-1], contact_history[i][-1]), extra_history[i][-1]])
            frames[i]["dynamics"].append(final_dynamics)
            frames[i]["patches"].append(np.zeros_like(patches[i]))  # terminal; never consumed by policy/bootstrap
            frames[i]["act_history"].append(raw_actions[i][-1].copy())
            if use_sparse:
                from .sparse_geometry import empty_sparse_frame
                blank = empty_sparse_frame(cfg)
                for name, value in zip(sparse_names, blank):
                    frames[i][name].append(value.copy())
            value = 0.
        else:
            row = None if sparse is None else tuple(plane[i] for plane in sparse)
            frame_inputs(i, patches[i], row)
            with torch.no_grad():
                if cfg.policy_backbone == "gru":
                    inputs = [tensor(frames[i][name][-1])[None, None] for name in ("dynamics", "patches", "act_history")]
                    feature, _ = model.recurrent_features(*inputs, hidden=hidden[i], **last_sparse_kwargs([i]))
                    _, val = model.distribution_from_features(feature[:, 0])
                else:
                    dyn, pats, acts, mask = windows([i])
                    _, val = model(dyn, pats, acts, mask, **sparse_kwargs([i]))
                value = val.item()
        if i in invalid_agents:
            continue
        buffer.add_episode(frames[i], records[i], value, cfg.gamma, cfg.gae_lambda)
        trajectories.append(Trajectory(np.asarray(raw_states[i]), np.asarray(raw_actions[i]),
                                       contact=np.asarray(contact_history[i], np.float32),
                                       corridor_progress=(np.asarray(corridor_credit[i], np.float32)
                                                          if trackers else None),
                                       guidance_index=(np.asarray(corridor_guidance[i], np.int8)
                                                       if trackers else None),
                                       corridor_credit_events=(np.asarray(corridor_events[i], dtype=bool)
                                                               if trackers else None),
                                       corridor_confirmed_progress=(np.asarray(corridor_confirmed[i], dtype=np.float64)
                                                                    if trackers else None),
                                       step_rewards=(np.asarray(corridor_rewards[i], dtype=np.float64)
                                                     if trackers else None),
                                       terminated=(np.asarray(corridor_terminated[i], dtype=bool)
                                                   if trackers else None),
                                       trajectory_valid=True,
                                       termination_known=True,
                                       progress_mode=progress_mode_of(cfg) if trackers else None))
    timings["bootstrap"] = time.perf_counter() - clock
    timings["total"] = time.perf_counter() - started
    worker.last_profile = {f"rollout_{name}_seconds": value for name, value in timings.items()}
    worker.last_profile.update(rollout_samples_per_second=buffer.environment_steps/timings["total"],
        rollout_environment_steps=float(buffer.environment_steps), rollout_decisions=float(buffer.size),
        rollout_mean_segment_steps=buffer.environment_steps/max(buffer.size, 1),
        rollout_contact_interruptions=float(sum(r["reason"] == "contact" for r in all_segments)),
        rollout_patch_storage_bytes=float(buffer.patch_storage_bytes()))
    worker.last_profile["rollout_cnn_frame_evaluations"] = float(model.encoded_frame_count - initial_encoded_frames)
    physical_actions = np.concatenate([np.asarray(raw_actions[i]) for i in agents])
    worker.last_profile["executed_action_saturation_fraction"] = float(
        np.any(np.abs(physical_actions) >= .999 * cfg.action_scale, axis=-1).mean())
    worker.last_corridor_metrics = {"corridor_progress_m": corridor_progress,
        "corridor_deviation_penalty": corridor_penalty, "corridor_inside_fraction": corridor_inside/max(corridor_count, 1),
        "corridor_gap_fraction": corridor_gap/max(corridor_count, 1),
        "corridor_valid_agents": float(len(agents) - len(invalid_agents)),
        "corridor_invalid_agents": float(len(invalid_agents))} if trackers else {}
    worker.last_feature_scales = scale_report(np.concatenate([np.asarray(extra_history[i]) for i in agents]), cfg)
    worker.last_segment_trace = all_segments
    if agents:
        ep = buffer.episodes[0]
        sample = min(16, len(ep["dynamics"]) - 1)
        scale_inputs = [tensor(ep[name][:sample])[None] for name in ("dynamics", "patches", "act_history")]
        scale_sparse = {}
        if use_sparse:
            scale_sparse = {name: tensor(ep[name][:sample])[None] for name in sparse_names}
        worker.last_profile.update(model.embedding_scales(*scale_inputs, **scale_sparse))
    return buffer, trajectories

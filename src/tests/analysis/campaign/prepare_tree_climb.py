"""Offline diagnostics + executable, hash-locked GRU ablation recipes. Never launches a game.

conda run --no-capture-output -n getting-over-it-analysis python -m \
    src.tests.analysis.campaign.prepare_tree_climb --output runs/tree_climb_preflight_20261004
"""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from dataclasses import asdict, replace
import json
import io
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from training.climb_diagnostics import PhaseRules, classify_motion, discount_summary, entry_state, phase_counts
from training.climb_experiment import ARMS, digest, identity, load_recipe, resolve_config, reset_experiment_critic
from training.config import TrainConfig
from training.dataset import Trajectory, TrajectoryDataset
from training.decision_buffer import DecisionRolloutBuffer
from training.deploy_sampling import extract_body_local, extract_tip_local
from training.elite_archive import EliteTrajectoryArchive
from training.experimental_policy import create_actor_critic
from training.gru_sil import discounted_returns, sequence_predictions, sequence_features, phase_chunk_groups
from training.policy_audit import audit_policy
from training.reference_anchor import scoring_context
from training.reference_corridor import ReferenceCorridor, replay_corridor_result, trajectory_endpoint
from training.reference_fingerprint import reference_fingerprint, sidecar_path, assert_fingerprint_compatible
from training.reward import ClimbingEfficiencyMap
from tests.control_interaction.test_l7_surface_airdrop import load_colliders, extract_polygons

DEFAULT_RUN = ROOT / "runs/gru_sil_8h_20261003"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def replay(traj, cfg, route):
    schema, meta = scoring_context(traj.raw_states, config=cfg, route=route)
    result = replay_corridor_result(route, traj.raw_states, schema, meta)
    if not result["valid"]:
        raise ValueError("Invalid physical trajectory")
    return result


def evaluation_trajectory(path, cfg, route):
    raw = json.loads(path.read_text(encoding="utf-8"))
    states = np.asarray(raw["states"], np.float32)
    actions = np.asarray([row["action"] for row in raw["trace"]], np.float32)
    traj = Trajectory(states, actions, contact=np.asarray(raw["contacts"], np.float32),
                      progress_mode="confirmed_endpoint_v2")
    result = replay(traj, cfg, route)
    traj.corridor_credit_events = result["credit_events"]
    traj.corridor_confirmed_progress = result["confirmed_progress"]
    traj.corridor_progress = result["legacy_credit"]
    traj.guidance_index = result["guidance_index"]
    traj.step_rewards = result["rewards"]
    # Eval JSON has no authoritative terminal array. It is never used for critic targets.
    recorded = np.asarray([row["reward"] for row in raw["trace"]])
    np.testing.assert_allclose(recorded, result["rewards"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(raw["metrics"]["cap_progress_m"], result["distance"], rtol=0, atol=1e-5)
    return traj


def describe(traj, cfg, rules, label, gammas):
    labels, motion = classify_motion(traj.raw_states, traj.contact, rules)
    endpoint = trajectory_endpoint(traj)
    y = traj.raw_states[:, 34]
    eligible = max(0, endpoint.cap + 1)
    starts = [0, eligible]
    starts.extend(max(0, f["start_state"] - rules.preparation_steps) for f in motion["flights"] if f["takeoff_candidate"])
    arrival = entry_state(traj.raw_states, rules)
    if arrival:
        starts.append(arrival["state_index"])
    return {"label": label, "source_iteration": traj.iteration, "steps": len(traj.actions),
            "D_m": endpoint.distance, "cap_action": endpoint.cap, "t_cap": endpoint.t_cap,
            "peak_state": int(y.argmax()), "peak_pot_xy": traj.raw_states[y.argmax(), 33:35].tolist(),
            "rise_above_spawn_m": float(y.max() - y[0]), "end_pot_xy": traj.raw_states[-1, 33:35].tolist(),
            "tree_entry": arrival, "motion": motion,
            "eligible_phase_counts": phase_counts(labels[:eligible]),
            "reward_before_cap": float(traj.step_rewards[:eligible].sum()),
            "reward_after_cap": float(traj.step_rewards[eligible:].sum()),
            "discount_comparisons": [discount_summary(traj.step_rewards, gamma, cfg.feature_step_seconds, starts) for gamma in gammas]}, labels


def historical_sil(run, cfg, fp, rules, evidence):
    counts = {key: Counter() for key in ("selected", "accepted", "zero_advantage_chunk", "kl_rejected_chunk", "eligible_pool_opportunities")}
    expected = Counter()
    records = []
    totals = Counter()
    label_cache = {}
    files = sorted((run / "sil_sampling").glob("iter_*.json"))
    if not files:
        raise ValueError("No historical GRU SIL sampling records")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("mode") != "gru_full_prefix_positive_advantage":
            raise ValueError(f"Not a GRU SIL record: {path}")
        iteration = int(data["iteration"])
        pool = run / "elites/full_episodes" / f"elites_iter_{iteration:04d}.npz"
        archive = EliteTrajectoryArchive(cfg, pool.parent, fp)
        archive.restore(pool)
        evidence.extend([identity(path), identity(pool), identity(sidecar_path(pool))])
        labels_by_ti = {}
        candidates = [i for i, t in enumerate(archive.trajectories) if trajectory_endpoint(t).cap >= 0]
        for ti in candidates:
            traj = archive.trajectories[ti]
            # Content identity, never pool index/source iteration alone (several agents per round).
            import hashlib
            key = hashlib.sha256(traj.raw_states.tobytes() + traj.contact.tobytes()).hexdigest()
            if key not in label_cache:
                label_cache[key] = classify_motion(traj.raw_states, traj.contact, rules)[0]
            labels = label_cache[key]
            labels_by_ti[ti] = labels
            cap = trajectory_endpoint(traj).cap + 1
            counts["eligible_pool_opportunities"].update(phase_counts(labels[:cap]))
            chunks = list(range(0, cap, cfg.recurrent_seq_len))
            # Expected selected frames for the original two-level uniform sampler.
            for a in chunks:
                for name, n in phase_counts(labels[a:min(a+cfg.recurrent_seq_len, cap)]).items():
                    expected[name] += n * len(data["chunks"]) / len(candidates) / len(chunks)
        for row in data["chunks"]:
            ti, a, b = int(row["trajectory"]), int(row["start"]), int(row["end"])
            traj = archive.trajectories[ti]
            cap = trajectory_endpoint(traj).cap + 1
            if traj.iteration != row["source_iteration"] or not 0 <= a < b <= cap or cap != row["cap"]:
                raise ValueError(f"SIL provenance mismatch at {path}: {row}")
            phases = phase_counts(labels_by_ti[ti][a:b])
            counts["selected"].update(phases)
            totals["selected_chunks"] += 1
            totals["positive_frames_recorded"] += row["positive_frames"]
            outcome = "accepted" if row["accepted"] else ("zero_advantage_chunk" if row["positive_frames"] == 0 else "kl_rejected_chunk")
            counts[outcome].update(phases)
            totals[outcome] += 1
            records.append({"iteration": iteration, "trajectory": ti, "source_iteration": traj.iteration,
                            "start": a, "end": b, "outcome": outcome, "phase_counts": phases,
                            "positive_frames": row["positive_frames"]})
    return {"totals": dict(totals), "frame_counts": {k: dict(v) for k,v in counts.items()},
            "expected_selected_frames_original_sampler": dict(expected), "records": records,
            "historical_positive_frames_by_phase": None, "historical_gradient_weights_by_phase": None,
            "limitation": "Old logs contain positive counts per chunk, not per-frame advantages; those cannot be reconstructed as historical truth.",
            "stage_labels": "offline geometry proxies; accepted frame count does not mean every frame had a positive gradient"}


def make_encoder(cfg, route):
    env, player = load_colliders(Path(cfg.game_root))
    polygons = extract_polygons(env)
    cache = Path(cfg.checkpoint_dir) / "cache/traversable_mask.npz"
    eff = ClimbingEfficiencyMap(polygons, grid_resolution=cfg.grid_resolution,
        diffusion_iterations=cfg.diffusion_iterations, diffusion_alpha=cfg.diffusion_alpha,
        cache_path=str(cache) if cache.exists() else None, height_prior_weight=cfg.height_prior_weight)
    encoder = TrajectoryDataset(cfg, eff.traversable, None, min_gx=eff._min_gx, min_gy=eff._min_gy,
        solid_polygons=polygons, tip_local=extract_tip_local(player), body_local=extract_body_local(player),
        reference_maps=route.guidance_maps(eff.traversable.shape, eff._min_gx, eff._min_gy, cfg.grid_resolution))
    return encoder, eff


def critic_diagnostic(model, traj, frames, labels, cfg):
    with torch.no_grad():
        means, stds, values = sequence_predictions(model, frames)
    bootstrap = 0. if traj.terminated[-1] else float(values[-1])
    returns = discounted_returns(traj.step_rewards, traj.terminated, bootstrap, cfg.gamma)
    values = values[:-1].cpu().numpy()
    advantage = np.clip(returns - values, 0, cfg.sil_advantage_cap)
    targets = np.clip(np.arctanh(np.clip(traj.actions/cfg.action_scale, -1+1e-6, 1-1e-6)), -cfg.sil_target_clamp, cfg.sil_target_clamp)
    loss = .5 * np.square((means[:-1].cpu().numpy()-targets)/stds[:-1].cpu().numpy()).sum(axis=-1)
    cap = trajectory_endpoint(traj).cap + 1
    phase_rows = {}
    for name in np.unique(labels[:cap]):
        mask = labels[:cap] == name
        phase_rows[name] = {"eligible_frames": int(mask.sum()), "positive_frames": int((advantage[:cap][mask] > 0).sum()),
                            "advantage_sum": float(advantage[:cap][mask].sum()),
                            "weighted_imitation_error_sum": float((advantage[:cap]*loss[:cap])[mask].sum())}
    return {"gamma": cfg.gamma, "bootstrap": bootstrap, "phases": phase_rows,
            "meaning": "Frozen FINAL critic on archived actions; not historical advantage and not policy performance"}


def report_text(report, rows, sil):
    det = next(row for row in rows if row["label"] == "deterministic_000_agent0")
    total = sil["totals"]
    lines = ["# 树下攀爬实验：离线预检", "", f"状态：{report['status']}。未启动游戏、未更新模型。", "",
        "## 已核对", "", "- 实际 checkpoint 的训练配置、权重、完整精英池、参考路线和当前插件身份。",
        "- 精英轨迹的逐步奖励、确认进度与引导段重算；精确出生的一致性。",
        "- 评估存档的冻结策略概率审计，以及共享入口重建输入的一致性。",
        "- 历史 SIL 片段与当轮精英池的来源对应；当前 critic 的独立离线诊断。", "",
        "## 关键结果", "",
        f"- 最终确定性轨迹：D={det['D_m']:.3f} m，封顶状态 {det['t_cap']}；封顶前奖励 {det['reward_before_cap']:.3f}，之后 {det['reward_after_cap']:.3f}。",
        f"- 历史 SIL：选中 {total['selected_chunks']} 个片段；接受 {total.get('accepted',0)}，无正优势 {total.get('zero_advantage_chunk',0)}，KL 回滚 {total.get('kl_rejected_chunk',0)}。",
        "- 每阶段的实际/期望采样帧数见 sil_coverage.json；当前模型的正优势和模仿误差见 episodes.json。",
        "- 不同折扣下只比较观察到的有限尾部收益，尾端取零；没有套用旧 critic 估计新折扣价值。", "",
        "## 实验配置", "",
        "四组仅分别改变一个因素：control、guidance_wide（引导 σ=3m）、sil_mixed（确认进度排序 + 质心非重力加速/转向）、long_discount（γ=0.9975）。",
        "全部继承同一 actor、GRU、归一化和 log_std；使用全新优化器、相同 critic 重置及两轮 critic 预热。精英池完整保留，SIL 收益按各组 γ 重算。",
        "共同将 sil_start_iter 设为 2，使第三轮首次 PPO 更新后可执行 SIL；若实际 PPO 预算不足或无正优势，必须把 SIL 在线验收标为未覆盖。",
        "配对种子 7/17/47；预检 3 轮、筛选 10 轮，均包含预热。配置中的步数是最多执行世界步，活动训练样本另记；早停/终止后必须核对实际执行量。",
        "每份配置可交给 main_ppo --experiment-config <recipe.json> --dry-run 校验；移除 --dry-run 才会真正训练。screen 是从相同源重新开始的独立十轮实验，不叠加 precheck 结果。", "",
        "## 尚未通过的游戏内项目", "",
        "- 新配置的采样、PPO/SIL 更新、KL 回滚和断点恢复冒烟测试。",
        "- 精英前缀回放后切换当前策略的接管入口尚未实现。",
        "- 树枝接触/保持的物理真值验收：现阶段只标几何候选，不能当作成功率。",
        "- 短训对照和多种子效果验证；此报告不证明任何改动有收益。", "",
        "阶段定义固定在 protocol.json：状态 t 对应动作 t 之前；树区域按 x 带定义，离地/上升依据几何接触和速度。",
        "8 个确定性 agent 的相同轨迹仅用于复现性检查，统计为一个行为结果。所有测试维持精确出生，无扰动或传送初始化。", "",
        "## 可复查产物", "",
        "- preflight.json：门禁、输入身份、审计误差和源文件哈希。",
        "- protocol.json、recipes/：阶段规则、分组配置与预算。",
        "- episodes.json：全部精英及最终评估的动作阶段、状态、收益和当前 critic 诊断。",
        "- sil_coverage.json：历史每个片段及来源追踪。",
        "- evidence.json：本次实际读取的证据路径及 SHA-256。", ""]
    return "\n".join(lines)


def write_recipes(output, saved, inputs, sources, rules):
    for source in sources:
        if digest(source["path"]) != source["sha256"]:
            raise ValueError(f"Source changed: {source['path']}")
    plans = []
    for stage, iterations in (("precheck", 3), ("screen", 10)):
        for seed in (7, 17, 47):
            for arm in ARMS:
                resolved = resolve_config(saved["config"], arm, seed, iterations)
                recipe = {"schema": "tree_climb_experiment_v1", "arm": arm, "seed": seed, "iterations": iterations,
                    "baseline_config": saved["config"], "resolved_config": asdict(resolved), "inputs": inputs,
                    "run_dir": str(output / "training" / f"{stage}_{arm}_s{seed}"),
                    "initialization": "same_actor_and_std_fresh_optimizer_reset_critic_two_warmup_rounds",
                    "budget": {"iterations": iterations, "maximum_executed_world_steps": iterations*8*1200,
                               "warmup_included": True, "unit": "executed_world_steps_not_active_transitions"},
                    "phase_rules": asdict(rules), "source_files": sources}
                path = output/"recipes"/f"{stage}_{arm}_s{seed}.json"
                write(path, recipe)
                load_recipe(path)
                # Exercise the actual production parser/resolver, stopping before model/game creation.
                from training.main_ppo import main as train_main
                old_argv = sys.argv
                stream = io.StringIO()
                try:
                    sys.argv = ["main_ppo", "--experiment-config", str(path), "--dry-run", "--torch-threads", "4"]
                    with contextlib.redirect_stdout(stream):
                        train_main()
                finally:
                    sys.argv = old_argv
                actual = json.loads(stream.getvalue())["resolved_config"]
                if actual != recipe["resolved_config"]:
                    changed = [key for key in actual if actual[key] != recipe["resolved_config"][key]]
                    raise ValueError(f"Production resolver changed recipe: {changed}")
                plans.append({"path": str(path), "stage": stage, "arm": arm, "seed": seed})
    return plans


def prepare(args):
    torch.set_num_threads(4)
    output = args.output.resolve()
    source_run = args.checkpoint.resolve().parent.parent
    if output == source_run or source_run in output.parents or output in source_run.parents:
        raise ValueError("Use a separate experiment directory")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running_offline", "online_preflight": "not_run", "training_started": False, "checks": {}}
    try:
        sources = [identity(p) for p in sorted((ROOT/"src/training").glob("*.py"))]
        sources.append(identity(Path(__file__)))
        evidence = [identity(args.checkpoint), identity(args.elite_pool), identity(sidecar_path(args.elite_pool)),
                    identity(args.evaluation/"manifest.json")]
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = TrainConfig(**saved["config"])
        resolve_config(saved["config"], "control", 7, 3)
        rules = PhaseRules()
        route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
        fp = reference_fingerprint(cfg, route_meta=route.metadata, schema_id="anchor35", raw_dim=35)
        archive = EliteTrajectoryArchive(cfg, args.elite_pool.parent, fp)
        archive.restore(args.elite_pool)
        inputs = {"checkpoint": identity(args.checkpoint), "elite_pool": identity(args.elite_pool),
                  "pool_fingerprint": identity(sidecar_path(args.elite_pool)), "route": identity(cfg.reference_trajectory),
                  "runtime_dll": identity(Path(cfg.game_root)/"BepInEx/plugins/GameRuntime_v2.dll")}
        launch = json.loads((source_run/"launch_manifest.json").read_text(encoding="utf-8"))
        evidence.append(identity(source_run/"launch_manifest.json"))
        if inputs["runtime_dll"]["sha256"] != launch["runtime_dll_sha256"]:
            raise ValueError("Installed runtime differs from the formal training binary")
        if route.metadata["sha256"] != saved["reference_corridor"]["sha256"]:
            raise ValueError("Reference changed since checkpoint")
        report["checks"]["checkpoint_pool_route_runtime"] = "passed"
        if getattr(args, "configs_only", False):
            plans = write_recipes(output, saved, inputs, sources, rules)
            report.update(status="configs_passed_full_diagnostics_not_run", inputs=inputs,
                          source_files=sources, recipes=plans)
            report["checks"]["validated_recipes"] = len(plans)
            write(output/"config_preflight.json", report)
            print(json.dumps({"status": report["status"], "recipes": len(plans), "output": str(output)}, ensure_ascii=False), flush=True)
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_actor_critic(cfg).to(device).eval()
        report["device"] = device
        model.load_state_dict(saved["model"], strict=True)
        original = {k: v.clone() for k,v in model.state_dict().items()}
        if not all(torch.isfinite(v).all() for v in original.values()):
            raise ValueError("Nonfinite source weights")
        rows = []; labels_by_pool = []
        gammas = sorted({cfg.gamma, .995, .9975, .999})
        initial = None; spawn_error = 0.; reward_error = 0.
        physical = [i for i in range(35) if i != 28]
        for i, traj in enumerate(archive.trajectories):
            replayed = replay(traj, cfg, route)
            reward_error = max(reward_error, float(np.max(np.abs(replayed["rewards"]-traj.step_rewards))))
            np.testing.assert_allclose(replayed["rewards"], traj.step_rewards, rtol=0, atol=1e-5)
            np.testing.assert_allclose(replayed["confirmed_progress"], traj.corridor_confirmed_progress, rtol=0, atol=1e-5)
            np.testing.assert_array_equal(replayed["credit_events"], traj.corridor_credit_events)
            np.testing.assert_array_equal(replayed["guidance_index"], traj.guidance_index)
            initial = traj.raw_states[0, physical].copy() if initial is None else initial
            spawn_error = max(spawn_error, float(np.max(np.abs(initial-traj.raw_states[0, physical]))))
            row, labels = describe(traj, cfg, rules, f"elite_{i:02d}", gammas)
            rows.append(row); labels_by_pool.append(labels)
        report["checks"]["elite_reward_replay_max_error"] = reward_error
        print("Elite integrity/reward replay passed", flush=True)
        encoder, eff = make_encoder(cfg, route)
        reconstruction = {}; frozen_audits = {}; deterministic_states = []
        for mode in ("deterministic", "exploration"):
            buffer_path = args.evaluation/f"{mode}_000.npz"
            evidence.append(identity(buffer_path))
            buffer = DecisionRolloutBuffer.load(buffer_path)
            frozen_audits[mode] = audit_policy(model, buffer, max_samples=None)
            for agent, frames in enumerate(buffer.episodes):
                path = args.evaluation/f"{mode}_000_agent{agent}.json"
                evidence.append(identity(path))
                traj = evaluation_trajectory(path, cfg, route)
                traj.terminated = np.asarray(frames["dones"], bool)
                np.testing.assert_array_equal(traj.actions, frames["actions"])
                np.testing.assert_allclose(traj.step_rewards, frames["step_rewards"], rtol=0, atol=1e-5)
                spawn_error = max(spawn_error, float(np.max(np.abs(initial-traj.raw_states[0, physical]))))
                if mode == "deterministic":
                    deterministic_states.append(traj.raw_states[:, physical])
                # All eight duplicates are audited above; only one contributes to behavior summaries.
                if mode == "deterministic" and agent > 0:
                    continue
                row, labels = describe(traj, cfg, rules, path.stem, gammas)
                row["critic_diagnostic"] = critic_diagnostic(model, traj, frames, labels, cfg)
                rows.append(row)
                if agent == 0:
                    rebuilt = encoder.encode_sequence(traj)
                    reconstruction[mode] = {name: float(np.max(np.abs(value-frames[name]))) for name,value in rebuilt.items()}
                    if any(error != 0 for error in reconstruction[mode].values()):
                        raise ValueError(f"Offline/online input mismatch: {reconstruction[mode]}")
                    del rebuilt
            del buffer
            print(f"{mode}: frozen behavior audit and input reconstruction passed", flush=True)
        if any(not np.array_equal(deterministic_states[0], s) for s in deterministic_states[1:]):
            raise ValueError("Deterministic worlds differ")
        if spawn_error > 1e-5:
            raise ValueError(f"Exact-spawn mismatch: {spawn_error}")
        report["checks"].update(spawn_max_error=spawn_error, frozen_policy=frozen_audits,
                                  shared_input_errors=reconstruction, deterministic_independent_behaviors=1)
        for i, traj in enumerate(archive.trajectories):
            frames = encoder.encode_sequence(traj)
            rows[i]["critic_diagnostic"] = critic_diagnostic(model, traj, frames, labels_by_pool[i], cfg)
            if i == 0:
                start = min(64, len(traj.actions)-1); end = min(start+32, len(traj.actions))
                with torch.no_grad():
                    full = sequence_features(model, frames, 0, end)[start:]
                    chunk = sequence_features(model, frames, start, end)
                error = float((full-chunk).abs().max())
                if error > 1e-5:
                    raise ValueError(f"GRU full-prefix mismatch: {error}")
                report["checks"]["full_prefix_feature_max_error"] = error
            del frames
            print(f"Frozen critic: elite {i+1}/{len(archive.trajectories)}", flush=True)
        sil = historical_sil(source_run, cfg, fp, rules, evidence)
        report["checks"]["historical_sil_provenance"] = "passed"
        if not all(torch.equal(original[k],v) for k,v in model.state_dict().items()):
            raise ValueError("Offline diagnostics mutated weights")
        report["checks"]["model_weights_unchanged"] = True
        # Experimental guidance changes observation only, verified on actual raw states.
        wide_route = ReferenceCorridor(route.segments, replace(cfg, reference_guidance_sigma=3.), route.metadata)
        for traj in archive.trajectories:
            a, b = replay(traj, cfg, route), replay(traj, wide_route.config, wide_route)
            np.testing.assert_array_equal(a["rewards"], b["rewards"])
        assert_fingerprint_compatible(fp, reference_fingerprint(wide_route.config, route_meta=route.metadata, schema_id="anchor35", raw_dim=35), context="guidance ablation")
        report["checks"]["guidance_keeps_reward_and_archive_semantics"] = True
        # Complete inherited weights; common critic-reset protocol must not perturb policy tensors.
        torch.manual_seed(7)
        reset_experiment_critic(model)
        if any(not torch.equal(original[k],v) for k,v in model.state_dict().items() if not k.startswith("critic_head.")):
            raise ValueError("Critic reset changed policy tensors")
        report["checks"]["critic_reset_preserves_policy"] = True
        model.load_state_dict(original)
        candidates = list(range(len(archive.trajectories)))
        groups = phase_chunk_groups(archive.trajectories, candidates, cfg.recurrent_seq_len, rules)
        for source in sources:
            if digest(source["path"]) != source["sha256"]:
                raise ValueError(f"Source changed during diagnostics: {source['path']}")
        plans = write_recipes(output, saved, inputs, sources, rules)
        report["checks"]["validated_recipes"] = len(plans)
        report.update(status="offline_passed_online_pending", inputs=inputs, source_files=sources,
                      limitations=["geometry contact candidates are not verified branch grips", "no historical per-frame gradients",
                                   "no game launch or optimizer steps in this preparation", "prefix-policy handoff is not implemented"])
        protocol = {"schema": "tree_climb_protocol_v1", "phase_rules": asdict(rules), "arms": ARMS,
                    "evaluation": {"exact_spawn": True, "agents": 8, "steps": 1200, "seed": 20261003,
                                   "modes": ["deterministic", "exploration"], "rounds_per_mode": 1,
                                   "deterministic_duplicates_are_not_independent": True},
                    "phase_sampler_groups_in_final_pool": {name: len(chunks) for name,chunks in groups.items()},
                    "recipes": plans, "online_preflight": "pending", "training_authorization": "not_requested_by_prepare"}
        write(output/"protocol.json", protocol); write(output/"episodes.json", rows)
        write(output/"sil_coverage.json", sil); write(output/"evidence.json", evidence)
        write(output/"preflight.json", report)
        (output/"REPORT.md").write_text(report_text(report, rows, sil), encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(output/"REPORT.md"), "recipes": len(plans)}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write(output/"preflight.json", report)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_RUN/"checkpoints/ppo_iter_0184.pt")
    p.add_argument("--elite-pool", type=Path, default=DEFAULT_RUN/"elites/full_episodes/elites_iter_0184.npz")
    p.add_argument("--evaluation", type=Path, default=DEFAULT_RUN/"post_training_review/checkpoint_0184")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--configs-only", action="store_true", help="Only validate frozen inputs and regenerate recipes; do not claim full diagnostics")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()

"""Read-only actual-data SIL/GRU consistency and fixed-history fitting audit.

No game, optimizer step, or change to formal checkpoints. Uses production
encoders, recurrent replay, return targets and existing diagnostic helpers.
"""
import csv
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from tests.analysis.campaign.prepare_tree_climb import make_encoder, evaluation_trajectory, replay
from training.config import TrainConfig
from training.decision_buffer import DecisionRolloutBuffer
from training.elite_archive import EliteTrajectoryArchive
from training.experimental_policy import create_actor_critic
from training.gru_sil import sequence_predictions, sequence_features, discounted_returns, _step
from training.reference_corridor import ReferenceCorridor, trajectory_endpoint

OUT = Path(__file__).resolve().parent
RUNS = [ROOT / 'runs' / x for x in (
    'gru_sil_8h_20261003', 'gru_sil_dynamics_1h_20261004', 'gru_sil_dynamics_1h2_20261004')]
ITERS = [184, 199, 209]
CHECKPOINTS = [r / f'checkpoints/ppo_iter_{i:04d}.pt' for r, i in zip(RUNS, ITERS)]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def trajectory_id(traj):
    return hashlib.sha256(traj.raw_states.tobytes() + traj.actions.tobytes()).hexdigest()

def write(report):
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

@torch.no_grad()
def precision_reference(model, frames, means, values, incremental_means, incremental_values):
    """Independent float64 arithmetic check using the same recurrent model."""
    reference = copy.deepcopy(model).cpu().double().eval()
    def step(a, b, hidden=None):
        def tensor(k): return torch.as_tensor(frames[k][a:b], dtype=torch.float64)[None]
        return reference.recurrent_features(tensor('dynamics'), tensor('patches'), tensor('act_history'),
            hidden=hidden, **{k:tensor(k) for k in ('contour_tokens','contour_mask','relation_tokens','relation_mask') if k in frames})
    output, _ = step(0, 224)
    dist, value = reference.distribution_from_features(output[0])
    hidden = None
    single_mean, single_value = [], []
    for t in range(224):
        output, hidden = step(t,t+1,hidden)
        d,v = reference.distribution_from_features(output[0])
        single_mean.append(d.mean); single_value.append(v)
    _, hidden = step(0,128)
    _, hidden = step(128,192,hidden)
    output, _ = step(192,224,hidden)
    chunk_dist, chunk_value = reference.distribution_from_features(output[0])
    logical = dict(incremental_mean=float((torch.cat(single_mean)-dist.mean).abs().max()),
        incremental_value=float((torch.cat(single_value)-value).abs().max()),
        burn_in_mean=float((chunk_dist.mean-dist.mean[192:224]).abs().max()),
        burn_in_value=float((chunk_value-value[192:224]).abs().max()))
    assert max(logical.values()) < 1e-9, logical
    arithmetic = dict(batch_mean=float((means[:224].cpu()-dist.mean).abs().max()),
        batch_value=float((values[:224].cpu()-value).abs().max()),
        incremental_mean=float((torch.cat(incremental_means).cpu()-dist.mean).abs().max()),
        incremental_value=float((torch.cat(incremental_values).cpu()-value).abs().max()))
    # Report real action-space differences, not just network latent units.
    action_error = float((100*torch.tanh(torch.cat(incremental_means))-100*torch.tanh(means[:224])).abs().max())
    assert max(arithmetic.values()) < 1e-3 and action_error < .01, (arithmetic,action_error)
    return dict(float64_recurrent_errors=logical, gpu_float32_errors_vs_float64=arithmetic,
                gpu_incremental_vs_batch_action_error=action_error,
                explanation='Same recurrence agrees in float64; GPU float32 arithmetic changes with batch/chunk shape.')

def summarize_fit(mean, std, value, traj, cfg, a, b):
    targets_unclipped = np.arctanh(np.clip(traj.actions / cfg.action_scale, -1+1e-6, 1-1e-6))
    targets = np.clip(targets_unclipped, -cfg.sil_target_clamp, cfg.sil_target_clamp)
    bootstrap = 0. if traj.terminated[-1] else float(value[-1])
    returns = discounted_returns(traj.step_rewards, traj.terminated, bootstrap, cfg.gamma)
    advantage = np.clip(returns - value[:-1], 0, cfg.sil_advantage_cap)
    error = mean[:-1] - targets
    action_error = cfg.action_scale * np.tanh(mean[:-1]) - traj.actions
    loss = .5 * np.square(error / std[:-1]).sum(-1)
    return dict(start=a, end=b, frames=b-a,
        action_rmse_per_component=float(np.sqrt(np.mean(action_error[a:b]**2))),
        raw_rmse_per_component=float(np.sqrt(np.mean(error[a:b]**2))),
        normalized_imitation_loss_mean=float(loss[a:b].mean()),
        positive_frames=int((advantage[a:b]>0).sum()), advantage_mean=float(advantage[a:b].mean()),
        critic_mean=float(value[a:b].mean()), return_mean=float(returns[a:b].mean()),
        weighted_actor_loss=float(cfg.gru_sil_loss_coef * np.mean(advantage[a:b]*loss[a:b])),
        target_clipped_frames=int(np.any(np.abs(targets_unclipped[a:b]) > cfg.sil_target_clamp, axis=-1).sum()),
        mean_std=std[a:b].mean(axis=0).tolist())

def main():
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    saved = [torch.load(p, map_location='cpu', weights_only=False) for p in CHECKPOINTS]
    cfg = TrainConfig(**saved[-1]['config'])
    report = dict(status='running', device=device, checkpoint_hashes={str(p): sha(p) for p in CHECKPOINTS},
        checks={}, fitting=[], coverage=[], limitations=[
            'Fixed recorded histories use elite actions as previous-action input, not closed-loop policy execution.',
            'Post-checkpoint critic advantages are not the historical per-frame SIL advantages.',
            'Unique accepted frames are candidate frames; logs do not store per-frame historical positive masks.',
            'Formal training buffers were not retained, so every historic training log probability cannot be independently replayed.',
            'No optimizer updates or gameplay in this audit; no causal isolation of PPO and SIL.'])
    previous = json.loads((OUT/'audit.json').read_text()) if (OUT/'audit.json').exists() else {}
    write(report)
    launch = json.loads((RUNS[-1]/'launch_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(s['path']) == s['sha256'] for s in launch['source_files'])
    report['checks']['unchanged_launch_source_files'] = len(launch['source_files'])
    route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
    assert sha(Path(cfg.game_root)/'BepInEx/plugins/GameRuntime_v2.dll') == launch['inputs']['runtime_dll']['sha256']
    assert all(s['reference_corridor']['sha256'] == route.metadata['sha256'] for s in saved)
    models = []
    for s in saved:
        c = TrainConfig(**s['config'])
        m = create_actor_critic(c).to(device).eval()
        m.load_state_dict(s['model'])
        m.exploration_boost = s.get('loop_state', {}).get('boost', 0.)
        assert m.exploration_boost == 0., 'This audit assumes the verified no-boost runs'
        models.append(m)
    report['checks']['route_runtime_and_zero_exploration_boost'] = True
    meta = json.loads((RUNS[-1]/'run_meta.json').read_text(encoding='utf-8'))
    pool = RUNS[-1]/'elites/full_episodes/elites_iter_0209.npz'
    archive = EliteTrajectoryArchive(cfg, pool.parent, meta['reference_fingerprint'])
    archive.restore(pool)
    ranked = sorted(archive.trajectories, key=lambda t: trajectory_endpoint(t).distance, reverse=True)
    encoder, _ = make_encoder(cfg, route)
    print('Source, checkpoint, route and runtime identities passed', flush=True)

    # Compare production offline/SIL rebuilding with persisted online observations.
    evaluation = RUNS[-1]/'post_training_review/checkpoint_0209'
    reconstruction = previous.get('checks', {}).get('online_offline_input_errors', {}) if '--reuse-input-checks' in sys.argv else {}
    if reconstruction:
        assert previous['checkpoint_hashes'] == report['checkpoint_hashes']
        assert previous['checks']['unchanged_launch_source_files'] == len(launch['source_files'])
        assert len(reconstruction) == 2 and all(e==0 for row in reconstruction.values() for e in row.values())
        report['checks']['reuse_earlier_this_audit_input_checks'] = True
    for mode, agent in ([] if reconstruction else [('deterministic', 0), ('exploration', 6)]):
        buffer = DecisionRolloutBuffer.load(evaluation/f'{mode}_000.npz')
        online = buffer.episodes[agent]
        traj = evaluation_trajectory(evaluation/f'{mode}_000_agent{agent}.json', cfg, route)
        traj.terminated = np.asarray(online['dones'], bool)
        np.testing.assert_array_equal(traj.actions, online['actions'])
        rebuilt = encoder.encode_sequence(traj)
        errors = {k: float(np.max(np.abs(v-online[k]))) for k,v in rebuilt.items()}
        assert all(e==0 for e in errors.values()), errors
        reconstruction[f'{mode}_agent{agent}'] = errors
        del buffer, online, rebuilt
        print(f'{mode} agent {agent}: exact online/offline input equality passed', flush=True)
    report['checks']['online_offline_input_errors'] = reconstruction
    write(report)

    for rank, traj in enumerate(ranked[:2], 1):
        result = replay(traj, cfg, route)
        np.testing.assert_allclose(result['rewards'], traj.step_rewards, rtol=0, atol=1e-5)
        np.testing.assert_array_equal(result['guidance_index'], traj.guidance_index)
        cache = OUT/f'elite_{trajectory_id(traj)}_frames.npz'
        if '--reuse-input-checks' in sys.argv and cache.exists():
            with np.load(cache) as stored: frames = {k:stored[k] for k in stored.files}
        else:
            frames = encoder.encode_sequence(traj)
            np.savez_compressed(cache, **frames)
        cap = trajectory_endpoint(traj).cap+1
        row = dict(rank=rank, source_id=trajectory_id(traj), D=trajectory_endpoint(traj).distance,
                   cap_exclusive=cap, checkpoints={})
        for iteration, model, s in zip(ITERS, models, saved):
            with torch.no_grad():
                means, stds, values = sequence_predictions(model, frames)
            windows = [(0,cap), (0,32), (160,192), (192,224), (224,256), (384,416), (416,min(448,cap))]
            row['checkpoints'][str(iteration)] = [summarize_fit(
                means.cpu().numpy(), stds.cpu().numpy(), values.cpu().numpy(), traj,
                TrainConfig(**s['config']), a,b) for a,b in windows if a < b <= cap]
            if iteration == ITERS[-1]:
                # Incremental playback, batched replay and detached full-prefix chunk.
                hidden = None
                incremental_means, incremental_values = [], []
                with torch.no_grad():
                    for t in range(224):
                        f, hidden = _step(model, frames, t, t+1, hidden)
                        dist, value = model.distribution_from_features(f[0])
                        incremental_means.append(dist.mean); incremental_values.append(value)
                    chunk = sequence_features(model, frames, 192, 224)
                    dist, value = model.distribution_from_features(chunk)
                errors = dict(incremental_mean=float((torch.cat(incremental_means)-means[:224]).abs().max()),
                    incremental_value=float((torch.cat(incremental_values)-values[:224]).abs().max()),
                    burn_in_mean=float((dist.mean-means[192:224]).abs().max()),
                    burn_in_value=float((value-values[192:224]).abs().max()))
                row['recurrent_errors'] = errors
                row['precision_reference'] = precision_reference(model,frames,means,values,incremental_means,incremental_values)
        report['fitting'].append(row)
        write(report)
        print(f'Elite rank {rank}: fixed-history fit and stateful/full-prefix consistency passed', flush=True)
        del frames

    # Restore every sampled pool to trace content identities across pool replacement/reordering.
    all_draws = []
    budgets, ppo_updates, physical_steps = 0, 0, 0
    for run in RUNS[1:]:
        metrics = list(csv.DictReader((run/'metrics.csv').open(encoding='utf-8')))
        for metric in metrics:
            iteration = int(metric['iteration'])
            budgets += int(float(metric['gru_sil_budget']))
            ppo_updates += int(float(metric['n_updates']))
            physical_steps += int(float(metric['rollout_environment_steps']))
            archive.restore(run/f'elites/full_episodes/elites_iter_{iteration:04d}.npz')
            record = json.loads((run/f'sil_sampling/iter_{iteration:04d}.json').read_text(encoding='utf-8'))
            for chunk in record['chunks']:
                all_draws.append(dict(chunk, run=run.name, training_iteration=iteration,
                                     source_id=trajectory_id(archive.trajectories[chunk['trajectory']])))
    accepted = [d for d in all_draws if d['accepted']]
    report['dose'] = dict(rounds=25, environment_steps=physical_steps, ppo_optimizer_steps=ppo_updates,
        sil_budget=budgets, sampled_chunks=len(all_draws), accepted_chunks=len(accepted),
        positive_frames_accepted=sum(d['positive_frames'] for d in accepted),
        accepted_frame_presentations=sum(d['end']-d['start'] for d in accepted),
        zero_positive_chunks=sum(d['positive_frames']==0 for d in all_draws),
        rejected_chunks=sum('kl' in d and not d['accepted'] for d in all_draws),
        accepted_start_distribution=dict(Counter('0_to_119' if d['start']<120 else '120_to_299' if d['start']<300 else '300_plus' for d in accepted)))
    for traj in ranked[:2]:
        key = trajectory_id(traj)
        draws = [d for d in all_draws if d['source_id']==key]
        acc = [d for d in draws if d['accepted']]
        coverage = np.zeros(trajectory_endpoint(traj).cap+1, np.int32)
        for d in acc: coverage[d['start']:d['end']] += 1
        report['coverage'].append(dict(source_id=key, D=trajectory_endpoint(traj).distance,
            eligible_frames=len(coverage), draws=len(draws), accepted_chunks=len(acc),
            unique_accepted_candidate_frames=int(np.count_nonzero(coverage)),
            accepted_frame_presentations=int(coverage.sum()),
            chunks=[dict(start=a, end=min(a+cfg.recurrent_seq_len,len(coverage)),
                accepted_repetitions=int(coverage[a])) for a in range(0,len(coverage),cfg.recurrent_seq_len)],
            important_192_draws=[dict(iteration=d['training_iteration'], accepted=d['accepted'],
                positive_frames=d['positive_frames']) for d in draws if d['start']==192]))
    report['discount'] = dict(gamma=cfg.gamma, step_seconds=cfg.feature_step_seconds,
        half_life_seconds=float(np.log(.5)/np.log(cfg.gamma)*cfg.feature_step_seconds),
        weight_after_1_second=cfg.gamma**120, weight_after_2_seconds=cfg.gamma**240,
        meaning='Future reward attenuation only; no counterfactual performance claim.')
    assert all(sha(p)==report['checkpoint_hashes'][str(p)] for p in CHECKPOINTS)
    for model, s in zip(models,saved):
        assert all(torch.equal(v.cpu(),s['model'][k]) for k,v in model.state_dict().items())
    report['checks']['formal_weights_and_loaded_models_unchanged'] = True
    report['status'] = 'passed'
    report['evidence'] = dict(regression_tests=str(OUT/'regression_tests.txt'), audit_source_sha256=sha(__file__),
        launch_manifest_sha256=sha(RUNS[-1]/'launch_manifest.json'), elite_pool_sha256=sha(pool))
    write(report)
    print(json.dumps(dict(status='passed', dose=report['dose'], report=str(OUT/'audit.json')), indent=2), flush=True)

if __name__ == '__main__':
    main()

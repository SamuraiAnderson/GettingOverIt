"""Reproduce this run's offline integrity and actual SIL sampling review."""
import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from training.config import TrainConfig
from training.elite_archive import EliteTrajectoryArchive
from training.gru_sil import GRUSILSamplingPlan, eligible_sil_trajectories
from training.reference_corridor import ReferenceCorridor, trajectory_endpoint
from training.local_evaluation import score_corridor_episode

torch.set_num_threads(2)
RUN = Path(__file__).resolve().parent.parent
OUT = RUN / 'post_training_review'
meta = json.loads((RUN / 'run_meta.json').read_text())
cfg = TrainConfig(**{k: v for k, v in meta['config'].items() if k in {f.name for f in fields(TrainConfig)}})
metrics = [{k: float(v) for k, v in row.items() if v} for row in csv.DictReader((RUN / 'metrics.csv').open())]
iterations = [int(x['iteration']) for x in metrics]
launch = json.loads((RUN / 'launch_manifest.json').read_text(encoding='utf-8'))
start_iteration = int(launch['resume_iteration'])
final_iteration = int(meta['last_iteration'])
assert iterations == list(range(start_iteration + 1, final_iteration + 1))
assert all(np.isfinite(list(x.values())).all() for x in metrics)
archive = EliteTrajectoryArchive(cfg, OUT / 'unused', meta['reference_fingerprint'])
archive.restore(Path(launch['elite_archive_resume']))
def identity(traj): return hashlib.sha256(traj.raw_states.tobytes() + traj.actions.tobytes()).hexdigest()
initial_ranked = sorted(archive.trajectories, key=lambda t: trajectory_endpoint(t).distance, reverse=True)
top_two_ids = {identity(t) for t in initial_ranked[:2]}
farthest_id = identity(initial_ranked[0])
route = ReferenceCorridor.from_file(cfg.reference_trajectory, cfg)
chunks = []
audited_probabilities = 0
finite_tensors = 0
def finite(value):
    global finite_tensors
    if torch.is_tensor(value):
        assert torch.isfinite(value).all()
        finite_tensors += 1
    elif isinstance(value, dict):
        for x in value.values(): finite(x)
    elif isinstance(value, (list, tuple)):
        for x in value: finite(x)

for metric in metrics:
    iteration = int(metric['iteration'])
    ckpt = RUN / f'checkpoints/ppo_iter_{iteration:04d}.pt'
    with zipfile.ZipFile(ckpt) as z: assert z.testzip() is None
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    assert saved['iteration'] == iteration
    assert all(saved.get(k) for k in ('rng_state', 'trainer_state', 'loop_state', 'optimizer'))
    finite(saved['model']); finite(saved['optimizer'])
    archive.restore(RUN / f'elites/full_episodes/elites_iter_{iteration:04d}.npz')
    record = json.loads((RUN / f'sil_sampling/iter_{iteration:04d}.json').read_text())
    assert record['sampling_mode'] == 'quality_dynamics_v3'
    plan = GRUSILSamplingPlan(archive.trajectories, eligible_sil_trajectories(archive.trajectories),
                             cfg.recurrent_seq_len, cfg.gru_sil_sampling_mode,
                             step_seconds=cfg.feature_step_seconds)
    assert abs(sum(plan.probabilities().values()) - 1) < 1e-12
    rng = np.random.default_rng(np.random.SeedSequence([cfg.gru_sil_seed, iteration, 714]))
    rows = record['chunks']
    for row in rows:
        ti, a, b, cap, phase, branch = plan.draw(rng)
        assert (ti, a, b, cap, branch) == (row['trajectory'], row['start'], row['end'], row['cap'], row['sampling_branch'])
        assert archive.trajectories[ti].iteration == row['source_iteration']
        for k, value in plan.chunk_details[(ti, a, b, cap)].items():
            assert np.isclose(value, row['sampling_weight'][k], atol=1e-12, rtol=0), (iteration, k)
        assert sum(row['positive_by_phase'].values()) == row['positive_frames']
        if row['accepted']:
            assert row['positive_frames'] > 0
            assert row['kl'] <= cfg.gru_sil_target_kl and row['frame_kl_max'] <= cfg.gru_sil_max_frame_kl
        elif 'kl' in row:
            assert row['kl'] > cfg.gru_sil_target_kl or row['frame_kl_max'] > cfg.gru_sil_max_frame_kl
        row = dict(row, training_iteration=iteration, source_id=identity(archive.trajectories[ti]),
                   source_D=trajectory_endpoint(archive.trajectories[ti]).distance)
        chunks.append(row)
        audited_probabilities += 1
    assert sum(x['end']-x['start'] for x in rows) == metric['gru_sil_sampled_frames']
    assert sum(x['positive_frames'] for x in rows) == metric['gru_sil_positive_frames']
    assert sum(x['accepted'] for x in rows) == metric['gru_sil_updates']
    assert sum('kl' in x for x in rows) == metric['gru_sil_attempts']
    assert sum('kl' in x and not x['accepted'] for x in rows) == metric['gru_sil_rejected_steps']
    assert metric['gru_sil_attempts'] <= metric['gru_sil_budget'] <= min(cfg.gru_sil_max_updates, int(metric['n_updates'] * cfg.gru_sil_budget_ratio))
    assert metric['corridor_invalid_agents'] == 0 and metric['elite_archive_rejected_incomplete'] == 0

entries = []
for index, traj in enumerate(archive.trajectories):
    ep = trajectory_endpoint(traj)
    score, _ = score_corridor_episode(traj, route)
    entries.append(dict(index=index, source_iteration=traj.iteration, D=ep.distance, t=ep.cap+1,
                        frames=len(traj.actions), complete=archive._complete(traj), metrics=score))

def aggregate(rows):
    phases, positives, dynamics = Counter(), Counter(), Counter()
    for row in rows:
        phases.update(row['phase_counts']); positives.update(row['positive_by_phase'])
        dynamics.update({k: row['sampling_weight'][k] for k in ('core_frames','context_frames','free_flight_frames')})
    return dict(chunks=len(rows), frames=sum(x['end']-x['start'] for x in rows),
                positive_frames=sum(x['positive_frames'] for x in rows), phase_frames=dict(phases),
                positive_phase_frames=dict(positives), dynamics_frames=dict(dynamics),
                starts=dict(Counter('0_to_119' if x['start']<120 else '120_to_299' if x['start']<300 else '300_plus' for x in rows)),
                top_two_historical_trajectories=sum(x['source_id'] in top_two_ids for x in rows),
                farthest_trajectory_192_chunk=sum(x['source_id']==farthest_id and x['start']==192 for x in rows),
                source_iterations=dict(Counter(x['source_iteration'] for x in rows)))

accepted = [x for x in chunks if x['accepted']]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = dict(status='training_audit_complete_evaluation_pending', start_iteration=start_iteration, final_iteration=final_iteration,
              rounds=len(metrics), physical_steps=int(sum(x['rollout_environment_steps'] for x in metrics)),
              finished_at=json.loads((RUN/'launcher_state.json').read_text(encoding='utf-8-sig'))['finished_at_utc'],
              final_checkpoint_sha256=sha(RUN/f'checkpoints/ppo_iter_{final_iteration:04d}.pt'),
              runtime_checks=dict(checkpoint_zip_crc_validated=len(iterations), pool_snapshots_validated=len(iterations),
                                  finite_model_optimizer_tensors=finite_tensors, contract_violations=[],
                                  reproduced_sampling_draws=audited_probabilities,
                                  preupdate_ratio_max_error=max(x['preupdate_ratio_max_error'] for x in metrics),
                                  max_action_saturation_fraction=max(x['action_saturation_fraction'] for x in metrics)),
              sil=dict(budget=int(sum(x['gru_sil_budget'] for x in metrics)),
                       attempted=int(sum(x['gru_sil_attempts'] for x in metrics)), accepted=len(accepted),
                       rejected=int(sum(x['gru_sil_rejected_steps'] for x in metrics)),
                       accepted_max_mean_kl=max(x['kl'] for x in accepted),
                       accepted_max_frame_kl=max(x['frame_kl_max'] for x in accepted),
                       sampled=aggregate(chunks), accepted_samples=aggregate(accepted),
                       zero_positive_chunks=sum(x['positive_frames']==0 for x in chunks)),
              pool=dict(size=len(entries), max_D=max(x['D'] for x in entries), entries=entries,
                        new_entries=[x for x in entries if x['source_iteration']>start_iteration]),
              limitations=['Training buffers were not persisted: runtime preupdate audits and sampling RNG/probabilities are checked; no full post-hoc replay of every training log probability.',
                           'Positive frames are logged by old motion phase, not by new dynamics core/context; do not equate sampled core frames with positive-advantage core frames.'])
(OUT/'training_review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(OUT/'actual_sampling.json').write_text(json.dumps(chunks,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('pool',)},indent=2))
print(json.dumps({'pool_size':len(entries),'max_D':report['pool']['max_D'],'new_entries':report['pool']['new_entries']},indent=2))

"""Actual checkpoint/elite observation validation. No game or optimizer steps."""
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from tests.analysis.campaign.prepare_tree_climb import make_encoder
from training.config import TrainConfig
from training.elite_archive import EliteTrajectoryArchive
from training.experimental_policy import create_actor_critic
from training.gru_sil import sequence_predictions
from training.main_ppo import main as training_main
from training.reference_corridor import ReferenceCorridor,trajectory_endpoint
from training.sequence_cache import SILObservationCache

def main():
    torch.set_num_threads(4)
    run=ROOT/'runs/gru_sil_dynamics_1h2_20261004'
    checkpoint=run/'checkpoints/ppo_iter_0209.pt'
    checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=TrainConfig(**saved['config'])
    meta=json.loads((run/'run_meta.json').read_text())
    archive=EliteTrajectoryArchive(cfg,Path(__file__).parent/'unused',meta['reference_fingerprint'])
    archive.restore(run/'elites/full_episodes/elites_iter_0209.npz')
    traj=max(archive.trajectories,key=lambda t:trajectory_endpoint(t).distance)
    source_id=hashlib.sha256(traj.raw_states.tobytes()+traj.actions.tobytes()).hexdigest()
    reference_file=ROOT/f'runs/gru_sil_consistency_audit_20261004/elite_{source_id}_frames.npz'
    with np.load(reference_file) as z: reference={k:z[k] for k in z.files}
    encoder,_=make_encoder(cfg,ReferenceCorridor.from_file(cfg.reference_trajectory,cfg))
    cache=SILObservationCache()
    limit=cfg.gru_sil_cache_mb*1024**2
    cache.begin_round(encoder,archive.trajectories,limit)
    frames=cache.get(encoder,traj)
    cold=cache.metrics()
    assert cold['gru_sil_cache_misses']==1
    assert cache.bytes<=limit
    errors={k:float(np.max(np.abs(v-reference[k]))) for k,v in frames.items()}
    assert all(e==0 for e in errors.values())
    cache.end_round()
    started=time.perf_counter()
    cache.begin_round(encoder,list(reversed(archive.trajectories)),limit)
    with patch.object(encoder,'encode_sequence',side_effect=AssertionError('warm cache rebuilt')):
        warm_frames=cache.get(encoder,traj)
    assert warm_frames is frames and cache.stats['misses']==0
    warm_seconds=time.perf_counter()-started
    warm=cache.metrics()
    print('Actual 1201-frame input equality and cross-round reuse passed',flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=create_actor_critic(cfg).to(device).eval()
    model.load_state_dict(saved['model'])
    model.exploration_boost=saved.get('loop_state',{}).get('boost',0.)
    original=[p.clone() for p in model.parameters()]
    fresh_prediction=sequence_predictions(model,reference)
    cached_prediction=sequence_predictions(model,warm_frames)
    prediction_errors={name:float((a-b).abs().max()) for name,a,b in zip(
        ('mean','std','value'),fresh_prediction,cached_prediction)}
    assert all(e==0 for e in prediction_errors.values()),prediction_errors
    assert all(torch.equal(p,b) for p,b in zip(model.parameters(),original))
    stream=io.StringIO()
    with patch.object(sys,'argv',['main_ppo','--resume',str(checkpoint),'--run-dir',str(Path(__file__).parent/'dry_run'),
                                  '--dry-run','--torch-threads','4']),contextlib.redirect_stdout(stream):
        training_main()
    options=json.loads(stream.getvalue())
    assert options['gru_sil_cache_mb']==1024
    assert options['gru_sil_sampling_mode']=='quality_dynamics_v3'
    assert options['gru_sil_max_updates']==4 and options['gru_sil_budget_ratio']==.25
    assert options['training_seed']==7 and options['rng_resume_mode']=='restore'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checkpoint_hash
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/training').glob('*.py')}
    report=dict(status='passed',device=device,checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,
        source_id=source_id,frames=len(traj.raw_states),input_errors=errors,prediction_errors=prediction_errors,
        cold=cold,warm=warm,warm_round_total_seconds=warm_seconds,
        actual_resume_config=options,source_hashes=sources,
        limitations=['One real elite validated; full PPO/SIL update equivalence tested separately with deterministic fixture physics.',
                     'No game launch, formal checkpoint change or training budget increase.'])
    output=Path(__file__).with_name('actual_validation.json')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='passed',prediction_errors=prediction_errors,cold=cold,warm=warm,
                         warm_round_total_seconds=warm_seconds,output=str(output)),indent=2),flush=True)

if __name__=='__main__':main()

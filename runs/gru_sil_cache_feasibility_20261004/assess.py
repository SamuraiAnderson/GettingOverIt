"""Offline cost measurement and shadow LRU simulation; no training changes."""
import csv
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from statistics import median

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from tests.analysis.campaign.prepare_tree_climb import make_encoder
from training.config import TrainConfig
from training.elite_archive import EliteTrajectoryArchive
from training.reference_corridor import ReferenceCorridor, trajectory_endpoint

OUT = Path(__file__).resolve().parent
RUNS = [ROOT/'runs'/x for x in ('gru_sil_dynamics_1h_20261004', 'gru_sil_dynamics_1h2_20261004')]
INPUTS = ('raw_states','actions','contact','guidance_index','terminated')

def array_digest(h, name, value):
    h.update(name.encode())
    if value is None:
        h.update(b'none'); return
    array = np.ascontiguousarray(value)
    h.update(str((array.dtype.str, array.shape)).encode())
    h.update(memoryview(array).cast('B'))

def input_key(traj):
    h = hashlib.sha256(b'gru_sil_raw_inputs_v1')
    for name in INPUTS: array_digest(h,name,getattr(traj,name))
    return h.hexdigest()

def shadow(rounds, bytes_per_frame, limit_mib, clear_on_run):
    limit = limit_mib*2**20
    cache = OrderedDict()
    resident = peak = hits = misses = evictions = obsolete = 0
    previous_run = None
    for row in rounds:
        if clear_on_run and previous_run != row['run']:
            cache.clear(); resident=0
        previous_run = row['run']
        for key in list(cache):
            if key not in row['active']:
                resident -= cache.pop(key); obsolete += 1
        for key, frames in row['requests']:
            if key in cache:
                cache.move_to_end(key); hits += 1
                continue
            misses += 1
            size = frames*bytes_per_frame
            while cache and resident+size > limit:
                resident -= cache.popitem(last=False)[1]; evictions += 1
            if size <= limit:
                cache[key]=size; resident += size; peak=max(peak,resident)
        assert resident <= limit
    local_builds=sum(r['current_round_builds'] for r in rounds)
    return dict(limit_mib=limit_mib, clear_on_run=clear_on_run, accesses=hits+misses,
        hits=hits, builds=misses, current_round_local_builds=local_builds,
        avoided_builds=local_builds-misses, avoided_build_fraction=(local_builds-misses)/local_builds,
        peak_resident_mib=peak/2**20, evictions=evictions, obsolete_removals=obsolete)

def main():
    torch.set_num_threads(4)
    checkpoint=RUNS[-1]/'checkpoints/ppo_iter_0209.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=TrainConfig(**saved['config'])
    meta=json.loads((RUNS[-1]/'run_meta.json').read_text())
    archive=EliteTrajectoryArchive(cfg,OUT/'unused',meta['reference_fingerprint'])
    archive.restore(RUNS[-1]/'elites/full_episodes/elites_iter_0209.npz')
    final_pool=list(archive.trajectories)
    traj=max(final_pool,key=lambda t:trajectory_endpoint(t).distance)
    content_id=hashlib.sha256(traj.raw_states.tobytes()+traj.actions.tobytes()).hexdigest()
    old_frames=ROOT/f'runs/gru_sil_consistency_audit_20261004/elite_{content_id}_frames.npz'
    with np.load(old_frames) as z: reference={k:z[k] for k in z.files}
    sizes={k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.nbytes) for k,v in reference.items()}
    entry_bytes=sum(v.nbytes for v in reference.values())
    bytes_per_frame=entry_bytes//len(traj.raw_states)
    assert entry_bytes == bytes_per_frame*len(traj.raw_states)
    rounds=[]; performance={}
    for run in RUNS:
        rows=list(csv.DictReader((run/'metrics.csv').open()))
        performance[run.name]={name:sum(float(r[name]) for r in rows) for name in ('rollout_total_seconds','rollout_patch_seconds','update_and_sil_seconds')}
        for row in rows:
            iteration=int(row['iteration'])
            archive.restore(run/f'elites/full_episodes/elites_iter_{iteration:04d}.npz')
            active={input_key(t) for t in archive.trajectories}
            logs=json.loads((run/f'sil_sampling/iter_{iteration:04d}.json').read_text())['chunks']
            requests=[(input_key(archive.trajectories[r['trajectory']]),len(archive.trajectories[r['trajectory']].raw_states)) for r in logs]
            rounds.append(dict(run=run.name,iteration=iteration,active=active,requests=requests,
                               current_round_builds=len({r['trajectory'] for r in logs})))
    simulations=[]
    for clear in (True,False):
        for limit in (128,256,512,1024):
            simulations.append(shadow(rounds,bytes_per_frame,limit,clear))
    print(json.dumps(dict(entry_mib=entry_bytes/2**20, simulations=simulations),indent=2),flush=True)
    report=dict(status='benchmark_running', memory=dict(entry_bytes=entry_bytes,entry_mib=entry_bytes/2**20,
        bytes_per_frame=bytes_per_frame,fields=sizes,current_pool_mib=sum(len(t.raw_states)*bytes_per_frame for t in final_pool)/2**20,
        capacity_20_equal_length_mib=20*entry_bytes/2**20,system_total_gib=psutil.virtual_memory().total/2**30,
        available_gib_at_start=psutil.virtual_memory().available/2**30),
        simulations=simulations,historical_timings_seconds=performance,
        per_run_512=[dict(run=run.name,**shadow([r for r in rounds if r['run']==run.name],bytes_per_frame,512,True)) for run in RUNS],
        limitations=['Shadow simulation counts builds using recorded draws, not a live training benchmark.',
            'Single-trajectory build timing is current machine timing, not the historical cost of every sampled trajectory.',
            'Resident cache budget excludes temporary build buffers, current rollout, models and Python container overhead.',
            'Cache alone does not increase per-round SIL budget or prove better closed-loop behavior.'])
    path=OUT/'assessment.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    encoder,_=make_encoder(cfg,ReferenceCorridor.from_file(cfg.reference_trajectory,cfg))
    # Time the exact production encoder, including its existing static geometry cache.
    timings={name:0. for name in ('patch','sparse')}
    for name,method in [('patch','_build_patch'),('sparse','_build_sparse')]:
        original=getattr(encoder,method)
        def wrapped(*args,_name=name,_fn=original,**kwargs):
            start=time.perf_counter()
            try: return _fn(*args,**kwargs)
            finally: timings[_name]+=time.perf_counter()-start
        setattr(encoder,method,wrapped)
    start=time.perf_counter(); fresh=encoder.encode_sequence(traj); elapsed=time.perf_counter()-start
    assert all(np.array_equal(fresh[k],reference[k]) for k in fresh)
    cache={input_key(traj):fresh}
    times=[]
    for _ in range(300):
        start=time.perf_counter(); cached=cache[input_key(traj)]; times.append(time.perf_counter()-start)
        assert cached is fresh
    # Hash validity does not depend on ndarray identity or pool position.
    from dataclasses import replace
    assert input_key(replace(traj,**{k:getattr(traj,k).copy() for k in INPUTS})) == input_key(traj)
    invalidation={}
    for name in INPUTS:
        arr=getattr(traj,name).copy()
        arr.flat[0]=not arr.flat[0] if arr.dtype.kind=='b' else arr.flat[0]+1
        invalidation[name]=input_key(replace(traj,**{name:arr})) != input_key(traj)
    assert all(invalidation.values())
    report['benchmark']=dict(frames=len(traj.raw_states),D=trajectory_endpoint(traj).distance,
        production_encode_seconds=elapsed,patch_seconds=timings['patch'],sparse_seconds=timings['sparse'],
        hashed_ram_lookup_median_seconds=median(times),hashed_ram_lookup_p95_seconds=float(np.quantile(times,.95)),
        fresh_equals_previously_validated_inputs=True,content_key_mutation_checks=invalidation,
        schema_note='Prototype key covers trajectory inputs. Production integration also requires immutable/versioned observation context.')
    report['status']='completed'
    report['evidence']=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        observations_file=str(old_frames),observations_sha256=hashlib.sha256(old_frames.read_bytes()).hexdigest())
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='completed',benchmark=report['benchmark'],report=str(path)),indent=2),flush=True)

if __name__=='__main__': main()

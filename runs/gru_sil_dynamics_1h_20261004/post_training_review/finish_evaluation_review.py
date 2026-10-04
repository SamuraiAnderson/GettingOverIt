"""Summarize the matched frozen evaluation using existing motion diagnostics."""
import datetime
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from training.climb_diagnostics import classify_motion
OUT = Path(__file__).resolve().parent
BASE = ROOT/'runs/gru_sil_8h_20261003/post_training_review/checkpoint_0184'
FINAL = OUT/'checkpoint_0199'
review = json.loads((OUT/'training_review.json').read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def inspect(path):
    results = json.loads((path/'results.json').read_text())
    assert results['failure'] is None and not results['updates']
    assert len(results['episodes']) == 16 and len(results['rollouts']) == 2
    motion=[]
    for p in sorted(path.glob('*_000_agent*.json')):
        data=json.loads(p.read_text()); states=np.asarray(data['states']); contacts=np.asarray(data['contacts'])
        labels, diag=classify_motion(states, contacts)
        tree=(states[:,33]>=-33)&(states[:,33]<=-27)
        row=dict(episode=p.name, mode=data['metrics']['mode'],agent=data['metrics']['agent'],
                 phase_frames=diag['counts'], takeoffs=[x for x in diag['flights'] if x['takeoff_candidate']],
                 elevated_contact_runs=diag['elevated_contact_runs'], tree_state_count=int(tree.sum()),
                 final_anchor=states[-1,33:35].tolist(), max_anchor_y=float(states[:,34].max()),
                 max_anchor_y_in_tree=float(states[tree,34].max()) if tree.any() else None)
        motion.append(row)
    return results,motion

base,bmotion=inspect(BASE); final,fmotion=inspect(FINAL)
bm=json.loads((BASE/'manifest.json').read_text()); fm=json.loads((FINAL/'manifest.json').read_text())
checks={k:bm[k]==fm[k] for k in ('seed','criterion','offsets','geometry_sha256','contact_source','native_diagnostics_requested','initialization','reference')}
assert all(checks.values())
differences={k:[bm['config'].get(k),v] for k,v in fm['config'].items() if bm['config'].get(k)!=v}
assert set(differences) == {'gru_sil_sampling_mode','log_dir','checkpoint_dir'}
source_old=json.loads((ROOT/'runs/gru_sil_8h_20261003/launch_manifest.json').read_text())
source_checks={name:sha(ROOT/f'src/training/{name}.py')==source_old['source_sha256'][f'src/training/{name}.py']
               for name in ('actor_critic','experimental_policy','experimental_rollout','dataset','rollout','local_evaluation','policy_features','policy_audit')}
assert all(source_checks.values())
initial=[]
for p in FINAL.glob('*_000_agent*.json'):
    b=np.array(json.loads((BASE/p.name).read_text())['states'][0])
    f=np.array(json.loads(p.read_text())['states'][0])
    initial.append(dict(episode=p.name,max_abs_state_delta=float(np.max(np.abs(np.delete(f-b,28))))))
checkpoint=OUT.parent/'checkpoints/ppo_iter_0199.pt'
assert sha(checkpoint)==review['final_checkpoint_sha256']
saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
behavior=torch.load(FINAL/'behavior.pt',map_location='cpu',weights_only=False)
assert all(torch.equal(v,behavior['model'][k]) for k,v in saved['model'].items())
for r in final['rollouts']:
    assert np.isfinite(list(v for v in r.values() if isinstance(v,(float,int)))).all()
    assert r['action_saturation_fraction']==0 and r['preupdate_ratio_max_error']<1e-3
review['evaluation']=dict(protocol='Exact spawn, 8 agents x 1200 steps, seed 20261003; deterministic/exploration one trial each; frozen weights',
                         summaries={'184':base['summary'],'199':final['summary']},
                         audits=final['rollouts'],baseline_reuse_checks=checks,source_checks=source_checks,
                         config_differences=differences,physical_initial_state_checks=initial,
                         checkpoint_unchanged=True,behavior_model_identical=True,
                         baseline_reference_source_note='Only added optional guidance sigma; None takes the same old corridor radius. No scoring or policy-input change is active.',
                         motion={'184':bmotion,'199':fmotion},
                         motion_limitations='Geometric contacts and tree box x=[-33,-27] are proxies; takeoff candidates are not proof of deliberate jumping or branch gripping.')
review['status']='completed'
review['review_completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(OUT/'training_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
compact={}
for key,results,motion in [('184',base,bmotion),('199',final,fmotion)]:
    compact[key]={}
    for mode,summary in results['summary'].items():
        rows=[r for r in motion if r['mode']==mode]
        compact[key][mode]={k:summary[k] for k in ('cap_progress_median_m','cap_progress_max_m','terminal_progress_median_m','arrival_count','hold_success_count','max_retreat_after_arrival_median_m')}
        compact[key][mode].update(mean_outside_fraction=float(np.mean([r['outside_fraction'] for r in results['episodes'] if r['mode']==mode])),
                                 episodes_with_takeoff=sum(bool(r['takeoffs']) for r in rows),
                                 episodes_with_tree_takeoff=sum(any(x['tree_at_release'] for x in r['takeoffs']) for r in rows),
                                 episodes_with_elevated_tip_contact=sum(bool(r['elevated_contact_runs']) for r in rows),
                                 takeoff_count=sum(len(r['takeoffs']) for r in rows))
print(json.dumps({'comparison':compact,'initial_state_max_difference':max(r['max_abs_state_delta'] for r in initial),'final_audits':final['rollouts']},indent=2))

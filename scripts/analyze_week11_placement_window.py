"""Independent routing and matched-prefix audits; bind physical reconstruction."""
from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from week11_placement_window import ARMS, audit_window
from week11_replay_common import read, sha, write

KEYS = ('agent_rgb','wrist_rgb','proprioception','mujoco_state','executed_action')


def first_difference(a, b):
    n = min(len(a),len(b)); a,b = a[:n],b[:n]
    mask = np.any(a != b, axis=tuple(range(1,a.ndim))) if a.ndim > 1 else a != b
    ix = np.flatnonzero(mask)
    return int(ix[0]) if len(ix) else None


def compare(left, right, at=None):
    with h5py.File(left) as a, h5py.File(right) as b:
        x,y = a['steps'],b['steps']; lengths = [len(s['executed_action']) for s in (x,y)]
        n = min(lengths); stop = at if at is not None else n
        assert stop <= n
        eq, current, differences = {}, {}, {}
        for k in KEYS:
            xa,ya = x[k][:n],y[k][:n]
            differences[k] = first_difference(xa,ya)
            eq[k] = bool(np.array_equal(xa[:stop],ya[:stop]))
            if at is not None and k != 'executed_action':
                current[k] = bool(np.array_equal(xa[at],ya[at]))
            if k == 'proprioception':
                eq['model_proprioception_float32'] = bool(np.array_equal(xa[:stop].astype('f4'),ya[:stop].astype('f4')))
                if at is not None:
                    current['model_proprioception_float32'] = bool(np.array_equal(xa[at].astype('f4'),ya[at].astype('f4')))
        chunks = bool(np.array_equal(x['predicted_action_chunk'][10:stop:8],y['predicted_action_chunk'][10:stop:8]))
        intervention_chunks = (bool(np.array_equal(x['predicted_action_chunk'][at],y['predicted_action_chunk'][at]))
                               if at is not None else None)
        meta = [json.loads(f['episode/metadata_json'][()]) for f in (a,b)]
    queries = [[json.loads(l)['policy'] for l in Path(p).with_suffix('.queries.jsonl').read_text().splitlines()]
               for p in (left,right)]
    qi = (stop-10)//8 if at is not None else min(map(len,queries))
    def event_equal(i):
        l,r = [qs[i] for qs in queries]
        return (l['prediction'] == r['prediction'] and l['decision'] == r['decision']
                and l['event']['categorical_probabilities'] == r['event']['categorical_probabilities'])
    forecasts_before = all(event_equal(i) for i in range(qi))
    forecast_at = event_equal(qi) if at is not None else None
    exact = all(eq.values()) and all(current.values()) and chunks and forecasts_before
    if at is not None: exact &= forecast_at
    return {'comparison_frame':at, 'lengths':lengths, 'prior_frames_equal':eq,
        'input_at_intervention_equal':current, 'prior_chunks_equal':chunks,
        'prior_forecasts_equal':forecasts_before, 'intervention_forecast_equal':forecast_at,
        'intervention_chunks_equal':intervention_chunks, 'all_prefix_and_inputs_exact':bool(exact),
        'first_differences':differences, 'outcomes':[m['success'] for m in meta],
        'same_outcome':meta[0]['success'] == meta[1]['success'],
        'full_recorded_trajectory_exact':bool(at is None and lengths[0] == lengths[1] and exact)}


def run(raw, source, output):
    output.mkdir(parents=True,exist_ok=False)
    cfg = read(raw/'contract.json'); done = read(raw/'completion.json'); adm = read(raw/'admission.json')
    assert done['status'] == 'complete' and done['completed_rollouts'] == 12
    assert done['contract_digest'] == cfg['digest'] and done['model_states_unchanged']
    rows=[]; indexed={}; queries=0
    for entry in done['rows']:
        report=read(entry['report']); assert report['digest'] == entry['report_digest']
        row=report['result']; path=Path(row['artifact']); assert sha(path) == row['artifact_sha256']
        assert sha(row['timing_artifact']) == row['timing_sha256']
        audit=audit_window(row,entry['arm'],adm); assert audit == report['routing_audit']
        assert report['artifact_audit']['error'] is None and not any(row['query_exceptions'])
        with h5py.File(path) as f:
            meta=json.loads(f['episode/metadata_json'][()]); assert meta['success'] == row['success']
        queries+=audit['query_count']; indexed[entry['repeat'],entry['arm']]=str(path)
        rows.append({**entry,'routing_audit':audit,'policy_steps':row['policy_step_count'],
                     'retry_count':row['retry_count'],'termination_reason':row['termination_reason']})
    prefixes=[]; initials=[]; historical=[]; repeats=[]
    for rep in (1,2,3):
        for arm,at in [('placement-only',138),('late-only',218)]:
            prefixes.append({'repeat':rep,'arm':arm,
                **compare(indexed[rep,'fixed8'],indexed[rep,arm],at)})
        for a,b in itertools.combinations(ARMS,2):
            initials.append({'repeat':rep,'arms':[a,b],**compare(indexed[rep,a],indexed[rep,b],10)})
        for arm in ('fixed8','adaptive'):
            historical.append({'repeat':rep,'arm':arm,**compare(indexed[rep,arm],cfg['historical'][arm])})
    for arm in ARMS:
        for a,b in itertools.combinations((1,2,3),2):
            repeats.append({'arm':arm,'repeats':[a,b],**compare(indexed[a,arm],indexed[b,arm])})
    summary={'contract_digest':cfg['digest'],'completion_digest':done['digest'],'rows':rows,
        'successes':{a:sum(r['success'] for r in rows if r['arm']==a) for a in ARMS},
        'queries_audited':queries,'prefixes':prefixes,'initials':initials,
        'historical_controls':historical,'repeat_comparisons':repeats,
        'all_intervention_prefixes_exact':all(r['all_prefix_and_inputs_exact'] for r in prefixes),
        'all_initial_inputs_exact':all(r['all_prefix_and_inputs_exact'] for r in initials),
        'historical_control_outcomes_reproduced':sum(r['same_outcome'] for r in historical),
        'historical_controls_full_trajectory_exact':sum(r['full_recorded_trajectory_exact'] for r in historical),
        'within_condition_repeat_pairs_exact':sum(r['full_recorded_trajectory_exact'] for r in repeats),
        'same_seed_repeats':True,'population_estimate':False,'baseline_changed':False}
    write(output/'summary.json',summary)
    inventory={str(p.relative_to(raw)):{'sha256':sha(p),'bytes':p.stat().st_size}
               for p in sorted(raw.rglob('*')) if p.is_file()}
    write(output/'raw-inventory.json',{'root':str(raw),'files':inventory})
    physical=output/'physical-contract';physical.mkdir()
    write(physical/'contract.json',{'selection':{'pairs':[
        {'id':f'clean155-repeat-{rep}','artifacts':{a:indexed[rep,a] for a in ARMS}} for rep in (1,2,3)]},
        'input_sha256':{r['artifact']:sha(r['artifact']) for r in rows},'summary_digest':read(output/'summary.json')['digest']})
    template=read(source/'annotation-template.json');template.pop('digest')
    template.update(replay_contract_digest=read(physical/'contract.json')['digest'],
        scope='All twelve placement-window recordings. Frozen saved-state geometry, no integration, rendering or model calls.')
    for name,h in template['code_sha256'].items():
        assert sha(source/name)==h
        dest=physical/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dest)
    write(physical/'annotation-contract.json',template)
    write(output/'completion.json',{'status':'complete','summary_digest':read(output/'summary.json')['digest'],
        'inventory_digest':read(output/'raw-inventory.json')['digest'],'model_calls':0,'simulator_steps':0})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','prefixes','initials','historical_controls','repeat_comparisons')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('raw','source','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.raw,a.source,a.output)

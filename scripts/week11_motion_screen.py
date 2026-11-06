"""Recording audits and frozen progression rules for the motion-adapter screen."""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
from week11_replay_common import read, sha

from eventground_wam.eventground_horizon import select_horizon

CONDITIONS = ('h8-fixed8', 'horizon-motion', 'h8-information-control')


def audit_queries(row, condition, admission):
    path = Path(row['artifact'])
    timing = read(path.with_suffix('.timing.json'))
    query_path = path.with_suffix('.queries.jsonl')
    assert sha(query_path) == timing['query_artifact_sha256']
    queries = [json.loads(line) for line in query_path.read_text().splitlines()]
    assert queries == timing['queries'] and queries
    counts = {str(h): 0 for h in (2, 4, 8, 16, 32)}
    for i, q in enumerate(queries):
        p = q['policy']
        assert q['query_index'] == i and q['chunk_start'] == i*8
        assert q['exception'] is p['exception'] is None
        assert p['branch_identities'] == admission
        decision = select_horizon(p['prediction'])
        assert p['decision'] == asdict(decision)
        horizon = 8 if condition == CONDITIONS[0] else decision.selected_horizon
        assert p['used_horizon'] == horizon
        assert p['proposal_actions'] == p['execute_actions'] == 8
        assert p['zero_padded_actions'] == 24
        assert p['future_calls'] == 1+int(condition != CONDITIONS[0] and horizon != 8)
        if condition == CONDITIONS[0]:
            assert p['arm'] == 'M4-EG-fixed8'
        else:
            assert p['adapter_arm'] == p['arm'] == condition
        for key, values in p['event']['categorical_probabilities'].items():
            assert len(values) == (4 if key == 'event' else 32)
            assert np.isfinite(values).all() and min(values) >= 0 and abs(sum(values)-1) < 1e-5
        counts[str(horizon)] += 1
    return {'passed': True, 'queries': len(queries), 'horizon_counts': counts}


def prefix_pair(left, right):
    """Audit consumed query inputs and physics through the first changed executed action.

    Images on non-query frames are retained but do not enter this consumed-input
    test. A changed unused tail of a predicted chunk is not an intervention.
    """
    with h5py.File(left) as a, h5py.File(right) as b:
        metadata = [json.loads(f['episode/metadata_json'][()]) for f in (a, b)]
        scene_equal = all(metadata[0][k] == metadata[1][k] for k in
                          ('bddl_sha256', 'initial_state_sha256', 'instruction', 'episode_seed'))
        x, y = a['steps'], b['steps']
        lengths = [len(t['executed_action']) for t in (x, y)]
        n = min(lengths)
        assert n > 10
        differences = np.flatnonzero(np.any(x['executed_action'][:n] != y['executed_action'][:n], axis=1))
        changed = int(differences[0]) if len(differences) else None
        stop = changed if changed is not None else n-1
        frames = np.arange(10, stop+1, 8)
        consumed = {k: bool(np.array_equal(x[k][frames], y[k][frames]))
                    for k in ('agent_rgb', 'wrist_rgb', 'proprioception')}
        physics = {k: bool(np.array_equal(x[k][:stop+1], y[k][:stop+1]))
                   for k in ('mujoco_state', 'proprioception')}
        initial = {k: bool(np.array_equal(x[k][10], y[k][10]))
                   for k in ('agent_rgb', 'wrist_rgb', 'proprioception', 'mujoco_state')}
        prior = bool(np.array_equal(x['executed_action'][:stop], y['executed_action'][:stop]))
    queries = [[json.loads(line)['policy'] for line in Path(p).with_suffix('.queries.jsonl').read_text().splitlines()]
               for p in (left, right)]
    forecasts = all(all(queries[0][i][k] == queries[1][i][k] for k in ('prediction', 'decision'))
                    and queries[0][i]['event']['categorical_probabilities'] ==
                    queries[1][i]['event']['categorical_probabilities'] for i in range(len(frames)))
    matched = scene_equal and all(initial.values()) and all(consumed.values()) and all(physics.values()) and prior and forecasts
    if changed is None:
        matched &= lengths[0] == lengths[1]
    return {'first_changed_executed_frame': changed, 'checked_through_frame': stop,
            'query_frames': frames.tolist(), 'initial_equal': initial, 'consumed_inputs_equal': consumed,
            'physics_equal': physics, 'prior_executed_actions_equal': prior, 'forecasts_equal': forecasts,
            'lengths': lengths, 'scene_equal': scene_equal, 'causally_matched': bool(matched)}


def decision(cases, rows, pairs, *, integrity_valid):
    """No population inference; unstable H8 controls or decisive mismatches cannot pass."""
    assert len(rows) == len(cases)*9
    indexed = {(r['case'], r['repeat'], r['condition']): r for r in rows}
    assert len(indexed) == len(rows)
    match = {(p['case'], p['repeat'], tuple(p['conditions'])): p['causally_matched'] for p in pairs}
    unstable, unstable_adapter_control, losses, rescues, unmatched = [], [], [], [], []
    per_case = []
    for c in cases:
        key = c['id']
        outcomes = {arm: [indexed[key, rep, arm]['success'] for rep in (1, 2, 3)] for arm in CONDITIONS}
        fixed, main = outcomes[CONDITIONS[0]], outcomes[CONDITIONS[1]]
        matched = all(match[key, rep, CONDITIONS[:2]] for rep in (1, 2, 3))
        if len(set(fixed)) != 1:
            unstable.append(key)
        if len(set(outcomes[CONDITIONS[2]])) != 1:
            unstable_adapter_control.append(key)
        if all(fixed) and not all(main):
            losses.append(key)
        if not any(fixed) and all(main):
            rescues.append(key)
        if (all(fixed) or not any(fixed) and all(main)) and not matched:
            unmatched.append(key)
        per_case.append({'case': key, 'outcomes': outcomes, 'main_fixed_all_repeats_matched': matched})
    status = ('invalid' if not integrity_valid else 'inconclusive' if unstable or unstable_adapter_control or unmatched else
              'failed' if losses or not rescues else 'passed')
    return {'status': status, 'advance_to_27_cases': status == 'passed',
            'unstable_h8_cases': unstable, 'h8_success_cases_lost': losses,
            'unstable_adapter_control_cases': unstable_adapter_control,
            'consistent_rescue_candidates': rescues, 'unmatched_decisive_cases': unmatched,
            'successes': {a: sum(r['success'] for r in rows if r['condition'] == a) for a in CONDITIONS},
            'per_case': per_case, 'baseline_promotion': False,
            'qualification': 'Selected nine-case same-seed reproducibility screen; not a population estimate.'}


def all_pairs(cases, rows):
    indexed = {(r['case'], r['repeat'], r['condition']): r for r in rows}
    result = []
    for case in cases:
        for rep in (1, 2, 3):
            for arms in itertools.combinations(CONDITIONS, 2):
                left, right = [indexed[case['id'], rep, a]['artifact'] for a in arms]
                result.append({'case': case['id'], 'repeat': rep, 'conditions': list(arms),
                               **prefix_pair(left, right)})
    return result

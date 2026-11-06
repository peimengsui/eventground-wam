"""Outcome-blind panel selection and paired analysis for fixed-H8 validation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

CONDITIONS = ('vanilla-m4', 'h8-continuation')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
SALT = 'week11-h8-validation-v1-20261031'


def rank(kind, value):
    return hashlib.sha256(f'{SALT}|{kind}|{value}'.encode()).hexdigest()


def select_cases(manifests, excluded):
    """Eight distinct canonical tasks per suite/panel, one remaining state each."""
    result = []
    for panel in ('targeted-467', 'clean-200'):
        tasks = manifests[panel].get('tasks', manifests[panel].get('pairs'))
        for suite in SUITES:
            groups = {}
            for ordinal, task in enumerate(tasks):
                key = f'{panel}-{ordinal:04d}'
                if task['suite'] != suite or key in excluded:
                    continue
                canonical = task['task_name'].split('_view_')[0]
                groups.setdefault(canonical, []).append({'id': key, 'panel': panel, 'ordinal': ordinal,
                    'canonical_task': canonical, 'task': task})
            assert len(groups) >= 8, (panel, suite, len(groups))
            chosen = sorted(groups, key=lambda name: rank('task', f'{panel}|{suite}|{name}'))[:8]
            for canonical in chosen:
                result.append(min(groups[canonical], key=lambda c: rank('state', c['id'])))
    assert len(result) == 64 and len({c['id'] for c in result}) == 64
    assert not ({c['id'] for c in result} & set(excluded))
    return result


def paired_summary(rows):
    assert rows and all(type(r[a]) is bool for r in rows for a in CONDITIONS)
    vanilla = np.array([r[CONDITIONS[0]] for r in rows], dtype=int)
    h8 = np.array([r[CONDITIONS[1]] for r in rows], dtype=int)
    delta = h8-vanilla
    wins, losses = int((delta == 1).sum()), int((delta == -1).sum())
    n = wins+losses
    pvalue = min(1., 2*sum(math.comb(n, k) for k in range(min(wins, losses)+1))/2**n) if n else 1.
    rng = np.random.Generator(np.random.PCG64(20261031)); bootstrap = np.zeros(10000)
    for suite in sorted({r['suite'] for r in rows}):
        values = delta[[r['suite'] == suite for r in rows]]
        bootstrap += values[rng.integers(0, len(values), size=(10000, len(values)))].sum(axis=1)
    return {'cases': len(rows), 'vanilla_successes': int(vanilla.sum()), 'h8_successes': int(h8.sum()),
            'h8_only_success': wins, 'vanilla_only_success': losses, 'net_successes': wins-losses,
            'gap_percentage_points': float(delta.mean()*100),
            'paired_95_bootstrap_percentage_points': np.quantile(bootstrap/len(rows)*100, [.025, .975]).tolist(),
            'exact_two_sided_mcnemar_p': pvalue, 'bootstrap_draws': 10000,
            'bootstrap_unit': 'one distinct canonical task per case within each panel, stratified by suite',
            'qualification': 'Exploratory selected-panel uncertainty, conditional on one training seed.'}


def prefix_pair(left, right):
    with h5py.File(left) as a, h5py.File(right) as b:
        x, y = a['steps'], b['steps']
        metadata = [json.loads(f['episode/metadata_json'][()]) for f in (a, b)]
        scene = all(metadata[0][k] == metadata[1][k] for k in
                    ('bddl_sha256', 'initial_state_sha256', 'instruction', 'episode_seed'))
        lengths = [len(z['executed_action']) for z in (x, y)]; n = min(lengths)
        assert n > 10
        changed = np.flatnonzero(np.any(x['executed_action'][:n] != y['executed_action'][:n], axis=1))
        first = int(changed[0]) if len(changed) else None
        stop = first if first is not None else n-1
        frames = np.arange(10, stop+1, 8)
        initial = {k: bool(np.array_equal(x[k][10], y[k][10]))
                   for k in ('agent_rgb', 'wrist_rgb', 'proprioception', 'mujoco_state')}
        consumed = {k: bool(np.array_equal(x[k][frames], y[k][frames]))
                    for k in ('agent_rgb', 'wrist_rgb', 'proprioception')}
        physics = {k: bool(np.array_equal(x[k][:stop+1], y[k][:stop+1])) for k in ('mujoco_state', 'proprioception')}
        prior = bool(np.array_equal(x['executed_action'][:stop], y['executed_action'][:stop]))
    no_event = all(json.loads(line)['policy']['event'] is None
                   for p in (left, right) for line in Path(p).with_suffix('.queries.jsonl').read_text().splitlines())
    matched = scene and all(initial.values()) and all(consumed.values()) and all(physics.values()) and prior and no_event
    if first is None:
        matched &= lengths[0] == lengths[1]
    return {'first_changed_executed_frame': first, 'scene_equal': scene, 'initial_inputs_equal': initial,
            'consumed_inputs_equal': consumed, 'physics_equal': physics, 'prior_actions_equal': prior,
            'event_branch_absent': no_event, 'lengths': lengths, 'causally_matched': bool(matched)}

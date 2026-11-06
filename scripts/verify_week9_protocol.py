#!/usr/bin/env python3
"""Verify the immutable Week 9 freeze and actual panel topology without a GPU."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_protocol import load_evaluation_protocol
from eventground_wam.evaluation_splits import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIGEST = '9f2de901e4eaeb27a15710014be4fdb71a196e126d9ab45c5e33f84649c5f179'


def verify(root=ROOT):
    contract = read_sealed(root / 'configs/evaluation/week9_factorial_v1.json')
    require(contract['digest'] == CONTRACT_DIGEST, 'Week 9 freeze changed; use a versioned amendment')
    for name, digest in contract['source_files_sha256'].items():
        require(sha256_file(root / name) == digest, f'frozen Week 9 input changed: {name}')
    panels = contract['panels']
    plus = json.loads((root / panels['perturbed']['path']).read_text())
    dev = json.loads((root / panels['development_overlap']['path']).read_text())
    protocol = load_evaluation_protocol(root / panels['perturbed']['protocol'])
    for manifest in (plus, dev):
        validate_manifest(manifest, protocol)
    keys = lambda rows: {(t['suite'], t['runtime_task_index']) for t in rows}
    require(len(keys(plus['tasks'])) == 1000 and len(keys(dev['tasks'])) == 210, 'panel counts changed')
    require(keys(dev['tasks']) <= keys(plus['tasks']), 'development overlap changed')
    indexed = {(t['suite'], t['runtime_task_index']): t for t in plus['tasks']}
    require(all(indexed[(t['suite'], t['runtime_task_index'])] == t for t in dev['tasks']),
            'overlapping task records or seeds changed')
    clean = read_sealed(root / panels['clean']['path'])
    require(clean['digest'] == panels['clean']['manifest_digest'] and len(clean['pairs']) == 2000,
            'clean panel changed')
    load_evaluation_protocol(root / panels['clean']['protocol'])
    clusters = defaultdict(list)
    for pair in clean['pairs']:
        clusters[(pair['suite'], pair['runtime_task_index'])].append(pair)
    require(len(clusters) == 40 and len({p['pair_id'] for p in clean['pairs']}) == 2000,
            'clean pair identities changed')
    require(all(len(pairs) == 50 and {p['initial_state_index'] for p in pairs} == set(range(50))
                and {p['trial_index'] for p in pairs} == set(range(50)) for pairs in clusters.values()),
            'clean task clusters or states changed')
    return seal_contract({'artifact_type': 'eventground_wam.week9_protocol_verification',
        'contract_digest': contract['digest'], 'passed': True, 'perturbed_tasks': 1000,
        'development_overlap': 210, 'additional_tasks': 790, 'clean_tasks': 40,
        'states_per_clean_task': 50, 'clean_trials': 2000, 'gpu_model_loaded': False,
        'evaluations_launched': False})


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True))

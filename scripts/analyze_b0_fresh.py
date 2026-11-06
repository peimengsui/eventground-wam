#!/usr/bin/env python3
"""Verify paired dev artifacts and apply the prospectively frozen v2 readiness rule."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import (
    read_episode_metadata,
    sha256_array,
    validate_episode_file,
)
from eventground_wam.evaluation_protocol import load_evaluation_protocol

ROOT = Path(__file__).resolve().parents[1]


def paired_dev_interval(differences, strata):
    rng = np.random.Generator(np.random.PCG64(20260922))
    draws = np.zeros(20000, dtype=np.int64)
    for label in sorted(set(strata)):
        values = np.asarray([v for v, s in zip(differences, strata, strict=True) if s == label], dtype=np.int64)
        draws += values[rng.integers(0, len(values), size=(20000, len(values)))].sum(axis=1)
    return (np.quantile(draws, [.025, .975], method='linear') * 100 / len(differences)).tolist()


def readiness(clean_rate, suite_floor, gap_pp, interval):
    low, high = interval
    if clean_rate < .80 or suite_floor < .70:
        return 'BELOW_TARGET'
    if clean_rate >= .85 and low >= -5:
        return 'READY'
    if gap_pp >= 5 and low > 0:
        return 'READY'
    if (clean_rate >= .85 and high >= -5) or (gap_pp >= 5 and high > 0):
        return 'INCONCLUSIVE'
    return 'BELOW_TARGET'


def checked_dev(path, row, identity, task, protocol):
    if not row['artifact'] or sha256_file(path) != row['artifact_sha256']:
        raise ValueError('missing or changed dev artifact')
    summary = validate_episode_file(path, protocol)
    meta = read_episode_metadata(path)
    if bool(summary['success']) != row['success']:
        raise ValueError('dev outcome differs from recorder')
    for key in ('checkpoint_id', 'checkpoint_revision', 'project_git_revision'):
        if meta[key] != identity[key]:
            raise ValueError(f'dev {key} changed')
    for key in ('suite', 'runtime_task_index', 'episode_seed', 'perturbation_category'):
        if meta[key] != task[key]:
            raise ValueError(f'dev task {key} changed')
    with h5py.File(path) as handle:
        steps = handle['steps']
        queries = np.flatnonzero(np.isfinite(steps['policy_latency_ms'][:]))
        if not len(queries):
            raise ValueError('no policy query in dev episode')
        inputs = {f'{i}:{key}': sha256_array(steps[key][i])
                  for i in (0, int(queries[0])) for key in ('agent_rgb', 'wrist_rgb', 'proprioception')}
    return {key: meta[key] for key in ('bddl_sha256', 'initial_state_sha256', 'instruction', 'episode_seed')}, inputs


def analyze(work):
    rule = read_sealed(ROOT / 'configs/evaluation/week4_fresh_v2_acceptance.json')
    clean = read_sealed(work / 'epochs/epoch-02/rollouts-clean-2000/evaluation-report.json')
    clean_identity = read_sealed(work / 'epochs/epoch-02/rollouts-clean-2000/identity.json')
    if not clean['all_pairs_valid'] or clean['completed'] != 2000 or clean_identity['fresh_acceptance_rule_digest'] != rule['digest']:
        raise ValueError('incomplete clean comparison')
    reports, identities = {}, {}
    for policy in ('b0', 'reference'):
        root = work / 'dev-210' / policy
        reports[policy] = read_sealed(root / 'evaluation-report.json')
        identities[policy] = read_sealed(root / 'identity.json')
        if (reports[policy]['episode_count'] != 210 or reports[policy]['identity_digest'] != identities[policy]['digest']
                or identities[policy]['acceptance_rule_digest'] != rule['digest']):
            raise ValueError('incomplete or changed dev report')
    tasks = json.loads((ROOT / rule['dev_panel']).read_text())['tasks']
    protocol = load_evaluation_protocol()
    differences, strata, evidence = [], [], []
    for i, task in enumerate(tasks):
        checked = []
        for policy in ('b0', 'reference'):
            row = reports[policy]['episodes'][i]
            checked.append(checked_dev(Path(row['artifact']), row, identities[policy], task, protocol))
        if checked[0] != checked[1]:
            raise ValueError(f'dev inputs or scene identity differ at ordinal {i}')
        left, right = [reports[p]['episodes'][i]['success'] for p in ('b0', 'reference')]
        differences.append(int(left) - int(right))
        strata.append((task['suite'], task['perturbation_category']))
        evidence.append({'ordinal': i, 'b0_success': left, 'reference_success': right,
                         'matched_scene_and_first_inputs': True})
    gap = 100 * sum(differences) / 210
    interval = paired_dev_interval(differences, strata)
    panel = read_sealed(ROOT / rule['clean_panel'])
    suites = defaultdict(list)
    for pair, check in zip(panel['pairs'], clean['comparisons']['openvla'], strict=True):
        suites[pair['suite']].append(check['b0_success'])
    rates = {key: sum(v) / len(v) for key, v in suites.items()}
    clean_rate = clean['counts']['fresh_v2'] / 2000
    result = seal_contract({
        'artifact_type': 'eventground_wam.fresh_v2_completion', 'status': 'complete',
        'acceptance_rule_digest': rule['digest'], 'checkpoint_sha256': clean['checkpoint_sha256'],
        'clean_report_digest': clean['digest'], 'dev_report_digests': {p: r['digest'] for p, r in reports.items()},
        'clean_success_rate': clean_rate, 'clean_suite_rates': rates, 'dev_gap_pp': gap,
        'dev_paired_stratified_bootstrap_95_pp': interval, 'dev_pairs': evidence,
        'dev_counts': {p: r['success_count'] for p, r in reports.items()},
        'readiness': readiness(clean_rate, min(rates.values()), gap, interval),
        'historical_clean_statistics': clean['historical_clean_statistics']['openvla'],
        'scope': 'Fixed development benchmarks; not unseen tasks or training-seed uncertainty. Historical decisions unchanged.',
    })
    atomic_json(work / 'completion.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'dev_pairs'}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        analyze(args.work_root)
    except (ValueError, FileNotFoundError) as error:
        # The service already reserves 78 for permanent integrity failures.
        # Preserve the frozen inputs and report the stop separately from completion.
        atomic_json(args.work_root / 'analysis-status.json', seal_contract({
            'artifact_type': 'eventground_wam.fresh_v2_analysis_stop',
            'status': 'integrity_stop',
            'acceptance': 'NOT_ASSESSED',
            'error_type': type(error).__name__,
            'error': str(error),
            'exit_code': 78,
            'analysis_script_sha256': sha256_file(Path(__file__)),
        }))
        print(f'Integrity stop: {error}', file=sys.stderr, flush=True)
        raise SystemExit(78) from error


if __name__ == '__main__':
    main()

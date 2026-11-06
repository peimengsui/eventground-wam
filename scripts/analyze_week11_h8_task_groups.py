"""Supplementary base-task dependence check; does not change the frozen analysis."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from week11_replay_common import read, sha, write


def group_cases(contract, clean_manifest):
    names = defaultdict(set)
    for task in clean_manifest['pairs']:
        names[task['suite']].add(task['task_name'])
    groups = {}
    for case in contract['cases']:
        name = case['task']['task_name']
        matches = [base for base in names[case['task']['suite']]
                   if name == base or name.startswith(base+'_')]
        assert len(matches) == 1, (name, matches)
        groups[case['id']] = {'panel': case['panel'], 'suite': case['task']['suite'], 'base_task': matches[0]}
    return groups


def summarize(groups, pairs):
    panels = {}
    for panel in ('targeted-467', 'clean-200'):
        suites = defaultdict(lambda: defaultdict(list))
        for pair in pairs:
            if pair['panel'] != panel:
                continue
            group = groups[pair['case']]
            suites[group['suite']][group['base_task']].append(
                int(pair['h8-continuation'])-int(pair['vanilla-m4']))
        rng = np.random.Generator(np.random.PCG64(20261031))
        cluster_boot = np.zeros(10000)
        equal_task = []
        for suite in sorted(suites):
            values = list(suites[suite].values())
            totals = np.array([sum(x) for x in values])
            sizes = np.array([len(x) for x in values])
            draws = rng.integers(0, len(values), (10000, len(values)))
            # Resample whole task clusters, retain all selected variants, then
            # preserve the original equal weight of the four suites.
            cluster_boot += totals[draws].sum(axis=1)/sizes[draws].sum(axis=1)/4
            equal_task.append(float(np.mean([np.mean(x) for x in values])))
        panels[panel] = {
            'base_tasks': sum(len(x) for x in suites.values()),
            'base_tasks_per_suite': {s: len(x) for s, x in suites.items()},
            'suite_stratified_task_cluster_bootstrap_95_percentage_points':
                np.quantile(cluster_boot*100, [.025, .975]).tolist(),
            'suite_balanced_equal_base_task_gap_percentage_points': float(np.mean(equal_task)*100),
            'base_task_differences': dict(suites),
        }
    return panels


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--clean-manifest', type=Path, required=True)
    p.add_argument('--analysis', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    contract, clean = read(a.contract), read(a.clean_manifest)
    groups = group_cases(contract, clean)
    result = {'contract_digest': contract['digest'], 'clean_manifest_sha256': sha(a.clean_manifest),
        'groups': groups, 'source_sha256': sha(__file__),
        'qualification': 'Supplementary dependence analysis; frozen selection, outcomes and decision rules unchanged.',
        'method': '10,000 PCG64 seed 20261031 draws of whole base-task clusters within suite; suites weighted equally.'}
    if a.analysis:
        analysis = read(a.analysis)
        result.update(analysis_digest=analysis['digest'], panels=summarize(groups, analysis['pairs']))
    write(a.output, result)

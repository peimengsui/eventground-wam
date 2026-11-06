"""Artifact-backed input, latency and paired-outcome summaries for completed panels."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
from eventground_wam.auxiliary_evaluation import artifact_inputs, require, paired_summary
from eventground_wam.demonstration_contracts import seal_contract


def latency_summary(episodes):
    groups = {'all_queries': [], 'first_queries': [], 'subsequent_queries': [], 'episode_medians': []}
    missing = 0
    for values in episodes:
        values = np.asarray(values, dtype=float)
        require(values.ndim == 1 and np.isfinite(values).all() and (values >= 0).all(), 'invalid latencies')
        if not len(values):
            missing += 1
            continue
        groups['all_queries'].extend(values.tolist())
        groups['first_queries'].append(float(values[0]))
        groups['subsequent_queries'].extend(values[1:].tolist())
        groups['episode_medians'].append(float(np.median(values)))
    return {**{key: {'count': len(values), 'median_ms': float(np.median(values)) if values else None,
                     'p95_ms': float(np.quantile(values, .95, method='linear')) if values else None}
               for key, values in groups.items()}, 'episodes_without_recorded_queries': missing,
            'unit': 'milliseconds_per_8_action_query', 'warmup_or_outlier_queries_removed': False}


def audit_panel(rows, identity, tasks, protocol):
    require(len(rows) == len(tasks), 'incomplete panel matrix')
    audits = []
    for ordinal, (row, task) in enumerate(zip(rows, tasks, strict=True)):
        require(type(row['success']) is bool, 'outcome is not binary')
        require(all(row[k] == task[k] for k in ('suite', 'runtime_task_index',
                    'perturbation_category', 'difficulty_level')), 'task order changed')
        inputs, error = None, None
        try:
            inputs = artifact_inputs(Path(row['artifact']) if row['artifact'] else Path('/missing'),
                                     row, identity, task, protocol)
            if inputs is None:
                error = 'no first policy query; pairing unverified'
            with h5py.File(row['artifact']) as artifact:
                latency = artifact['steps/policy_latency_ms'][:]
                values = latency[np.isfinite(latency)].tolist()
                require(values == row.get('recorded_action_query_latencies_ms', row['policy_latencies_ms']),
                        'latency differs from HDF5')
                if 'cohort_digest' in identity:
                    from eventground_wam.episode_recorder import read_episode_metadata
                    meta = read_episode_metadata(Path(row['artifact']))
                    for key in ('cohort_digest', 'week9_protocol_digest', 'policy_objective', 'admission_digest'):
                        require(meta[key] == identity[key], f'Week 9 binding changed: {key}')
                    from eventground_wam.b0_training import sha256_file, read_sealed
                    require(sha256_file(Path(row['timing_artifact'])) == row['timing_sha256'], 'timing artifact changed')
                    timing = read_sealed(Path(row['timing_artifact']))
                    require(timing['cohort_digest'] == identity['cohort_digest'] and
                            timing['checkpoint_revision'] == identity['checkpoint_revision'], 'timing identity changed')
                    require([q['latency_ms'] for q in timing['queries']] == row['policy_latencies_ms'] and
                            [q['latency_ms'] for q in timing['queries'] if q['exception'] is None] == values,
                            'full query timing differs from sidecar or HDF5')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            error = str(exc)
        audits.append({'ordinal': ordinal, 'inputs': inputs,
                       'input_digest': seal_contract(inputs)['digest'] if inputs else None, 'error': error})
    return audits


def summarize_rows(rows):
    from evaluate_reference_policy import wilson_interval
    successes = sum(r['success'] for r in rows)
    groups = {}
    for field in ('suite', 'perturbation_category', 'difficulty_level', 'termination_reason'):
        values = defaultdict(list)
        for row in rows:
            values[str(row[field])].append(row['success'])
        groups[field] = {key: {'count': len(v), 'successes': sum(v), 'success_rate': sum(v)/len(v)}
                         for key, v in values.items()}
    return {'count': len(rows), 'successes': successes, 'success_rate': successes/len(rows),
            'wilson_95_descriptive': wilson_interval(successes, len(rows)), 'breakdowns': groups,
            'episode_wall_seconds': {'available_count': sum(r.get('episode_wall_time_seconds') is not None for r in rows),
                'sum_available': sum(r.get('episode_wall_time_seconds') or 0 for r in rows)},
            'latency': latency_summary([r['policy_latencies_ms'] for r in rows])}


def compare_inputs(left, right):
    require(len(left) == len(right), 'audit matrix length changed')
    pairs = []
    for a, b in zip(left, right, strict=True):
        require(a['ordinal'] == b['ordinal'], 'audit order changed')
        match = not a['error'] and not b['error'] and a['inputs'] == b['inputs']
        fields = []
        if a['inputs'] and b['inputs']:
            fields = [key for key in a['inputs']['prefix']
                      if a['inputs']['prefix'][key] != b['inputs']['prefix'].get(key)]
            fields += ['scene:'+key for key in a['inputs']['scene']
                       if a['inputs']['scene'][key] != b['inputs']['scene'].get(key)]
        pairs.append({'ordinal': a['ordinal'], 'exact_match': bool(match),
                      'different_fields': fields, 'left_error': a['error'], 'right_error': b['error']})
    return {'exact_count': sum(p['exact_match'] for p in pairs), 'count': len(pairs),
            'all_exact': all(p['exact_match'] for p in pairs), 'pairs': pairs,
            'qualification': 'exact canonical inputs' if all(p['exact_match'] for p in pairs)
                             else 'whole-panel contrast descriptive'}


def paired_effect(left, right, *, tasks=None, clean=False):
    if not clean:
        return paired_summary(left, right)
    a, b = np.asarray(left, dtype=int), np.asarray(right, dtype=int)
    require(a.shape == b.shape and len(a) == len(tasks) and np.isin(a, [0, 1]).all()
            and np.isin(b, [0, 1]).all(), 'invalid clean outcome matrix')
    clusters = defaultdict(list)
    for i, task in enumerate(tasks):
        clusters[(task['suite'], task['runtime_task_index'])].append(i)
    require(len(clusters) == 40 and all(len(v) == 50 for v in clusters.values()), 'clean clusters changed')
    delta = a-b
    cluster_delta = np.array([delta[v].mean() for v in clusters.values()])
    indices = np.random.Generator(np.random.PCG64(8)).integers(0, 40, (10000, 40))
    return {'trial_count': len(a), 'cluster_count': 40, 'gap_percentage_points': float(delta.mean()*100),
            'paired_95_interval_percentage_points': np.quantile(cluster_delta[indices].mean(1)*100,
                [.025, .975], method='linear').tolist(),
            'left_only_success': int((delta == 1).sum()), 'right_only_success': int((delta == -1).sum()),
            'both_success': int(((a == 1) & (b == 1)).sum()), 'both_failure': int(((a == 0) & (b == 0)).sum()),
            'bootstrap': {'draws': 10000, 'rng': 'PCG64', 'seed': 8, 'unit': 'canonical_task_50_states'}}

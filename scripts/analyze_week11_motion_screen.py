"""Independently re-audit retained motion-screen recordings, including an early stop."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from week11_motion_screen import CONDITIONS, audit_queries, prefix_pair
from week11_replay_common import read, sha, write


def run(root, output):
    import run_week11_formal as formal
    from week11_multihorizon_runtime import ROOT

    from eventground_wam.evaluation_comparisons import audit_panel, latency_summary

    cfg = read(root/'bundle/contract.json'); screen = root/'screen'
    completion = read(screen/'completion.json')
    assert completion['contract_digest'] == cfg['digest']
    for p, h in cfg['code_sha256'].items():
        assert sha(root/'bundle'/p) == h, p
    for p, h in cfg['input_sha256'].items():
        assert sha(p) == h, p
    preflight = read(screen/'preflight.json'); assert preflight['passed']
    admissions = read(screen/'admissions.json')
    inputs = {p: formal.inventory(read(ROOT/'configs/evaluation/week11_integration_v1.json'), p)
              for p in ('targeted-467', 'clean-200')}
    checked, rows = {}, completion['rows']
    assert len(rows) == completion['completed_rollouts']
    for item, row in zip(cfg['schedule'], rows, strict=False):
        case = cfg['cases'][item['case_index']]
        assert row['case'] == case['id'] and row['condition'] == item['condition'] and row['repeat'] == item['repeat']
        report = read(row['report']); assert report['digest'] == row['report_digest']
        identity = read(Path(row['report']).parent/'identity.json')
        task = inputs[case['panel']][0][case['ordinal']]; protocol = inputs[case['panel']][1]
        audit = audit_panel([report['result']], identity, [task], protocol)[0]
        assert audit['error'] is None and audit == report['artifact_audit']
        assert audit_queries(report['result'], row['condition'], admissions[row['condition']]) == report['query_audit']
        assert row['success'] == report['result']['success'] and row['artifact'] == report['result']['artifact']
        assert row['policy_latencies_ms'] == report['result']['policy_latencies_ms']
        checked[row['report']] = sha(row['report'])
        checked[row['artifact']] = sha(row['artifact'])
        for suffix in ('.timing.json', '.queries.jsonl'):
            p = Path(row['artifact']).with_suffix(suffix); checked[str(p)] = sha(p)
    lat = read(screen/'latency.json')
    for arm, summary in lat['summary'].items():
        values = [r['ms'] for r in lat['rows'] if r['condition'] == arm]
        assert summary == {'queries': len(values), 'median_ms': float(np.median(values)),
                           'p95_ms': float(np.percentile(values, 95))}
    cumulative = {a: latency_summary([r['policy_latencies_ms'] for r in rows if r['condition'] == a]) for a in CONDITIONS}
    recorded_lat = read(screen/f'rollout-latency-{len(rows):02d}.json')
    assert recorded_lat['summary'] == cumulative
    ratios = {a: cumulative[a]['all_queries']['p95_ms']/lat['summary']['original-fixed']['p95_ms'] for a in CONDITIONS[1:]}
    assert ratios == recorded_lat['ratios']
    indexed = {(r['case'], r['repeat'], r['condition']): r for r in rows}
    pairs, per_case = [], []
    for c in cfg['cases']:
        outcomes = {a: [] for a in CONDITIONS}
        for rep in (1, 2, 3):
            for a in CONDITIONS:
                if (c['id'], rep, a) in indexed:
                    outcomes[a].append(indexed[c['id'], rep, a]['success'])
            for a, b in itertools.combinations(CONDITIONS, 2):
                if (c['id'], rep, a) in indexed and (c['id'], rep, b) in indexed:
                    pairs.append({'case': c['id'], 'repeat': rep, 'conditions': [a, b],
                        **prefix_pair(indexed[c['id'], rep, a]['artifact'], indexed[c['id'], rep, b]['artifact'])})
        per_case.append({'case': c['id'], 'outcomes': outcomes})
    receipt = {'artifact_type': 'eventground_wam.week11_motion_screen_independent_audit',
        'contract_digest': cfg['digest'], 'completion_digest': completion['digest'],
        'recording_integrity_passed': True, 'completed_episodes': len(rows),
        'successes': {a: sum(r['success'] for r in rows if r['condition'] == a) for a in CONDITIONS},
        'per_case': per_case, 'pairs': pairs, 'unmatched_pairs': sum(not p['causally_matched'] for p in pairs),
        'latency': cumulative, 'latency_ratios_to_preflight_original_fixed': ratios,
        'latency_gate_passed': recorded_lat['passed'], 'preflight_passed': True,
        'checkpoint_files_and_frozen_source_unchanged': True,
        'end_of_rollout_in_memory_weight_hashes_available': completion.get('model_states_unchanged', False),
        'input_sha256': checked, 'analysis_source_sha256': sha(__file__),
        'status': 'stopped_latency' if not recorded_lat['passed'] else completion['status'],
        'retention_rescue_gate': 'not evaluable: three repeats incomplete' if len(rows) < 81 else 'see complete analysis',
        'new_rollouts': 0, 'model_calls': 0, 'simulator_steps': 0,
        'qualification': 'Retained partial outcomes are descriptive; no replacement or progression.'}
    write(output, receipt)
    print({k: v for k, v in receipt.items() if k in ('status', 'completed_episodes', 'successes', 'unmatched_pairs', 'recording_integrity_passed')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); run(a.root, a.output)

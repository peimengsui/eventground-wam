"""Recording-backed independent audit of the completed fixed-H8 validation."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from week11_h8_validation import CONDITIONS, paired_summary, prefix_pair, select_cases
from week11_replay_common import read, sha, write


def run(root, output):
    import run_week11_formal as formal
    from week11_multihorizon_runtime import ROOT, admission

    from eventground_wam.evaluation_comparisons import audit_panel, latency_summary
    from eventground_wam.week11_execution import validate_query_artifact

    cfg = read(root/'bundle/contract.json'); dest = root/'validation'
    done = read(dest/'completion.json'); analysis = read(dest/'analysis.json')
    assert done['status'] == 'complete' and done['completed_rollouts'] == 128
    assert done['models_unchanged'] and done['analysis_digest'] == analysis['digest']
    assert done['contract_digest'] == cfg['digest']
    for p, h in cfg['code_sha256'].items():
        assert sha(root/'bundle'/p) == h, p
    for p, h in cfg['input_sha256'].items():
        assert sha(p) == h, p
    assert admission()[1]['digest'] == cfg['training_source_digest']
    manifests = {p: read(ROOT/e['path']) for p, e in cfg['panel_manifests'].items()}
    assert select_cases(manifests, set(cfg['excluded_case_ids'])) == cfg['cases']
    inputs = {p: formal.inventory(read(ROOT/'configs/evaluation/week11_integration_v1.json'), p) for p in manifests}
    admissions = read(dest/'admissions.json'); checked = {}; retries = 0
    for item, row in zip(cfg['schedule'], done['rows'], strict=True):
        case = cfg['cases'][item['case_index']]
        assert row['case'] == case['id'] and row['condition'] == item['condition']
        report = read(row['report']); assert report['digest'] == row['report_digest']
        identity = read(Path(row['report']).parent/'identity.json')
        task = inputs[case['panel']][0][case['ordinal']]; protocol = inputs[case['panel']][1]
        actual = audit_panel([report['result']], identity, [task], protocol)[0]
        assert actual['error'] is None and actual == report['artifact_audit']
        assert row['success'] == report['result']['success'] and row['artifact'] == report['result']['artifact']
        assert row['policy_latencies_ms'] == report['result']['policy_latencies_ms']
        assert validate_query_artifact(report['result'], 'M4-fixed', admissions[row['condition']])
        assert not any(report['result']['query_exceptions'])
        retries += report['result']['retry_count']
        for p in (Path(row['report']), Path(row['artifact']), Path(row['artifact']).with_suffix('.timing.json'),
                  Path(row['artifact']).with_suffix('.queries.jsonl')):
            checked[str(p)] = sha(p)
    indexed = {(r['case'], r['condition']): r for r in done['rows']}; pairs = []
    for case in cfg['cases']:
        a, b = [indexed[case['id'], arm] for arm in CONDITIONS]
        pairs.append({'case': case['id'], 'panel': case['panel'], 'suite': case['task']['suite'],
            CONDITIONS[0]: a['success'], CONDITIONS[1]: b['success'], **prefix_pair(a['artifact'], b['artifact'])})
    assert pairs == analysis['pairs']
    for panel in manifests:
        selected = [p for p in pairs if p['panel'] == panel]; matched = [p for p in selected if p['causally_matched']]
        assert paired_summary(selected) == analysis['panels'][panel]['all_outcomes']
        assert (paired_summary(matched) if matched else None) == analysis['panels'][panel]['matched_only']
    latency = read(dest/'latency.json')
    for a, summary in latency['summary'].items():
        values = [r['ms'] for r in latency['rows'] if r['condition'] == a]
        assert summary == {'queries': len(values), 'median_ms': float(np.median(values)), 'p95_ms': float(np.percentile(values, 95))}
    rollout_lat = {a: latency_summary([r['policy_latencies_ms'] for r in done['rows'] if r['condition'] == a]) for a in CONDITIONS}
    assert rollout_lat == read(dest/'latency-128.json')['summary']
    targeted, clean = [analysis['panels'][p]['all_outcomes'] for p in ('targeted-467', 'clean-200')]
    directional = all(p['causally_matched'] for p in pairs) and targeted['net_successes'] >= 2 and clean['net_successes'] >= 0 and latency['h8_to_vanilla_p95_ratio'] <= 1.1
    assert analysis['directional_support'] == done['directional_support'] == directional
    stronger = directional and targeted['paired_95_bootstrap_percentage_points'][0] > 0 and targeted['exact_two_sided_mcnemar_p'] < .05
    assert analysis['stronger_targeted_statistical_support'] == stronger
    result = {'status': 'pass', 'contract_digest': cfg['digest'], 'completion_digest': done['digest'],
        'analysis_digest': analysis['digest'], 'episodes_verified': 128, 'cases': 64, 'retries': retries,
        'pairs_matched': sum(p['causally_matched'] for p in pairs), 'panels': analysis['panels'],
        'latency_summary': latency['summary'], 'rollout_latency': rollout_lat,
        'directional_support': directional, 'stronger_targeted_statistical_support': stronger,
        'event_off_numerical_parity_passed': read(dest/'recorded-parity.json')['passed'],
        'checkpoint_files_and_source_unchanged': True, 'no_event_branch_in_recorded_queries': True,
        'input_sha256': checked, 'analysis_source_sha256': sha(__file__),
        'model_calls': 0, 'new_rollouts': 0, 'simulator_steps': 0, 'baseline_promotion': False}
    write(output, result)
    print({k: result[k] for k in ('status', 'episodes_verified', 'pairs_matched', 'directional_support')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); run(a.root, a.output)

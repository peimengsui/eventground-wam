"""Consolidate retained evidence with CPU-only numerical cross-checks.

No model/runtime imports, remote access, checkpoint selection or rollout calls.
Run with Python, NumPy and --output pointing to a new reporting directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {}
CHECKS = []
ARMS = ['B0', 'M1', 'M2', 'M3', 'M4']
W9 = 'docs/evaluation/week9-final-v1/'
W10 = 'docs/evaluation/week10-sft-continuation-report-v1/'
W11 = 'docs/evaluation/week11-closeout-v1/'
H8 = 'docs/evaluation/week11-h8-validation-v1/'


def source(path):
    p = ROOT / path
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p


def read(path):
    return json.loads(source(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def same(actual, expected, label):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, err_msg=label)
    CHECKS.append(label)


def interval(values):
    return np.quantile(values, [.025, .975], method='linear').tolist()


def timing(values, published, label):
    same([len(values), np.median(values), np.percentile(values, 95)],
         [published.get('count', published.get('queries')), published['median_ms'], published['p95_ms']], label)


def load_week9():
    a = read(W9 + 'analysis.json')
    out = {'panels': {}, 'bootstrap': a['bootstrap'], 'robustness_ratios': a['robustness_ratios'],
           'source': W9 + 'analysis.json', 'cost': read(W9 + 'cost-ledger.json')}
    weights = {'M1_minus_B0': [-1, 1, 0, 0, 0], 'M2_minus_B0': [-1, 0, 1, 0, 0],
               'M3_minus_B0': [-1, 0, 0, 1, 0], 'M4_minus_B0': [-1, 0, 0, 0, 1],
               'M2_minus_M1': [0, -1, 1, 0, 0], 'M3_minus_M1': [0, -1, 0, 1, 0],
               'M4_minus_M2': [0, 0, -1, 0, 1], 'M4_minus_M3': [0, 0, 0, -1, 1],
               'latent_minus_pixel_main_effect': [0, -.5, .5, -.5, .5],
               'coupling_main_effect': [0, -.5, -.5, .5, .5], 'interaction': [0, 1, -1, -1, 1]}
    draws_by_panel = {}
    for panel, seed, n in [('perturbed', 7, 1000), ('clean', 8, 2000)]:
        values, units, summaries = [], [], {}
        reference_tasks = None
        for arm in ARMS:
            prefix = 'cohort' if arm == 'M2' else 'comparison-context'
            base = f'outputs/week9-m2-v1-metadata/{prefix}/panels/{arm}/{panel}/'
            r = read(base + 'evaluation-report.json')
            tasks = read(base + 'tasks.json')['tasks']
            assert r['digest'] == a['panels'][panel]['report_digests'][arm]
            assert r['status'] == 'audited' and all(x['error'] is None for x in r['audits'])
            if reference_tasks is None:
                reference_tasks = tasks
            assert tasks == reference_tasks, f'Unpaired Week 9 tasks: {arm}/{panel}'
            v = np.array([int(x['success']) for x in r['episodes']])
            assert len(v) == len(tasks) == n and np.isin(v, [0, 1]).all()
            s = a['panels'][panel]['summaries'][arm]
            same([v.sum(), v.mean()], [s['successes'], s['success_rate']], f'w9 {arm}/{panel} outcomes')
            timing([t for row in r['episodes'] for t in row['policy_latencies_ms']],
                   s['latency']['all_queries'], f'w9 {arm}/{panel} timing')
            values.append(v)
            if panel == 'clean':
                groups = defaultdict(list)
                for i, task in enumerate(tasks):
                    groups[task['suite'], task['runtime_task_index']].append(i)
                assert len(groups) == 40 and all(len(g) == 50 for g in groups.values())
                units.append(np.array([v[g].mean() for g in groups.values()]))
            else:
                units.append(v)
            summaries[arm] = {k: s[k] for k in ['count', 'successes', 'success_rate', 'latency', 'breakdowns']}
            if 'development_overlap' in s:
                summaries[arm]['development_overlap'] = {k: {key: x[key] for key in ['count', 'successes', 'success_rate']}
                                                        for k, x in s['development_overlap'].items()}
        x = np.stack(units, axis=1)
        idx = np.random.Generator(np.random.PCG64(seed)).integers(0, len(x), (10000, len(x)))
        sampled = np.stack([x[:, i][idx].mean(axis=1) for i in range(5)], axis=1)
        draws_by_panel[panel] = sampled
        fact = a['panels'][panel]['factorial']
        for i, arm in enumerate(ARMS):
            same([x[:, i].mean(), *interval(sampled[:, i])],
                 [fact['success_rates'][arm]['estimate'], *fact['success_rates'][arm]['percentile_95']],
                 f'w9 {panel}/{arm} rate bootstrap')
        for name, w in weights.items():
            c = fact['contrasts_percentage_points'][name]
            same([float(x.mean(axis=0) @ w) * 100, *interval(sampled @ w * 100)],
                 [c['estimate'], *c['percentile_95']], f'w9 {panel}/{name} paired bootstrap')
        comparisons = {}
        for name, c in a['panels'][panel]['comparisons'].items():
            left, right = name.split('-')
            d = values[ARMS.index(left)] - values[ARMS.index(right)]
            comparisons[name] = {'exact_inputs': c['inputs']['all_exact'],
                                 'exact_prefixes': c['inputs']['exact_count'],
                                 'n': n, 'left_only_success': int((d == 1).sum()),
                                 'right_only_success': int((d == -1).sum())}
        out['panels'][panel] = {'summaries': summaries, 'factorial': fact, 'input_audits': comparisons}
    for i, arm in enumerate(ARMS):
        expected = a['robustness_ratios'][arm]
        same(interval(draws_by_panel['perturbed'][:, i] / draws_by_panel['clean'][:, i]),
             expected['percentile_95'], f'w9 {arm} robustness-ratio bootstrap')
    return out


def prediction_records(base):
    c = read(base + '/completion.json')
    p = source(base + '/predictions.jsonl')
    assert hashlib.sha256(p.read_bytes()).hexdigest() == c['predictions_sha256']
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(rows) == c['metrics']['n'] and len({r['prompt_id'] for r in rows}) == len(rows)
    for r in rows:
        t, pred = r['target'], r['prediction']
        match = t['next_event'] == pred['next_event']
        assert match == r['correct_event']
        if t['next_event'] != 'none':
            if match:
                lo, hi = t['interval']; pl, ph = pred['interval']
                iou = max(0, min(hi, ph) - max(lo, pl)) / (max(hi, ph) - min(lo, pl))
            else:
                iou = 0.
            same(iou, r['positive_iou'], 'event typed IoU ' + r['prompt_id'])
    m = c['metrics']
    macro = np.mean([np.mean([r['positive_iou'] for r in rows if r['target']['next_event'] == k])
                     for k in ['contact', 'grasp', 'gripper_close']])
    none = [r for r in rows if r['target']['next_event'] == 'none']
    same(macro, m['macro_positive_iou'], base + ' macro IoU')
    fp = sum(not r['correct_event'] for r in none)
    same(fp / len(none), m['none_false_positive_rate'], base + ' none false positives')
    positives = [r for r in rows if r['target']['next_event'] != 'none']
    matched = [r for r in positives if r['correct_event']]
    same(len(matched) / len(positives), m['matched_positive_coverage'], base + ' positive coverage')
    same(np.mean([r['start_error'] for r in matched]), m['matched_positive_start_mae'], base + ' matched onset MAE')
    return rows, {**m, 'source_count': len({r['source_id'] for r in rows}),
                  'none_false_positives': fp, 'none_count': len(none), 'source': base}


def source_bootstrap(first, second):
    assert [(r['prompt_id'], r['source_id'], r['target']) for r in first] == [
        (r['prompt_id'], r['source_id'], r['target']) for r in second]
    sources = sorted({r['source_id'] for r in first})
    kinds = ['contact', 'grasp', 'gripper_close']
    nums = np.zeros((len(sources), 3)); dens = np.zeros_like(nums)
    for a, b in zip(first, second):
        if a['target']['next_event'] != 'none':
            i, j = sources.index(a['source_id']), kinds.index(a['target']['next_event'])
            nums[i, j] += b['positive_iou'] - a['positive_iou']; dens[i, j] += 1
    indices = np.random.Generator(np.random.PCG64(10)).integers(0, len(sources), (10000, len(sources)))
    den = dens[indices].sum(axis=1); valid = (den != 0).all(axis=1)
    vals = (nums[indices].sum(axis=1)[valid] / den[valid]).mean(axis=1)
    return interval(vals), int((~valid).sum())


def load_week10():
    result = read(W10 + 'analysis/results.json')
    retrospective = read(W10 + 'analysis/retrospective.json')
    dev = {}; records = {}
    for key, base in [('sft', 'week10-future-onset-v1'), ('primary', 'week10-future-onset-v1'),
                      ('calibration', 'week10-future-onset-v1'), ('hybrid', 'week10-preservation-pilot-v1'),
                      ('control', 'week10-preservation-pilot-v1')]:
        records[key], dev[key] = prediction_records(f'docs/evaluation/{base}/eval/dev/{key}')
    for row in retrospective['history']:
        if row['dev_n'] == 396:
            assert any(math.isclose(row['metrics']['macro_positive_iou'], v['macro_positive_iou'], abs_tol=1e-12)
                       for v in dev.values())
    first, before = prediction_records('docs/evaluation/week10-future-onset-v1/eval/report/sft')
    second, after = prediction_records(W10.rstrip('/') + '/eval/report/control')
    ci, undefined = source_bootstrap(first, second)
    same(ci, result['bootstrap']['interval95'], 'w10 report paired source bootstrap')
    assert undefined == result['bootstrap']['undefined_draws'] == 0
    same(after['macro_positive_iou'] - before['macro_positive_iou'], result['delta_iou'], 'w10 report delta')
    preservation = read('docs/evaluation/week10-preservation-pilot-v1/analysis/results.json')
    for name, (left, right) in {'control_minus_sft': ('control', 'sft'),
                               'hybrid_minus_sft': ('hybrid', 'sft'),
                               'hybrid_minus_control': ('hybrid', 'control')}.items():
        ci, undefined = source_bootstrap(records[right], records[left])
        same(ci, preservation['comparisons'][name]['interval95'], 'w10 dev paired source bootstrap ' + name)
        assert undefined == preservation['comparisons'][name]['undefined_draws']
    return {'dev': dev, 'report': {'sft': before, 'control': after}, 'report_comparison': result,
            'dev_comparison_bootstraps': preservation['comparisons'],
            'historical_original_task': [r for r in retrospective['history'] if r['dev_n'] != 396]}


def load_week11():
    a = read(W11 + 'remote/analysis.json')
    matrix = read(W11 + 'remote/paired-matrices.json')
    for panel in ['targeted-467', 'clean-200']:
        m = matrix[panel]; v = np.array(m['outcomes'], dtype=int)
        assert m['arms'] == ['M4-fixed', 'M4-EG-fixed8', 'M4-EG-adaptive']
        assert v.shape == (len(m['tasks']), 3) and np.isin(v, [0, 1]).all()
        units = v
        if panel == 'clean-200':
            groups = defaultdict(list)
            for i, t in enumerate(m['tasks']):
                groups[t['suite'], t['runtime_task_index']].append(i)
            assert len(groups) == 40 and all(len(g) == 5 for g in groups.values())
            units = np.array([v[g].mean(axis=0) for g in groups.values()])
        idx = np.random.Generator(np.random.PCG64(8 if panel == 'clean-200' else 7)).integers(0, len(units), (10000, len(units)))
        for left, right in [(2, 0), (2, 1), (1, 0)]:
            key = m['arms'][left] + ' minus ' + m['arms'][right]
            s = a['statistics'][panel][key]; delta = v[:, left] - v[:, right]
            same([delta.mean() * 100, *interval((units[:, left] - units[:, right])[idx].mean(axis=1) * 100)],
                 [s['difference_pp'], *s['paired_95_ci_pp']], f'w11 {panel}/{key} bootstrap')
            same([sum(delta == 1), sum(delta == -1)], [s['left_only_success'], s['right_only_success']], f'w11 {panel}/{key} discordance')
        for i, arm in enumerate(m['arms']):
            r = read(W11 + f'remote/panels/{arm}/{panel}.json')
            assert [int(x['success']) for x in r['episodes']] == v[:, i].tolist()
            s = a['summaries'][arm + '/' + panel]
            same([v[:, i].sum(), len(v)], [s['successes'], s['count']], f'w11 {arm}/{panel} counts')
            timing([t for row in r['episodes'] for t in row['policy_latencies_ms']], s['latency']['all_queries'], f'w11 {arm}/{panel} timing')
    return {k: a[k] for k in ['summaries', 'statistics', 'control_summary', 'decision', 'latency_gates', 'formal_episode_count']}


def load_followups():
    h8 = read(H8 + 'remote/validation/analysis.json')
    sensitivity = read(H8 + 'base-task-sensitivity.json')
    lat = read(H8 + 'remote/validation/latency.json')
    for arm, s in lat['summary'].items():
        timing([r['ms'] for r in lat['rows'] if r['condition'] == arm], s, 'h8 matched-input timing ' + arm)
    for panel, s in h8['panels'].items():
        rows = [r for r in h8['pairs'] if r['panel'] == panel]
        d = np.array([int(r['h8-continuation']) - int(r['vanilla-m4']) for r in rows])
        expected = s['all_outcomes']
        same([d.mean() * 100, sum(d == 1), sum(d == -1)],
             [expected['gap_percentage_points'], expected['h8_only_success'], expected['vanilla_only_success']], 'h8 counts ' + panel)
        rng = np.random.Generator(np.random.PCG64(20261031)); boot = np.zeros(10000)
        for suite in sorted({r['suite'] for r in rows}):
            values = d[[r['suite'] == suite for r in rows]]
            boot += values[rng.integers(0, len(values), (10000, len(values)))].sum(axis=1)
        same(interval(boot / len(rows) * 100), expected['paired_95_bootstrap_percentage_points'], 'h8 frozen bootstrap ' + panel)
        suites = defaultdict(lambda: defaultdict(list))
        for r in rows:
            g = sensitivity['groups'][r['case']]
            suites[g['suite']][g['base_task']].append(int(r['h8-continuation']) - int(r['vanilla-m4']))
        rng = np.random.Generator(np.random.PCG64(20261031)); boot = np.zeros(10000); equal = []
        for suite in sorted(suites):
            values = list(suites[suite].values())
            totals = np.array([sum(x) for x in values]); sizes = np.array([len(x) for x in values])
            idx = rng.integers(0, len(values), (10000, len(values)))
            boot += totals[idx].sum(axis=1) / sizes[idx].sum(axis=1) / 4
            equal.append(np.mean([np.mean(x) for x in values]))
        pub = sensitivity['panels'][panel]
        same(interval(boot * 100), pub['suite_stratified_task_cluster_bootstrap_95_percentage_points'], 'h8 task-cluster bootstrap ' + panel)
        same(np.mean(equal) * 100, pub['suite_balanced_equal_base_task_gap_percentage_points'], 'h8 equal-task sensitivity ' + panel)
    multi = read('docs/evaluation/week11-multihorizon-results-v1/summary.json')
    multi_raw = read('docs/evaluation/week11-multihorizon-results-v1/remote/screen/analysis.json')
    assert multi_raw['decision'] == multi['decision']
    for arm, count in multi['decision']['counts'].items():
        same(sum(r['outcomes'][arm] for r in multi_raw['per_case']), count, 'continuation counts ' + arm)
    for name, c in multi['all_pairwise_contrasts'].items():
        left, right = name.split(' minus ')
        wins = [r['id'] for r in multi_raw['per_case'] if r['outcomes'][left] and not r['outcomes'][right]]
        losses = [r['id'] for r in multi_raw['per_case'] if r['outcomes'][right] and not r['outcomes'][left]]
        assert wins == c['wins'] and losses == c['losses']
        CHECKS.append('continuation paired wins/losses ' + name)
    adapter = read('docs/evaluation/week11-motion-screen-v1/remote/screen/completion.json')
    status = read('docs/evaluation/week11-motion-screen-v1/status.json')
    assert len(adapter['rows']) == status['completed_episodes'] == 27
    counts = {arm: {'n': sum(r['condition'] == arm for r in adapter['rows']),
                    'successes': sum(r['success'] for r in adapter['rows'] if r['condition'] == arm)}
              for arm in sorted({r['condition'] for r in adapter['rows']})}
    adapter_latency = read('docs/evaluation/week11-motion-screen-v1/remote/screen/rollout-latency-27.json')
    for arm in counts:
        timing([t for r in adapter['rows'] if r['condition'] == arm for t in r['policy_latencies_ms']],
               adapter_latency['summary'][arm]['all_queries'], 'adapter screen timing ' + arm)
    return {'h8': h8, 'h8_task_sensitivity': sensitivity, 'h8_matched_latency': lat['summary'],
            'continuation': multi, 'adapter': {'status': status, 'counts': counts, 'latency': adapter_latency}}


def export_tables(data, out):
    tables = {}
    rates, contrasts, latency, costs, event = [], [], [], [], []
    for panel, p in data['week9']['panels'].items():
        for arm, s in p['summaries'].items():
            lo, hi = p['factorial']['success_rates'][arm]['percentile_95']
            rates.append(dict(study='Week 9', panel=panel, arm=arm, n=s['count'], successes=s['successes'],
                              rate_percent=s['success_rate'] * 100, ci_low_percent=lo * 100, ci_high_percent=hi * 100,
                              interval_unit=p['factorial']['bootstrap']['unit'], evidence='planned; single seed; reused panels', source=W9+'analysis.json'))
            latency.append(dict(study='Week 9', panel=panel, arm=arm, **s['latency']['all_queries'],
                                protocol='all rollout queries; includes first and failed; no removals', source=W9+'analysis.json'))
        for name, c in p['factorial']['contrasts_percentage_points'].items():
            audit = p['input_audits'].get(name.replace('_minus_', '-'))
            contrasts.append(dict(study='Week 9', panel=panel, contrast=name, estimate_pp=c['estimate'],
                                  ci_low_pp=c['percentile_95'][0], ci_high_pp=c['percentile_95'][1],
                                  unit=p['factorial']['bootstrap']['unit'],
                                  evidence='exact initial inputs; descriptive' if audit and audit['exact_inputs'] else 'descriptive; contributing RGB mismatches',
                                  source=W9+'analysis.json'))
    for key, s in data['week11']['summaries'].items():
        arm, panel = key.split('/')
        rates.append(dict(study='Week 11 formal', panel=panel, arm=arm, n=s['count'], successes=s['successes'],
                          rate_percent=s['success_rate'] * 100, ci_low_percent=None, ci_high_percent=None,
                          interval_unit='See paired contrast table; do not use naive episode CI for clean',
                          evidence='frozen protocol; reused panels; single seed', source=W11+'remote/analysis.json'))
        latency.append(dict(study='Week 11 formal', panel=panel, arm=arm, **s['latency']['all_queries'],
                            protocol='all rollout queries; includes first and failed; no removals', source=W11+'remote/analysis.json'))
    for panel, cs in data['week11']['statistics'].items():
        for name, c in cs.items():
            contrasts.append(dict(study='Week 11 formal', panel=panel, contrast=name, estimate_pp=c['difference_pp'],
                                  ci_low_pp=c['paired_95_ci_pp'][0], ci_high_pp=c['paired_95_ci_pp'][1], unit=c['bootstrap']['unit'],
                                  evidence='frozen protocol; descriptive conditional panel interval', source=W11+'remote/analysis.json'))
    cost = data['week9']['cost']
    for arm in ARMS:
        costs.append(dict(arm=arm, hardware=cost['training_hardware'][arm], active_hours=cost['training_active_hours'][arm],
                          scope=cost['training_time_scope'], source=W9+'cost-ledger.json'))
    for kind, c in cost['original_shared_CPT'].items():
        costs.append(dict(arm='shared '+kind+' CPT', hardware='RTX 5090', active_hours=c['hours'],
                          scope='Count once per module, shared across two policy arms', source=W9+'cost-ledger.json'))
    labels = {'sft': 'Starting SFT', 'primary': 'RLVR primary', 'calibration': 'RLVR calibration',
              'hybrid': 'RLVR + supervised preservation', 'control': 'SFT-only continuation'}
    for panel in ['dev', 'report']:
        for key, m in data['week10'][panel].items():
            event.append(dict(panel=panel, arm=labels[key], prompts=m['n'], sources=m['source_count'],
                              macro_positive_iou_percent=m['macro_positive_iou'] * 100,
                              none_false_positives=m['none_false_positives'], none_count=m['none_count'],
                              matched_onset_mae_frames=m['matched_positive_start_mae'],
                              matched_positive_coverage_percent=m['matched_positive_coverage'] * 100,
                              evidence='exploratory automatic weak labels; prior panel exposure', source=m['source']))
    r = data['week10']['report_comparison']
    contrasts.append(dict(study='Week 10', panel='report', contrast='SFT-only continuation minus starting SFT',
                          estimate_pp=r['delta_iou'] * 100, ci_low_pp=r['bootstrap']['interval95'][0] * 100,
                          ci_high_pp=r['bootstrap']['interval95'][1] * 100, unit='physical source; 80 sources',
                          evidence='exploratory automatic weak labels', source=W10+'analysis/results.json'))
    for panel, s in data['followups']['h8']['panels'].items():
        p = s['all_outcomes']; grouped = data['followups']['h8_task_sensitivity']['panels'][panel]
        for arm, key in [('Vanilla M4', 'vanilla_successes'), ('H8 continuation', 'h8_successes')]:
            rates.append(dict(study='Week 11 fresh H8 follow-up', panel=panel+' subset', arm=arm, n=p['cases'],
                              successes=p[key], rate_percent=100*p[key]/p['cases'], ci_low_percent=None, ci_high_percent=None,
                              interval_unit='See paired contrast table', evidence='exploratory; existing formal pool', source=H8+'remote/validation/analysis.json'))
        for label, ci, unit in [('frozen variant analysis',p['paired_95_bootstrap_percentage_points'],'suite-stratified variants'),
                                ('task-cluster sensitivity',grouped['suite_stratified_task_cluster_bootstrap_95_percentage_points'],'base-task clusters within suite')]:
            contrasts.append(dict(study='Week 11 fresh H8 follow-up', panel=panel+' subset',
                                  contrast='H8 minus vanilla; '+label, estimate_pp=p['gap_percentage_points'],
                                  ci_low_pp=ci[0], ci_high_pp=ci[1], unit=unit,
                                  evidence=f'exploratory; {grouped["base_tasks"]} base tasks', source=H8+('base-task-sensitivity.json' if 'sensitivity' in label else 'remote/validation/analysis.json')))
    for arm, s in data['followups']['h8_matched_latency'].items():
        latency.append(dict(study='Week 11 H8 matched inputs', panel='108 observations × 3 passes', arm=arm,
                            count=s['queries'], median_ms=s['median_ms'], p95_ms=s['p95_ms'],
                            protocol='after warmup; rotating order; no outlier removal; separate from rollout timing', source=H8+'remote/validation/latency.json'))
    for name, c in data['week10']['dev_comparison_bootstraps'].items():
        left, right = name.split('_minus_')
        delta = data['week10']['dev'][left]['macro_positive_iou'] - data['week10']['dev'][right]['macro_positive_iou']
        contrasts.append(dict(study='Week 10', panel='reused dev', contrast=name, estimate_pp=100*delta,
                              ci_low_pp=100*c['interval95'][0], ci_high_pp=100*c['interval95'][1], unit='103 physical sources',
                              evidence='exploratory weak labels; repeated dev use', source='docs/evaluation/week10-preservation-pilot-v1/analysis/results.json'))
    for arm, count in data['followups']['continuation']['decision']['counts'].items():
        rates.append(dict(study='Week 11 continuation screen', panel='27 selected cases', arm=arm,
                          n=27, successes=count, rate_percent=count/27*100, ci_low_percent=None, ci_high_percent=None,
                          interval_unit='Selected diagnostic counts; no new population interval', evidence='exploratory; preservation gate failed',
                          source='docs/evaluation/week11-multihorizon-results-v1/summary.json'))
    for arm, c in data['followups']['adapter']['counts'].items():
        rates.append(dict(study='Week 11 adapter screen', panel='first repeat of 9 cases', arm=arm,
                          n=c['n'], successes=c['successes'], rate_percent=c['successes']/c['n']*100,
                          ci_low_percent=None, ci_high_percent=None, interval_unit='Incomplete 27/81 screen; no new population interval',
                          evidence='exploratory; stopped at latency gate', source='docs/evaluation/week11-motion-screen-v1/remote/screen/completion.json'))
        latency.append(dict(study='Week 11 adapter screen', panel='first repeat of 9 cases', arm=arm,
                            **data['followups']['adapter']['latency']['summary'][arm]['all_queries'],
                            protocol='all rollout queries; event branch running; partial screen',
                            source='docs/evaluation/week11-motion-screen-v1/remote/screen/rollout-latency-27.json'))
    tables.update(success_rates=rates, contrasts=contrasts, latency=latency, training_cost=costs, event_forecasting=event)
    for name, rows in tables.items():
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / 'tables' / (name+'.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return {k: len(v) for k, v in tables.items()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    marker = out / 'week12-build.json'
    if out.exists() and not marker.exists():
        raise ValueError('Output must be new or an existing Week 12 build directory')
    out.mkdir(parents=True, exist_ok=True)
    if marker.exists():
        assert read_marker(marker)['artifact_type'] == 'eventground_wam.week12_results_build'
    else:
        write(marker, {'artifact_type': 'eventground_wam.week12_results_build'})
    for folder in ['tables', 'figures', 'examples', 'before', 'source']:
        (out / folder).mkdir(exist_ok=True)
    for name in ['README.md', 'docs/ROADMAP.md', 'docs/EventGround-WAM_12-Week_Execution_Plan.md', 'docs/WEEK12_STATUS.md']:
        p = out / 'before' / name
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / name, p)
    frozen = read('docs/evaluation/week12-scope-v1/package-index.json')
    for name, digest in frozen['files_sha256'].items():
        assert hashlib.sha256(source('docs/evaluation/week12-scope-v1/'+name).read_bytes()).hexdigest() == digest
    for p in ['docs/WEEK12_SCOPE.md', 'docs/WEEK12_REPORT_OUTLINE.md', 'docs/WEEK9_FINAL_RESULTS.md',
              'docs/WEEK10_CLOSEOUT.md', 'docs/WEEK11_CLOSEOUT.md', 'docs/WEEK8_COUPLING_INTERFACE.md']:
        source(p)
    data = {'artifact_type': 'eventground_wam.week12_consolidated_results', 'week9': load_week9(),
            'week10': load_week10(), 'week11': load_week11(), 'followups': load_followups(),
            'scope': 'Existing evidence only; distinct cohorts are never pooled; no new model calls or rollouts.'}
    counts = export_tables(data, out)
    write(out / 'consolidated.json', data)
    write(out / 'numerical-verification.json', {'status': 'pass', 'checks': CHECKS, 'check_count': len(CHECKS),
          'week9_outcomes': 15000, 'week11_formal_outcomes': 2001, 'h8_followup_outcomes': 128,
          'tables': counts, 'tolerance': {'absolute': 1e-10, 'relative': 1e-10},
          'scope': 'CPU replay of audited local outcome, prediction and timing metadata. Prior raw recording/input audits remain authoritative; no remote bytes rehashed.',
          'numpy_version': np.__version__, 'new_model_calls': 0, 'new_rollouts': 0, 'new_training_updates': 0})
    source('scripts/build_week12_results.py')
    write(out / 'source-manifest.json', {'files_sha256': SOURCES})
    for name, digest in SOURCES.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    shutil.copyfile(__file__, out / 'source/build_week12_results.py')
    print(json.dumps({'status': 'pass', 'checks': len(CHECKS), 'sources': len(SOURCES), 'tables': counts}))


def read_marker(path):
    return json.loads(path.read_text())


if __name__ == '__main__':
    main()

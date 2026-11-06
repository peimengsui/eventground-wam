"""Independent read-only formal re-audit, frozen statistics and control comparison."""
from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_continuation import ActiveBudget
from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.clean_comparison import atomic_json, utc_now
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_comparisons import audit_panel, compare_inputs, summarize_rows
from eventground_wam.evaluation_protocol import load_evaluation_protocol, protocol_sha256
from eventground_wam.week11_analysis import (
    ARMS,
    control_summary,
    control_trajectory_pair,
    paired_statistics,
    progression_decision,
    secondary_statistics,
)
from eventground_wam.week11_execution import validate_query_artifact
from eventground_wam.week11_formal import all_attempt_latencies

PANELS = ('targeted-467', 'clean-200')


def write(path, value):
    value = seal_contract(value)
    atomic_json(path, value)
    return value


def file_inventory(root):
    files = {}
    for i, path in enumerate(sorted(root.rglob('*'))):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {'sha256': sha256_file(path), 'bytes': path.stat().st_size}
        if i and i % 1000 == 0:
            print(json.dumps({'phase': 'raw_inventory', 'files_seen': i}), flush=True)
    return files


def query_summary(rows):
    used, selected, events, fallbacks, cadence = Counter(), Counter(), Counter(), Counter(), Counter()
    components, costs, per_episode = defaultdict(list), defaultdict(float), []
    exceptions = 0
    for ordinal, row in enumerate(rows):
        timing = read_sealed(Path(row['timing_artifact']))
        for k, v in timing['costs'].items():
            costs[k] += v
        episode_used, episode_events, episode_fallbacks = Counter(), Counter(), Counter()
        for q in timing['queries']:
            p = q['policy']
            exceptions += q['exception'] is not None
            used[str(p['used_horizon'])] += 1
            episode_used[str(p['used_horizon'])] += 1
            cadence[f"h={p['used_horizon']},remaining={q['remaining_chunk_actions']}"] += 1
            if p['prediction']:
                event = p['prediction']['next_event']
                events[event] += 1; episode_events[event] += 1
            if p['decision']:
                selected[str(p['decision']['selected_horizon'])] += 1
                if p['decision']['fallback_reason']:
                    reason = p['decision']['fallback_reason']
                    fallbacks[reason] += 1; episode_fallbacks[reason] += 1
            flat = {**p.get('component_ms', {}), **{k: v for k, v in p.items()
                    if k.endswith('_ms') and isinstance(v, (int, float))}}
            if p['event']:
                flat.update({k: v for k, v in p['event'].items() if k.endswith('_ms')})
            for k, v in flat.items():
                if v is not None:
                    components[k].append(v)
        per_episode.append({'ordinal': ordinal, 'queries': len(timing['queries']), 'used_horizons': dict(episode_used),
                            'events': dict(episode_events), 'fallbacks': dict(episode_fallbacks)})
    return {'used_horizons': dict(used), 'selected_horizons': dict(selected), 'events': dict(events),
            'fallbacks': dict(fallbacks), 'horizon_vs_remaining_actions': dict(cadence), 'query_exceptions': exceptions,
            'per_episode': per_episode, 'simulator_and_recording_seconds': dict(costs),
            'component_latency_ms': {k: {'count': len(v), 'median': float(np.median(v)),
                'p95': float(np.quantile(v, .95, method='linear'))} for k, v in components.items()}}


class AuditBudget:
    def __init__(self, output, identity, limits, completion, recordings, prior_runs=()):
        self.output, self.limits = output, limits
        self.prior_seconds = completion['resources']['total_active_seconds']
        self.prior_bytes = sum(p.stat().st_size for p in recordings.rglob('*') if p.is_file()) + completion['resources']['qualification_output_bytes']
        self.prior_analysis = []
        self.budget = ActiveBudget(output/'active-budget.json', identity)
        self.panel_budgets = {}
        self.prior_panels = {}
        for arm in ARMS:
            for panel in PANELS:
                key = arm+'/'+panel
                previous = read_sealed(recordings/'panel-budgets'/f'{key}.json')
                require(not previous['running'], 'Evaluation panel budget still running')
                self.prior_panels[key] = previous['active_seconds']
        for previous_root in prior_runs:
            previous = read_sealed(previous_root/'active-budget.json')
            require(not previous['running'], 'Prior analysis is still running')
            self.prior_seconds += previous['active_seconds']
            self.prior_bytes += sum(p.stat().st_size for p in previous_root.rglob('*') if p.is_file())
            self.prior_analysis.append({'root': str(previous_root), 'budget_digest': previous['digest'],
                                        'active_seconds': previous['active_seconds'], 'source_digest': previous['identity']})
            for path in (previous_root/'panel-budgets').glob('*/*.json'):
                panel_budget = read_sealed(path)
                require(not panel_budget['running'], 'Prior panel re-audit still running')
                key = str(path.relative_to(previous_root/'panel-budgets').with_suffix(''))
                self.prior_panels[key] += panel_budget['active_seconds']
        self.active_panel = None
        self.peak_rss = 0

    def set_panel(self, key):
        if self.active_panel is not None:
            self.panel_budgets[self.active_panel].close()
        self.active_panel = key
        self.panel_budgets[key] = ActiveBudget(self.output/'panel-budgets'/f'{key}.json', self.budget.identity+'/'+key)

    def check(self):
        import psutil
        process = psutil.Process()
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        self.peak_rss = max(self.peak_rss, rss)
        require(rss <= self.limits['host_process_tree_rss_limit_bytes'], 'Analysis process-tree RSS cap')
        require(self.prior_seconds+self.budget.seconds < self.limits['aggregate_active_seconds'], 'Aggregate active time cap')
        if self.active_panel is not None:
            require(self.prior_panels[self.active_panel]+self.panel_budgets[self.active_panel].seconds
                    < self.limits['per_formal_arm_panel_active_seconds'], 'Panel active time cap including re-audit')
        require(shutil.disk_usage(self.output).free >= self.limits['minimum_free_bytes'], 'Disk reserve cap')

    def close(self):
        self.check()
        if self.active_panel is not None:
            self.panel_budgets[self.active_panel].close()
        self.budget.close()
        extra_bytes = sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())
        require(extra_bytes+self.prior_bytes < self.limits['output_limit_bytes'], 'Aggregate output cap')
        return {'prior_evaluation_and_qualification_seconds': self.prior_seconds,
                'analysis_active_seconds': read_sealed(self.output/'active-budget.json')['active_seconds'],
                'total_active_seconds': self.prior_seconds+read_sealed(self.output/'active-budget.json')['active_seconds'],
                'analysis_peak_process_tree_rss_bytes': self.peak_rss, 'analysis_gpu_inference_calls': 0,
                'prior_recording_bytes_including_qualification': self.prior_bytes,
                'analysis_bytes_at_close': extra_bytes, 'free_bytes': shutil.disk_usage(self.output).free,
                'per_panel_evaluation_seconds': self.prior_panels,
                'per_panel_reaudit_seconds': {k: read_sealed(v.path)['active_seconds'] for k, v in self.panel_budgets.items()},
                'prior_analysis_attempts': self.prior_analysis,
                'control_comparison_seconds_charged_to_fixed8_panel': True}


def analyze(source_root, recordings, output, workers, prior_runs=()):
    output.mkdir(parents=True, exist_ok=True)
    require(not (output/'analysis.json').exists(), 'Completed analysis already exists; preserve it')
    cfg = read_sealed(source_root/'configs/evaluation/week11_integration_v1.json')
    source = read_sealed(source_root/'week11-formal-source-manifest.json')
    completion = read_sealed(recordings/'completion.json')
    cohort = read_sealed(recordings/'cohort.json')
    admission = read_sealed(recordings/'admission.json')
    require(completion['status'] == 'complete' and completion['formal_episodes'] == 2001,
            'Formal matrix incomplete')
    require(completion['source_digest'] == source['digest'] == cohort['source_digest']
            and cohort['parent_digest'] == cfg['digest'] and completion['cohort_digest'] == cohort['digest']
            and cohort['admission_digest'] == admission['digest'], 'Evaluation provenance differs')
    analysis_source = read_sealed(Path(__file__).resolve().parents[1]/'analysis-source-manifest.json')
    for name, expected in analysis_source['files_sha256'].items():
        require(sha256_file(Path(__file__).resolve().parents[1]/name) == expected, 'Analysis source differs: '+name)
    require(analysis_source['formal_source_digest'] == source['digest'], 'Analysis parent changed')
    for name, expected in source['files'].items():
        require(sha256_file(source_root/name) == expected, 'Executed evaluation source differs: '+name)
        require(sha256_file(Path(__file__).resolve().parents[1]/name) == expected,
                'Analysis dependency differs from executed evaluation: '+name)
    budget = AuditBudget(output, analysis_source['digest'], cfg['resources'], completion, recordings, prior_runs)
    started = time.monotonic()
    try:
        initial_stats = {str(p.relative_to(recordings)): (p.stat().st_size, p.stat().st_mtime_ns)
                         for p in recordings.rglob('*') if p.is_file()}
        inventory = write(output/'raw-inventory.json', {'artifact_type': 'eventground_wam.week11_formal_raw_inventory',
            'root': str(recordings), 'observed_at': utc_now(), 'files': file_inventory(recordings)})
        reports, tasks_by_panel, protocols = {}, {}, {}
        for panel in PANELS:
            manifest = read_sealed(source_root/cfg['panels'][panel]['path'])
            require(manifest['digest'] == cfg['panels'][panel]['digest'], 'Panel manifest changed')
            tasks_by_panel[panel] = manifest['tasks'] if panel == 'targeted-467' else [
                {**p, 'perturbation_category': 'Clean', 'difficulty_level': None} for p in manifest['pairs']]
            protocols[panel] = load_evaluation_protocol(source_root/'configs/evaluation'/(
                'libero_plus_v1.json' if panel == 'targeted-467' else 'week4_clean_comparison_v1.json'))
        for arm in ARMS:
            for panel in PANELS:
                key = arm+'/'+panel
                budget.set_panel(key)
                target = recordings/'panels'/arm/panel
                identity = read_sealed(target/'identity.json')
                report = read_sealed(target/'report.json')
                tasks, protocol = tasks_by_panel[panel], protocols[panel]
                require(report['digest'] == completion['reports'][key] and report['identity_digest'] == identity['digest']
                        and identity['cohort_digest'] == cohort['digest'] and identity['admission_digest'] == admission['digest']
                        and identity['protocol_sha256'] == protocol_sha256(protocol), 'Panel report binding differs')
                require(read_sealed(target/'tasks.json') == seal_contract({'tasks': tasks, 'selected_ordinals': list(range(len(tasks)))}),
                        'Executed task list differs')
                require(len(report['episodes']) == len(tasks), 'Panel count differs')
                for ordinal, row in enumerate(report['episodes']):
                    budget.check()
                    receipt = read_sealed(target/f'result-{ordinal:04d}.json')
                    journal_path = target/f'episode-{ordinal:04d}.attempts.json'
                    journal = read_sealed(journal_path)
                    key_fields = {'identity_digest': identity['digest'], 'ordinal': ordinal, 'task': tasks[ordinal]}
                    require(receipt['result'] == row and all(receipt[k] == v for k, v in key_fields.items())
                            and journal['identity'] == key_fields
                            and sha256_file(journal_path) == receipt['attempt_journal_sha256'], 'Episode receipt/journal differs')
                    require(journal['attempts'] == [0] and row['retry_count'] == 0, 'Unexpected retry; extend audit explicitly')
                    expected_path = target/f'episode-{ordinal:04d}-attempt-0.hdf5'
                    require(row['artifact'] == str(expected_path)
                            and row['timing_artifact'] == str(expected_path.with_suffix('.timing.json')), 'Artifact location differs')
                    require(validate_query_artifact(row, arm, admission), 'Invalid query artifact')
                audits = audit_panel(report['episodes'], identity, tasks, protocol)
                require(audits == report['audits'] and all(a['error'] is None for a in audits), 'Independent artifact audit differs')
                require(summarize_rows(report['episodes']) == report['summary'], 'Summary differs from episodes')
                require(all_attempt_latencies(target, identity) == report['all_attempt_query_timing'], 'Attempt timing differs')
                reports[key] = {'original_report_digest': report['digest'], 'episodes': report['episodes'],
                    'summary': report['summary'], 'audits': audits, 'query_summary': query_summary(report['episodes']),
                    'model_load_seconds': report['model_load_seconds_current_invocation']}
                write(output/'panels'/arm/(panel+'.json'), reports[key])
                print(json.dumps({'phase': 'independent_panel_audit', 'arm': arm, 'panel': panel,
                                  'episodes': len(tasks), 'passed': True}), flush=True)
        comparison_audits = {panel: {a+' vs '+b: compare_inputs(reports[a+'/'+panel]['audits'], reports[b+'/'+panel]['audits'])
            for a, b in ((ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2]))} for panel in PANELS}
        require(comparison_audits == completion['comparisons'], 'Cross-arm input comparison differs')
        statistics, matrices, latency, controls = {}, {}, {}, {}
        for panel in PANELS:
            tasks = tasks_by_panel[panel]
            outcomes = np.array([[reports[a+'/'+panel]['episodes'][i]['success'] for a in ARMS] for i in range(len(tasks))])
            statistics[panel] = paired_statistics(outcomes, tasks, clean=panel == 'clean-200')
            matrices[panel] = {'arms': list(ARMS), 'tasks': tasks, 'outcomes': outcomes.astype(int).tolist()}
            fixed = reports['M4-fixed/'+panel]['summary']['latency']['all_queries']['p95_ms']
            adaptive = reports['M4-EG-adaptive/'+panel]['summary']['latency']['all_queries']['p95_ms']
            latency[panel] = {'fixed_p95_ms': fixed, 'adaptive_p95_ms': adaptive, 'adaptive_to_fixed_p95_ratio': adaptive/fixed}
            if panel == 'targeted-467':
                # This inherited manifest is file-hash bound by the source;
                # its original schema predates seal_contract/digest.
                dev = json.loads((source_root/'evaluation/splits/libero-plus-v1/dev-210.json').read_text())['tasks']
                contact = read_sealed(source_root/cfg['panels']['contact-proxy-98']['path'])['tasks']
                secondary = secondary_statistics(outcomes, tasks,
                    {(t['suite'], t['runtime_task_index']) for t in dev},
                    {(t['suite'], t['runtime_task_index']) for t in contact})
            budget.set_panel('M4-EG-fixed8/'+panel)
            jobs = [(i, task, reports['M4-fixed/'+panel]['episodes'][i], reports['M4-EG-fixed8/'+panel]['episodes'][i])
                    for i, task in enumerate(tasks)]
            pairs = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for pair in pool.map(control_trajectory_pair, jobs, chunksize=1):
                    pairs.append(pair)
                    budget.check()
                    if len(pairs) % 25 == 0:
                        print(json.dumps({'phase': 'full_control_trajectory_audit', 'panel': panel,
                                          'completed': len(pairs), 'total': len(jobs)}), flush=True)
            controls[panel] = {'summary': control_summary(pairs), 'pairs': pairs}
            write(output/'controls'/(panel+'.json'), controls[panel])
        final_stats = {str(p.relative_to(recordings)): (p.stat().st_size, p.stat().st_mtime_ns)
                       for p in recordings.rglob('*') if p.is_file()}
        require(initial_stats == final_stats, 'Raw recording directory changed during read-only audit')
        evaluation_limits = cfg['resources']
        r = completion['resources']
        pilot_resources = read_sealed(source_root/'docs/evaluation/week11-integration-v1/analysis.json')['resources']
        resources_passed = (not r['stop_reason']
            and r['total_active_seconds'] < evaluation_limits['aggregate_active_seconds']
            and r['total_output_bytes'] < evaluation_limits['output_limit_bytes']
            and min(r['free_bytes'], pilot_resources['free_bytes']) >= evaluation_limits['minimum_free_bytes']
            and max(r['peak_process_tree_rss_bytes'], pilot_resources['peak_process_tree_rss_bytes']) <= evaluation_limits['host_process_tree_rss_limit_bytes']
            and max(r['peak_allocated_bytes'], r['peak_reserved_bytes'], pilot_resources['peak_allocated_bytes'], pilot_resources['peak_reserved_bytes'])
                < r['gpu_total_bytes']*evaluation_limits['gpu_peak_fraction_strictly_below']
            and all(v < evaluation_limits['per_formal_arm_panel_active_seconds'] for v in budget.prior_panels.values()))
        all_inputs = all(c['all_exact'] for ps in comparison_audits.values() for c in ps.values())
        decision = progression_decision(statistics['targeted-467']['M4-EG-adaptive minus M4-fixed'],
            statistics['clean-200']['M4-EG-adaptive minus M4-fixed'], latency, cfg['decision'],
            audit_passed=all_inputs, resources_passed=resources_passed)
        matrix = write(output/'paired-matrices.json', matrices)
        write(output/'secondary-statistics.json', secondary)
        control_checks = {p: c['summary'] for p, c in controls.items()}
        cases = {}
        for panel, matrix_panel in matrices.items():
            selected = {}
            for ordinal, row in enumerate(matrix_panel['outcomes']):
                category = ('both_success' if row[0] and row[2] else 'both_failure' if not row[0] and not row[2]
                            else 'adaptive_only_success' if row[2] else 'fixed_only_success')
                if category not in selected:
                    selected[category] = {'ordinal': ordinal, 'task': matrix_panel['tasks'][ordinal],
                        'artifacts': {a: reports[a+'/'+panel]['episodes'][ordinal]['artifact'] for a in ARMS}}
            cases[panel] = selected
        write(output/'selected-case-index.json', {'selection': 'first manifest ordinal in each paired outcome class; no quality selection', 'panels': cases})
        budget.check()
        resources = budget.close()
        result = write(output/'analysis.json', {'artifact_type': 'eventground_wam.week11_formal_closeout',
            'created_at': utc_now(), 'status': 'complete', 'analysis_source_digest': analysis_source['digest'],
            'source_digest': source['digest'], 'cohort_digest': cohort['digest'], 'completion_digest': completion['digest'],
            'raw_inventory_digest': inventory['digest'], 'paired_matrix_digest': matrix['digest'],
            'formal_episode_count': 2001, 'all_artifacts_independently_validated': True,
            'all_pairwise_initial_inputs_exact': all_inputs, 'statistics': statistics, 'latency_gates': latency,
            'decision': decision, 'control_summary': control_checks,
            'summaries': {k: v['summary'] for k, v in reports.items()},
            'evaluation_resources': r, 'qualification_resources': pilot_resources, 'analysis_resources': resources,
            'analysis_wall_seconds_current_invocation': time.monotonic()-started,
            'new_training_or_rollouts': 0, 'on_policy_event_accuracy_claimed': False,
            'secondary_interpretation': 'descriptive one-way subsets; no multiplicity adjustment or new test set',
            'step3_causal_failure_diagnosis_performed': False})
        print(json.dumps({'phase': 'analysis_complete', 'digest': result['digest'], 'statistics': statistics,
                          'decision': decision, 'control_summary': control_checks}), flush=True)
    except BaseException as exc:
        resources = budget.close()
        write(output/'failure.json', {'source_digest': analysis_source['digest'], 'at': utc_now(),
            'exception': f'{type(exc).__name__}: {exc}', 'resources': resources})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--recordings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--prior-analysis', type=Path, action='append', default=[])
    args = parser.parse_args()
    with (args.recordings/'.run.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        analyze(args.source_root, args.recordings, args.output, args.workers, args.prior_analysis)


if __name__ == '__main__':
    main()

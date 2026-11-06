"""Revalidate mirrored pilot artifacts and report engineering-only conclusions."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from verify_week11_horizon import verify

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.clean_comparison import atomic_json
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import sha256_array
from eventground_wam.evaluation_comparisons import audit_panel, compare_inputs, summarize_rows
from eventground_wam.evaluation_protocol import load_evaluation_protocol
from eventground_wam.eventground_inference import ARMS
from eventground_wam.week11_execution import validate_query_artifact

ROOT = Path(__file__).resolve().parents[1]


def analyze(recordings, source_manifest):
    verify(ROOT)
    cfg = read_sealed(ROOT/'configs/evaluation/week11_integration_v1.json')
    cohort = read_sealed(recordings/'cohort.json')
    source = read_sealed(source_manifest)
    require(cohort['source_digest'] == source['digest'], 'Executed source binding differs')
    admission = read_sealed(recordings/'admission.json')
    numerical = read_sealed(recordings/'numerical-qualification.json')
    completion = read_sealed(recordings/'latest-completion.json')
    require(completion['status'] == 'complete' and completion['qualification_episodes'] == 24
            and completion['source_digest'] == source['digest'], 'Pilot incomplete')
    require(numerical['passed'] and numerical['branch_identities'] == admission, 'Wrong numerical qualification')
    tasks = read_sealed(recordings/'tasks.json')['tasks']
    expected = read_sealed(ROOT/cfg['panels']['qualification-8']['path'])['pairs']
    require(tasks == [{**r, 'perturbation_category': 'Clean', 'difficulty_level': None} for r in expected], 'Panel differs')
    protocol = load_evaluation_protocol(ROOT/'configs/evaluation/week4_clean_comparison_v1.json')
    reports, all_queries, prefix_audits = {}, {}, []
    inventory = {p.relative_to(recordings).as_posix(): sha256_file(p)
                 for p in sorted(recordings.rglob('*')) if p.is_file()}
    remote_inventory = read_sealed(recordings/'artifact-inventory.json')
    require({k: v for k, v in inventory.items() if k != 'artifact-inventory.json'}
            == remote_inventory['files_sha256'], 'Mirrored bytes differ from remote inventory')
    for name in ('src/eventground_wam/eventground_inference.py',
                 'src/eventground_wam/week11_execution.py',
                 'src/eventground_wam/week11_recording.py',
                 'src/eventground_wam/week11_resources.py', 'scripts/run_week11_pilot.py'):
        require(sha256_file(ROOT/name) == source['files'][name], 'Local implementation differs from executed source')
    for ordinal, task in enumerate(tasks):
        path = recordings/'renderer-prefixes'/f'prefix-{ordinal:04d}.hdf5'
        receipt = read_sealed(path.with_suffix('.json'))
        require(receipt['sha256'] == sha256_file(path) and receipt['task'] == task, 'Prefix artifact differs')
        with h5py.File(path) as handle:
            prefix = {f'{i}:{key}': sha256_array(row) for key in handle for i, row in enumerate(handle[key])}
        require(prefix == receipt['inputs']['prefix'], 'Prefix hashes differ from bytes')
        prefix_audits.append({'ordinal': ordinal, 'inputs': receipt['inputs'], 'error': None})
    for arm in ARMS:
        identity = read_sealed(recordings/arm/'identity.json')
        report = read_sealed(recordings/arm/'report.json')
        require(report['digest'] == completion['arm_report_digests'][arm]
                and report['identity_digest'] == identity['digest']
                and identity['cohort_digest'] == cohort['digest'], 'Arm report binding differs')
        rows, queries = [], []
        for ordinal, remote in enumerate(report['episodes']):
            receipt = read_sealed(recordings/arm/f'result-{ordinal:04d}.json')
            require(receipt['result'] == remote and receipt['task'] == tasks[ordinal]
                    and receipt['identity_digest'] == identity['digest'], 'Episode receipt differs')
            journal = recordings/arm/f'episode-{ordinal:04d}.attempts.json'
            require(sha256_file(journal) == receipt['attempt_journal_sha256'], 'Attempt journal differs')
            require(remote['artifact'] is not None, 'Missing completed artifact')
            row = {**remote, 'artifact': str(recordings/arm/Path(remote['artifact']).name),
                   'timing_artifact': str(recordings/arm/Path(remote['timing_artifact']).name)}
            require(validate_query_artifact(row, arm, admission), 'Invalid event artifact')
            queries.extend(json.loads(line) for line in Path(row['artifact']).with_suffix('.queries.jsonl').read_text().splitlines())
            rows.append(row)
        audits = audit_panel(rows, identity, tasks, protocol)
        require(audits == report['audits'] and all(not a['error'] for a in audits), 'Artifact input audit differs')
        summary = summarize_rows(rows)
        require(summary == report['summary'], 'Outcome or latency summary differs')
        selected = Counter(str(q['policy']['decision']['selected_horizon']) for q in queries if q['policy']['decision'])
        used = Counter(str(q['policy']['used_horizon']) for q in queries)
        events = Counter(q['policy']['prediction']['next_event'] for q in queries if q['policy']['prediction'])
        fallbacks = Counter(q['policy']['decision']['fallback_reason'] for q in queries
                            if q['policy']['decision'] and q['policy']['decision']['fallback_reason'])
        components = {}
        for q in queries:
            p = q['policy']
            flat = {**p.get('component_ms', {}), 'action_branch': p.get('action_branch_ms')}
            if p['event']:
                flat.update({k: v for k, v in p['event'].items() if k.endswith('_ms')})
            for key, value in flat.items():
                if value is not None:
                    components.setdefault(key, []).append(value)
        reports[arm] = {'summary': summary, 'audits': audits,
            'renderer_repeat_audit': compare_inputs(audits, prefix_audits),
            'selected_horizons': dict(selected), 'used_horizons': dict(used), 'events': dict(events),
            'fallbacks': dict(fallbacks), 'query_failures': sum(q['exception'] is not None for q in queries),
            'components_ms': {k: {'median': float(np.median(v)), 'p95': float(np.quantile(v, .95))}
                              for k, v in components.items()},
            'per_episode_horizons': [dict(Counter(str(q['policy']['used_horizon']) for q in
                map(json.loads, Path(row['artifact']).with_suffix('.queries.jsonl').read_text().splitlines()))) for row in rows]}
        all_queries[arm] = queries
    comparisons = {a+' vs '+b: compare_inputs(reports[a]['audits'], reports[b]['audits'])
                   for a, b in itertools.combinations(ARMS, 2)}
    control_trajectories = []
    left = read_sealed(recordings/'M4-fixed/report.json')['episodes']
    right = read_sealed(recordings/'M4-EG-fixed8/report.json')['episodes']
    for ordinal, (a, b) in enumerate(zip(left, right, strict=True)):
        with h5py.File(recordings/'M4-fixed'/Path(a['artifact']).name) as ah, h5py.File(
                recordings/'M4-EG-fixed8'/Path(b['artifact']).name) as bh:
            fields = {name: sha256_array(ah['steps'][name][:]) == sha256_array(bh['steps'][name][:])
                      for name in ah['steps'] if name != 'policy_latency_ms'}
            indices = np.flatnonzero(np.isfinite(ah['steps/policy_latency_ms'][:]))
            other_indices = np.flatnonzero(np.isfinite(bh['steps/policy_latency_ms'][:]))
            query_inputs_exact = np.array_equal(indices, other_indices) and all(
                sha256_array(ah['steps'][name][indices]) == sha256_array(bh['steps'][name][other_indices])
                for name in ('agent_rgb', 'wrist_rgb', 'proprioception'))
            rendering_differences = {}
            for name in ('agent_rgb', 'wrist_rgb'):
                if not fields[name]:
                    delta = ah['steps'][name][:].astype(np.int16)-bh['steps'][name][:].astype(np.int16)
                    different = np.flatnonzero(np.any(delta != 0, axis=(1, 2, 3))).tolist()
                    rendering_differences[name] = {'recorded_frame_indices': different,
                        'different_channel_values': int(np.count_nonzero(delta)),
                        'max_channel_abs_delta': int(np.abs(delta).max()),
                        'policy_query_frame_indices': [i for i in different if i in indices]}
        control_trajectories.append({'ordinal': ordinal, 'all_non_timing_steps_exact': all(fields.values()),
                                     'all_policy_query_observations_exact': bool(query_inputs_exact),
                                     'fields': fields, 'rendering_differences': rendering_differences})
    all_cross_exact = all(v['all_exact'] for v in comparisons.values())
    all_repeat_exact = all(v['renderer_repeat_audit']['all_exact'] for v in reports.values())
    resources = completion['resources']
    limits = cfg['resources']
    within = (resources['active_seconds'] <= limits['qualification_active_seconds']
              and resources['peak_process_tree_rss_bytes'] <= limits['host_process_tree_rss_limit_bytes']
              and max(resources['peak_reserved_bytes'], resources['peak_allocated_bytes'])
                    < resources['gpu_total_bytes']*limits['gpu_peak_fraction_strictly_below']
              and resources['free_bytes'] >= limits['minimum_free_bytes']
              and resources['output_bytes'] < limits['output_limit_bytes'] and not resources['stop_reason'])
    fixed_p95 = reports['M4-fixed']['summary']['latency']['all_queries']['p95_ms']
    adaptive_p95 = reports['M4-EG-adaptive']['summary']['latency']['all_queries']['p95_ms']
    # Resource projection charges each formal episode for its full suite step
    # limit. This avoids assuming that the pilot's short successful trajectories
    # predict the duration of harder perturbation tasks.
    formal = {name: read_sealed(ROOT/cfg['panels'][name]['path']) for name in ('targeted-467', 'clean-200')}
    projection = {}
    for arm in ARMS:
        rows = read_sealed(recordings/arm/'report.json')['episodes']
        seconds_per_step = max(r['episode_wall_time_seconds']/max(r['policy_step_count'], 1) for r in rows)
        bytes_per_step = max((recordings/arm/Path(r['artifact']).name).stat().st_size/max(r['policy_step_count'], 1) for r in rows)
        projection[arm] = {}
        for name, panel in formal.items():
            members = panel.get('tasks', panel.get('pairs'))
            steps = sum(protocol['benchmark']['suites'][r['suite']]['max_policy_steps'] for r in members)
            projection[arm][name] = {'max_policy_steps': steps,
                'active_seconds_1p2': 1.2*(steps*seconds_per_step+completion['model_load_seconds']),
                'hdf5_bytes_1p2': 1.2*steps*bytes_per_step}
    projected_seconds = resources['active_seconds']+sum(p['active_seconds_1p2'] for a in projection.values() for p in a.values())
    # Include measured non-HDF5 overhead and all archived source bytes separately.
    overhead_ratio = sum(p.stat().st_size for p in recordings.rglob('*') if p.is_file()) / sum(p.stat().st_size for p in recordings.rglob('*.hdf5'))
    projected_bytes = resources['output_bytes']+overhead_ratio*sum(p['hdf5_bytes_1p2'] for a in projection.values() for p in a.values())
    projection_fits = (projected_seconds <= limits['aggregate_active_seconds']
        and all(p['active_seconds_1p2'] <= limits['per_formal_arm_panel_active_seconds'] for a in projection.values() for p in a.values())
        and projected_bytes <= limits['output_limit_bytes']
        and projected_bytes-resources['output_bytes'] <= resources['free_bytes']-limits['minimum_free_bytes'])
    return seal_contract({'artifact_type': 'eventground_wam.week11_pilot_reaudit',
        'source_digest': source['digest'], 'cohort_digest': cohort['digest'],
        'remote_inventory_digest': remote_inventory['digest'],
        'completion_digest': completion['digest'], 'numerical_digest': numerical['digest'],
        'reports': reports, 'comparisons': comparisons, 'resources': resources,
        'fixed8_full_trajectory_parity': control_trajectories,
        'all_cross_arm_inputs_exact': all_cross_exact, 'all_renderer_repeats_exact': all_repeat_exact,
        'all_artifacts_valid': True, 'within_resource_caps': within,
        'engineering_pilot_passed': numerical['passed'] and within,
        'input_audits_allow_positive_gate': all_cross_exact and all_repeat_exact,
        'pilot_adaptive_p95_ms': adaptive_p95, 'pilot_adaptive_fixed_p95_ratio': adaptive_p95/fixed_p95,
        'pilot_meets_future_latency_thresholds': adaptive_p95 <= 200 and adaptive_p95/fixed_p95 <= 2.5,
        'formal_step_limit_projection': projection, 'projected_total_seconds_1p2': projected_seconds,
        'projected_total_bytes_1p2': projected_bytes, 'projection_fits': projection_fits,
        'runner_projection_fits': completion['projected_formal_fits'],
        'runner_projected_total_seconds_1p2': completion['projected_total_active_seconds'],
        'runner_projected_total_bytes_1p2': completion['projected_total_bytes'],
        'formal_resource_readiness': projection_fits and completion['projected_formal_fits'],
        'formal_additional_bytes_projected_1p2': projected_bytes-resources['output_bytes'],
        'available_bytes_above_reserve': resources['free_bytes']-limits['minimum_free_bytes'],
        'additional_free_bytes_needed': max(0., projected_bytes-resources['output_bytes']
                                           +limits['minimum_free_bytes']-resources['free_bytes']),
        'projection_method': 'maximum observed per-step episode wall/storage costs times full suite step limits, plus load and sidecar overhead; 1.2 margin',
        'formal_episodes': 0, 'training_updates': 0, 'robot_benefit_established': False,
        'inventory_sha256': inventory})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--recordings', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.recordings, args.source_manifest)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('reports', 'comparisons', 'inventory_sha256', 'formal_step_limit_projection')}, indent=2))

"""Audit and summarize the fixed 27-case three-arm diagnostic; no model or simulator."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.clean_comparison import atomic_json
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import read_episode_metadata
from eventground_wam.week11_closing_gate import ADAPTIVE, ARMS, FIXED, GATED, audit_gate


def write(path, value):
    value = seal_contract(value)
    atomic_json(path, value)
    return value


def paired(rows, left, right):
    differences = [int(r['outcomes'][left])-int(r['outcomes'][right]) for r in rows]
    return {'n': len(rows), 'left': left, 'right': right,
            'left_successes': sum(r['outcomes'][left] for r in rows),
            'right_successes': sum(r['outcomes'][right] for r in rows),
            'left_only': sum(d == 1 for d in differences),
            'right_only': sum(d == -1 for d in differences),
            'both_success': sum(r['outcomes'][left] and r['outcomes'][right] for r in rows),
            'both_failure': sum(not r['outcomes'][left] and not r['outcomes'][right] for r in rows),
            'gap_pp': 100*sum(differences)/len(rows) if rows else None,
            'left_only_cases': [r['id'] for r, d in zip(rows, differences) if d == 1],
            'right_only_cases': [r['id'] for r, d in zip(rows, differences) if d == -1]}


def prefix_agreement(fixed_path, gated_path, gated_queries):
    records = []
    with h5py.File(fixed_path) as lf, h5py.File(gated_path) as rf:
        left, right = lf['steps'], rf['steps']
        for i, q in enumerate(gated_queries):
            if q['policy']['closing_history']['active']:
                break
            if q['exception']:
                continue
            step = 10+8*i
            if step >= len(left['executed_action']):
                records.append({'query': i, 'fixed_recording_ended': True})
                continue
            inputs_equal = all(np.array_equal(left[k][step], right[k][step])
                               for k in ('agent_rgb', 'wrist_rgb'))
            inputs_equal &= np.array_equal(left['proprioception'][step].astype(np.float32),
                                           right['proprioception'][step].astype(np.float32))
            actions_equal = np.array_equal(left['predicted_action_chunk'][step],
                                            right['predicted_action_chunk'][step])
            records.append({'query': i, 'fixed_recording_ended': False,
                'inputs_equal': bool(inputs_equal), 'actions_equal': bool(actions_equal),
                'simulator_state_equal': bool(np.array_equal(left['mujoco_state'][step], right['mujoco_state'][step])),
                'previous_actions_equal': bool(np.array_equal(left['executed_action'][:step],
                                                             right['executed_action'][:step]))})
            if inputs_equal:
                assert actions_equal, 'Same-input inactive gate altered fixed8 actions'
    return {'queries': records, 'all_inputs_and_actions_exact': all(
                not r['fixed_recording_ended'] and r['inputs_equal'] and r['actions_equal'] for r in records),
            'same_input_queries': sum(r.get('inputs_equal', False) for r in records),
            'first_difference': next((r for r in records if r['fixed_recording_ended']
                                      or not r['inputs_equal'] or not r['actions_equal']), None)}


def run(raw, output):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    cfg, completion = read_sealed(raw/'contract.json'), read_sealed(raw/'completion.json')
    assert completion['status'] == 'complete' and completion['contract_digest'] == cfg['digest']
    assert completion['completed_rollouts'] == 81
    admission = read_sealed(raw/'admission.json')
    reports = {}
    for key, digest in completion['reports'].items():
        report = read_sealed(raw/'panels'/key/'report.json')
        assert report['digest'] == digest
        reports[key] = report
    rows, query_count, repeated_audits = [], 0, 0
    for case in cfg['cases']:
        row = {'id': case['id'], 'panel': case['panel'], 'ordinal': case['ordinal'],
               'task': case['task'], 'task_group': case['task_group'],
               'prior_outcome_class': case['outcome_class'], 'outcomes': {}, 'arms': {},
               'historical_outcomes': {}}
        current_paths, all_queries = {}, {}
        for arm in ARMS:
            report = reports[arm+'/'+case['panel']]
            ix = report['case_ids'].index(case['id'])
            result = report['episodes'][ix]
            assert not report['audits'][ix]['error']
            assert result['artifact'] and sha256_file(Path(result['artifact'])) == result['artifact_sha256']
            assert sha256_file(Path(result['timing_artifact'])) == result['timing_sha256']
            gate = audit_gate(result, arm, admission)
            assert gate == report['gate_audits'][ix] and gate['valid'] and not gate['query_exceptions']
            repeated_audits += 1
            path = Path(result['artifact']); current_paths[arm] = path
            queries = [json.loads(line) for line in path.with_suffix('.queries.jsonl').read_text().splitlines()]
            all_queries[arm] = queries
            query_count += len(queries)
            row['outcomes'][arm] = result['success']
            assert read_episode_metadata(path)['success'] == result['success']
            row['arms'][arm] = {'success': result['success'], 'artifact': str(path),
                'policy_step_count': result['policy_step_count'], 'retry_count': result['retry_count'],
                'termination_reason': result['termination_reason'], 'gate': gate,
                'used_horizon_counts': dict(Counter(str(q['policy']['used_horizon']) for q in queries)),
                'first_non8_query': next((i for i,q in enumerate(queries) if q['policy']['used_horizon'] != 8), None),
                'inactive_non8_queries': sum(not q['policy']['closing_history']['active']
                                            and q['policy']['used_horizon'] != 8 for q in queries)}
        for arm, path in case['artifacts'].items():
            assert sha256_file(Path(path)) == cfg['input_sha256'][path]
            row['historical_outcomes'][arm] = read_episode_metadata(Path(path))['success']
        row['control_outcomes_reproduced'] = {a: row['outcomes'][a] == row['historical_outcomes'][a]
                                             for a in (FIXED, ADAPTIVE)}
        row['inactive_gate_vs_fixed8'] = prefix_agreement(current_paths[FIXED], current_paths[GATED], all_queries[GATED])
        rows.append(row)
    groups = {'all': rows}
    for field in ('panel', 'task_group', 'prior_outcome_class'):
        for value in sorted({r[field] for r in rows}):
            groups[field+':'+value] = [r for r in rows if r[field] == value]
    summaries = {name: {'n': len(rs), 'successes': {arm: sum(r['outcomes'][arm] for r in rs) for arm in ARMS},
        'contrasts': [paired(rs, GATED, ADAPTIVE), paired(rs, GATED, FIXED), paired(rs, ADAPTIVE, FIXED)]}
        for name, rs in groups.items()}
    gate_summary = {'activated_episodes': sum(r['arms'][GATED]['gate']['active_query_count'] > 0 for r in rows),
        'closed_but_episode_ended_before_next_query': [r['id'] for r in rows
            if r['arms'][GATED]['gate']['first_closing_step'] is not None
            and not r['arms'][GATED]['gate']['active_query_count']],
        'never_closed': [r['id'] for r in rows if r['arms'][GATED]['gate']['first_closing_step'] is None],
        'inactive_non8_queries': sum(r['arms'][GATED]['inactive_non8_queries'] for r in rows),
        'inactive_prefixes_fully_exact': sum(r['inactive_gate_vs_fixed8']['all_inputs_and_actions_exact'] for r in rows),
        'inactive_same_input_queries_exact': sum(r['inactive_gate_vs_fixed8']['same_input_queries'] for r in rows),
        'control_outcome_reproduction': {a: sum(r['control_outcomes_reproduced'][a] for r in rows) for a in (FIXED, ADAPTIVE)}}
    summary = write(output/'summary.json', {'contract_digest': cfg['digest'], 'completion_digest': completion['digest'],
        'rows': rows, 'groups': summaries, 'gate_summary': gate_summary,
        'all_initial_inputs_exact': completion['all_initial_inputs_exact'],
        'queries_audited': query_count, 'episode_gate_audits_repeated': repeated_audits,
        'posthoc_selected_panel': True, 'population_estimate': False, 'baseline_changed': False})
    files = {str(p.relative_to(raw)): {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
             for p in sorted(raw.rglob('*')) if p.is_file()}
    inventory = write(output/'raw-inventory.json', {'root': str(raw), 'files': files})
    write(output/'completion.json', {'status': 'complete', 'summary_digest': summary['digest'],
        'inventory_digest': inventory['digest'], 'raw_files_hashed': len(files),
        'seconds': time.monotonic()-started, 'model_calls': 0, 'simulator_steps': 0,
        'source_sha256': sha256_file(Path(__file__))})
    print(json.dumps({'all': summaries['all'], 'gate': gate_summary}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.raw, args.output)

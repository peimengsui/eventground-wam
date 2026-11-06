"""Frozen Week 11 paired statistics and recording-only fixed-control audit."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.episode_recorder import read_episode_metadata

ARMS = ('M4-fixed', 'M4-EG-fixed8', 'M4-EG-adaptive')
CONTRASTS = ((2, 0), (2, 1), (1, 0))


def paired_statistics(outcomes, tasks, *, clean=False):
    """Use one common 10,000-draw index matrix for all three contrasts."""
    values = np.asarray(outcomes)
    require(values.shape == (len(tasks), 3) and len(tasks) > 0
            and np.isin(values, [0, 1]).all(), 'Invalid paired outcome matrix')
    values = values.astype(np.int64)
    if clean:
        groups = defaultdict(list)
        for i, task in enumerate(tasks):
            groups[(task['suite'], task['runtime_task_index'])].append(i)
        require(len(groups) == 40 and all(len(g) == 5 for g in groups.values())
                and all(sorted(tasks[i]['initial_state_index'] for i in g) == list(range(5))
                        for g in groups.values()), 'Clean bootstrap must retain 40 tasks x 5 states')
        units = np.array([values[g].mean(axis=0) for g in groups.values()])
        seed, unit = 8, 'canonical_task_with_all_five_states'
    else:
        units = values
        seed, unit = 7, 'variant_task'
    indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(units), (10000, len(units)))
    results = {}
    for left, right in CONTRASTS:
        delta = values[:, left] - values[:, right]
        sampled = (units[:, left]-units[:, right])[indices].mean(axis=1)*100
        results[ARMS[left]+' minus '+ARMS[right]] = {
            'count': len(tasks), 'bootstrap_unit_count': len(units),
            'left_successes': int(values[:, left].sum()), 'right_successes': int(values[:, right].sum()),
            'difference_pp': float(delta.mean()*100),
            'paired_95_ci_pp': np.quantile(sampled, [.025, .975], method='linear').tolist(),
            'left_only_success': int((delta == 1).sum()), 'right_only_success': int((delta == -1).sum()),
            'both_success': int(((values[:, left] == 1) & (values[:, right] == 1)).sum()),
            'both_failure': int(((values[:, left] == 0) & (values[:, right] == 0)).sum()),
            'bootstrap': {'draws': 10000, 'rng': 'PCG64', 'seed': seed, 'unit': unit,
                          'common_indices_across_arms': True, 'quantile_method': 'linear'},
        }
    return results


def secondary_statistics(outcomes, tasks, dev_keys, contact_keys):
    def key(t):
        return t['suite'], t['runtime_task_index']
    memberships = {
        'suite': [t['suite'] for t in tasks],
        'category': [t['perturbation_category'] for t in tasks],
        'difficulty': [str(t['difficulty_level']) if t['difficulty_level'] is not None else 'Long/unlabeled'
                       for t in tasks],
        'dev_overlap': ['dev-overlap90' if key(t) in dev_keys else 'other377' for t in tasks],
        'contact_proxy': ['contact-proxy98' if key(t) in contact_keys else 'other369' for t in tasks],
    }
    require(memberships['dev_overlap'].count('dev-overlap90') == 90, 'Dev overlap changed')
    require(memberships['contact_proxy'].count('contact-proxy98') == 98, 'Contact proxy changed')
    result = {}
    values = np.asarray(outcomes)
    for field, labels in memberships.items():
        result[field] = {}
        for label in sorted(set(labels)):
            positions = [i for i, x in enumerate(labels) if x == label]
            result[field][label] = paired_statistics(values[positions], [tasks[i] for i in positions])
    return result


def progression_decision(primary, clean, latency, limits, *, audit_passed, resources_passed):
    checks = {
        'artifacts_and_pairwise_inputs': bool(audit_passed),
        'resources': bool(resources_passed),
        'targeted_gain': primary['difference_pp'] >= limits['targeted_gain_pp_at_least'],
        'targeted_ci': primary['paired_95_ci_pp'][0] > limits['targeted_paired_ci_lower_pp_strictly_above'],
        'clean_gain': clean['difference_pp'] >= limits['clean_gain_pp_at_least'],
        'clean_ci': clean['paired_95_ci_pp'][0] >= limits['clean_paired_ci_lower_pp_at_least'],
    }
    for panel, value in latency.items():
        checks[panel+'_latency_ms'] = value['adaptive_p95_ms'] <= limits['adaptive_p95_ms_at_most_each_panel']
        checks[panel+'_latency_ratio'] = value['adaptive_to_fixed_p95_ratio'] <= limits['adaptive_to_fixed_p95_ratio_at_most_each_panel']
    passed = all(checks.values())
    return {'promising_for_week12': passed, 'checks': checks,
            'failed_checks': [k for k, v in checks.items() if not v],
            'recommendation': 'positive progression' if passed else 'no positive progression under v1',
            'classification': 'observed performance regression' if min(primary['difference_pp'], clean['difference_pp']) < 0 else 'inconclusive',
            'automatic_followup_launch': False,
            'inference_scope': 'conditional descriptive panel uncertainty; no training-seed or unseen-task generalization'}


def _row_bits_equal(left, right):
    require(left.shape == right.shape and left.dtype == right.dtype, 'Compared row schema differs')
    return np.all(np.ascontiguousarray(left).view(np.uint8).reshape(len(left), -1)
                  == np.ascontiguousarray(right).view(np.uint8).reshape(len(right), -1), axis=1)


def control_trajectory_pair(job):
    """Stream all non-timing fields; trace query input/action divergence exactly."""
    ordinal, task, left_row, right_row = job
    left_path, right_path = Path(left_row['artifact']), Path(right_row['artifact'])
    left_meta, right_meta = read_episode_metadata(left_path), read_episode_metadata(right_path)
    require(all(left_meta[k] == right_meta[k] for k in (
        'instruction', 'bddl_sha256', 'initial_state_sha256', 'episode_seed', 'suite', 'runtime_task_index')),
        'Control scene identity differs')
    result = {'ordinal': ordinal, 'task': task,
              'outcomes_equal': left_row['success'] == right_row['success'],
              'left_success': left_row['success'], 'right_success': right_row['success'],
              'left_policy_steps': left_row['policy_step_count'], 'right_policy_steps': right_row['policy_step_count']}
    with h5py.File(left_path) as left_h, h5py.File(right_path) as right_h:
        left, right = left_h['steps'], right_h['steps']
        require(set(left) == set(right), 'Control dataset inventory differs')
        nleft, nright = len(left['step_index']), len(right['step_index'])
        common = min(nleft, nright)
        masks, fields = {}, {}
        for name in left:
            if name == 'policy_latency_ms':
                continue
            a, b = left[name], right[name]
            require(a.shape[1:] == b.shape[1:] and a.dtype == b.dtype, 'Control field schema differs: '+name)
            masks[name] = np.ones(common, dtype=bool)
            changed_values, max_delta = 0, 0.
            for start in range(0, common, 32):
                aa, bb = a[start:min(common, start+32)], b[start:min(common, start+32)]
                equal = _row_bits_equal(aa, bb)
                masks[name][start:start+len(aa)] = equal
                if not equal.all():
                    # Byte equality above includes signed zero and NaN payloads;
                    # numeric magnitude is a separate descriptive measure.
                    valid = np.isfinite(aa) & np.isfinite(bb)
                    changed_values += int(np.count_nonzero((aa != bb) & valid))
                    if valid.any():
                        max_delta = max(max_delta, float(np.max(np.abs(aa[valid].astype(np.float64)-bb[valid].astype(np.float64)))))
            positions = np.flatnonzero(~masks[name]).tolist()
            fields[name] = {'all_rows_bitwise_equal': nleft == nright and not positions,
                'first_different_common_step': positions[0] if positions else None,
                'different_common_rows': len(positions), 'different_finite_values': changed_values,
                'max_finite_abs_delta': max_delta, 'left_extra_rows': max(0, nleft-common),
                'right_extra_rows': max(0, nright-common)}
        qleft = np.flatnonzero(np.isfinite(left['policy_latency_ms'][:]))
        qright = np.flatnonzero(np.isfinite(right['policy_latency_ms'][:]))
        nquery = min(len(qleft), len(qright))
        require(np.array_equal(qleft[:nquery], qright[:nquery]), 'Control common query step indices differ')
        steps = qleft[:nquery]
        raw_state_equal = masks['proprioception'][steps]
        effective_state_equal = _row_bits_equal(left['proprioception'][steps].astype(np.float32),
                                                right['proprioception'][steps].astype(np.float32))
        input_equal = masks['agent_rgb'][steps] & masks['wrist_rgb'][steps] & effective_state_equal
        action_equal = masks['predicted_action_chunk'][steps]
        input_diff = np.flatnonzero(~input_equal)
        action_diff = np.flatnonzero(~action_equal)
        unexplained = np.flatnonzero(input_equal & ~action_equal)
        first_context = None
        if len(input_diff):
            q = int(input_diff[0]); step = int(steps[q])
            pixel = {}
            for camera in ('agent_rgb', 'wrist_rgb'):
                delta = left[camera][step].astype(np.int16)-right[camera][step].astype(np.int16)
                pixel[camera] = {'different_channel_values': int(np.count_nonzero(delta)),
                                 'max_abs_delta': int(np.abs(delta).max())}
            first_context = {'query_index': q, 'recorded_step': step, 'camera_differences': pixel,
                'raw_proprioception_equal': bool(raw_state_equal[q]),
                'effective_float32_proprioception_equal': bool(effective_state_equal[q]),
                'simulator_state_equal_at_query': bool(masks['mujoco_state'][step]),
                'all_previous_executed_actions_equal': bool(masks['executed_action'][:step].all()),
                'action_chunk_equal_at_query': bool(action_equal[q])}
        result.update(left_recorded_steps=nleft, right_recorded_steps=nright, fields=fields,
            all_non_timing_steps_bitwise_equal=all(v['all_rows_bitwise_equal'] for v in fields.values()),
            common_query_count=nquery, left_query_count=len(qleft), right_query_count=len(qright),
            common_queries_with_equal_inputs=int(input_equal.sum()),
            equal_input_queries_with_different_actions=unexplained.tolist(),
            first_input_difference_query=int(input_diff[0]) if len(input_diff) else None,
            first_action_difference_query=int(action_diff[0]) if len(action_diff) else None,
            first_input_difference_context=first_context,
            input_difference_query_indices=input_diff.tolist(),
            action_difference_query_indices=action_diff.tolist())
    return result


def control_summary(pairs):
    return {'episode_count': len(pairs),
        'outcome_disagreements': [p['ordinal'] for p in pairs if not p['outcomes_equal']],
        'episode_length_disagreements': [p['ordinal'] for p in pairs if p['left_policy_steps'] != p['right_policy_steps']],
        'fully_bitwise_equal_trajectories': sum(p['all_non_timing_steps_bitwise_equal'] for p in pairs),
        'episodes_with_query_input_differences': sum(p['first_input_difference_query'] is not None for p in pairs),
        'episodes_with_action_differences': sum(p['first_action_difference_query'] is not None for p in pairs),
        'equal_input_query_count': sum(p['common_queries_with_equal_inputs'] for p in pairs),
        'equal_input_queries_with_different_actions': sum(len(p['equal_input_queries_with_different_actions']) for p in pairs),
        'left_total_queries': sum(p['left_query_count'] for p in pairs),
        'right_total_queries': sum(p['right_query_count'] for p in pairs),
        'first_input_differences_with_equal_state_and_previous_actions': sum(
            p['first_input_difference_context'] is not None
            and p['first_input_difference_context']['simulator_state_equal_at_query']
            and p['first_input_difference_context']['raw_proprioception_equal']
            and p['first_input_difference_context']['all_previous_executed_actions_equal'] for p in pairs),
        'timing_excluded_from_trajectory_equality': True,
        'input_definition': 'recorded uint8 camera bytes, float32 proprioception as consumed by action adapter, identical instruction'}

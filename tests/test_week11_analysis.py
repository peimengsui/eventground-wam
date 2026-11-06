"""Known paired effects, cluster sampling and causal-order recording comparisons."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from analyze_week11_formal import AuditBudget, query_summary

from eventground_wam.clean_comparison import atomic_json
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.week11_analysis import (
    control_summary,
    control_trajectory_pair,
    paired_statistics,
    progression_decision,
    secondary_statistics,
)

ROOT = Path(__file__).resolve().parents[1]


def test_common_indices_and_paired_counts():
    values = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [1, 1, 1]])
    result = paired_statistics(values, [{}]*4)
    primary = result['M4-EG-adaptive minus M4-fixed']
    assert primary['difference_pp'] == -25
    assert (primary['left_only_success'], primary['right_only_success'], primary['both_success'], primary['both_failure']) == (1, 2, 1, 0)
    indices = np.random.Generator(np.random.PCG64(7)).integers(0, 4, (10000, 4))
    expected = np.quantile(np.array([-1, -1, 1, 0])[indices].mean(1)*100, [.025, .975], method='linear')
    assert primary['paired_95_ci_pp'] == expected.tolist()
    assert result['M4-EG-adaptive minus M4-EG-fixed8'] == primary
    assert result['M4-EG-fixed8 minus M4-fixed']['paired_95_ci_pp'] == [0, 0]


def test_clean_bootstrap_samples_tasks_not_episodes():
    tasks = [{'suite': 'clean', 'runtime_task_index': i, 'initial_state_index': s}
             for i in range(40) for s in range(5)]
    values = np.ones((200, 3), dtype=int)
    values[:25, 2] = 0  # Five whole tasks fail; strong within-task dependence.
    result = paired_statistics(values, tasks, clean=True)['M4-EG-adaptive minus M4-fixed']
    units = np.r_[-np.ones(5), np.zeros(35)]
    indices = np.random.Generator(np.random.PCG64(8)).integers(0, 40, (10000, 40))
    expected = np.quantile(units[indices].mean(1)*100, [.025, .975], method='linear')
    assert result['paired_95_ci_pp'] == expected.tolist()
    assert result['bootstrap_unit_count'] == 40 and result['difference_pp'] == -12.5
    assert paired_statistics(values, tasks)['M4-EG-adaptive minus M4-fixed']['paired_95_ci_pp'] != expected.tolist()
    tasks[0]['initial_state_index'] = 1
    with pytest.raises(ValueError, match='40 tasks x 5 states'):
        paired_statistics(values, tasks, clean=True)


@pytest.mark.parametrize('values', [[], [[1, 0]], [[1, 1, 2]], [[1, 1, float('nan')]]])
def test_invalid_outcomes_rejected(values):
    with pytest.raises(ValueError):
        paired_statistics(values, [{}]*len(values))


def test_predeclared_subsets_keep_all_rows():
    tasks = json.loads((ROOT/'evaluation/splits/week11-integration-v1/targeted-467.json').read_text())['tasks']
    dev = json.loads((ROOT/'evaluation/splits/libero-plus-v1/dev-210.json').read_text())['tasks']
    contact = json.loads((ROOT/'evaluation/splits/week11-integration-v1/contact-proxy-98.json').read_text())['tasks']
    key = lambda t: (t['suite'], t['runtime_task_index'])
    result = secondary_statistics(np.zeros((467, 3), dtype=int), tasks, set(map(key, dev)), set(map(key, contact)))
    for groups in result.values():
        assert sum(g['M4-EG-adaptive minus M4-fixed']['count'] for g in groups.values()) == 467
    assert result['dev_overlap']['dev-overlap90']['M4-EG-adaptive minus M4-fixed']['count'] == 90
    assert result['contact_proxy']['contact-proxy98']['M4-EG-adaptive minus M4-fixed']['count'] == 98


def test_negative_point_estimates_cannot_pass_with_good_latency():
    limits = json.loads((ROOT/'configs/evaluation/week11_integration_v1.json').read_text())['decision']
    d = progression_decision({'difference_pp': -6.9, 'paired_95_ci_pp': [-10, -3]},
        {'difference_pp': -5, 'paired_95_ci_pp': [-8, -1]},
        {'targeted-467': {'adaptive_p95_ms': 108, 'adaptive_to_fixed_p95_ratio': 1.9}}, limits,
        audit_passed=True, resources_passed=True)
    assert not d['promising_for_week12']
    assert d['failed_checks'] == ['targeted_gain', 'targeted_ci', 'clean_gain', 'clean_ci']
    assert d['checks']['targeted-467_latency_ms'] and not d['automatic_followup_launch']


def recordings(tmp_path, nright=26):
    paths = [tmp_path/'left.hdf5', tmp_path/'right.hdf5']
    for path, n in zip(paths, [26, nright], strict=True):
        with h5py.File(path, 'w') as f:
            f.create_dataset('episode/metadata_json', data=json.dumps({k: 'same' for k in (
                'instruction', 'bddl_sha256', 'initial_state_sha256', 'episode_seed', 'suite', 'runtime_task_index')}))
            g = f.create_group('steps')
            data = {'step_index': np.arange(n), 'agent_rgb': np.zeros((n, 3, 3, 3), np.uint8),
                    'wrist_rgb': np.zeros((n, 3, 3, 3), np.uint8), 'proprioception': np.ones((n, 8)),
                    'mujoco_state': np.zeros((n, 12)), 'executed_action': np.zeros((n, 7)),
                    'predicted_action_chunk': np.zeros((n, 8, 7)), 'policy_latency_ms': np.full(n, np.nan)}
            data['predicted_action_chunk'][:10] = np.nan
            data['policy_latency_ms'][10::8] = 50 if path == paths[0] else 100
            for k, v in data.items():
                g.create_dataset(k, data=v)
    return (0, {'task_name': 'fixture'},
            {'artifact': str(paths[0]), 'success': True, 'policy_step_count': 16},
            {'artifact': str(paths[1]), 'success': True, 'policy_step_count': nright-10})


def test_timing_and_matching_nan_do_not_break_control_parity(tmp_path):
    pair = control_trajectory_pair(recordings(tmp_path))
    assert pair['all_non_timing_steps_bitwise_equal']
    assert pair['common_queries_with_equal_inputs'] == 2
    assert control_summary([pair])['equal_input_queries_with_different_actions'] == 0


def test_pixel_difference_precedes_action_divergence(tmp_path):
    job = recordings(tmp_path)
    with h5py.File(job[3]['artifact'], 'r+') as f:
        f['steps/wrist_rgb'][18, 0, 0, 0] = 1
        f['steps/predicted_action_chunk'][18:, 0, 0] = .25
        f['steps/executed_action'][18:, 0] = .25
    pair = control_trajectory_pair(job)
    assert pair['first_input_difference_query'] == pair['first_action_difference_query'] == 1
    assert pair['equal_input_queries_with_different_actions'] == []
    context = pair['first_input_difference_context']
    assert context['all_previous_executed_actions_equal'] and context['simulator_state_equal_at_query']
    assert context['camera_differences']['wrist_rgb'] == {'different_channel_values': 1, 'max_abs_delta': 1}


def test_action_difference_on_equal_inputs_is_flagged(tmp_path):
    job = recordings(tmp_path)
    with h5py.File(job[3]['artifact'], 'r+') as f:
        f['steps/predicted_action_chunk'][10, 0, 0] = .25
        # Differences removed by the adapter's float32 cast do not excuse actions.
        f['steps/proprioception'][10, 0] = 1.+1e-12
    pair = control_trajectory_pair(job)
    assert pair['equal_input_queries_with_different_actions'] == [0]
    assert pair['first_input_difference_query'] is None


def test_unequal_lengths_preserve_common_queries_and_extras(tmp_path):
    pair = control_trajectory_pair(recordings(tmp_path, nright=34))
    assert not pair['all_non_timing_steps_bitwise_equal']
    assert (pair['left_query_count'], pair['right_query_count'], pair['common_query_count']) == (2, 3, 2)
    assert pair['fields']['wrist_rgb']['right_extra_rows'] == 8
    assert pair['equal_input_queries_with_different_actions'] == []


def test_component_summary_flattens_nested_timing_dictionary(tmp_path):
    path = tmp_path/'timing.json'
    policy = {'used_horizon': 8, 'prediction': None, 'decision': None, 'event': None,
              'component_ms': {'action_forward': 44., 'future_core': 2.},
              'action_branch_ms': 52., 'wrapper_wall_ms': 53.}
    queries = [{'policy': policy, 'exception': None, 'remaining_chunk_actions': 8}]*2
    atomic_json(path, seal_contract({'costs': {'simulator_seconds': 1.}, 'queries': queries}))
    summary = query_summary([{'timing_artifact': str(path)}])
    assert summary['component_latency_ms']['future_core'] == {'count': 2, 'median': 2., 'p95': 2.}
    assert 'component_ms' not in summary['component_latency_ms']
    assert summary['used_horizons'] == {'8': 2}


def test_failed_analysis_time_remains_charged(tmp_path, monkeypatch):
    fake_process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1), children=lambda recursive: [])
    monkeypatch.setitem(sys.modules, 'psutil', SimpleNamespace(Process=lambda: fake_process))
    recordings_root, prior, output = tmp_path/'recordings', tmp_path/'prior', tmp_path/'output'
    output.mkdir()
    for arm in ('M4-fixed', 'M4-EG-fixed8', 'M4-EG-adaptive'):
        for panel in ('targeted-467', 'clean-200'):
            atomic_json(recordings_root/'panel-budgets'/arm/(panel+'.json'),
                        seal_contract({'identity': 'evaluation', 'running': False, 'active_seconds': 10.}))
    atomic_json(prior/'active-budget.json', seal_contract({'identity': 'old-analysis', 'running': False, 'active_seconds': 20.}))
    atomic_json(prior/'panel-budgets/M4-fixed/targeted-467.json',
                seal_contract({'identity': 'old-analysis/panel', 'running': False, 'active_seconds': 5.}))
    limits = json.loads((ROOT/'configs/evaluation/week11_integration_v1.json').read_text())['resources']
    limits['minimum_free_bytes'] = 0  # Unit test's local disk is not Feifei's reserve.
    budget = AuditBudget(output, 'new-analysis', limits,
        {'resources': {'total_active_seconds': 100., 'qualification_output_bytes': 10}}, recordings_root, [prior])
    assert budget.prior_seconds == 120.
    assert budget.prior_panels['M4-fixed/targeted-467'] == 15.
    result = budget.close()
    assert result['total_active_seconds'] >= 120.
    assert result['prior_analysis_attempts'][0]['source_digest'] == 'old-analysis'

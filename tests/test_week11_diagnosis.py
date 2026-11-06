"""Boundary checks for the exploratory recorder-only diagnostic."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np

spec = importlib.util.spec_from_file_location('diagnosis', Path(__file__).parents[1]/'scripts/investigate_week11_adaptive.py')
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def data(n=50, onset=10):
    a, p = np.zeros((n, 7)), np.zeros((n, 8))
    a[:, 6] = -1; a[onset:, 6] = 1
    p[:, 6:8] = [.04, -.04]; p[onset+5:, 6:8] = [.02, -.02]
    return a, p


def test_physical_closing_proxy():
    a, p = data()
    x = d.closing_proxy(a, p)
    assert x['command_onsets'] == x['confirmed_onsets'] == [10]


def test_no_initial_close_onset():
    a, p = data(onset=0)
    x = d.closing_proxy(a, p)
    assert x['command_onsets'] == [] and x['initial_close_left_censored']


def test_weak_and_short_and_incomplete():
    a, p = data(); p[:] = 0
    assert d.closing_proxy(a, p)['weak_response'] == [10]
    a[12:, 6] = -1
    assert d.closing_proxy(a, p)['short_command'] == [10]
    a, p = data(n=15)
    assert d.closing_proxy(a, p)['incomplete_response'] == [10]


def test_proxy_censor_boundary_and_current_onset_exclusion():
    proxy = {'command_onsets': [10, 42], 'confirmed_onsets': [10, 42]}
    assert not d.proxy_at_query(10, 32, 47, proxy)['fully_observed_32_plus_response']
    x = d.proxy_at_query(10, 32, 48, proxy)
    assert x['fully_observed_32_plus_response'] and x['confirmed_next_offset'] == 32
    assert x['confirmed_nearest_prediction_error'] == 0


def test_censored_not_false_negative():
    x = d.proxy_at_query(10, 2, 20, {'command_onsets': [], 'confirmed_onsets': []})
    assert x['confirmed_any_in_window'] is None


def test_action_comparison_separates_gripper_and_motion():
    a = np.zeros((8, 7)); b = a.copy(); b[3, 6] = 1
    x = d.action_difference(a, b)
    assert not x['equal'] and x['translation_rms'] == 0 and x['gripper_disagree_actions'] == 1


def test_outcome_and_sequence_metrics():
    assert d.outcome_class([1, 1, 0]) == 'adaptive_loss'
    assert d.outcome_class([0, 0, 1]) == 'adaptive_win'
    qs = [{'used': h, 'event': 'contact', 'start': s} for h, s in [(32, 30), (4, 4), (32, 30)]]
    x = d.sequence_metrics(qs)
    assert x['switch_fraction'] == 1 and x['short_long_switches'] == 2
    assert x['same_type_offset_not_decreasing'] == 1


def test_seal_matches_repository():
    from eventground_wam.demonstration_contracts import seal_contract
    assert d.seal({'x': [1, None, 'é']}) == seal_contract({'x': [1, None, 'é']})


def pair_files(tmp_path, *, camera_difference=False, effective_state_difference=False):
    paths = [tmp_path/'fixed.h5', tmp_path/'adaptive.h5']
    for i, path in enumerate(paths):
        with h5py.File(path, 'w') as h:
            s = h.create_group('steps')
            s['step_index'] = np.arange(26)
            s['agent_rgb'] = np.zeros((26, 2, 2, 3), dtype=np.uint8)
            s['wrist_rgb'] = np.zeros((26, 2, 2, 3), dtype=np.uint8)
            s['proprioception'] = np.ones((26, 8))
            s['mujoco_state'] = np.zeros((26, 12))
            s['executed_action'] = np.zeros((26, 7))
            s['predicted_action_chunk'] = np.zeros((26, 8, 7))
            if i:
                s['predicted_action_chunk'][18, 0, 0] = .1
                s['proprioception'][18, 0] = 1.1 if effective_state_difference else 1.+1e-9
                if camera_difference:
                    s['wrist_rgb'][18, 0, 0, 0] = 1
    adaptive = {'queries': [{'query': q, 'step': step, 'used': used,
                            'event': 'contact', 'start': 32} for q, step, used in [(0, 10, 8), (1, 18, 32)]]}
    return paths, adaptive


def test_same_effective_input_horizon_change(tmp_path):
    paths, adaptive = pair_files(tmp_path)
    x = d.first_divergence(*paths, adaptive)
    assert len(x['identical_input_queries']) == 2
    assert x['identical_input_queries'][0]['action']['equal']
    assert x['first_action_difference']['query'] == 1
    assert x['first_non8_at_shared_prefix']['inputs_equal']
    assert x['first_input_difference'] is None


def test_render_difference_not_called_horizon_only(tmp_path):
    paths, adaptive = pair_files(tmp_path, camera_difference=True)
    x = d.first_divergence(*paths, adaptive)
    assert len(x['identical_input_queries']) == 1
    assert not x['first_non8_at_shared_prefix']['inputs_equal']
    assert x['first_input_difference']['previous_executed_actions_equal']


def test_effective_proprio_difference_excludes_matched_query(tmp_path):
    paths, adaptive = pair_files(tmp_path, effective_state_difference=True)
    x = d.first_divergence(*paths, adaptive)
    assert len(x['identical_input_queries']) == 1
    assert x['first_input_difference']['query'] == 1

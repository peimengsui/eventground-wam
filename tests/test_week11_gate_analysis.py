import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_week11_gate_comparison import paired, prefix_agreement


def test_paired_contrast_keeps_wins_and_losses_separate():
    rows = [{'id': str(i), 'outcomes': {'gated': a, 'original': b}}
            for i, (a, b) in enumerate([(True, False), (True, False),
                                        (False, True), (True, True), (False, False)])]
    result = paired(rows, 'gated', 'original')
    assert result['n'] == 5 and result['gap_pp'] == 20
    assert result['left_only_cases'] == ['0', '1'] and result['right_only_cases'] == ['2']
    assert result['both_success'] == result['both_failure'] == 1
    assert paired(rows, 'original', 'gated')['gap_pp'] == -20


def recordings(tmp_path):
    paths = [tmp_path/'fixed.hdf5', tmp_path/'gated.hdf5']
    for path in paths:
        with h5py.File(path, 'w') as handle:
            s = handle.create_group('steps')
            for name, shape in [('agent_rgb', (27, 2, 2, 3)), ('wrist_rgb', (27, 2, 2, 3)),
                                ('proprioception', (27, 8)), ('mujoco_state', (27, 10)),
                                ('executed_action', (27, 7)), ('predicted_action_chunk', (27, 8, 7))]:
                s.create_dataset(name, data=np.zeros(shape))
    queries = [{'exception': None, 'policy': {'closing_history': {'active': i == 2}}} for i in range(3)]
    return paths, queries


def test_postactivation_difference_is_allowed_but_same_input_pre_activation_must_match(tmp_path):
    paths, queries = recordings(tmp_path)
    with h5py.File(paths[1], 'a') as h:
        h['steps/predicted_action_chunk'][26] = 1
    result = prefix_agreement(*paths, queries)
    assert result['all_inputs_and_actions_exact'] and result['same_input_queries'] == 2
    with h5py.File(paths[1], 'a') as h:
        h['steps/predicted_action_chunk'][18] = 1
    with pytest.raises(AssertionError, match='inactive gate altered'):
        prefix_agreement(*paths, queries)


def test_camera_difference_is_preserved_as_a_qualification(tmp_path):
    paths, queries = recordings(tmp_path)
    with h5py.File(paths[1], 'a') as h:
        h['steps/agent_rgb'][18] = 1
    result = prefix_agreement(*paths, queries)
    assert not result['all_inputs_and_actions_exact']
    assert result['first_difference']['query'] == 1
    assert result['first_difference']['actions_equal']
    assert result['same_input_queries'] == 1

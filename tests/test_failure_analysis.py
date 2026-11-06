import numpy as np
import pytest

from eventground_wam.failure_analysis import grouped_success, motion_metrics, success_summary


def test_wilson_counts_zero_and_full_are_not_certain():
    for successes in (0, 30):
        rows = [{'success': i < successes, 'termination_reason': 'done' if i < successes else 'timeout'} for i in range(30)]
        r = success_summary(rows)
        assert r['successes'] == successes and r['count'] == 30
        assert r['wilson_95'][0] < r['wilson_95'][1]
        assert 0 <= r['wilson_95'][0] <= r['success_rate'] + 1e-15
        assert r['success_rate'] <= r['wilson_95'][1] + 1e-15 <= 1 + 1e-15
    assert success_summary([])['success_rate'] is None


def test_motion_flags_distinguish_low_command_and_no_motion():
    actions = np.zeros((50, 7)); actions[:, 6] = -1
    state = np.zeros((50, 8))
    quiet = motion_metrics(actions, state)
    assert quiet['late_low_motion'] and not quiet['late_low_motion_with_command']
    assert quiet['first_close_command_policy_step'] is None
    actions[:, 0] = 1
    actions[20:, 6] = 1
    blocked = motion_metrics(actions, state)
    assert blocked['late_low_motion_with_command']
    assert blocked['continuous_saturation_fraction'] == pytest.approx(1 / 6)
    assert blocked['first_close_command_policy_step'] == 20
    assert blocked['gripper_command_switches'] == 1
    state[:, 0] = np.arange(50) / 100
    assert not motion_metrics(actions, state)['late_low_motion']


def test_short_and_empty_rollouts_do_not_get_stall_label():
    assert motion_metrics(np.zeros((0, 7)), np.zeros((0, 8)))['late_low_motion'] is None
    assert not motion_metrics(np.zeros((10, 7)), np.zeros((10, 8)))['late_low_motion']
    with pytest.raises(ValueError):
        motion_metrics(np.full((50, 7), np.nan), np.zeros((50, 8)))


def test_category_difficulty_denominators_remain_separate():
    rows = [{'category': 'A', 'difficulty': 1, 'success': True, 'termination_reason': 'done'},
            {'category': 'A', 'difficulty': 2, 'success': False, 'termination_reason': 'timeout'}]
    groups = grouped_success(rows, ('category', 'difficulty'))
    assert groups['A | 1']['count'] == groups['A | 2']['count'] == 1
    assert groups['A | 1']['success_rate'] == 1 and groups['A | 2']['success_rate'] == 0


def test_hdf5_policy_phase_is_numeric_and_settling_is_excluded(tmp_path):
    from test_episode_recorder import episode_metadata, step_record

    from eventground_wam.episode_recorder import EpisodeRecorder
    from eventground_wam.evaluation_protocol import load_evaluation_protocol
    from eventground_wam.failure_analysis import inspect_trajectory
    protocol = load_evaluation_protocol()
    path = tmp_path / 'episode.hdf5'
    with EpisodeRecorder(path, protocol, episode_metadata(protocol)) as recorder:
        for i in range(13):
            recorder.append(step_record(protocol, i, policy=i >= 10, done=i == 12))
        recorder.finalize(success=True, termination_reason='environment_done')
    metrics = inspect_trajectory(path)
    assert metrics['policy_steps'] == 3
    assert metrics['policy_query_count'] == 1
    assert metrics['policy_latency_median_ms'] == 1.25

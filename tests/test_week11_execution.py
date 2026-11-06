from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_week9_execution import Environment
from test_week9_execution import case as original_case

from eventground_wam.b0_training import read_sealed
from eventground_wam.week11_execution import execute_episode, validate_query_artifact
from eventground_wam.week11_recording import run_episode
from eventground_wam.week11_resources import QualificationStopped, competing_gpu_jobs


@pytest.fixture
def case(tmp_path):
    return original_case.__wrapped__(tmp_path)


def test_contract_errors_are_not_retried(case):
    def wrong_contract():
        raise ValueError('bad contract')
    case['environment_factory'] = wrong_contract
    with pytest.raises(ValueError, match='bad contract'):
        execute_episode(**case)
    journal = read_sealed(case['root']/'episode-0000.attempts.json')
    assert journal['attempts'] == [0]
    error = read_sealed(case['root']/'episode-0000-attempt-0-error.json')
    assert not error['classified_infrastructure']


def test_interruption_retains_attempt_then_resumes_with_one_retry(case):
    def stop():
        raise QualificationStopped('interrupt')
    case['environment_factory'] = stop
    with pytest.raises(QualificationStopped):
        execute_episode(**case)
    case['environment_factory'] = lambda: SimpleNamespace(close=lambda: None)
    row = execute_episode(**case)
    assert row['retry_count'] == 1
    assert execute_episode(**case) == row


def test_exhausted_infrastructure_failure_stays_terminal(case):
    def unavailable():
        raise OSError('renderer unavailable')
    case['environment_factory'] = unavailable
    row = execute_episode(**case)
    assert row['artifact'] is None and not row['success']
    assert execute_episode(**case) == row


@pytest.mark.parametrize('fail', [False, True])
def test_durable_query_sidecar_resume_and_failure_scoring(case, fail, monkeypatch):
    monkeypatch.setattr('evaluate_reference_policy.proprioception_from_observation',
                        lambda _obs: np.zeros(8))
    monkeypatch.setattr('eventground_wam.week11_recording.proprioception_from_observation',
                        lambda _obs: np.zeros(8))
    admission = {'m4': 'pinned'}
    policy = SimpleNamespace(last_query=None)
    def predict(**_kw):
        policy.last_query = {'arm': 'M4-fixed', 'branch_identities': admission, 'prediction': None,
            'decision': None, 'event': None, 'used_horizon': 8, 'future_calls': 1,
            'proposal_actions': 8, 'zero_padded_actions': 24, 'exception': None}
        if fail:
            policy.last_query['exception'] = 'ValueError: policy failure'
            raise ValueError('policy failure')
        return np.zeros((8, 7))
    policy.predict_chunk = predict
    case['policy'] = policy
    case['environment_factory'] = EnvironmentWithClose
    case['episode_runner'] = lambda **kw: run_episode(**kw, timing_synchronize=lambda: None)
    row = execute_episode(**case)
    assert row['success'] is (not fail)
    assert row['retry_count'] == 0
    assert len(row['policy_latencies_ms']) == 1
    assert validate_query_artifact(row, 'M4-fixed', admission)
    assert execute_episode(**case) == row
    Path(row['artifact']).with_suffix('.queries.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='Query artifact changed'):
        validate_query_artifact(row, 'M4-fixed', admission)


class EnvironmentWithClose(Environment):
    def close(self):
        pass


def test_desktop_gpu_classification_does_not_ignore_small_compute_jobs():
    rows = '1, /usr/bin/ptyxis, 52\n2, python, 20\n3, /usr/bin/ptyxis, 257\n4, python, 1000'
    assert competing_gpu_jobs(rows, own_pid=4) == [2, 3]

from types import SimpleNamespace

import numpy as np
import pytest

from eventground_wam.week11_closing_gate import (
    ADAPTIVE,
    FIXED,
    GATED,
    ClosingGatePolicy,
    ClosingHistory,
    ExecutedCommandObserver,
)


def action(command):
    return np.array([0., 0., 0., 0., 0., 0., command])


def wrappers():
    def make(arm):
        p = SimpleNamespace(checkpoint=None, last_query=None, calls=0)
        def predict(**kw):
            p.calls += 1
            p.last_query = {'arm': arm}
            return np.stack([action(-1), action(1)]+[action(-1)]*6)
        p.predict_chunk = predict
        return p
    return make(FIXED), make(ADAPTIVE)


OBS = {'agent_rgb': None, 'wrist_rgb': None, 'proprioception': None, 'instruction': 'x', 'suite': 'x'}


def test_only_executed_transition_enables_next_query_and_reset_clears():
    fixed, adaptive = wrappers()
    p = ClosingGatePolicy(fixed, adaptive, GATED)
    p.history.observe(action(-1))
    chunk = p.predict_chunk(**OBS)
    assert not p.last_query['closing_history']['active']
    assert not p.history.active  # The proposed chunk already contains a close.
    p.history.observe(chunk[0])
    p.predict_chunk(**OBS)  # Interrupted before the close: still fixed.
    assert adaptive.calls == 0
    p.history.observe(chunk[1])
    assert p.last_query['closing_history']['active'] is False  # Snapshot, not mutable alias.
    p.predict_chunk(**OBS)
    assert adaptive.calls == 1 and p.last_query['closing_history']['first_closing_step'] == 2
    p.history.observe(action(-1))
    p.predict_chunk(**OBS)
    assert adaptive.calls == 2  # Reopening does not reset the latch.
    p.reset()
    p.predict_chunk(**OBS)
    assert not p.history.active and adaptive.calls == 2


def test_initial_closed_command_is_not_an_open_to_close_transition():
    h = ClosingHistory()
    h.observe(action(1)); h.observe(action(1))
    assert not h.active
    h.observe(action(-1)); h.observe(action(1))
    assert h.first_closing_step == 3


def test_environment_failure_cannot_activate_gate():
    history = ClosingHistory()
    history.observe(action(-1))
    def fail(a):
        raise OSError('step failed')
    env = ExecutedCommandObserver(SimpleNamespace(step=fail), history)
    with pytest.raises(OSError):
        env.step(action(1))
    assert not history.active and history.executed_steps == 1
    env.environment.step = lambda a: ('obs', 0, True, {})
    assert env.step(action(1))[2]
    assert history.active and history.executed_steps == 2


@pytest.mark.parametrize('arm,expected', [(FIXED, FIXED), (ADAPTIVE, ADAPTIVE)])
def test_original_controls_ignore_latch_for_routing(arm, expected):
    p = ClosingGatePolicy(*wrappers(), arm)
    p.predict_chunk(**OBS)
    assert p.last_query['arm'] == expected
    p.history.observe(action(-1)); p.history.observe(action(1))
    p.predict_chunk(**OBS)
    assert p.last_query['arm'] == expected


@pytest.mark.parametrize('bad', [action(0), action(float('nan')), np.zeros(8)])
def test_invalid_commands_rejected(bad):
    with pytest.raises(ValueError):
        ClosingHistory().observe(bad)


def test_future_observation_or_labels_cannot_be_passed_to_policy():
    p = ClosingGatePolicy(*wrappers(), GATED)
    with pytest.raises(TypeError):
        p.predict_chunk(**OBS, future_event='grasp')


def test_recorder_activates_next_chunk_and_independent_audit_rejects_early_activation(tmp_path, monkeypatch):
    import json
    from dataclasses import asdict

    from test_week9_execution import Environment
    from test_week9_execution import case as original_case

    from eventground_wam.b0_training import read_sealed, sha256_file
    from eventground_wam.clean_comparison import atomic_json
    from eventground_wam.demonstration_contracts import seal_contract
    from eventground_wam.eventground_horizon import select_horizon
    from eventground_wam.week11_closing_gate import audit_gate, run_gated_episode

    case = original_case.__wrapped__(tmp_path)
    monkeypatch.setattr('eventground_wam.week11_recording.proprioception_from_observation',
                        lambda obs: np.zeros(8))
    monkeypatch.setattr('evaluate_reference_policy.proprioception_from_observation',
                        lambda obs: np.zeros(8))

    class TwoChunks(Environment):
        def step(self, a):
            self.n += 1
            return self.obs(), 0., self.n == 26, {}

    pred = {'next_event': 'contact', 'interval': [32, 33], 'horizon': 32}
    def wrapper(arm):
        p = SimpleNamespace(checkpoint=None, last_query=None)
        def predict(**kw):
            p.last_query = {'arm': arm, 'branch_identities': {}, 'exception': None,
                'prediction': pred, 'decision': asdict(select_horizon(pred)),
                'used_horizon': 8 if arm == FIXED else 32, 'future_calls': 1,
                'proposal_actions': 8, 'execute_actions': 8, 'zero_padded_actions': 24,
                'event': {'categorical_probabilities': {'event': [0., 0., 1., 0.],
                          'start': [0.]*31+[1.], 'end': [0.]*31+[1.]}}}
            return np.stack([action(-1)]*3+[action(1)]*5)
        p.predict_chunk = predict
        return p
    policy = ClosingGatePolicy(wrapper(FIXED), wrapper(ADAPTIVE), GATED)
    path = tmp_path/'recording.hdf5'
    run_gated_episode(environment=TwoChunks(), policy=policy, protocol=case['protocol'],
        metadata=case['metadata'], initial_state=case['initial_state'], artifact_path=path,
        timing_synchronize=lambda: None)
    audit = audit_gate({'artifact': str(path)}, GATED, {})
    assert audit['first_closing_step'] == 13
    assert audit['first_query_after_closing'] == 1 and audit['horizons'] == [8, 32]
    qpath = path.with_suffix('.queries.jsonl')
    queries = [json.loads(line) for line in qpath.read_text().splitlines()]
    queries[0]['policy']['closing_history'].update(first_closing_step=13, active=True)
    qpath.write_text(''.join(json.dumps(q)+'\n' for q in queries))
    tpath = path.with_suffix('.timing.json')
    timing = read_sealed(tpath)
    timing.pop('digest')
    atomic_json(tpath, seal_contract({**timing, 'queries': queries,
                                     'query_artifact_sha256': sha256_file(qpath)}))
    with pytest.raises(ValueError, match='unexecuted or stale'):
        audit_gate({'artifact': str(path)}, GATED, {})

from types import SimpleNamespace

import numpy as np
import pytest

from eventground_wam.week11_closing_gate import ADAPTIVE, FIXED, GATED
from eventground_wam.week11_first_chunk import ARMS, ONLY, SKIP, FirstChunkPolicy, audit_first_chunk

OBS = {'agent_rgb': None, 'wrist_rgb': None, 'proprioception': None, 'instruction': 'x', 'suite': 'x'}


def action(command):
    return np.array([0., 0., 0., 0., 0., 0., command])


def wrapper(arm, horizon=4):
    from dataclasses import asdict

    from eventground_wam.eventground_horizon import select_horizon
    pred = {'next_event': 'grasp', 'interval': [horizon, 33], 'horizon': horizon}
    p = SimpleNamespace(checkpoint=None, last_query=None)
    def predict(**kwargs):
        p.last_query = {'arm': arm, 'branch_identities': {}, 'exception': None,
            'prediction': pred, 'decision': asdict(select_horizon(pred)),
            'used_horizon': 8 if arm == FIXED else horizon, 'future_calls': 1,
            'proposal_actions': 8, 'execute_actions': 8, 'zero_padded_actions': 24,
            'event': {'categorical_probabilities': {'event': [0., 0., 0., 1.],
                'start': [0.]*(horizon-1)+[1.]+[0.]*(32-horizon), 'end': [0.]*31+[1.]}}}
        return np.stack([action(-1)]*3+[action(1)]*5)
    p.predict_chunk = predict
    return p


@pytest.mark.parametrize('arm,expected', [(FIXED,[FIXED]*4), (GATED,[FIXED]+[ADAPTIVE]*3),
    (SKIP,[FIXED,FIXED,ADAPTIVE,ADAPTIVE]), (ONLY,[FIXED,ADAPTIVE,FIXED,FIXED])])
def test_one_chunk_schedule_reopening_and_reset(arm, expected):
    p=FirstChunkPolicy(wrapper(FIXED),wrapper(ADAPTIVE),arm)
    observed=[]
    p.predict_chunk(**OBS);observed.append(p.last_query['arm'])
    assert not p.history.active  # Proposed closing must not trigger activation.
    p.history.observe(action(-1));p.history.observe(action(1))
    for _ in range(3):
        p.predict_chunk(**OBS);observed.append(p.last_query['arm'])
        p.history.observe(action(-1));p.history.observe(action(1))
    assert observed==expected and p.active_queries==3
    p.reset();p.predict_chunk(**OBS)
    assert p.last_query['arm']==FIXED and p.active_queries==0


def test_changed_first_forecast_fails_without_consuming_intervention():
    p=FirstChunkPolicy(wrapper(FIXED,16),wrapper(ADAPTIVE,16),ONLY)
    p.history.observe(action(-1));p.history.observe(action(1))
    with pytest.raises(ValueError,match='frozen horizon 4'):p.predict_chunk(**OBS)
    assert p.active_queries==0 and 'frozen horizon 4' in p.last_query['exception']


@pytest.mark.parametrize('arm,expected', [(FIXED,[8,8,8,8]), (GATED,[8,4,4,4]),
    (SKIP,[8,8,4,4]), (ONLY,[8,4,8,8])])
def test_real_recorder_and_independent_audit(tmp_path,monkeypatch,arm,expected):
    import json

    from test_week9_execution import Environment
    from test_week9_execution import case as original_case

    from eventground_wam.b0_training import read_sealed, sha256_file
    from eventground_wam.clean_comparison import atomic_json
    from eventground_wam.demonstration_contracts import seal_contract
    from eventground_wam.week11_closing_gate import run_gated_episode
    case=original_case.__wrapped__(tmp_path)
    monkeypatch.setattr('eventground_wam.week11_recording.proprioception_from_observation',lambda obs:np.zeros(8))
    monkeypatch.setattr('evaluate_reference_policy.proprioception_from_observation',lambda obs:np.zeros(8))
    class FourChunks(Environment):
        def step(self,a):
            self.n+=1
            return self.obs(),0.,self.n==42,{}
    p=FirstChunkPolicy(wrapper(FIXED),wrapper(ADAPTIVE),arm)
    path=tmp_path/'recording.hdf5'
    run_gated_episode(environment=FourChunks(),policy=p,protocol=case['protocol'],metadata=case['metadata'],
        initial_state=case['initial_state'],artifact_path=path,timing_synchronize=lambda:None)
    a=audit_first_chunk({'artifact':str(path)},arm,{})
    assert a['horizons']==expected and a['first_closing_step']==13 and a['first_query_after_closing']==1
    # A consistent reseal does not hide an incorrect counter from the independent command-stream audit.
    qpath=path.with_suffix('.queries.jsonl');queries=[json.loads(l) for l in qpath.read_text().splitlines()]
    queries[2]['policy']['post_close_query_index']=0
    qpath.write_text(''.join(json.dumps(q)+'\n' for q in queries))
    tpath=path.with_suffix('.timing.json');t=read_sealed(tpath);t.pop('digest')
    atomic_json(tpath,seal_contract({**t,'queries':queries,'query_artifact_sha256':sha256_file(qpath)}))
    with pytest.raises(ValueError,match='post-close query index'):audit_first_chunk({'artifact':str(path)},arm,{})


def test_unknown_arm_and_future_labels_rejected():
    with pytest.raises(ValueError):FirstChunkPolicy(wrapper(FIXED),wrapper(ADAPTIVE),'unknown')
    for arm in ARMS:
        p=FirstChunkPolicy(wrapper(FIXED),wrapper(ADAPTIVE),arm)
        with pytest.raises(TypeError):p.predict_chunk(**OBS,future_event='grasp')

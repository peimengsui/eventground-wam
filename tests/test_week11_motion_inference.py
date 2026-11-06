from types import SimpleNamespace

import numpy as np
import pytest
import torch
from week11_motion_adapter import MotionInference


class Action:
    checkpoint = 'frozen'
    _device = 'cpu'

    def __init__(self):
        self._policy = torch.nn.Linear(1, 1)
        self._policy.arm = 'horizon-motion'
        self._policy.horizons = None
        self.seen = []
        self.fail = False

    def predict_chunk(self, **_):
        self.seen.append(self._policy.horizons[0])
        if self.fail:
            raise RuntimeError('action failure')
        return np.zeros((8,7),dtype=np.float64)


def setup():
    action = Action()
    event = SimpleNamespace(model=torch.nn.Linear(1,1), prediction=None)
    event.predict = lambda _: (event.prediction, {})
    policy = MotionInference(action,event)
    obs = {'agent_rgb':np.zeros((256,256,3),dtype=np.uint8),
           'wrist_rgb':np.zeros((256,256,3),dtype=np.uint8),
           'proprioception':np.zeros(8),'instruction':'test','suite':'test'}
    return action,event,policy,obs


def test_all_onsets_none_and_eight_action_cadence():
    action,event,policy,obs=setup()
    bins=[2]*3+[4]*3+[8]*6+[16]*12+[32]*8
    for start,expected in enumerate(bins,1):
        event.prediction={'next_event':'contact','interval':[start,start+1],'horizon':start}
        assert policy.predict_chunk(**obs).shape==(8,7)
        assert action.seen[-1]==expected and action._policy.horizons is None
        assert policy.last_query['execute_actions']==8
        assert policy.last_query['future_calls']==1+int(expected!=8)
    event.prediction={'next_event':'none','interval':None,'horizon':None}
    policy.predict_chunk(**obs)
    assert action.seen[-1]==8 and policy.last_query['decision']['fallback_reason']=='no_new_onset'


@pytest.mark.parametrize('prediction',[{},None,{'next_event':'contact','interval':[33,34],'horizon':33}])
def test_invalid_forecast_does_not_execute_action_and_unlocks(prediction):
    action,event,policy,obs=setup();event.prediction=prediction
    with pytest.raises(ValueError):policy.predict_chunk(**obs)
    assert not action.seen and action._policy.horizons is None and not policy.lock.locked()
    assert policy.last_query['exception']
    event.prediction={'next_event':'none','interval':None,'horizon':None}
    policy.predict_chunk(**obs);assert action.seen==[8]


def test_action_failure_resets_horizon_and_lock():
    action,event,policy,obs=setup()
    event.prediction={'next_event':'grasp','interval':[24,25],'horizon':24}
    action.fail=True
    with pytest.raises(RuntimeError):policy.predict_chunk(**obs)
    assert action._policy.horizons is None and not policy.lock.locked()
    action.fail=False;policy.predict_chunk(**obs)
    assert action.seen==[16,16]

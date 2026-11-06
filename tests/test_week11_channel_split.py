"""Check channel composition, boundaries, candidate equality and reset."""
import numpy as np
import pytest
from week11_channel_split import ARMS, WindowPolicy


class Candidate:
    checkpoint='same'

    def __init__(self,motion,grip):
        self.value=np.full((8,7),motion);self.value[:,6]=grip;self.calls=0
        self.last_query={'prediction':'same','decision':{},'event':{'categorical_probabilities':{}},'exception':None}

    def predict_chunk(self,**_):
        self.calls+=1;return self.value


@pytest.mark.parametrize('arm',ARMS)
def test_hybrid_uses_current_candidates_and_stops_at_138(arm):
    fixed=Candidate(.1,-1);adaptive=Candidate(.2,1);p=WindowPolicy(fixed,adaptive,arm)
    for frame in (10,34,130,138,218,522):
        p.history.executed_steps=frame;before=(fixed.calls,adaptive.calls);got=p.predict_chunk()
        assert np.all(got[:,:6]==(.2 if frame<138 and arm in ('early-only','motion-only') else .1))
        assert np.all(got[:,6]==(1 if frame<138 and arm in ('early-only','gripper-only') else -1))
        assert (fixed.calls-before[0],adaptive.calls-before[1])==(1,int(frame<138))
    p.reset();assert p.history.executed_steps==0 and p.last_query is None
    with pytest.raises(ValueError):p.predict_chunk()


def test_disagreeing_event_predictions_are_rejected():
    fixed=Candidate(.1,-1);adaptive=Candidate(.2,1)
    adaptive.last_query={**adaptive.last_query,'prediction':'different'}
    p=WindowPolicy(fixed,adaptive,'motion-only');p.history.executed_steps=34
    with pytest.raises(AssertionError):p.predict_chunk()
    assert 'AssertionError' in p.last_query['exception']

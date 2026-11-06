"""Boundary, cadence and executed-step regressions for the causal intervention."""
from types import SimpleNamespace

import numpy as np
import pytest
from week11_phase_split import ARMS, WindowPolicy

from eventground_wam.week11_closing_gate import ExecutedCommandObserver


class FakePolicy:
    checkpoint = 'same'

    def __init__(self, value):
        self.value = value
        self.last_query = {'arm': str(value)}

    def predict_chunk(self, **_):
        return self.value


@pytest.mark.parametrize('arm', ARMS)
def test_all_query_frames_and_reset(arm):
    p = WindowPolicy(FakePolicy(0), FakePolicy(1), arm)
    seen = []
    env = ExecutedCommandObserver(SimpleNamespace(step=lambda _: None), p.history)
    for frame in range(530):
        if frame >= 10 and (frame-10) % 8 == 0 and p.predict_chunk():
            seen.append(frame)
        env.step(np.array([0.,0.,0.,0.,0.,0.,-1.]))
    expected = {'fixed8':[], 'early-only':list(range(10,138,8)),
                'late-sustained':list(range(138,530,8)), 'adaptive':list(range(10,530,8))}
    assert seen == expected[arm]
    p.reset(); assert p.history.executed_steps == 0 and p.last_query is None
    with pytest.raises(ValueError): p.predict_chunk()


def test_failed_step_does_not_advance_intervention_clock():
    p = WindowPolicy(FakePolicy(0), FakePolicy(1), 'early-only')
    p.history.executed_steps = 137
    def fail(_): raise OSError('step failed')
    env = ExecutedCommandObserver(SimpleNamespace(step=fail), p.history)
    with pytest.raises(OSError): env.step(np.array([0.,0.,0.,0.,0.,0.,-1.]))
    assert p.history.executed_steps == 137
    with pytest.raises(ValueError): p.predict_chunk()

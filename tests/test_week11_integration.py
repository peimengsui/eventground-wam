from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_coupled_policy import activate, setup

from eventground_wam.b0_inference import decode_b0_action_chunk
from eventground_wam.eventground_horizon import InvalidEventPrediction
from eventground_wam.eventground_inference import (
    EventGroundInferencePolicy,
    require_disjoint_models,
    routed_horizon,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    model, _, batch, images = setup('M4')
    activate(model)
    model.eval()
    inputs = {k: v[:1] for k, v in batch['inputs'].items()}
    return model, inputs, batch['proprioception'][:1], images[:1]


def test_exact_normalized_decoded_parity_and_horizon_dependency():
    model, inputs, state, images = fixture()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        original = model(inputs, state, images).normalized_actions
        outputs = {}
        for h in (2, 4, 8, 16, 32):
            record = {}
            with routed_horizon(model, h, record, device='cpu', capture=True):
                outputs[h] = model(inputs, state, images).normalized_actions
            assert record['used_horizon'] == h and record['future_calls'] == 1
            assert record['condition'].actions[:, 8:].count_nonzero() == 0
        assert torch.equal(original, outputs[8])
        decode = lambda t: decode_b0_action_chunk(t[0].numpy(), model.proposal.normalizer)[0]
        assert np.array_equal(decode(original), decode(outputs[8]))
        assert all(not torch.equal(outputs[8], outputs[h]) for h in (2, 4, 16, 32))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert not model.future_core._forward_pre_hooks
    assert not model._forward_hooks


def test_short_horizon_masks_and_long_horizon_padding():
    model, inputs, state, images = fixture()
    with torch.no_grad():
        record = {}
        with routed_horizon(model, 2, record, device='cpu', capture=True):
            model(inputs, state, images)
        condition = record['condition']
        altered = condition.actions.clone()
        altered[:, 2:] = 100
        assert torch.equal(model.future_core(condition),
                           model.future_core(replace(condition, actions=altered)))
        # The same perturbation is visible when those positions are in the horizon.
        c32 = replace(condition, horizons=torch.tensor([32]))
        assert not torch.equal(model.future_core(c32),
                               model.future_core(replace(c32, actions=altered)))


def test_hook_cleanup_on_failure_and_reject_missing_core():
    model, _, _, _ = fixture()
    with pytest.raises(RuntimeError), routed_horizon(model, 8, {}, device='cpu'):
        raise RuntimeError('interrupted')
    assert not model.future_core._forward_pre_hooks
    with pytest.raises(ValueError, match='Exactly one'), routed_horizon(model, 8, {}, device='cpu'):
        pass
    assert not model._forward_hooks


def test_shared_branch_storage_is_rejected():
    a = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match='share'):
        require_disjoint_models(a, a)
    require_disjoint_models(a, torch.nn.Linear(2, 2))


@pytest.mark.parametrize('arm', ['M4-EG-fixed8', 'M4-EG-adaptive'])
def test_invalid_prediction_never_falls_back_or_runs_action(arm):
    model = torch.nn.Linear(2, 2)
    action = SimpleNamespace(_policy=model, _device='cpu', checkpoint=None,
                             predict_chunk=lambda **kw: pytest.fail('action must not run'))
    event = SimpleNamespace(model=torch.nn.Linear(2, 2), predict=lambda o:
        ({'next_event': 'contact', 'interval': [33, 34], 'horizon': 33}, {}))
    policy = EventGroundInferencePolicy(action, arm, event)
    args = {'agent_rgb': np.zeros((256, 256, 3), np.uint8),
                'wrist_rgb': np.zeros((256, 256, 3), np.uint8), 'proprioception': np.zeros(8),
                'instruction': 'move', 'suite': 'libero_spatial'}
    with pytest.raises(InvalidEventPrediction):
        policy.predict_chunk(**args)
    assert policy.last_query['used_horizon'] is None
    assert policy.last_query['future_calls'] == 0
    assert policy.last_query['exception']
    # Lock is released even after a malformed prediction.
    with pytest.raises(InvalidEventPrediction):
        policy.predict_chunk(**args)


def test_interface_rejects_privileged_inputs():
    policy = EventGroundInferencePolicy(SimpleNamespace(checkpoint=None), 'M4-fixed')
    with pytest.raises(TypeError):
        policy.predict_chunk(future_observation=np.zeros(1))

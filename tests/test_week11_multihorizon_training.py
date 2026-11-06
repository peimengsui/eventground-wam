from dataclasses import replace

import pytest
import torch
from test_auxiliary_training import complete_batch, targets
from test_b0_oft import recipe
from test_coupled_policy import LORA_NAMES, activate, setup

from eventground_wam.b0_training import CheckpointStore, seed_training
from eventground_wam.coupled_policy import CoupledPolicy, coupled_full_batch_backward
from eventground_wam.pixel_training import tree_digest
from eventground_wam.week11_multihorizon_training import (
    ContinuationEngine,
    ContinuationPolicy,
    route_candidates,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model_and_predictor():
    original, predictor, _, _ = setup('M4')
    model = ContinuationPolicy(original.policy, predictor, original.proposal.normalizer, arm='M4')
    activate(model)
    return model, predictor


def engine(arm):
    model, predictor = model_and_predictor()
    return ContinuationEngine(model, LORA_NAMES, predictor, recipe(), identity_digest='c'*64,
        device=torch.device('cpu'), continuation_arm=arm, design_digest='d'*64,
        plan=[complete_batch(i)['tokens'] for i in range(8)])


def forward(model, batch):
    return model.forward_with_features(batch['inputs'], batch['proprioception'],
        batch['future_inputs'].current_images, normalized_action_targets=batch['targets'])


def test_h8_forward_and_full_objective_gradients_are_exact():
    model, predictor = model_and_predictor()
    b = complete_batch(0)
    truth = targets('jepa')
    reference = CoupledPolicy.forward_with_features(model, b['inputs'], b['proprioception'],
        b['future_inputs'].current_images, normalized_action_targets=b['targets'])
    raw = reference.action_output.raw_action_predictions.detach().clone()
    coupled_full_batch_backward(model, reference, predictor, b['future_inputs'], truth)
    expected = [p.grad.clone() for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    model.candidate_horizons = [8, 8]
    routed = forward(model, b)
    assert torch.equal(routed.action_output.raw_action_predictions, raw)
    coupled_full_batch_backward(model, routed, predictor, b['future_inputs'], truth)
    assert all(torch.equal(a, p.grad) for a, p in zip(expected,
               (p for p in model.parameters() if p.requires_grad), strict=True))
    assert not model.future_core._forward_pre_hooks


def test_candidate_routing_is_separate_from_auxiliary_and_targets():
    model, predictor = model_and_predictor()
    b = complete_batch(0)
    model.candidate_horizons = [2, 32]
    seen = []
    handle = model.future_core.register_forward_hook(lambda _m, a, _o: seen.extend(a[0].horizons.tolist()))
    result = forward(model, b)
    assert seen == [2, 32] and not result.candidate_actions.requires_grad
    assert not model.future_core._forward_pre_hooks
    raw = result.action_output.raw_action_predictions.detach().clone()
    b['targets'] = -b['targets']
    b['future_mask'] = ~b['future_mask']
    assert torch.equal(forward(model, b).action_output.raw_action_predictions, raw)
    altered = replace(b['future_inputs'], horizons=torch.full_like(b['future_inputs'].horizons, 4))
    predictor(altered.condition(result.features))
    assert seen == [2, 32, 2, 32]
    handle.remove()


@pytest.mark.parametrize('arm', ['h8-continuation', 'mixed-continuation'])
def test_complete_resume_partial_failure_and_cross_arm_rejection(arm, tmp_path):
    current = engine(arm)
    seed_training(17)
    current.step(complete_batch(0), targets('jepa'))
    store = CheckpointStore(tmp_path, 'c'*64, keep=3)
    first = current.snapshot()
    store.save(first)
    current.step(complete_batch(1), targets('jepa'))
    expected = tree_digest(current.snapshot())
    restored = engine(arm)
    restored.restore(store.load())
    restored.step(complete_batch(1), targets('jepa'))
    assert tree_digest(restored.snapshot()) == expected
    assert restored.scheduler.base_lrs == [1e-5]*4
    before = restored.snapshot()
    def fail(*_):
        assert not restored.model.future_core._forward_pre_hooks
        raise RuntimeError('interrupt before commit')
    with pytest.raises(RuntimeError):
        restored.step(complete_batch(2), targets('jepa'), hook=fail)
    assert restored.model.candidate_horizons is None
    assert not restored.model.future_core._forward_pre_hooks
    with pytest.raises(ValueError, match='partial'):
        restored.snapshot()
    restored.restore(before)
    assert tree_digest(restored.snapshot()) == expected
    damaged = store.save(before)
    (tmp_path/damaged['checkpoint']).write_bytes(b'torn')
    assert store.load()['global_step'] == 1
    wrong = dict(first, continuation_arm='mixed-continuation' if arm=='h8-continuation' else 'h8-continuation')
    with pytest.raises(ValueError, match='identity'):
        restored.restore(wrong)


def test_invalid_stream_is_rejected_before_any_update():
    current = engine('mixed-continuation')
    b = complete_batch(0)
    b['tokens'][0] = (0, 999, 0)
    with pytest.raises(ValueError, match='exposure'):
        current.step(b, targets('jepa'))
    assert current.global_step == 0 and not current.optimizer.state


@pytest.mark.parametrize('count', [1, 23, 32])
def test_routing_preserves_microbatch_offsets_and_cleans_up_on_failure(count):
    from types import SimpleNamespace

    from eventground_wam.pixel_predictor import PixelCondition
    core=torch.nn.Identity()
    model=SimpleNamespace(future_core=core)
    horizons=([2,4,8,16,32]*7)[:count]
    def condition(n):
        return PixelCondition(torch.zeros(n,1,2048),torch.zeros(n,1,dtype=torch.bool),
            torch.zeros(n,2,3,128,128),torch.zeros(n,8),torch.zeros(n,32,7),torch.full((n,),8))
    seen=[]
    with route_candidates(model,horizons):
        for start in range(0,count,10): seen.extend(core(condition(min(10,count-start))).horizons.tolist())
    assert seen==horizons and not core._forward_pre_hooks
    with pytest.raises(RuntimeError), route_candidates(model,horizons):
        raise RuntimeError('forward failure')
    assert not core._forward_pre_hooks

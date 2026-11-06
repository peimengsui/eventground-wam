import pytest
import torch
from test_coupled_policy import activate, setup
from week11_motion_adapter import ARMS, FrozenMotionPolicy, MotionEngine

from eventground_wam.b0_training import CheckpointStore, seed_training
from eventground_wam.pixel_training import tree_digest


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def case(arm=ARMS[0]):
    ref, _, batch, images = setup('M4'); activate(ref)
    model = FrozenMotionPolicy(ref, arm)
    batch.update(tokens=[(0, 1, 0), (0, 1, 1)], sample_count=2, identity_digest='a'*64,
                 index=0, current_images=images)
    return model, batch


def call(model, b):
    return model(b['inputs'], b['proprioception'], b['current_images'])


@pytest.mark.parametrize('arm', ARMS)
def test_zero_initialization_h8_bypass_gripper_and_frozen_gradients(arm):
    model, b = case(arm)
    expected = call(model.reference, b).normalized_actions.detach().clone()
    for hs in ([2, 32], [4, 16], [8, 8], [8, 2]):
        model.horizons = hs
        assert torch.equal(call(model, b).normalized_actions, expected)
    engine = MotionEngine(model, [b['tokens']]*8, 'c'*64)
    frozen = tree_digest(model.reference.state_dict())
    engine.step(b)
    assert tree_digest(model.reference.state_dict()) == frozen
    assert all(p.grad is None for p in model.reference.parameters())
    model.horizons = [8, 8]
    assert torch.equal(call(model, b).normalized_actions, expected)
    model.horizons = [2, 32]
    actual = call(model, b).normalized_actions
    assert not torch.equal(actual[..., :6], expected[..., :6])
    assert torch.equal(actual[..., 6], expected[..., 6])
    model.horizons = [8, 2]
    assert torch.equal(call(model, b).normalized_actions[0], expected[0])
    assert not model.reference.future_core._forward_pre_hooks


def test_control_discards_non8_information_and_matches_compute():
    model, b = case(ARMS[1])
    with torch.no_grad(): model.adapter.output.weight.normal_(std=.01)
    calls = []
    hook = model.reference.future_core.register_forward_hook(lambda _m,a,_o: calls.append(a[0].horizons.tolist()))
    model.horizons = [2, 2]; a = call(model, b).normalized_actions.detach().clone()
    model.horizons = [32, 32]; z = call(model, b).normalized_actions.detach().clone()
    hook.remove()
    assert torch.equal(a, z) and calls == [[8,8],[2,2],[8,8],[32,32]]
    model.arm = ARMS[0]
    model.horizons = [2, 2]; a = call(model, b).normalized_actions.detach().clone()
    model.horizons = [32, 32]; z = call(model, b).normalized_actions.detach().clone()
    assert not torch.equal(a, z)


@pytest.mark.parametrize('arm', ARMS)
def test_complete_restart_torn_recovery_and_partial_update(arm, tmp_path):
    model, b = case(arm); engine = MotionEngine(model, [b['tokens']]*8, 'b'*64)
    seed_training(17); engine.step(b)
    first = engine.snapshot(); store = CheckpointStore(tmp_path, 'b'*64)
    store.save(first); engine.step(dict(b, index=1)); expected = tree_digest(engine.snapshot())
    engine.restore(store.load()); engine.step(dict(b, index=1))
    assert tree_digest(engine.snapshot()) == expected
    saved = engine.snapshot()
    def interrupt(): raise RuntimeError('interrupt')
    with pytest.raises(RuntimeError): engine.step(dict(b,index=2), hook=interrupt)
    with pytest.raises(ValueError, match='partial'): engine.snapshot()
    engine.restore(saved); assert tree_digest(engine.snapshot()) == expected
    last = store.save(saved); (tmp_path/last['checkpoint']).write_bytes(b'torn')
    assert store.load()['global_step'] == 1
    with pytest.raises(ValueError, match='identity'): engine.restore(dict(first, arm='other'))


def test_bad_horizons_and_reference_failure_clean_up_hooks(monkeypatch):
    model, b = case()
    model.horizons = [2, 3]
    with pytest.raises(ValueError, match='horizon'): call(model, b)
    model.horizons = [2, 4]
    def broken(*_, **__): raise RuntimeError('inference failed')
    monkeypatch.setattr(model.reference, 'forward_with_features', broken)
    with pytest.raises(RuntimeError): call(model, b)
    assert not model.reference.future_core._forward_pre_hooks
    assert not model.reference.coupling._forward_pre_hooks

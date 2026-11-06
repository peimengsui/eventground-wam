from __future__ import annotations

import pytest
import torch
from test_auxiliary_training import complete_batch, targets
from test_b0_oft import recipe
from test_coupled_policy import LORA_NAMES, setup

from eventground_wam.b0_training import CheckpointStore, seed_training
from eventground_wam.coupled_training import CoupledTrainingEngine
from eventground_wam.pixel_training import tree_digest


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def engine(arm):
    model, predictor, _, _ = setup(arm)
    config = {**recipe(), 'auxiliary_learning_rate': 1e-4}
    return CoupledTrainingEngine(model, LORA_NAMES, predictor, config,
        identity_digest='c'*64, steps_per_epoch=8, device=torch.device('cpu'))


@pytest.mark.parametrize('arm,method', [('M3', 'pixel'), ('M4', 'jepa')])
def test_coupled_update_and_complete_recovery(arm, method, tmp_path):
    current = engine(arm)
    seed_training(17)
    row = current.step(complete_batch(0), targets(method))
    assert len(row['learning_rates']) == 4
    assert len(set(map(id, current.parameters))) == len(current.parameters)
    assert current.model.coupling.output.weight.count_nonzero() > 0
    saved = current.snapshot()
    store = CheckpointStore(tmp_path, 'c'*64, keep=3)
    store.save(saved)
    current.step(complete_batch(1), targets(method))
    expected = tree_digest(current.snapshot())
    # Restore into an independently initialized model, retaining predictor/core aliases.
    restored = engine(arm)
    seed_training(999)
    restored.restore(store.load())
    restored.step(complete_batch(1), targets(method))
    assert tree_digest(restored.snapshot()) == expected
    assert restored.samples_seen == 4 and restored.future_pairs_seen == 6
    before = restored.snapshot()
    def fail(a, b):
        raise RuntimeError('interrupted microbatch')
    with pytest.raises(RuntimeError):
        restored.step(complete_batch(2), targets(method), hook=fail)
    with pytest.raises(ValueError, match='partial'):
        restored.snapshot()
    restored.restore(before)
    assert tree_digest(restored.snapshot()) == tree_digest(before)
    damaged = store.save(before)
    (tmp_path/damaged['checkpoint']).write_bytes(b'torn checkpoint')
    assert store.load()['global_step'] == 1 and store.recovery_notes
    assert restored.model.future_core.action_projection[0].weight is restored.predictor.action_projection[0].weight


@pytest.mark.parametrize('field', ['split', 'digest', 'cursor', 'epoch'])
def test_reject_bad_batch_before_coupled_update(field):
    current = engine('M4')
    batch = complete_batch(0)
    if field == 'split':
        batch['future_identities'][0]['split'] = 'test'
    elif field == 'digest':
        batch['future_digest'] = 'f'*64
    elif field == 'cursor':
        batch['index'] = 1
    else:
        batch['tokens'][0] = (1, 4, 0)
    with pytest.raises(ValueError):
        current.step(batch, targets('jepa'))
    assert not current._in_update and current.global_step == 0 and not current.optimizer.state


def test_reject_cross_arm_or_corrupted_recovery_before_mutation():
    current = engine('M4')
    saved = current.snapshot()
    saved['arm'] = 'M3'
    with pytest.raises(ValueError, match='recipe'):
        current.restore(saved)
    saved['arm'] = 'M4'
    saved['coupling']['output.weight'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='tensors'):
        current.restore(saved)
    assert torch.isfinite(current.model.coupling.output.weight).all()

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_auxiliary_production import Batches, recipe
from test_auxiliary_training import complete_batch, targets
from test_coupled_training import engine

from eventground_wam.b0_training import CheckpointStore, read_sealed, seed_training
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.pixel_training import tree_digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import train_auxiliary_policy as shared
import train_coupled_policy as production
from coupled_runtime import project_hours

IDENTITY = seal_contract({'test': 'coupled-production'})


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fresh():
    value = engine('M4')
    value.steps_per_epoch = 2
    value.recipe['epochs'] = 2
    value.identity_digest = IDENTITY['digest']
    seed_training(17)
    return value


@pytest.mark.parametrize('failure_point', ['before_export', 'after_export', 'after_finish'])
def test_coupled_epoch_export_boundary_recovers_without_repeated_updates(failure_point, tmp_path):
    seed_training(4)
    rows = [complete_batch(i) for i in range(2)]
    truths = [targets('jepa').detach() for _ in rows]
    config = {**recipe(), 'arm': 'M4', 'method': 'jepa'}
    kwargs = {'recipe': config, 'identity': IDENTITY, 'budget': SimpleNamespace(seconds=0),
              'batch_factory': lambda epoch: Batches(epoch, rows),
              'target_fn': lambda b: (truths[b['index']], {}),
              'resource_check': lambda *a, **kw: None}
    reference = fresh()
    root = tmp_path/'reference'
    store = CheckpointStore(root/'recovery', IDENTITY['digest'])
    assert shared.train_loop(reference, output=root, store=store, exporter=production.export_epoch, **kwargs) == 0
    expected = tree_digest(reference.snapshot())
    current = fresh()
    root = tmp_path/'interrupted'
    store = CheckpointStore(root/'recovery', IDENTITY['digest'])
    original_save = store.save
    def save(value):
        if failure_point == 'after_finish' and value['epoch'] == 1:
            raise RuntimeError('crash after finish')
        return original_save(value)
    store.save = save
    def export(*args):
        if failure_point == 'before_export':
            raise RuntimeError('crash before export')
        result = production.export_epoch(*args)
        if failure_point == 'after_export':
            raise RuntimeError('crash after export')
        return result
    with pytest.raises(RuntimeError, match='crash'):
        shared.train_loop(current, output=root, store=store, exporter=export, **kwargs)
    resumed = fresh()
    store.save = original_save
    resumed.restore(store.load())
    assert resumed.global_step == 2
    assert shared.train_loop(resumed, output=root, store=store, exporter=production.export_epoch, **kwargs) == 0
    assert tree_digest(resumed.snapshot()) == expected
    report = read_sealed(root/'epochs/epoch-02/training-report.json')
    assert report['decoder_free_coupled_export']
    artifact = torch.load(root/'epochs/epoch-02/M4.ckpt', weights_only=True)
    assert set(artifact['states']) == {'policy', 'future_core', 'coupling'}
    assert not any('output_projection' in k for k in artifact['states']['future_core'])
    assert shared.train_loop(resumed, output=root, store=store, exporter=production.export_epoch, **kwargs) == 0


def test_projection_includes_cold_outliers_and_measured_checkpoint_allowance():
    durations = [1.]*100+[5.]*28
    mean, p95, hours = project_hours(durations, 10.)
    assert mean > 1 and p95 == 5
    assert hours == 1.2*5*117132/3600+590/3600
    for bad in ([1.]*127, [float('nan')]*128, [0.]*128):
        with pytest.raises(ValueError):
            project_hours(bad, 10.)


def test_production_requires_qualification_before_any_model_or_payload_load(tmp_path):
    args = SimpleNamespace(arm='M4')
    with pytest.raises(FileNotFoundError):
        production.verify_qualification(tmp_path, {}, {}, {}, args)

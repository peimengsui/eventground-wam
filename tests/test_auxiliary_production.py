from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_auxiliary_training import complete_batch, engine, targets

from eventground_wam.b0_training import CheckpointStore, read_sealed, seed_training
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.pixel_training import tree_digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import train_auxiliary_policy as production

IDENTITY = seal_contract({'test': 'auxiliary-production'})


def recipe():
    return {'arm': 'M1', 'method': 'pixel', 'digest': 'b'*64, 'epochs': 2,
            'total_updates': 4, 'updates_per_epoch': 2, 'action_presentations': 8,
            'future_pairs': 12, 'checkpoint_selection': 'fixed-epoch-2-endpoint',
            'resources': {'maximum_active_hours': 1},
            'recovery': {'checkpoint_interval_seconds': 300, 'checkpoint_interval_steps': 2000}}


class Batches:
    def __init__(self, epoch, rows):
        self.rows = copy.deepcopy(rows)
        for row in self.rows:
            row['tokens'] = [(epoch, ep, anchor) for _, ep, anchor in row['tokens']]
        self.plan = [row['tokens'] for row in self.rows]
        # Synthetic future identities have three available pairs per batch.
        self.pairs = SimpleNamespace(sources={4: {'episode_length': 5}})

    def __getitem__(self, index):
        return self.rows[index]


def fresh():
    seed_training(17)
    value = engine('pixel')
    value.steps_per_epoch = 2
    value.recipe['epochs'] = 2
    value.identity_digest = IDENTITY['digest']
    return value


@pytest.mark.parametrize('failure_point', ['before_export', 'after_export', 'after_finish'])
def test_production_epoch_boundary_crash_resumes_exactly(failure_point, tmp_path):
    seed_training(4)
    rows = [complete_batch(i) for i in range(2)]
    truths = [targets('pixel').detach() for _ in rows]
    make_batches = lambda epoch: Batches(epoch, rows)
    identity = IDENTITY
    kwargs = {'recipe': recipe(), 'identity': identity, 'budget': SimpleNamespace(seconds=0),
              'batch_factory': make_batches, 'target_fn': lambda b: (truths[b['index']], {}),
              'resource_check': lambda *a, **kw: None}
    reference = fresh()
    reference_root = tmp_path/'reference'
    store = CheckpointStore(reference_root/'recovery', identity['digest'])
    assert production.train_loop(reference, output=reference_root, store=store, **kwargs) == 0
    expected = tree_digest(reference.snapshot())
    trial = fresh()
    root = tmp_path/'interrupted'
    store = CheckpointStore(root/'recovery', identity['digest'])
    original_save = store.save
    def save(value):
        if failure_point == 'after_finish' and value['epoch'] == 1:
            raise RuntimeError('crash after finish before durable commit')
        return original_save(value)
    store.save = save
    def export(*args):
        if failure_point == 'before_export':
            raise RuntimeError('crash before export')
        report = production.export_epoch(*args)
        if failure_point == 'after_export':
            raise RuntimeError('crash after export')
        return report
    with pytest.raises(RuntimeError, match='crash'):
        production.train_loop(trial, output=root, store=store, exporter=export, **kwargs)
    resumed = fresh()
    store.save = original_save
    resumed.restore(store.load())
    assert resumed.global_step == 2
    assert production.train_loop(resumed, output=root, store=store, **kwargs) == 0
    assert tree_digest(resumed.snapshot()) == expected
    report = read_sealed(root/'training-report.json')
    assert report['samples_seen'] == 8 and report['future_pairs_seen'] == 12
    deployed = torch.load(root/'epochs/epoch-02/M1.ckpt', weights_only=True)
    assert 'predictor' not in deployed and 'optimizer' not in deployed
    assert production.train_loop(resumed, output=root, store=store, **kwargs) == 0


def test_budget_stop_saves_without_consuming_another_batch(tmp_path):
    current = fresh()
    store = CheckpointStore(tmp_path/'recovery', IDENTITY['digest'])
    batches = SimpleNamespace(plan=[], pairs=None)
    code = production.train_loop(current, recipe(), IDENTITY, tmp_path, store,
        SimpleNamespace(seconds=3600), lambda epoch: batches,
        lambda batch: pytest.fail('must not read payloads after budget expiration'),
        resource_check=lambda *a, **kw: None)
    assert code == 78 and store.load()['global_step'] == 0


def test_resumed_future_counter_mismatch_is_rejected():
    current = fresh()
    current.future_pairs_seen = 1
    with pytest.raises(ValueError, match='future exposure mismatch'):
        production.check_exposure(current, recipe(), SimpleNamespace(plan=[]))


def test_gpu_check_allows_only_small_known_desktop_allocations_and_own_process():
    rows = ('1, /usr/bin/python, 10000\n'
            '2, /opt/google/chrome/chrome --type=gpu-process --ozone-platform=wayland, 133\n'
            '3, /usr/bin/gnome-control-center, 36\n')
    assert production.other_gpu_jobs(rows, 1) == []
    assert production.other_gpu_jobs(rows+'4, /usr/bin/python, 10\n', 1) == [4]
    assert production.other_gpu_jobs(rows.replace('133', '300'), 1) == [2]
    assert production.other_gpu_jobs(rows.replace('gpu-process', 'renderer'), 1) == [2]

import json
from pathlib import Path

import pytest
import torch
from test_b0_data import _dataset
from test_b0_overfit import _tiny_policy
from test_b0_training import assert_tree_equal, batch, recipe

from eventground_wam import b0_data
from eventground_wam.b0_continuation import MixedBatches
from eventground_wam.b0_fresh import aggregate_validation, fresh_plan, load_fresh_recipe
from eventground_wam.b0_training import CheckpointStore, TrainingEngine, cpu_tree, seed_training
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.trajectory_split import expand_groups

ROOT = Path(__file__).resolve().parents[1]


def test_v2_loader_replaces_every_membership_and_rejects_changed_groups(monkeypatch, tmp_path):
    root = ROOT / 'configs/datasets/libero-plus-trajectory-v2'
    rows = expand_groups([json.loads(x) for x in (root / 'groups.jsonl').read_text().splitlines()])
    old = b0_data.B0SplitIndex(tuple(b0_data.B0SplitAssignment(
        r['episode_index'], 0, r['episode_length'], 'train', 'a' * 64, 'b' * 64, 'c' * 64) for r in rows))
    monkeypatch.setattr(b0_data, 'load_frozen_b0_split_index', lambda path: old)
    index = b0_data.load_trajectory_v2_split_index('/unused', root)
    assert index.episode_counts == {'train': 11435, 'validation': 1528, 'test': 1384}
    groups = {}
    for assignment in index.assignments:
        groups.setdefault(assignment.initial_condition_digest, set()).add(assignment.split)
    assert len(groups) == 1681 and all(len(s) == 1 for s in groups.values())
    b0_data._validate_b0_source_identity(index.source_identity)
    with pytest.raises(ValueError, match='frozen split'):
        b0_data._validate_b0_source_identity({**index.source_identity, 'split_assignments_sha256': '0' * 64})
    (tmp_path / 'manifest.json').write_bytes((root / 'manifest.json').read_bytes())
    (tmp_path / 'groups.jsonl').write_bytes((root / 'groups.jsonl').read_bytes() + b'\n')
    with pytest.raises(ValueError, match='membership'):
        b0_data.load_trajectory_v2_split_index('/unused', tmp_path)


def test_two_fresh_epoch_plans_cover_every_chunk_and_mixed_loader_accepts_epoch_zero():
    _, data = _dataset([12, 14, 10], ['train', 'train', 'validation'])
    config = {'batch_size': 32, 'seed': 17}
    plans = [fresh_plan(data, config, epoch) for epoch in range(2)]
    for epoch, plan in enumerate(plans):
        assert sorted(t for b in plan for t in b) == [(epoch, e, a) for e, n in [(0, 5), (1, 7)] for a in range(n)]
        assert batches_digest(plan) == batches_digest(fresh_plan(data, config, epoch))
        MixedBatches(data, plan, '/unused', expected_epoch=epoch)
    with pytest.raises(ValueError, match='outside'):
        MixedBatches(data, plans[0], '/unused')


def test_grouped_validation_does_not_overweight_long_or_repeated_variants():
    values = {0: {'sample_count': 100, 'sums': {'loss': 100.}},
              1: {'sample_count': 10, 'sums': {'loss': 30.}},
              2: {'sample_count': 2, 'sums': {'loss': 12.}},
              3: {'sample_count': 1, 'sums': {'loss': 10.}}}
    result = aggregate_validation(values, {0: ('a', 'x'), 1: ('a', 'x'), 2: ('a', 'y'), 3: ('b', 'z')})
    assert result['micro']['loss'] == pytest.approx(152 / 113)
    assert result['task_macro']['loss'] == 7  # (((1+3)/2 + 6)/2 + 10)/2
    assert result['group_count'] == 3 and result['task_count'] == 2


def test_fresh_cosine_optimizer_resume_across_epoch_boundary(tmp_path):
    config = {**recipe(), 'epochs': 2, 'warmup_fraction': .25, 'minimum_learning_rate_ratio': .1}
    def create():
        policy, names = _tiny_policy()
        return TrainingEngine(policy, names, config, identity_digest='f' * 64,
                              steps_per_epoch=2, device=torch.device('cpu'))
    seed_training(17)
    current = create()
    current.step(batch(0)); current.step(batch(1))
    current.finish_epoch({'objective': 1.})
    store = CheckpointStore(tmp_path, 'f' * 64)
    store.save(current.snapshot())
    current.step(batch(0))
    expected = cpu_tree(current.snapshot())
    restored = create()
    restored.restore(store.load())
    restored.step(batch(0))
    assert_tree_equal(expected, cpu_tree(restored.snapshot()))


def test_frozen_fresh_recipe_and_normalizers_are_v2_train_only():
    value = load_fresh_recipe(ROOT / 'configs/training/week4_b0_fresh_all_l1_v2.json')
    normalizer = b0_data.load_b0_action_normalizer(ROOT / 'docs/model/week4-fresh-v2/action-statistics.json')
    assert value['maximum_updates'] == 117132
    assert normalizer.source_identity == value['state_normalization']['source']
    assert normalizer.source_identity['dataset_version'] == 'libero-plus-trajectory-v2'

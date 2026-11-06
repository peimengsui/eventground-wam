from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from eventground_wam.b0_training import cpu_tree, seed_training
from eventground_wam.future_pairs import HORIZONS, sha256_file
from eventground_wam.pixel_predictor import PixelCondition, PixelPredictor, PixelPredictorConfig
from eventground_wam.pixel_training import (
    RECOVERY_FORMAT,
    CPTOrder,
    PixelCheckpointStore,
    PixelTrainingEngine,
    accumulate_loss,
    build_order,
    pilot_indices,
    rate_factor,
    tree_digest,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sources():
    return [{"episode_index": i, "episode_length": 35+i, "split": "train",
             "instruction": "move "*(i+1), "digest": f"{i:064x}"} for i in range(9)]


@pytest.fixture()
def order(tmp_path):
    root = tmp_path/"order"
    build_order(sources(), root, pair_manifest_sha256="a"*64)
    return CPTOrder(root, sha256_file(root/"manifest.json"))


def test_order_exactly_covers_all_anchors_and_each_valid_horizon_once(order):
    expected = {(s["episode_index"], t) for s in sources() for t in range(s["episode_length"]-2)}
    observed = [ref for i in range(len(order)) for ref in order.references(i)]
    assert len(observed) == len(expected) and set(observed) == expected
    lengths = {s["episode_index"]: s["episode_length"] for s in sources()}
    counts = {str(h): 0 for h in HORIZONS}
    for i in range(len(order)):
        refs = order.references(i)
        assert len({ep for ep, _ in refs}) <= 4 and 1 <= len(refs) <= 8
        pairs = [(ep, t, h) for ep, t in refs for h in HORIZONS if t+h < lengths[ep]]
        assert len(pairs) == order.pair_counts[i]
        for _, _, h in pairs:
            counts[str(h)] += 1
    assert counts == order.manifest["pairs_by_horizon"]
    assert any(len(order.references(i)) < 8 for i in range(len(order)))
    assert max(order.pair_counts) <= 40


def test_order_regeneration_is_independent_of_source_order_and_global_rng(order, tmp_path):
    seed_training(444)
    second = tmp_path/"again"
    record = build_order(list(reversed(sources())), second, pair_manifest_sha256="a"*64)
    assert record == order.manifest
    assert sha256_file(second/"manifest.json") == sha256_file(tmp_path/"order/manifest.json")


def test_order_rejects_source_and_manifest_drift(order, tmp_path):
    bad = sources(); bad[0]["split"] = "validation"
    with pytest.raises(ValueError, match="train"):
        build_order(bad, tmp_path/"bad", pair_manifest_sha256="a"*64)
    with pytest.raises(ValueError, match="manifest changed"):
        CPTOrder(tmp_path/"order", "b"*64)
    (tmp_path/"order/anchors.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="array changed"):
        CPTOrder(tmp_path/"order", sha256_file(tmp_path/"order/manifest.json"))


def test_pilot_has_unique_windows_and_extreme_episode_and_prompt(order):
    indices = pilot_indices(order, {s["episode_index"]: s for s in sources()})
    assert indices == sorted(set(indices))
    assert 8 in {ep for index in indices for ep, _ in order.references(index)}


def tiny_model():
    return PixelPredictor(PixelPredictorConfig(feature_dimension=8, dimension=32, heads=4, layers=1))


def condition(count):
    return PixelCondition(torch.rand(count, 3, 8), torch.zeros(count, 3, dtype=torch.bool),
                          torch.full((count, 2, 3, 128, 128), .5), torch.rand(count, 8),
                          torch.rand(count, 32, 7), torch.tensor([8]*count))


def test_accumulation_matches_full_logical_batch_including_short_remainder():
    seed_training(11); model = tiny_model()
    with torch.no_grad():
        model.residual_head.weight.normal_(0, .001)
    x = condition(3); target = .1+.8*torch.rand_like(x.current_images)
    whole = accumulate_loss(model, x, target, microbatch=3, ssim_weight=.1)
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    split = accumulate_loss(model, x, target, microbatch=2, ssim_weight=.1)
    assert abs(whole-split) < 1e-6
    for before, parameter in zip(gradients, model.parameters(), strict=True):
        torch.testing.assert_close(before, parameter.grad, rtol=2e-4, atol=2e-7)


class TinyOrder:
    pair_counts = [2]*8
    @staticmethod
    def references(index):
        return [(0, index), (1, index)]


def engine():
    recipe = {"optimizer": {"learning_rate": .0001, "betas": [.9, .999], "epsilon": 1e-8,
                             "weight_decay": .0001, "warmup_updates": 2, "minimum_lr_ratio": .1,
                             "gradient_clip": 1.},
              "total_updates": 8, "predictor_microbatch": 1, "loss": {"ssim_weight": 0.}}
    return PixelTrainingEngine(tiny_model(), recipe, identity_digest="a"*64,
                               indices=range(8), order=TinyOrder())


def step(current, **kwargs):
    index = current.global_step
    # Changing every global RNG changes the next update, testing real restoration.
    x = condition(2)
    x = replace(x, state=x.state+random.random()+np.random.random())
    identities = [{"episode_index": ep, "anchor": index, "split": "train",
                   "pair_digest": f"{index*2+ep:064x}"} for ep in range(2)]
    return current.step(x, x.current_images*.8, identities, **kwargs)


def test_complete_disk_resume_matches_weights_moments_schedule_rng_cursor_and_exposure(tmp_path):
    seed_training(42); reference = engine()
    for _ in range(5):step(reference)
    expected = tree_digest(reference.snapshot())
    seed_training(42); interrupted = engine()
    for _ in range(2):step(interrupted)
    store = PixelCheckpointStore(tmp_path, "a"*64)
    store.save(interrupted.snapshot())
    seed_training(91); resumed = engine(); saved = store.load(); original = tree_digest(saved)
    resumed.restore(saved)
    for _ in range(3):step(resumed)
    assert tree_digest(resumed.snapshot()) == expected
    assert tree_digest(saved) == original
    assert resumed.pairs_seen == 10 and resumed.global_step == 5


def test_partial_update_cannot_commit_and_replay_starts_from_last_complete_boundary(tmp_path):
    current = engine(); step(current)
    store = PixelCheckpointStore(tmp_path, "a"*64); store.save(current.snapshot())
    def crash(start, stop):
        raise RuntimeError("simulated interruption during accumulation")
    with pytest.raises(RuntimeError):step(current, hook=crash)
    with pytest.raises(ValueError, match="partial"):current.snapshot()
    restored = engine(); restored.restore(store.load())
    assert restored.global_step == 1 and restored.pairs_seen == 2


@pytest.mark.parametrize("field,value", [("identity_digest", "b"*64), ("global_step", 5), ("pairs_seen", 9)])
def test_resume_rejects_identity_and_cursor_drift(field, value):
    current = engine(); saved = cpu_tree(current.snapshot()); saved[field] = value
    with pytest.raises(ValueError, match="changed"):current.restore(saved)


def payload(step):
    return {"format": RECOVERY_FORMAT, "identity_digest": "a"*64, "global_step": step,
            "tensor": torch.tensor([step])}


def test_recovery_fallback_partial_files_and_bounded_retention(tmp_path):
    store = PixelCheckpointStore(tmp_path, "a"*64, keep=3)
    for i in range(5):store.save(payload(i))
    latest = store.save(payload(5))
    (tmp_path/latest["checkpoint"]).write_bytes(b"crash")
    (tmp_path/".unfinished.partial").write_bytes(b"partial")
    torch.save(payload(999), tmp_path/"checkpoint-999.pt")
    assert store.load()["global_step"] == 4 and store.notes
    store.save(payload(6))
    assert len(list(tmp_path.glob("checkpoint-*.pt"))) == 3
    assert len(list(tmp_path.glob("checkpoint-*.json"))) == 3
    assert not list(tmp_path.glob(".*.partial"))
    with pytest.raises(ValueError, match="different"):PixelCheckpointStore(tmp_path, "b"*64).load()


def test_crash_before_commit_preserves_previous_checkpoint(tmp_path, monkeypatch):
    store = PixelCheckpointStore(tmp_path, "a"*64); store.save(payload(1))
    def fail(*args, **kwargs):raise OSError("crash before durable commit")
    monkeypatch.setattr("eventground_wam.pixel_training.atomic_json", fail)
    with pytest.raises(OSError):store.save(payload(2))
    assert store.load()["global_step"] == 1


def test_all_corrupt_commits_fail_closed(tmp_path):
    store = PixelCheckpointStore(tmp_path, "a"*64); record = store.save(payload(1))
    (tmp_path/record["checkpoint"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="no intact"):store.load()


def test_warmup_cosine_and_fixed_endpoint():
    assert rate_factor(0, 100, 5, .1) == .2
    assert rate_factor(4, 100, 5, .1) == 1.
    assert rate_factor(99, 100, 5, .1) == .1
    assert rate_factor(1000, 100, 5, .1) == .1

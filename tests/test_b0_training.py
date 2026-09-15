from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_b0_data import _dataset
from test_b0_overfit import _tiny_policy

from eventground_wam.b0_data import B0DataConfig
from eventground_wam.b0_model import B0ObjectiveConfig
from eventground_wam.b0_training import (
    TRAINING_FORMAT,
    CheckpointStore,
    EpochBatches,
    TrainingContractError,
    TrainingEngine,
    build_batch_plan,
    cpu_tree,
    evaluate_validation,
    load_recipe,
    plan_digest,
    seed_training,
)

ROOT = Path(__file__).parents[1]


def recipe() -> dict:
    return load_recipe(ROOT / "configs/training/week4_b0_5epoch_v1.json")


def engine() -> TrainingEngine:
    policy, names = _tiny_policy()
    policy.objective_config = B0ObjectiveConfig("continuous-l1-plus-gripper-bce-v1")
    return TrainingEngine(policy, names, recipe(), identity_digest="a" * 64,
                          steps_per_epoch=8, device=torch.device("cpu"))


def batch(index: int) -> dict:
    # All three global RNGs contribute: resume must restore them, not only weights.
    values = torch.rand(2, 4) + random.random() + np.random.random()
    targets = torch.zeros(2, 8, 7)
    targets[..., 6] = 1
    return {"index": index, "inputs": {"input_ids": (values * 10).long(),
                                          "attention_mask": torch.ones(2, 4)},
            "proprioception": torch.rand(2, 8), "targets": targets,
            "sample_count": 2, "identity_digest": f"{index + 1:064x}"}


def assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right, strict=True):
            assert_tree_equal(x, y)
    else:
        assert left == right


def test_disk_resume_matches_uninterrupted_weights_optimizer_scheduler_rng_and_cursor(tmp_path):
    seed_training(912)
    baseline = engine()
    for index in range(8):
        baseline.step(batch(index))
    expected = cpu_tree(baseline.snapshot())

    seed_training(912)
    interrupted = engine()
    for index in range(3):
        interrupted.step(batch(index))
    store = CheckpointStore(tmp_path, "a" * 64)
    store.save(interrupted.snapshot())
    # New process construction can consume any amount of RNG before restoration.
    seed_training(7)
    resumed = engine()
    resumed.restore(store.load())
    for index in range(resumed.next_batch, 8):
        resumed.step(batch(index))
    assert_tree_equal(expected, cpu_tree(resumed.snapshot()))
    assert resumed.global_step == 8
    assert resumed.samples_seen == 16


def payload(step: int) -> dict:
    return {"format": TRAINING_FORMAT, "identity_digest": "a" * 64,
            "global_step": step, "epoch": 0, "next_batch": step,
            "tensor": torch.tensor([step])}


def test_recovery_falls_back_from_damaged_newest_and_ignores_uncommitted_files(tmp_path):
    store = CheckpointStore(tmp_path, "a" * 64)
    store.save(payload(1))
    last = store.save(payload(2))
    (tmp_path / last["checkpoint"]).write_bytes(b"interrupted or corrupted")
    (tmp_path / ".checkpoint-999.partial").write_bytes(b"partial")
    torch.save(payload(999), tmp_path / "checkpoint-999.pt")
    assert store.load()["global_step"] == 1
    assert store.recovery_notes


def test_crash_before_commit_keeps_previous_checkpoint_usable(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path, "a" * 64)
    store.save(payload(1))

    def fail(*args, **kwargs):
        raise OSError("simulated disk interruption")

    monkeypatch.setattr("eventground_wam.b0_training.atomic_json", fail)
    with pytest.raises(OSError):
        store.save(payload(2))
    assert store.load()["global_step"] == 1


def test_recovery_rejects_identity_drift_and_all_corrupt_checkpoints(tmp_path):
    store = CheckpointStore(tmp_path, "a" * 64)
    record = store.save(payload(1))
    with pytest.raises(TrainingContractError, match="different training identity"):
        CheckpointStore(tmp_path, "b" * 64).load()
    (tmp_path / record["checkpoint"]).write_bytes(b"bad")
    with pytest.raises(TrainingContractError, match="no intact"):
        store.load()


def test_checkpoint_retention_leaves_three_complete_generations(tmp_path):
    store = CheckpointStore(tmp_path, "a" * 64, keep=3)
    for index in range(6):
        store.save(payload(index))
    assert len(list(tmp_path.glob("checkpoint-*.json"))) == 3
    assert len(list(tmp_path.glob("checkpoint-*.pt"))) == 3
    assert store.load()["global_step"] == 5


def test_resume_rejects_skipped_batch_and_inconsistent_cursor():
    current = engine()
    with pytest.raises(TrainingContractError, match="batch order"):
        current.step(batch(1))
    saved = cpu_tree(current.snapshot())
    saved["global_step"] = 1
    with pytest.raises(TrainingContractError, match="cursor"):
        current.restore(saved)


def test_epoch_plan_covers_all_chunks_once_and_resume_does_not_decode_prior_episodes():
    _, dataset = _dataset([12, 14, 10], ["train", "train", "validation"],
                       config=B0DataConfig(seed=17))
    plan = build_batch_plan(dataset.ordered_episodes(), batch_size=4, seed=17, epoch=0,
                            shuffle=True)
    pairs = [(episode, anchor) for episode, anchors in plan for anchor in anchors]
    expected = [(item.episode_index, item.anchor_frame_index)
                for item in dataset.iter_sample_identities()]
    assert sorted(pairs) == sorted(expected)
    assert len(pairs) == len(set(pairs)) == 12
    assert len(plan) == 4  # final short batches are retained
    assert plan_digest(plan) == plan_digest(build_batch_plan(
        dataset.ordered_episodes(), batch_size=4, seed=17, epoch=0, shuffle=True,
    ))
    assert plan_digest(plan) != plan_digest(build_batch_plan(
        dataset.ordered_episodes(), batch_size=4, seed=17, epoch=1, shuffle=True,
    ))

    batches = EpochBatches(dataset, plan, cache_dir="/unused", maximum_cached_episode_frames=32)
    batches._builder = lambda instructions, front, wrist: {"input_ids": torch.ones(len(front), 2)}
    episode_index = plan[-1][0]
    result = batches[len(plan) - 1]
    assert result["index"] == len(plan) - 1
    assert dataset.reader.step_opens == [episode_index]
    assert result["sample_count"] == len(plan[-1][1])
    assert result["targets"].shape[1:] == (8, 7)


def test_validation_weights_partial_batches_by_samples():
    class Policy:
        def eval(self):
            pass

        def __call__(self, inputs, state, normalized_action_targets):
            predictions = normalized_action_targets.clone()
            predictions[..., :6] += inputs["error"]
            return SimpleNamespace(loss=inputs["error"], normalized_actions=predictions)

    batches = []
    for count, error in ((4, 1.0), (1, 3.0)):
        targets = torch.zeros(count, 8, 7)
        targets[..., 6] = 1
        batches.append({"inputs": {"error": torch.tensor(error)},
                        "proprioception": torch.zeros(count, 8), "targets": targets,
                        "sample_count": count})
    result = evaluate_validation(Policy(), batches, torch.device("cpu"))
    assert result["objective"] == pytest.approx(1.4)
    assert result["continuous_action_mae"] == pytest.approx(1.4)
    assert result["gripper_accuracy"] == 1.0


def test_recipe_rejects_contract_drift_and_service_is_host_owned(tmp_path):
    value = recipe()
    value["epochs"] = 4
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value))
    with pytest.raises(TrainingContractError, match="epochs"):
        load_recipe(path)
    spec = importlib.util.spec_from_file_location("manage_job", ROOT / "scripts/manage_b0_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = SimpleNamespace(python=Path("/runtime/python"), recipe=Path("/recipe.json"),
                           repository_root=Path("/data/source"), frozen_root=Path("/data/frozen"),
                           normalization=Path("/data/stats.json"), cache_dir=Path("/data/cache"),
                           output_root=Path("/data/run with spaces"), pilot_steps=1000,
                           pilot_report=None)
    unit = module.render_unit(args)
    assert "Restart=on-failure" in unit
    assert "RestartPreventExitStatus=78" in unit
    assert "WantedBy=default.target" in unit
    assert "KillMode=mixed" in unit
    assert "ssh" not in unit
    assert '"/data/run with spaces"' in unit
    assert module.unit_quote("a%b") == '"a%%b"'


def test_epoch_completion_requires_all_updates_and_preserves_validation_history():
    current = engine()
    with pytest.raises(TrainingContractError, match="incomplete"):
        current.finish_epoch({"objective": 1.0})
    for index in range(8):
        current.step(batch(index))
    current.finish_epoch({"objective": 1.0})
    assert current.epoch == 1 and current.next_batch == 0
    assert current.history[0]["training_samples"] == 16
    current.restore(cpu_tree(current.snapshot()))
    current.step(batch(0))
    assert current.global_step == 9

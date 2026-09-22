from pathlib import Path

import pytest
import torch
from test_b0_data import _dataset
from test_b0_overfit import _tiny_policy
from test_b0_training import assert_tree_equal, batch, recipe

from eventground_wam.b0_continuation import (
    OBJECTIVE,
    ActiveBudget,
    MixedBatches,
    action_metrics,
    adopt_parent,
    validation_metrics,
)
from eventground_wam.b0_model import B0ObjectiveConfig
from eventground_wam.b0_training import (
    CheckpointStore,
    TrainingEngine,
    capture_rng,
    cpu_tree,
    seed_training,
)


def engine():
    policy, names = _tiny_policy()
    policy.objective_config = B0ObjectiveConfig(OBJECTIVE)
    config = {**recipe(), "objective": OBJECTIVE, "minimum_learning_rate_ratio": 1.,
              "warmup_fraction": 0., "decoder_projector_learning_rate": 1e-5,
              "lora_learning_rate": 1e-5}
    return TrainingEngine(policy, names, config, identity_digest="a" * 64,
                          steps_per_epoch=8, device=torch.device("cpu"))


def test_l1_uses_raw_gripper_and_decodes_binary_without_changing_gradient():
    current = engine()
    b = batch(0)
    b["targets"][0, :, 6] = -1
    out = current.policy(b["inputs"], b["proprioception"], normalized_action_targets=b["targets"])
    expected = torch.nn.functional.l1_loss(out.raw_action_predictions, b["targets"])
    torch.testing.assert_close(out.loss, expected)
    torch.testing.assert_close(out.loss, (6 * out.continuous_l1_loss + out.gripper_l1_loss) / 7)
    assert out.gripper_bce_loss is None and out.gripper_logits is None
    assert set(out.normalized_actions[..., 6].detach().unique().tolist()) <= {-1., 1.}
    actual_grad = torch.autograd.grad(out.loss, out.raw_action_predictions, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, out.raw_action_predictions)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad[..., 6].abs().sum() > 0
    assert action_metrics(out, b["targets"])["all_action_l1"] == pytest.approx(float(expected.detach()))


def test_parent_adoption_keeps_weights_adam_and_rng_with_new_child_counters():
    current = engine()
    current.step(batch(0))
    parent = cpu_tree(current.snapshot())
    parent.update(global_step=72000, epoch=1, next_batch=13063, samples_seen=2096581)
    clone = cpu_tree(parent)
    adopted = engine()
    result = adopt_parent(adopted, parent)
    assert result["learning_rates"] == [1e-5, 1e-5]
    assert adopted.global_step == adopted.next_batch == adopted.samples_seen == 0
    assert_tree_equal(adopted.snapshot()["state_dict"], parent["state_dict"])
    assert_tree_equal(adopted.snapshot()["optimizer"]["state"], parent["optimizer"]["state"])
    assert_tree_equal(capture_rng(), parent["rng"])
    assert_tree_equal(parent, clone)


def test_l1_disk_resume_matches_next_weights_adam_rng_and_constant_lr(tmp_path):
    seed_training(14)
    current = engine()
    for i in range(3):
        assert current.step(batch(i))["learning_rates"] == [1e-5, 1e-5]
    store = CheckpointStore(tmp_path, "a" * 64)
    store.save(current.snapshot())
    current.step(batch(3))
    expected = cpu_tree(current.snapshot())
    resumed = engine()
    resumed.restore(store.load())
    resumed.step(batch(3))
    assert_tree_equal(expected, cpu_tree(resumed.snapshot()))


def test_mixed_batches_preserve_tokens_and_resume_with_bounded_cache():
    _, data = _dataset([12, 14, 10], ["train", "train", "validation"])
    plan = [[(1, 0, 0), (1, 1, 3)], [(1, 1, 0), (1, 0, 4)]]
    mixed = MixedBatches(data, plan, "/unused", maximum_frames=14)
    mixed.builder = lambda instructions, front, wrist: {"input_ids": torch.ones(len(front), 2)}
    first = mixed[0]
    assert first["sample_count"] == 2 and mixed.cached_frames <= 14
    resumed = MixedBatches(data, plan, "/unused", maximum_frames=14)
    resumed.builder = mixed.builder
    a, b = mixed[1], resumed[1]
    assert a["identity_digest"] == b["identity_digest"]
    assert torch.equal(a["targets"], b["targets"])
    assert torch.equal(a["proprioception"], b["proprioception"])
    assert mixed.cached_frames <= 14
    with pytest.raises(ValueError, match="outside"):
        MixedBatches(data, [[(1, 2, 0)]], "/unused")


def test_validation_preserves_rng_modes_and_uses_binary_gripper_accuracy():
    current = engine()
    batches = [batch(0), batch(1)]
    before = cpu_tree(current.snapshot())
    modes = [m.training for m in current.policy.modules()]
    metrics = validation_metrics(current.policy, batches, torch.device("cpu"))
    assert metrics["sample_count"] == 4
    assert 0 <= metrics["gripper_accuracy"] <= 1
    assert metrics["all_action_l1"] == pytest.approx(
        (6 * metrics["motion_l1"] + metrics["gripper_l1"]) / 7)
    assert_tree_equal(before, cpu_tree(current.snapshot()))
    assert modes == [m.training for m in current.policy.modules()]


def test_budget_accumulates_across_process_lifetimes(tmp_path: Path):
    first = ActiveBudget(tmp_path / "budget.json", "x")
    first.prior = 100.
    first.close()
    resumed = ActiveBudget(tmp_path / "budget.json", "x")
    try:
        assert resumed.seconds >= 100.
    finally:
        resumed.close()

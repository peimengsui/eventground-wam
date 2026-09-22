import numpy as np
import pytest
import torch

from eventground_wam.b0_diagnostics import (
    gradient_statistics,
    noop_mask,
    numeric_digest,
    tensor_tree_digest,
)


def test_stationary_gripper_change_is_not_noop():
    actions = np.zeros((4, 7), dtype=np.float32)
    actions[:, 6] = [-1, 1, 1, 1]
    actions[3, 0] = .1
    assert noop_mask(actions).tolist() == [True, False, True, False]


def test_numeric_hash_ignores_storage_dtype_but_preserves_sequence():
    x = np.arange(14, dtype=np.float32).reshape(2, 7)
    assert numeric_digest(x) == numeric_digest(x.astype(np.float64))
    assert numeric_digest(x) != numeric_digest(x[::-1])


def test_gradient_sums_match_direct_combined_objective_and_do_not_populate_grad():
    a = torch.nn.Parameter(torch.tensor([1., 2.]))
    b = torch.nn.Parameter(torch.tensor([3.]))
    motion = a.square().sum() + b.sum()
    gripper = -a.sum()
    left = torch.autograd.grad(motion, [a, b], retain_graph=True, allow_unused=True)
    right = torch.autograd.grad(gripper, [a, b], retain_graph=True, allow_unused=True)
    direct = torch.autograd.grad(motion + gripper, [a, b])
    stats = gradient_statistics([
        "conditioning_adapter.backbone.language_model.x.lora_A.weight",
        "action_decoder.weight"], left, right)
    expected = torch.cat(direct).norm().item()
    assert stats["all_trainable"]["combined_norm"] == pytest.approx(expected)
    assert stats["action_head"]["cosine"] is None
    assert a.grad is None and b.grad is None
    assert stats["backbone_lora"]["cosine"] < 0


def test_undefined_ratios_and_nonfinite_are_explicit():
    name = "action_decoder.weight"
    result = gradient_statistics([name], [None], [torch.ones(2)])
    assert result["all_trainable"]["bce_motion_ratio"] is None
    with pytest.raises(ValueError, match="nonfinite"):
        gradient_statistics([name], [torch.tensor([float("nan")])], [None])


def test_tensor_hash_handles_bfloat16_and_detects_changes():
    a = {"p": torch.tensor([1, 2], dtype=torch.bfloat16)}
    assert tensor_tree_digest(a) == tensor_tree_digest({"p": a["p"].clone()})
    assert tensor_tree_digest(a) != tensor_tree_digest({"p": a["p"] + 1})

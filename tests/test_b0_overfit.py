from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from eventground_wam.b0_data import B0Batch, B0SampleIdentity
from eventground_wam.b0_model import (
    ActionQueryTransformer,
    ActionQueryTransformerConfig,
    B0ActionPolicy,
    QwenConditioningAdapter,
    QwenConditioningConfig,
)
from eventground_wam.b0_overfit import (
    B0OverfitError,
    OneBatchOverfitConfig,
    array_sha256,
    build_fixed_batch_manifest,
    evaluate_fixed_batch,
    run_one_batch_optimization,
)
from eventground_wam.demonstration_contracts import contract_digest


class ConstantPolicy(nn.Module):
    def __init__(self, predictions: Tensor, *, loss_offset: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("predictions", predictions)
        self.loss_offset = loss_offset

    def forward(
        self,
        _inputs: object,
        _proprioception: Tensor,
        *,
        normalized_action_targets: Tensor,
    ) -> SimpleNamespace:
        loss = torch.abs(self.predictions - normalized_action_targets).mean() + self.loss_offset
        return SimpleNamespace(normalized_actions=self.predictions, loss=loss)


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_scale = nn.Parameter(torch.linspace(0.5, 1.5, 16))
        self.lora_gain = nn.Parameter(torch.tensor(0.2))

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        use_cache: bool,
        return_dict: bool,
        **_: Tensor,
    ) -> SimpleNamespace:
        assert use_cache is False
        assert return_dict is True
        hidden = torch.sin(
            input_ids.float().unsqueeze(-1) / 7.0 + self.base_scale + self.lora_gain
        )
        assert hidden.shape[:2] == attention_mask.shape
        return SimpleNamespace(last_hidden_state=hidden)


def _tiny_policy() -> tuple[B0ActionPolicy, tuple[str, ...]]:
    backbone = TinyBackbone()
    adapter = QwenConditioningAdapter(
        backbone,
        config=QwenConditioningConfig(
            backbone_hidden_size=16,
            decoder_hidden_size=16,
            initialization_seed=3,
        ),
        trainable_backbone_parameter_names=("lora_gain",),
    )
    decoder = ActionQueryTransformer(
        config=ActionQueryTransformerConfig(
            model_dimension=16,
            layer_count=1,
            attention_head_count=4,
            feedforward_dimension=32,
            initialization_seed=5,
        )
    )
    return (
        B0ActionPolicy(adapter, decoder),
        ("conditioning_adapter.backbone.lora_gain",),
    )


def _tiny_inputs() -> dict[str, Tensor]:
    return {
        "input_ids": torch.tensor([[1, 2, 3, 0]] * 4),
        "attention_mask": torch.tensor([[1, 1, 1, 0]] * 4),
    }


def _batch() -> B0Batch:
    identities = tuple(
        B0SampleIdentity(
            split="train",
            episode_index=0,
            task_index=1,
            anchor_frame_index=index,
            action_frame_end_exclusive=index + 8,
            split_assignment_digest="a" * 64,
        )
        for index in range(4)
    )
    return B0Batch(
        identities=identities,
        instructions=("pick up the red block",) * 4,
        front_rgb=np.zeros((4, 3, 256, 256), dtype=np.uint8),
        wrist_rgb=np.ones((4, 3, 256, 256), dtype=np.uint8),
        proprioception=np.arange(32, dtype=np.float32).reshape(4, 8),
        actions=np.zeros((4, 8, 7), dtype=np.float32),
    )


def test_one_batch_config_locks_acceptance_thresholds() -> None:
    config = OneBatchOverfitConfig()

    assert config.loss_ratio_threshold == 0.05
    assert config.action_mae_threshold == 0.05
    assert config.to_dict()["scheduler"] == "none"
    with pytest.raises(B0OverfitError, match="loss-ratio threshold"):
        OneBatchOverfitConfig(loss_ratio_threshold=0.06)
    with pytest.raises(B0OverfitError, match="MAE threshold"):
        OneBatchOverfitConfig(action_mae_threshold=0.06)
    with pytest.raises(B0OverfitError, match="zero weight decay"):
        OneBatchOverfitConfig(weight_decay=0.01)


def test_array_digest_includes_dtype_shape_and_values() -> None:
    values = np.arange(12, dtype=np.float32).reshape(3, 4)

    assert array_sha256(values) == array_sha256(values.copy())
    assert array_sha256(values) != array_sha256(values.astype(np.float64))
    assert array_sha256(values) != array_sha256(values.reshape(2, 6))
    changed = values.copy()
    changed[0, 0] = 99
    assert array_sha256(values) != array_sha256(changed)


def test_fixed_batch_manifest_seals_exact_training_samples_and_tensors() -> None:
    batch = _batch()
    manifest = build_fixed_batch_manifest(
        batch,
        config=OneBatchOverfitConfig(),
        project_git_revision="b" * 40,
        source_identity={"dataset_version": "test-v1"},
        normalization_artifact_digest="c" * 64,
        normalization_file_sha256="d" * 64,
    )

    assert manifest["digest"] == contract_digest(manifest)
    assert manifest["selection"]["algorithm"] == "first-complete-canonical-b0-batch-v1"
    assert len(manifest["samples"]) == 4
    assert manifest["samples"][0]["anchor_frame_index"] == 0
    assert manifest["tensor_shapes"]["normalized_actions"] == [4, 8, 7]
    assert manifest["tensor_sha256"]["normalized_actions"] == array_sha256(batch.actions)

    bad_batch = B0Batch(
        identities=batch.identities[:3],
        instructions=batch.instructions[:3],
        front_rgb=batch.front_rgb[:3],
        wrist_rgb=batch.wrist_rgb[:3],
        proprioception=batch.proprioception[:3],
        actions=batch.actions[:3],
    )
    with pytest.raises(B0OverfitError, match="expected 4"):
        build_fixed_batch_manifest(
            bad_batch,
            config=OneBatchOverfitConfig(),
            project_git_revision="b" * 40,
            source_identity={"dataset_version": "test-v1"},
            normalization_artifact_digest="c" * 64,
            normalization_file_sha256="d" * 64,
        )


def test_diagnostic_uses_direct_normalized_action_mae_and_both_thresholds() -> None:
    targets = torch.zeros(4, 8, 7)
    policy = ConstantPolicy(torch.full_like(targets, 0.02))
    config = OneBatchOverfitConfig()

    metrics = evaluate_fixed_batch(
        policy,  # type: ignore[arg-type]
        {},
        torch.zeros(4, 8),
        targets,
        step=20,
        initial_loss=1.0,
        config=config,
    )

    assert metrics.l1_loss == pytest.approx(0.02)
    assert metrics.normalized_action_mae == pytest.approx(0.02)
    assert metrics.loss_ratio_to_initial == pytest.approx(0.02)
    assert metrics.passed is True

    absolute_failure = evaluate_fixed_batch(
        ConstantPolicy(torch.full_like(targets, 0.06)),  # type: ignore[arg-type]
        {},
        torch.zeros(4, 8),
        targets,
        step=20,
        initial_loss=2.0,
        config=config,
    )
    assert absolute_failure.loss_ratio_to_initial < 0.05
    assert absolute_failure.normalized_action_mae > 0.05
    assert absolute_failure.passed is False


def test_diagnostic_rejects_loss_that_differs_from_direct_mae() -> None:
    targets = torch.zeros(4, 8, 7)
    policy = ConstantPolicy(torch.full_like(targets, 0.02), loss_offset=0.1)

    with pytest.raises(B0OverfitError, match="differs"):
        evaluate_fixed_batch(
            policy,  # type: ignore[arg-type]
            {},
            torch.zeros(4, 8),
            targets,
            step=0,
            initial_loss=None,
            config=OneBatchOverfitConfig(),
        )


def test_tiny_one_batch_training_checks_all_gradient_groups() -> None:
    policy, exact_lora_names = _tiny_policy()
    config = OneBatchOverfitConfig(
        max_steps=2,
        evaluation_interval=1,
        decoder_projector_learning_rate=1e-2,
        lora_learning_rate=1e-3,
    )

    result = run_one_batch_optimization(
        policy,
        _tiny_inputs(),
        torch.linspace(-0.5, 0.5, 32).reshape(4, 8),
        torch.zeros(4, 8, 7),
        exact_lora_parameter_names=exact_lora_names,
        config=config,
    )

    assert result.steps_completed >= 1
    assert result.initial.step == 0
    assert result.final.step == result.steps_completed
    assert result.training_seconds >= 0.0
    assert len(result.curve) == result.steps_completed + 1
    assert result.last_gradient_summary["zero_signal_groups"] == []
    assert set(result.last_gradient_summary["nonzero_signal_groups"]) == {
        "qwen_upper_lora",
        "backbone_projection",
        "proprioception_projector",
        "action_decoder",
    }
    assert policy.conditioning_adapter.backbone.base_scale.requires_grad is False
    assert policy.conditioning_adapter.backbone.base_scale.grad is None

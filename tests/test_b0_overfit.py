from __future__ import annotations

import json
from pathlib import Path
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
    B0ObjectiveConfig,
    QwenConditioningAdapter,
    QwenConditioningConfig,
)
from eventground_wam.b0_overfit import (
    B0OverfitError,
    CohortTensorBatch,
    OneBatchOverfitConfig,
    TenDemonstrationOverfitConfig,
    array_sha256,
    build_fixed_batch_manifest,
    build_ten_demonstration_manifest,
    evaluate_fixed_batch,
    evaluate_ten_demonstration_cohort,
    run_one_batch_optimization,
    run_ten_demonstration_optimization,
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


class InputPredictionPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.objective_config = B0ObjectiveConfig()

    def forward(
        self,
        inputs: dict[str, Tensor],
        _proprioception: Tensor,
        *,
        normalized_action_targets: Tensor,
    ) -> SimpleNamespace:
        predictions = inputs["predictions"]
        loss = torch.abs(predictions - normalized_action_targets).mean()
        return SimpleNamespace(
            normalized_actions=predictions,
            gripper_logits=predictions[..., 6],
            loss=loss,
        )


class HybridInputPredictionPolicy(nn.Module):
    def __init__(self, *, gripper_bce_weight: float) -> None:
        super().__init__()
        self.objective_config = B0ObjectiveConfig(
            objective="continuous-l1-plus-gripper-bce-v1",
            gripper_bce_weight=gripper_bce_weight,
        )

    def forward(
        self,
        inputs: dict[str, Tensor],
        _proprioception: Tensor,
        *,
        normalized_action_targets: Tensor,
    ) -> SimpleNamespace:
        raw = inputs["predictions"]
        logits = raw[..., 6]
        decoded = raw.clone()
        decoded[..., 6] = torch.where(logits >= 0, 1.0, -1.0)
        continuous_l1 = torch.abs(raw[..., :6] - normalized_action_targets[..., :6]).mean()
        labels = (normalized_action_targets[..., 6] > 0).to(dtype=logits.dtype)
        gripper_bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return SimpleNamespace(
            normalized_actions=decoded,
            gripper_logits=logits,
            loss=continuous_l1 + self.objective_config.gripper_bce_weight * gripper_bce,
        )


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
        hidden = torch.sin(input_ids.float().unsqueeze(-1) / 7.0 + self.base_scale + self.lora_gain)
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


def _ten_sample_cohort() -> tuple[B0Batch, list[dict[str, object]]]:
    episode_indices = (0, 1, 3, 4, 5, 8, 9, 10, 11, 12)
    identities = tuple(
        B0SampleIdentity(
            split="train",
            episode_index=episode_index,
            task_index=position,
            anchor_frame_index=0,
            action_frame_end_exclusive=8,
            split_assignment_digest=f"{position + 1:064x}",
        )
        for position, episode_index in enumerate(episode_indices)
    )
    cohort = B0Batch(
        identities=identities,
        instructions=tuple(f"instruction {index}" for index in range(10)),
        front_rgb=np.zeros((10, 3, 256, 256), dtype=np.uint8),
        wrist_rgb=np.ones((10, 3, 256, 256), dtype=np.uint8),
        proprioception=np.arange(80, dtype=np.float32).reshape(10, 8),
        actions=np.zeros((10, 8, 7), dtype=np.float32),
    )
    summaries: list[dict[str, object]] = [
        {
            "episode_index": identity.episode_index,
            "task_index": identity.task_index,
            "instruction": cohort.instructions[position],
            "episode_length": 8,
            "sample_count": 1,
            "split_assignment_digest": identity.split_assignment_digest,
            "first_sample_identity_digest": identity.digest,
            "last_sample_identity_digest": identity.digest,
        }
        for position, identity in enumerate(identities)
    ]
    return cohort, summaries


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


def test_ten_demonstration_config_locks_all_three_acceptance_thresholds() -> None:
    config = TenDemonstrationOverfitConfig()

    assert config.demonstration_count == 10
    assert config.loss_ratio_threshold == 0.20
    assert config.action_mae_threshold == 0.10
    assert config.gripper_sign_accuracy_threshold == 1.0
    assert config.scheduler == "none"
    assert config.objective == "normalized-action-l1"
    with pytest.raises(B0OverfitError, match="exactly ten"):
        TenDemonstrationOverfitConfig(demonstration_count=9)
    with pytest.raises(B0OverfitError, match="loss-ratio"):
        TenDemonstrationOverfitConfig(loss_ratio_threshold=0.21)
    with pytest.raises(B0OverfitError, match="MAE threshold"):
        TenDemonstrationOverfitConfig(action_mae_threshold=0.11)
    with pytest.raises(B0OverfitError, match="gripper-sign"):
        TenDemonstrationOverfitConfig(gripper_sign_accuracy_threshold=0.99)
    with pytest.raises(B0OverfitError, match="scheduler"):
        TenDemonstrationOverfitConfig(scheduler="linear")
    with pytest.raises(B0OverfitError, match="learning_rate_ratio"):
        TenDemonstrationOverfitConfig(scheduler_minimum_learning_rate_ratio=0.0)
    with pytest.raises(B0OverfitError, match="objective"):
        TenDemonstrationOverfitConfig(objective="gripper-mse")
    with pytest.raises(B0OverfitError, match="gripper_bce_weight"):
        TenDemonstrationOverfitConfig(gripper_bce_weight=0.0)


def test_ten_demonstration_manifest_seals_all_episodes_and_tensor_values() -> None:
    cohort, summaries = _ten_sample_cohort()
    manifest = build_ten_demonstration_manifest(
        cohort,
        summaries,
        config=TenDemonstrationOverfitConfig(),
        project_git_revision="a" * 40,
        source_identity={"dataset_version": "test-v1"},
        normalization_artifact_digest="b" * 64,
        normalization_file_sha256="c" * 64,
    )

    assert manifest["digest"] == contract_digest(manifest)
    assert manifest["selection"]["demonstration_count"] == 10
    assert manifest["selection"]["sample_count"] == 10
    assert len(manifest["episodes"]) == 10
    assert manifest["tensor_shapes"]["normalized_actions"] == [10, 8, 7]
    assert manifest["tensor_sha256"]["normalized_actions"] == array_sha256(cohort.actions)


def test_cohort_diagnostic_weights_every_action_and_requires_perfect_gripper_sign() -> None:
    targets_one = torch.ones(1, 8, 7)
    targets_three = torch.ones(3, 8, 7)
    batches = (
        CohortTensorBatch(
            backbone_inputs={"predictions": targets_one.clone()},
            proprioception=torch.zeros(1, 8),
            targets=targets_one,
            sample_identity_digests=("a" * 64,),
        ),
        CohortTensorBatch(
            backbone_inputs={"predictions": targets_three + 0.1},
            proprioception=torch.zeros(3, 8),
            targets=targets_three,
            sample_identity_digests=("b" * 64, "c" * 64, "d" * 64),
        ),
    )
    config = TenDemonstrationOverfitConfig()

    metrics = evaluate_ten_demonstration_cohort(
        InputPredictionPolicy(),  # type: ignore[arg-type]
        batches,
        device=torch.device("cpu"),
        epoch=1,
        step=2,
        initial_loss=1.0,
        config=config,
    )

    assert metrics.normalized_action_mae == pytest.approx(0.075)
    assert metrics.l1_loss == pytest.approx(0.075)
    assert metrics.sample_count == 4
    assert metrics.gripper_sign_accuracy == 1.0
    assert metrics.passed is True

    sign_predictions = torch.ones(4, 8, 7)
    sign_predictions[0, 0, 6] = -1.0
    sign_failure = evaluate_ten_demonstration_cohort(
        InputPredictionPolicy(),  # type: ignore[arg-type]
        (
            CohortTensorBatch(
                backbone_inputs={"predictions": sign_predictions},
                proprioception=torch.zeros(4, 8),
                targets=torch.ones(4, 8, 7),
                sample_identity_digests=("a" * 64,) * 4,
            ),
        ),
        device=torch.device("cpu"),
        epoch=1,
        step=1,
        initial_loss=1.0,
        config=config,
    )
    assert sign_failure.normalized_action_mae < 0.10
    assert sign_failure.loss_ratio_to_initial < 0.20
    assert sign_failure.gripper_sign_accuracy < 1.0
    assert sign_failure.passed is False


def test_cohort_diagnostic_aggregates_hybrid_bce_and_exact_sign_errors() -> None:
    targets = torch.ones(3, 8, 7)
    targets[0, ..., 6] = -1.0
    raw = targets.clone()
    raw[..., :6] += 0.03
    raw[..., 6] = torch.tensor(
        [
            [-2.0] * 8,
            [2.0] * 8,
            [2.0] * 7 + [-0.25],
        ]
    )
    config = TenDemonstrationOverfitConfig(
        objective="continuous-l1-plus-gripper-bce-v1",
        gripper_bce_weight=0.75,
    )
    metrics = evaluate_ten_demonstration_cohort(
        HybridInputPredictionPolicy(gripper_bce_weight=0.75),  # type: ignore[arg-type]
        (
            CohortTensorBatch(
                backbone_inputs={"predictions": raw[:1]},
                proprioception=torch.zeros(1, 8),
                targets=targets[:1],
                sample_identity_digests=("a" * 64,),
            ),
            CohortTensorBatch(
                backbone_inputs={"predictions": raw[1:]},
                proprioception=torch.zeros(2, 8),
                targets=targets[1:],
                sample_identity_digests=("b" * 64, "c" * 64),
            ),
        ),
        device=torch.device("cpu"),
        epoch=1,
        step=2,
        initial_loss=1.0,
        config=config,
    )

    labels = (targets[..., 6] > 0).float()
    expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(raw[..., 6], labels)
    assert metrics.objective == "continuous-l1-plus-gripper-bce-v1"
    assert metrics.continuous_action_mae == pytest.approx(0.03)
    assert metrics.gripper_bce_loss == pytest.approx(expected_bce.item())
    assert metrics.objective_loss == pytest.approx(0.03 + 0.75 * expected_bce.item())
    assert metrics.gripper_sign_error_count == 1
    assert metrics.gripper_sign_accuracy == pytest.approx(23 / 24)
    assert metrics.minimum_gripper_signed_logit_margin == pytest.approx(-0.25)
    assert metrics.normalized_action_mae == pytest.approx((0.03 * 144 + 2.0) / 168)
    assert metrics.passed is False


def test_tiny_ten_demonstration_training_uses_every_cached_minibatch() -> None:
    policy, exact_lora_names = _tiny_policy()
    inputs = _tiny_inputs()
    targets = torch.zeros(4, 8, 7)
    targets[..., 6] = 1.0
    batches = tuple(
        CohortTensorBatch(
            backbone_inputs={name: value[start : start + 2] for name, value in inputs.items()},
            proprioception=torch.linspace(-0.5, 0.5, 32).reshape(4, 8)[start : start + 2],
            targets=targets[start : start + 2],
            sample_identity_digests=(f"{start + 1:064x}", f"{start + 2:064x}"),
        )
        for start in (0, 2)
    )
    config = TenDemonstrationOverfitConfig(
        max_epochs=1,
        evaluation_interval_epochs=1,
        decoder_projector_learning_rate=1e-2,
        lora_learning_rate=1e-3,
    )

    result = run_ten_demonstration_optimization(
        policy,
        batches,
        device=torch.device("cpu"),
        exact_lora_parameter_names=exact_lora_names,
        config=config,
    )

    assert result.epochs_completed == 1
    assert result.steps_completed == 2
    assert result.initial.sample_count == 4
    assert result.final.sample_count == 4
    assert sum(record["record_type"] == "training" for record in result.curve) == 2
    assert result.last_gradient_summary["zero_signal_groups"] == []


def test_ten_demonstration_continuation_retains_baseline_and_decays_learning_rate() -> None:
    policy, exact_lora_names = _tiny_policy()
    inputs = _tiny_inputs()
    targets = torch.zeros(4, 8, 7)
    targets[..., 6] = 1.0
    batches = tuple(
        CohortTensorBatch(
            backbone_inputs={name: value[start : start + 2] for name, value in inputs.items()},
            proprioception=torch.linspace(-0.5, 0.5, 32).reshape(4, 8)[start : start + 2],
            targets=targets[start : start + 2],
            sample_identity_digests=(f"{start + 1:064x}", f"{start + 2:064x}"),
        )
        for start in (0, 2)
    )
    config = TenDemonstrationOverfitConfig(
        max_epochs=3,
        decoder_projector_learning_rate=1e-2,
        lora_learning_rate=1e-3,
        scheduler="cosine",
        scheduler_minimum_learning_rate_ratio=0.1,
    )

    result = run_ten_demonstration_optimization(
        policy,
        batches,
        device=torch.device("cpu"),
        exact_lora_parameter_names=exact_lora_names,
        config=config,
        starting_epoch=2,
        starting_step=4,
        acceptance_initial_l1=2.0,
    )

    training_records = [record for record in result.curve if record["record_type"] == "training"]
    assert result.initial.epoch == 2
    assert result.initial.step == 4
    assert result.initial.loss_ratio_to_initial == pytest.approx(result.initial.l1_loss / 2.0)
    assert result.acceptance_initial_l1 == 2.0
    assert result.epochs_completed == 3
    assert result.steps_completed == 6
    assert len(training_records) == 2
    assert training_records[0]["applied_decoder_projector_learning_rate"] == pytest.approx(1e-2)
    assert training_records[-1]["next_decoder_projector_learning_rate"] == pytest.approx(1e-3)


def test_committed_one_batch_overfit_evidence_is_sealed_and_passes_locked_gates() -> None:
    path = Path(__file__).parents[1] / "docs/evaluation/week3-b0-overfit-1batch-v1.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["digest"] == contract_digest(evidence)
    assert evidence["status"] == "passed"
    assert evidence["implementation"]["project_tree_clean"] is True
    assert evidence["implementation"]["project_git_revision"] == (
        "795f152b71e62bbbf61f1cf5f330e5be8847f0d1"
    )
    assert evidence["configuration"]["loss_ratio_threshold"] == 0.05
    assert evidence["configuration"]["action_mae_threshold"] == 0.05
    assert evidence["validation"]["initial"]["l1_loss"] > 0.0
    assert evidence["validation"]["final"]["loss_ratio_to_initial"] <= 0.05
    assert evidence["validation"]["final"]["normalized_action_mae"] <= 0.05
    assert evidence["validation"]["final"]["gripper_sign_accuracy"] == 1.0
    assert evidence["validation"]["gradients"]["frozen_backbone_gradient_count"] == 0
    assert evidence["validation"]["gradients"]["zero_signal_group_count"] == 0
    assert evidence["validation"]["checkpoint"]["reload_max_absolute_error"] <= 1e-3
    assert evidence["artifacts"]["overfit_report"]["file_sha256"] == (
        "1abcbd154f338698cd2be19f4d8c458251ad62c9f5a2714fd87bbbe73e6064ea"
    )

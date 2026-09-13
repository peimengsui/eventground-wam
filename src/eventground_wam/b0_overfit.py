"""Reproducible optimization and evidence contracts for the B0 fixed-batch gate."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from eventground_wam.b0_data import B0Batch
from eventground_wam.b0_model import (
    B0ActionPolicy,
    B0ModelError,
    validate_b0_trainable_parameters,
)
from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract

LOCKED_ONE_BATCH_LOSS_RATIO_THRESHOLD = 0.05
LOCKED_ONE_BATCH_ACTION_MAE_THRESHOLD = 0.05
FIXED_BATCH_SELECTION_ALGORITHM = "first-complete-canonical-b0-batch-v1"


class B0OverfitError(ValueError):
    """Raised when a fixed-batch run violates its declared optimization contract."""


@dataclass(frozen=True)
class OneBatchOverfitConfig:
    """Predeclared settings for one immutable fixed-batch diagnostic run."""

    seed: int = 17
    batch_size: int = 4
    max_steps: int = 2_000
    evaluation_interval: int = 20
    decoder_projector_learning_rate: float = 1e-3
    lora_learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    maximum_gradient_norm: float = 1.0
    loss_ratio_threshold: float = LOCKED_ONE_BATCH_LOSS_RATIO_THRESHOLD
    action_mae_threshold: float = LOCKED_ONE_BATCH_ACTION_MAE_THRESHOLD

    def __post_init__(self) -> None:
        for name, value in (
            ("seed", self.seed),
            ("batch_size", self.batch_size),
            ("max_steps", self.max_steps),
            ("evaluation_interval", self.evaluation_interval),
        ):
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise B0OverfitError(f"{name} must be a valid nonnegative/positive integer")
        for name, value in (
            ("decoder_projector_learning_rate", self.decoder_projector_learning_rate),
            ("lora_learning_rate", self.lora_learning_rate),
            ("adam_epsilon", self.adam_epsilon),
            ("maximum_gradient_norm", self.maximum_gradient_norm),
        ):
            if not math.isfinite(value) or value <= 0:
                raise B0OverfitError(f"{name} must be finite and positive")
        for name, value in (("adam_beta1", self.adam_beta1), ("adam_beta2", self.adam_beta2)):
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise B0OverfitError(f"{name} must be finite and in [0,1)")
        if not math.isfinite(self.weight_decay) or self.weight_decay != 0.0:
            raise B0OverfitError("the one-batch diagnostic requires zero weight decay")
        if self.loss_ratio_threshold != LOCKED_ONE_BATCH_LOSS_RATIO_THRESHOLD:
            raise B0OverfitError("the one-batch loss-ratio threshold is locked at 0.05")
        if self.action_mae_threshold != LOCKED_ONE_BATCH_ACTION_MAE_THRESHOLD:
            raise B0OverfitError("the one-batch normalized-action MAE threshold is locked at 0.05")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-native optimizer contract."""

        return {
            **asdict(self),
            "optimizer": "AdamW",
            "scheduler": "none",
            "augmentation": "none",
            "early_stop": "first-diagnostic-evaluation-passing-both-locked-thresholds",
        }


@dataclass(frozen=True)
class FixedBatchDiagnostics:
    """Evaluation-mode action errors on the predeclared training batch."""

    step: int
    l1_loss: float
    normalized_action_mae: float
    loss_ratio_to_initial: float
    maximum_absolute_error: float
    per_action_dimension_mae: tuple[float, ...]
    gripper_sign_accuracy: float | None
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OneBatchTrainingResult:
    """In-memory result used to write curves, checkpoints, and sealed evidence."""

    initial: FixedBatchDiagnostics
    final: FixedBatchDiagnostics
    steps_completed: int
    passed: bool
    curve: tuple[dict[str, Any], ...]
    trainable_parameter_names: tuple[str, ...]
    trainable_parameter_count: int
    last_gradient_summary: Mapping[str, Any]
    training_seconds: float


def array_sha256(array: NDArray[Any]) -> str:
    """Hash an array's dtype, shape, and C-order bytes without lossy conversion."""

    contiguous = np.ascontiguousarray(array)
    header = canonical_json_bytes(
        {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def build_fixed_batch_manifest(
    batch: B0Batch,
    *,
    config: OneBatchOverfitConfig,
    project_git_revision: str,
    source_identity: Mapping[str, Any],
    normalization_artifact_digest: str,
    normalization_file_sha256: str,
) -> dict[str, Any]:
    """Seal the exact diagnostic batch before any optimization is attempted."""

    if len(batch.identities) != config.batch_size:
        raise B0OverfitError(
            f"fixed batch contains {len(batch.identities)} samples, expected {config.batch_size}"
        )
    if len(batch.instructions) != config.batch_size:
        raise B0OverfitError("fixed batch instruction count does not match its sample count")
    if any(identity.split != "train" for identity in batch.identities):
        raise B0OverfitError("the fixed overfit batch must contain only training samples")
    manifest = {
        "artifact_type": "eventground_wam.b0_fixed_batch_manifest",
        "schema_version": 1,
        "selection": {
            "algorithm": FIXED_BATCH_SELECTION_ALGORITHM,
            "epoch": 0,
            "shuffle_episodes": False,
            "batch_size": config.batch_size,
        },
        "optimizer_contract": config.to_dict(),
        "project_git_revision": project_git_revision,
        "source": dict(source_identity),
        "normalization": {
            "artifact_digest": normalization_artifact_digest,
            "file_sha256": normalization_file_sha256,
        },
        "samples": [identity.to_dict() for identity in batch.identities],
        "instructions": list(batch.instructions),
        "tensor_sha256": {
            "front_rgb": array_sha256(batch.front_rgb),
            "wrist_rgb": array_sha256(batch.wrist_rgb),
            "proprioception": array_sha256(batch.proprioception),
            "normalized_actions": array_sha256(batch.actions),
        },
        "tensor_shapes": {
            "front_rgb": list(batch.front_rgb.shape),
            "wrist_rgb": list(batch.wrist_rgb.shape),
            "proprioception": list(batch.proprioception.shape),
            "normalized_actions": list(batch.actions.shape),
        },
    }
    return seal_contract(manifest)


def evaluate_fixed_batch(
    policy: B0ActionPolicy,
    backbone_inputs: Mapping[str, Tensor],
    proprioception: Tensor,
    targets: Tensor,
    *,
    step: int,
    initial_loss: float | None,
    config: OneBatchOverfitConfig,
) -> FixedBatchDiagnostics:
    """Compute the locked evaluation-mode loss and direct normalized-action MAE."""

    if type(step) is not int or step < 0:
        raise B0OverfitError("diagnostic step must be a nonnegative integer")
    policy.eval()
    with torch.no_grad():
        output = policy(
            backbone_inputs,
            proprioception,
            normalized_action_targets=targets,
        )
    if output.loss is None or not torch.isfinite(output.loss):
        raise B0OverfitError("fixed-batch evaluation produced a missing or non-finite L1 loss")
    predictions = output.normalized_actions
    direct_mae = torch.abs(predictions - targets).mean()
    if not torch.isfinite(direct_mae):
        raise B0OverfitError("fixed-batch evaluation produced a non-finite action MAE")
    l1_loss = output.loss.item()
    normalized_action_mae = direct_mae.item()
    if abs(l1_loss - normalized_action_mae) > 1e-6:
        raise B0OverfitError("locked L1 loss differs from direct normalized-action MAE")
    if initial_loss is None:
        loss_ratio = 1.0
    else:
        if not math.isfinite(initial_loss) or initial_loss <= 0.0:
            raise B0OverfitError("initial diagnostic L1 loss must be finite and positive")
        loss_ratio = l1_loss / initial_loss
    dimension_mae = torch.abs(predictions - targets).mean(dim=(0, 1))
    gripper_targets = targets[..., 6]
    gripper_mask = gripper_targets.abs() >= 0.5
    gripper_sign_accuracy = None
    if gripper_mask.any().item():
        correct = (predictions[..., 6] >= 0) == (gripper_targets >= 0)
        gripper_sign_accuracy = correct[gripper_mask].float().mean().item()
    passed = (
        loss_ratio <= config.loss_ratio_threshold
        and normalized_action_mae <= config.action_mae_threshold
    )
    return FixedBatchDiagnostics(
        step=step,
        l1_loss=l1_loss,
        normalized_action_mae=normalized_action_mae,
        loss_ratio_to_initial=loss_ratio,
        maximum_absolute_error=torch.abs(predictions - targets).max().item(),
        per_action_dimension_mae=tuple(value.item() for value in dimension_mae),
        gripper_sign_accuracy=gripper_sign_accuracy,
        passed=passed,
    )


def run_one_batch_optimization(
    policy: B0ActionPolicy,
    backbone_inputs: Mapping[str, Tensor],
    proprioception: Tensor,
    targets: Tensor,
    *,
    exact_lora_parameter_names: Sequence[str],
    config: OneBatchOverfitConfig,
    progress_callback: Callable[[FixedBatchDiagnostics], None] | None = None,
) -> OneBatchTrainingResult:
    """Optimize only the allowlisted B0 path on one in-memory batch."""

    if targets.shape != (config.batch_size, 8, 7):
        raise B0OverfitError(
            f"fixed-batch targets have shape {tuple(targets.shape)}, expected "
            f"({config.batch_size}, 8, 7)"
        )
    if proprioception.shape != (config.batch_size, 8):
        raise B0OverfitError("fixed-batch proprioception must have shape [batch,8]")
    trainable_names = validate_b0_trainable_parameters(
        policy,
        allowed_exact_names=exact_lora_parameter_names,
    )
    parameters = dict(policy.named_parameters())
    exact_lora_set = set(exact_lora_parameter_names)
    lora_parameters = [parameters[name] for name in trainable_names if name in exact_lora_set]
    decoder_projector_parameters = [
        parameters[name] for name in trainable_names if name not in exact_lora_set
    ]
    if not lora_parameters or not decoder_projector_parameters:
        raise B0OverfitError("both LoRA and decoder/projector optimizer groups must be non-empty")
    optimizer = torch.optim.AdamW(
        [
            {"params": decoder_projector_parameters, "lr": config.decoder_projector_learning_rate},
            {"params": lora_parameters, "lr": config.lora_learning_rate},
        ],
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    initial = evaluate_fixed_batch(
        policy,
        backbone_inputs,
        proprioception,
        targets,
        step=0,
        initial_loss=None,
        config=config,
    )
    if initial.l1_loss <= 0.0:
        raise B0OverfitError("the fixed batch has zero initial loss and cannot demonstrate a reduction")
    if progress_callback is not None:
        progress_callback(initial)
    curve: list[dict[str, Any]] = [
        {
            "step": 0,
            "training_l1_loss": None,
            "diagnostic": initial.to_dict(),
        }
    ]
    last_diagnostic = initial
    last_gradient_summary: Mapping[str, Any] = {}
    _synchronize(targets.device)
    started = time.perf_counter()
    steps_completed = 0
    for step in range(1, config.max_steps + 1):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        output = policy(
            backbone_inputs,
            proprioception,
            normalized_action_targets=targets,
        )
        if output.loss is None or not torch.isfinite(output.loss):
            raise B0OverfitError(f"training step {step} produced a missing or non-finite loss")
        output.loss.backward()
        validate_gradients = (
            step == 1
            or step % config.evaluation_interval == 0
            or step == config.max_steps
        )
        if validate_gradients:
            last_gradient_summary = validate_training_gradients(
                policy,
                trainable_names=trainable_names,
                exact_lora_parameter_names=exact_lora_parameter_names,
            )
        gradient_norm = nn.utils.clip_grad_norm_(
            [parameters[name] for name in trainable_names],
            max_norm=config.maximum_gradient_norm,
            error_if_nonfinite=True,
        )
        if not torch.isfinite(gradient_norm):
            raise B0OverfitError(f"training step {step} produced a non-finite gradient norm")
        optimizer.step()
        steps_completed = step
        record: dict[str, Any] = {
            "step": step,
            "training_l1_loss": output.loss.detach().item(),
            "gradient_norm_before_clip": gradient_norm.item(),
        }
        if step % config.evaluation_interval == 0 or step == config.max_steps:
            last_diagnostic = evaluate_fixed_batch(
                policy,
                backbone_inputs,
                proprioception,
                targets,
                step=step,
                initial_loss=initial.l1_loss,
                config=config,
            )
            record["diagnostic"] = last_diagnostic.to_dict()
            if progress_callback is not None:
                progress_callback(last_diagnostic)
        curve.append(record)
        if last_diagnostic.step == step and last_diagnostic.passed:
            break
    _synchronize(targets.device)
    elapsed = time.perf_counter() - started
    policy.eval()
    return OneBatchTrainingResult(
        initial=initial,
        final=last_diagnostic,
        steps_completed=steps_completed,
        passed=last_diagnostic.passed,
        curve=tuple(curve),
        trainable_parameter_names=trainable_names,
        trainable_parameter_count=sum(parameters[name].numel() for name in trainable_names),
        last_gradient_summary=last_gradient_summary,
        training_seconds=elapsed,
    )


def validate_training_gradients(
    policy: B0ActionPolicy,
    *,
    trainable_names: Sequence[str],
    exact_lora_parameter_names: Sequence[str],
) -> dict[str, Any]:
    """Enforce finite gradients and a nonzero signal in every declared trainable group."""

    parameters = dict(policy.named_parameters())
    missing = [name for name in trainable_names if parameters[name].grad is None]
    nonfinite = [
        name
        for name in trainable_names
        if parameters[name].grad is not None and not torch.isfinite(parameters[name].grad).all()
    ]
    exact_lora_set = set(exact_lora_parameter_names)
    forbidden = [
        name
        for name, parameter in parameters.items()
        if name.startswith("conditioning_adapter.backbone.")
        and name not in exact_lora_set
        and parameter.grad is not None
    ]
    groups = {
        "qwen_upper_lora": tuple(exact_lora_parameter_names),
        "backbone_projection": tuple(
            name
            for name in trainable_names
            if name.startswith("conditioning_adapter.backbone_projection.")
        ),
        "proprioception_projector": tuple(
            name
            for name in trainable_names
            if name.startswith("conditioning_adapter.proprioception_projector.")
        ),
        "action_decoder": tuple(
            name for name in trainable_names if name.startswith("action_decoder.")
        ),
    }
    zero_signal_groups = [
        group
        for group, names in groups.items()
        if not names
        or not any(
            parameters[name].grad is not None
            and bool((parameters[name].grad.detach().abs() > 0).any())
            for name in names
        )
    ]
    if missing or nonfinite or forbidden or zero_signal_groups:
        raise B0ModelError(
            "fixed-batch gradient contract failed: "
            f"missing={missing}, nonfinite={nonfinite}, forbidden={forbidden}, "
            f"zero_groups={zero_signal_groups}"
        )
    return {
        "all_trainable_gradients_present": True,
        "all_trainable_gradients_finite": True,
        "frozen_backbone_gradient_names": forbidden,
        "nonzero_signal_groups": list(groups),
        "zero_signal_groups": zero_signal_groups,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

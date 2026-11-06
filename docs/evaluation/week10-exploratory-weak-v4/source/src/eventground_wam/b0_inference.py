"""Verified B0 checkpoint loading and LIBERO action-chunk inference."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from eventground_wam.b0_data import ActionNormalizer, load_b0_action_normalizer
from eventground_wam.b0_model import (
    QWEN_REVISION,
    ActionQueryTransformer,
    B0ActionPolicy,
    B0ModelError,
    B0ObjectiveConfig,
    QwenB0InputBuilder,
    QwenConditioningAdapter,
    install_qwen_upper_lora,
    load_b0_trainable_state_dict,
    load_pinned_qwen_model,
    load_pinned_qwen_processor,
)
from eventground_wam.demonstration_contracts import contract_digest

B0_ROLLOUT_CHECKPOINT_ID = "eventground-wam/b0-ten-demonstration-bce-v1"
B0_COHORT_CHECKPOINT_FORMAT = "eventground-wam-b0-ten-demonstration-trainable-state-v1"
B0_ACCEPTED_REVIEW_ARTIFACT_TYPE = "eventground_wam.b0_ten_demonstration_overfit_review"
B0_TRAINING_REPORT_ARTIFACT_TYPE = "eventground_wam.b0_ten_demonstration_overfit"
B0_COHORT_MANIFEST_ARTIFACT_TYPE = "eventground_wam.b0_ten_demonstration_manifest"
B0_ROLLOUT_OBJECTIVE = "continuous-l1-plus-gripper-bce-v1"
B0_ROLLOUT_GRIPPER_BCE_WEIGHT = 1.0
B0_SIMULATOR_ACTION_MINIMUM = -1.0
B0_SIMULATOR_ACTION_MAXIMUM = 1.0


class B0InferenceError(RuntimeError):
    """Raised when a B0 inference bundle or action violates the rollout contract."""


@dataclass(frozen=True)
class VerifiedB0Checkpoint:
    """Immutable identities proven before model tensors are loaded."""

    checkpoint_root: Path
    checkpoint_path: Path
    checkpoint_file_sha256: str
    checkpoint_id: str
    checkpoint_revision: str
    training_report_digest: str
    training_report_file_sha256: str
    cohort_manifest_digest: str
    cohort_episodes: tuple[tuple[int, int, str], ...]
    training_project_git_revision: str
    objective: str
    gripper_bce_weight: float
    normalization_artifact_digest: str
    normalization_file_sha256: str
    review_evidence_digest: str
    review_evidence_file_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_revision": self.checkpoint_revision,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "training_report_digest": self.training_report_digest,
            "training_report_file_sha256": self.training_report_file_sha256,
            "cohort_manifest_digest": self.cohort_manifest_digest,
            "cohort_episodes": [
                {
                    "episode_index": episode_index,
                    "task_index": task_index,
                    "instruction": instruction,
                }
                for episode_index, task_index, instruction in self.cohort_episodes
            ],
            "training_project_git_revision": self.training_project_git_revision,
            "objective": self.objective,
            "gripper_bce_weight": self.gripper_bce_weight,
            "normalization_artifact_digest": self.normalization_artifact_digest,
            "normalization_file_sha256": self.normalization_file_sha256,
            "review_evidence_digest": self.review_evidence_digest,
            "review_evidence_file_sha256": self.review_evidence_file_sha256,
        }

    def cohort_matches_for_task_name(self, task_name: str) -> tuple[dict[str, Any], ...]:
        normalized_task = f"_{_task_slug(task_name)}_"
        return tuple(
            {
                "episode_index": episode_index,
                "task_index": task_index,
                "instruction": instruction,
            }
            for episode_index, task_index, instruction in self.cohort_episodes
            if f"_{_task_slug(instruction)}_" in normalized_task
        )


def _task_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_sealed_json(path: Path, *, artifact_type: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B0InferenceError(f"cannot read sealed JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise B0InferenceError(f"sealed JSON artifact {path} is not an object")
    if value.get("artifact_type") != artifact_type:
        raise B0InferenceError(f"sealed JSON artifact {path} has the wrong type")
    if value.get("digest") != contract_digest(value):
        raise B0InferenceError(f"sealed JSON artifact {path} has an invalid digest")
    return value


def _require_sha256_sidecar(path: Path) -> str:
    actual = _sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        expected = sidecar.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise B0InferenceError(f"cannot read SHA-256 sidecar {sidecar}") from error
    if actual != expected:
        raise B0InferenceError(f"artifact {path} fails its SHA-256 sidecar")
    return actual


def verify_b0_checkpoint_bundle(
    checkpoint_root: str | Path,
    normalization_path: str | Path,
    review_evidence_path: str | Path,
) -> VerifiedB0Checkpoint:
    """Verify accepted review, training report, cohort, checkpoint, and normalizer."""

    root = Path(checkpoint_root).expanduser().resolve()
    normalization = Path(normalization_path).expanduser().resolve()
    review_path = Path(review_evidence_path).expanduser().resolve()
    report_path = root / "overfit-report.json"
    manifest_path = root / "cohort-manifest.json"
    checkpoint_path = root / "trainable-state.pt"

    review = _read_sealed_json(
        review_path,
        artifact_type=B0_ACCEPTED_REVIEW_ARTIFACT_TYPE,
    )
    if (
        review.get("status") != "accepted-with-documented-residuals"
        or review.get("reviewed_disposition", {}).get("accepted_with_documented_residuals")
        is not True
    ):
        raise B0InferenceError("B0 checkpoint has no accepted overfit review")
    report = _read_sealed_json(report_path, artifact_type=B0_TRAINING_REPORT_ARTIFACT_TYPE)
    manifest = _read_sealed_json(manifest_path, artifact_type=B0_COHORT_MANIFEST_ARTIFACT_TYPE)
    report_sha256 = _require_sha256_sidecar(report_path)
    manifest_sha256 = _require_sha256_sidecar(manifest_path)
    checkpoint_sha256 = _require_sha256_sidecar(checkpoint_path)
    normalization_sha256 = _sha256_file(normalization)
    review_sha256 = _sha256_file(review_path)

    review_training = review.get("artifacts", {}).get("training", {})
    review_report = review_training.get("overfit_report", {})
    review_manifest = review_training.get("cohort_manifest", {})
    review_checkpoint = review_training.get("checkpoint", {})
    if (
        review_report.get("artifact_digest") != report.get("digest")
        or review_report.get("file_sha256") != report_sha256
        or review_manifest.get("artifact_digest") != manifest.get("digest")
        or review_manifest.get("file_sha256") != manifest_sha256
        or review_checkpoint.get("file_sha256") != checkpoint_sha256
    ):
        raise B0InferenceError("accepted review does not identify the supplied checkpoint bundle")

    configuration = report.get("configuration", {})
    objective = configuration.get("objective")
    gripper_bce_weight = configuration.get("gripper_bce_weight")
    if objective != B0_ROLLOUT_OBJECTIVE or gripper_bce_weight != B0_ROLLOUT_GRIPPER_BCE_WEIGHT:
        raise B0InferenceError("B0 rollout requires the accepted unit-weight BCE objective")
    if report.get("cohort", {}).get("manifest_digest") != manifest.get("digest"):
        raise B0InferenceError("training report does not reference the supplied cohort manifest")
    if report.get("cohort", {}).get("manifest_file_sha256") != manifest_sha256:
        raise B0InferenceError("training report records a different cohort manifest file")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 10:
        raise B0InferenceError("B0 cohort manifest must identify exactly ten demonstrations")
    cohort_episodes: list[tuple[int, int, str]] = []
    for position, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise B0InferenceError(f"B0 cohort episode {position} is not an object")
        episode_index = episode.get("episode_index")
        task_index = episode.get("task_index")
        instruction = episode.get("instruction")
        if (
            type(episode_index) is not int
            or episode_index < 0
            or type(task_index) is not int
            or task_index < 0
            or not isinstance(instruction, str)
            or not instruction
        ):
            raise B0InferenceError(f"B0 cohort episode {position} has invalid task identity")
        cohort_episodes.append((episode_index, task_index, instruction))
    if len({record[0] for record in cohort_episodes}) != len(cohort_episodes):
        raise B0InferenceError("B0 cohort manifest repeats an episode identity")
    if report.get("checkpoint", {}).get("file_sha256") != checkpoint_sha256:
        raise B0InferenceError("training report records a different checkpoint file")
    if report.get("checkpoint", {}).get("format") != B0_COHORT_CHECKPOINT_FORMAT:
        raise B0InferenceError("training report has an unsupported checkpoint format")
    if report.get("checkpoint", {}).get("reload_max_absolute_error", float("inf")) > 1e-3:
        raise B0InferenceError("training report did not pass checkpoint reload validation")

    source = report.get("source", {})
    normalizer = load_b0_action_normalizer(normalization)
    if (
        source.get("normalization_file_sha256") != normalization_sha256
        or source.get("normalization_artifact_digest") != normalizer.artifact_digest
    ):
        raise B0InferenceError("normalization artifact differs from the accepted training run")
    project_revision = report.get("project", {}).get("git_revision")
    if not isinstance(project_revision, str) or len(project_revision) != 40:
        raise B0InferenceError("training report has an invalid project revision")

    return VerifiedB0Checkpoint(
        checkpoint_root=root,
        checkpoint_path=checkpoint_path,
        checkpoint_file_sha256=checkpoint_sha256,
        checkpoint_id=B0_ROLLOUT_CHECKPOINT_ID,
        checkpoint_revision=checkpoint_sha256,
        training_report_digest=str(report["digest"]),
        training_report_file_sha256=report_sha256,
        cohort_manifest_digest=str(manifest["digest"]),
        cohort_episodes=tuple(cohort_episodes),
        training_project_git_revision=project_revision,
        objective=str(objective),
        gripper_bce_weight=float(gripper_bce_weight),
        normalization_artifact_digest=str(normalizer.artifact_digest),
        normalization_file_sha256=normalization_sha256,
        review_evidence_digest=str(review["digest"]),
        review_evidence_file_sha256=review_sha256,
    )


def decode_b0_action_chunk(
    normalized_actions: Any,
    normalizer: ActionNormalizer,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Invert normalization and enforce the LIBERO execution-time action bounds."""

    normalized = np.asarray(normalized_actions, dtype=np.float64)
    if normalized.shape != (8, 7):
        raise B0InferenceError(f"B0 must predict one 8x7 action chunk, found {normalized.shape}")
    if not np.isfinite(normalized).all():
        raise B0InferenceError("B0 normalized action chunk contains non-finite values")
    if not np.isin(normalized[:, 6], (-1.0, 1.0)).all():
        raise B0InferenceError("B0 gripper predictions must be decoded to exact -1 or +1")
    denormalized = normalizer.denormalize(normalized)
    unclipped_continuous = denormalized[:, :6].copy()
    denormalized[:, :6] = np.clip(
        denormalized[:, :6],
        B0_SIMULATOR_ACTION_MINIMUM,
        B0_SIMULATOR_ACTION_MAXIMUM,
    )
    clipped_delta = np.abs(denormalized[:, :6] - unclipped_continuous)
    if not np.isfinite(denormalized).all():
        raise B0InferenceError("B0 executable action chunk contains non-finite values")
    if np.any(denormalized < -1.0) or np.any(denormalized > 1.0):
        raise B0InferenceError("B0 executable action chunk is outside [-1,1]")
    if not np.array_equal(denormalized[:, 6], normalized[:, 6]):
        raise B0InferenceError("B0 action decoding changed gripper polarity")
    return np.ascontiguousarray(denormalized), {
        "continuous_value_count": int(unclipped_continuous.size),
        "clipped_continuous_value_count": int(np.count_nonzero(clipped_delta)),
        "maximum_continuous_clipping_delta": float(clipped_delta.max(initial=0.0)),
        "gripper_values": sorted(float(value) for value in np.unique(denormalized[:, 6])),
    }


InputBuilder = Callable[[list[str], list[Any], list[Any]], Mapping[str, Tensor]]


class B0InferencePolicy:
    """Evaluator-compatible B0 policy with verified checkpoint provenance."""

    def __init__(
        self,
        *,
        policy: B0ActionPolicy,
        input_builder: InputBuilder,
        normalizer: ActionNormalizer,
        device: torch.device,
        checkpoint: VerifiedB0Checkpoint,
    ) -> None:
        self._policy = policy
        self._input_builder = input_builder
        self._normalizer = normalizer
        self._device = device
        self.checkpoint = checkpoint
        self._prediction_count = 0
        self._clipped_continuous_value_count = 0
        self._maximum_continuous_clipping_delta = 0.0

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_root: str | Path,
        normalization_path: str | Path,
        review_evidence_path: str | Path,
        cache_dir: str | Path | None,
        local_files_only: bool,
        device: str = "cuda:0",
    ) -> B0InferencePolicy:
        """Build Qwen+LoRA and strictly restore the accepted trainable state."""

        torch_device = torch.device(device)
        if torch_device.type != "cuda" or not torch.cuda.is_available():
            raise B0InferenceError("real B0 rollout inference requires an available CUDA device")
        verified = verify_b0_checkpoint_bundle(
            checkpoint_root,
            normalization_path,
            review_evidence_path,
        )
        resolved_cache = (
            str(Path(cache_dir).expanduser().resolve()) if cache_dir is not None else None
        )
        processor = load_pinned_qwen_processor(
            cache_dir=resolved_cache,
            local_files_only=local_files_only,
        )
        full_model = load_pinned_qwen_model(
            cache_dir=resolved_cache,
            local_files_only=local_files_only,
        )
        core_lora_names = install_qwen_upper_lora(full_model)
        policy = B0ActionPolicy(
            QwenConditioningAdapter(
                full_model.model,
                trainable_backbone_parameter_names=core_lora_names,
            ),
            ActionQueryTransformer(),
            objective_config=B0ObjectiveConfig(
                objective=verified.objective,
                gripper_bce_weight=verified.gripper_bce_weight,
            ),
        ).to(device=torch_device)
        exact_lora_names = tuple(
            f"conditioning_adapter.backbone.{name}" for name in core_lora_names
        )
        checkpoint = torch.load(
            verified.checkpoint_path,
            map_location=torch_device,
            weights_only=True,
        )
        if (
            checkpoint.get("format") != B0_COHORT_CHECKPOINT_FORMAT
            or checkpoint.get("qwen_revision") != QWEN_REVISION
            or checkpoint.get("project_git_revision")
            != verified.training_project_git_revision
            or checkpoint.get("cohort_manifest_digest") != verified.cohort_manifest_digest
        ):
            raise B0InferenceError("B0 checkpoint tensor envelope has invalid provenance")
        try:
            load_b0_trainable_state_dict(
                policy,
                checkpoint["state_dict"],
                allowed_exact_names=exact_lora_names,
            )
        except (B0ModelError, KeyError) as error:
            raise B0InferenceError("B0 trainable state cannot be restored strictly") from error
        policy.eval()
        return cls(
            policy=policy,
            input_builder=QwenB0InputBuilder(processor),
            normalizer=load_b0_action_normalizer(normalization_path),
            device=torch_device,
            checkpoint=verified,
        )

    def predict_chunk(
        self,
        *,
        agent_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        proprioception: np.ndarray,
        instruction: str,
        suite: str,
    ) -> NDArray[np.float64]:
        """Predict one executable open-loop chunk from a recorder observation."""

        for name, image in (("agent_rgb", agent_rgb), ("wrist_rgb", wrist_rgb)):
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise B0InferenceError(f"{name} must be uint8 HWC 256x256")
        state = np.asarray(proprioception, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise B0InferenceError("proprioception must be a finite 8-vector")
        if not isinstance(instruction, str) or not instruction:
            raise B0InferenceError("instruction must be a nonempty string")
        if not isinstance(suite, str) or not suite:
            raise B0InferenceError("suite must be a nonempty string")
        encoding = self._input_builder([instruction], [agent_rgb], [wrist_rgb])
        inputs = {
            name: value.to(
                device=self._device,
                dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
            )
            for name, value in encoding.items()
        }
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device=self._device)
        self._policy.eval()
        with torch.inference_mode():
            output = self._policy(inputs, state_tensor)
        normalized = output.normalized_actions.detach().float().cpu().numpy()
        if normalized.shape != (1, 8, 7):
            raise B0InferenceError(
                f"B0 policy output has shape {normalized.shape}, expected (1,8,7)"
            )
        actions, decode_summary = decode_b0_action_chunk(normalized[0], self._normalizer)
        self._prediction_count += 1
        self._clipped_continuous_value_count += decode_summary[
            "clipped_continuous_value_count"
        ]
        self._maximum_continuous_clipping_delta = max(
            self._maximum_continuous_clipping_delta,
            decode_summary["maximum_continuous_clipping_delta"],
        )
        return actions

    def memory_summary(self) -> dict[str, Any]:
        return {
            "device": torch.cuda.get_device_name(self._device),
            "capability": list(torch.cuda.get_device_capability(self._device)),
            "allocated_bytes": torch.cuda.memory_allocated(self._device),
            "reserved_bytes": torch.cuda.memory_reserved(self._device),
        }

    def inference_summary(self) -> dict[str, Any]:
        return {
            "prediction_count": self._prediction_count,
            "decoded_action_count": self._prediction_count * 8,
            "continuous_value_count": self._prediction_count * 8 * 6,
            "clipped_continuous_value_count": self._clipped_continuous_value_count,
            "maximum_continuous_clipping_delta": self._maximum_continuous_clipping_delta,
            "gripper_decoding": "nonnegative-logit-to-+1-negative-logit-to--1",
            "continuous_decoding": "invert-train-q01-q99-affine-then-clip-to-simulator-range",
        }

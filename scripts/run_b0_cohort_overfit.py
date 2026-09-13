#!/usr/bin/env python3
"""Overfit the fixed ten-demonstration B0 cohort and seal acceptance evidence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from eventground_wam.b0_data import (
    PINNED_TRAIN_SAMPLE_COUNT,
    B0Batch,
    B0DataConfig,
    B0StreamingDataset,
    collate_b0_samples,
    load_b0_action_normalizer,
    load_frozen_b0_split_index,
)
from eventground_wam.b0_model import (
    QWEN_REVISION,
    ActionQueryTransformer,
    B0ActionPolicy,
    B0ModelError,
    QwenB0InputBuilder,
    QwenConditioningAdapter,
    b0_trainable_state_dict,
    install_qwen_upper_lora,
    load_b0_trainable_state_dict,
    load_pinned_qwen_model,
    load_pinned_qwen_processor,
)
from eventground_wam.b0_overfit import (
    PINNED_TEN_DEMONSTRATION_EPISODE_INDICES,
    PINNED_TEN_DEMONSTRATION_SAMPLE_COUNT,
    CohortDiagnostics,
    CohortTensorBatch,
    TenDemonstrationOverfitConfig,
    build_ten_demonstration_manifest,
    evaluate_ten_demonstration_cohort,
    run_ten_demonstration_optimization,
)
from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN_MANIFEST_DIGEST = "a6d4cddc5ab0dfb45fb34b3d91160fad15ea8aba8e18989eee29601a32859835"
CHECKPOINT_FORMAT = "eventground-wam-b0-ten-demonstration-trainable-state-v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--minibatch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--evaluation-interval-epochs", type=int, default=1)
    parser.add_argument("--decoder-projector-learning-rate", type=float, default=3e-4)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--maximum-gradient-norm", type=float, default=1.0)
    return parser


def _git_identity() -> tuple[str, bool]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, not bool(status.strip())


def _config(args: argparse.Namespace) -> TenDemonstrationOverfitConfig:
    return TenDemonstrationOverfitConfig(
        seed=args.seed,
        minibatch_size=args.minibatch_size,
        max_epochs=args.max_epochs,
        evaluation_interval_epochs=args.evaluation_interval_epochs,
        decoder_projector_learning_rate=args.decoder_projector_learning_rate,
        lora_learning_rate=args.lora_learning_rate,
        maximum_gradient_norm=args.maximum_gradient_norm,
    )


def _seed_everything(seed: int) -> None:
    if seed > np.iinfo(np.uint32).max:
        raise B0ModelError("seed must fit in an unsigned 32-bit integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_fixed_cohort(
    args: argparse.Namespace,
    config: TenDemonstrationOverfitConfig,
) -> tuple[B0Batch, list[dict[str, Any]], Any, Any, Any]:
    reader = DemonstrationReader(args.repository_root)
    split_index = load_frozen_b0_split_index(args.frozen_root)
    normalizer = load_b0_action_normalizer(args.normalization)
    dataset = B0StreamingDataset(
        reader,
        split_index,
        normalizer,
        config=B0DataConfig(seed=config.seed, shuffle_episodes=False),
    )
    if dataset.sample_count != PINNED_TRAIN_SAMPLE_COUNT:
        raise B0ModelError("ten-demonstration loader sample count differs from the frozen source")
    episodes = dataset.episodes[: config.demonstration_count]
    episode_indices = tuple(episode.episode_index for episode in episodes)
    if episode_indices != PINNED_TEN_DEMONSTRATION_EPISODE_INDICES:
        raise B0ModelError(
            f"canonical cohort episodes are {episode_indices}, "
            f"expected {PINNED_TEN_DEMONSTRATION_EPISODE_INDICES}"
        )
    selected = set(episode_indices)
    samples = []
    iterator = dataset.iter_samples(epoch=0)
    try:
        for sample in iterator:
            if sample.identity.episode_index not in selected:
                break
            samples.append(sample)
    finally:
        iterator.close()
    if len(samples) != PINNED_TEN_DEMONSTRATION_SAMPLE_COUNT:
        raise B0ModelError(
            f"canonical ten-demonstration cohort has {len(samples)} samples, "
            f"expected {PINNED_TEN_DEMONSTRATION_SAMPLE_COUNT}"
        )
    cohort = collate_b0_samples(samples)
    episode_summaries = []
    for episode in episodes:
        episode_samples = [
            sample for sample in samples if sample.identity.episode_index == episode.episode_index
        ]
        expected_count = episode.length - 8 + 1
        if len(episode_samples) != expected_count:
            raise B0ModelError(
                f"cohort episode {episode.episode_index} has {len(episode_samples)} samples, "
                f"expected {expected_count}"
            )
        episode_summaries.append(
            {
                "episode_index": episode.episode_index,
                "task_index": episode.task_index,
                "instruction": episode.instruction,
                "episode_length": episode.length,
                "sample_count": len(episode_samples),
                "split_assignment_digest": (
                    episode_samples[0].identity.split_assignment_digest
                ),
                "first_sample_identity_digest": episode_samples[0].identity.digest,
                "last_sample_identity_digest": episode_samples[-1].identity.digest,
            }
        )
    if len({summary["task_index"] for summary in episode_summaries}) != 10:
        raise B0ModelError("the pinned ten-demonstration cohort must span ten tasks")
    return cohort, episode_summaries, split_index, normalizer, dataset


def _cache_processor_batches(
    cohort: B0Batch,
    processor: Any,
    config: TenDemonstrationOverfitConfig,
) -> tuple[tuple[CohortTensorBatch, ...], dict[str, Any]]:
    builder = QwenB0InputBuilder(processor)
    batches = []
    sequence_lengths = []
    image_grid_rows = 0
    cached_bytes = 0
    sample_count = len(cohort.identities)
    for batch_number, start in enumerate(range(0, sample_count, config.minibatch_size), start=1):
        end = min(start + config.minibatch_size, sample_count)
        encoding = builder(
            cohort.instructions[start:end],
            list(cohort.front_rgb[start:end]),
            list(cohort.wrist_rgb[start:end]),
        )
        cached_inputs = {
            name: value.to(dtype=torch.bfloat16).contiguous()
            if value.is_floating_point()
            else value.contiguous()
            for name, value in encoding.items()
        }
        proprioception = torch.from_numpy(cohort.proprioception[start:end]).clone()
        targets = torch.from_numpy(cohort.actions[start:end]).clone()
        identities = tuple(identity.digest for identity in cohort.identities[start:end])
        batch = CohortTensorBatch(
            backbone_inputs=cached_inputs,
            proprioception=proprioception,
            targets=targets,
            sample_identity_digests=identities,
        )
        batches.append(batch)
        sequence_lengths.extend(encoding["attention_mask"].sum(dim=1).tolist())
        image_grid_rows += int(encoding["image_grid_thw"].shape[0])
        cached_bytes += sum(value.numel() * value.element_size() for value in cached_inputs.values())
        cached_bytes += proprioception.numel() * proprioception.element_size()
        cached_bytes += targets.numel() * targets.element_size()
        if batch_number % 50 == 0:
            print(
                f"processor-cache batches={batch_number} samples={end}/{sample_count}",
                flush=True,
            )
    if sum(batch.sample_count for batch in batches) != sample_count:
        raise B0ModelError("processor cache omitted cohort samples")
    return tuple(batches), {
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "image_backend": processor.image_processor.backend,
        "padding_side": processor.tokenizer.padding_side,
        "batch_count": len(batches),
        "sample_count": sample_count,
        "minimum_sequence_length": min(sequence_lengths),
        "maximum_sequence_length": max(sequence_lengths),
        "image_grid_row_count": image_grid_rows,
        "cached_tensor_bytes": cached_bytes,
        "floating_dtype": "torch.bfloat16",
    }


def _build_policy(
    *,
    cache_dir: str | None,
    local_files_only: bool,
    device: torch.device,
) -> tuple[B0ActionPolicy, tuple[str, ...]]:
    full_model = load_pinned_qwen_model(
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    core_lora_names = install_qwen_upper_lora(full_model)
    policy = B0ActionPolicy(
        QwenConditioningAdapter(
            full_model.model,
            trainable_backbone_parameter_names=core_lora_names,
        ),
        ActionQueryTransformer(),
    ).to(device=device)
    exact_lora_names = tuple(
        f"conditioning_adapter.backbone.{name}" for name in core_lora_names
    )
    return policy, exact_lora_names


def _predict_cohort(
    policy: B0ActionPolicy,
    batches: Sequence[CohortTensorBatch],
    device: torch.device,
) -> Tensor:
    predictions = []
    policy.eval()
    with torch.no_grad():
        for batch in batches:
            inputs = {
                name: value.to(device=device) for name, value in batch.backbone_inputs.items()
            }
            predictions.append(
                policy(
                    inputs,
                    batch.proprioception.to(device=device),
                ).normalized_actions.cpu()
            )
    return torch.cat(predictions, dim=0)


def _save_and_verify_checkpoint(
    output: Path,
    policy: B0ActionPolicy,
    batches: Sequence[CohortTensorBatch],
    device: torch.device,
    *,
    exact_lora_names: Sequence[str],
    project_git_revision: str,
    cohort_manifest_digest: str,
) -> dict[str, Any]:
    policy.zero_grad(set_to_none=True)
    expected = _predict_cohort(policy, batches, device)
    state_dict = {
        name: value.cpu()
        for name, value in b0_trainable_state_dict(
            policy,
            allowed_exact_names=exact_lora_names,
        ).items()
    }
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "project_git_revision": project_git_revision,
        "qwen_revision": QWEN_REVISION,
        "cohort_manifest_digest": cohort_manifest_digest,
        "state_dict": state_dict,
    }
    with output.open("xb") as handle:
        torch.save(checkpoint, handle)
    file_sha256 = _write_sha256_sidecar(output)
    parameters = dict(policy.named_parameters())
    with torch.no_grad():
        for name in state_dict:
            parameters[name].zero_()
    perturbed = _predict_cohort(policy, batches, device)
    perturbed_error = (perturbed - expected).abs().max().item()
    if perturbed_error <= 1e-4:
        raise B0ModelError("zeroing the cohort checkpoint did not change predictions")
    restored = torch.load(output, map_location=device, weights_only=True)
    if (
        restored.get("format") != CHECKPOINT_FORMAT
        or restored.get("project_git_revision") != project_git_revision
        or restored.get("qwen_revision") != QWEN_REVISION
        or restored.get("cohort_manifest_digest") != cohort_manifest_digest
    ):
        raise B0ModelError("saved cohort checkpoint identity is invalid")
    load_b0_trainable_state_dict(
        policy,
        restored["state_dict"],
        allowed_exact_names=exact_lora_names,
    )
    reloaded = _predict_cohort(policy, batches, device)
    reload_error = (reloaded - expected).abs().max().item()
    if reload_error > 1e-3:
        raise B0ModelError(f"cohort checkpoint reload error is {reload_error}, expected <= 1e-3")
    return {
        "format": CHECKPOINT_FORMAT,
        "relative_path": output.name,
        "file_sha256": file_sha256,
        "tensor_count": len(state_dict),
        "perturbed_max_absolute_error": perturbed_error,
        "reload_max_absolute_error": reload_error,
    }


def _print_diagnostic(metrics: CohortDiagnostics) -> None:
    print(
        "cohort-diagnostic "
        f"epoch={metrics.epoch} step={metrics.step} "
        f"l1={metrics.l1_loss:.8f} ratio={metrics.loss_ratio_to_initial:.8f} "
        f"mae={metrics.normalized_action_mae:.8f} "
        f"gripper_sign={metrics.gripper_sign_accuracy:.8f} "
        f"passed={str(metrics.passed).lower()}",
        flush=True,
    )


def _write_json_with_sidecar(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    return _write_sha256_sidecar(path)


def _write_curve_with_sidecar(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    with path.open("xb") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record) + b"\n")
    return _write_sha256_sidecar(path)


def _write_sha256_sidecar(path: Path) -> str:
    digest = _sha256_file(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(digest + "\n")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": ".".join(str(value) for value in __import__("sys").version_info[:3]),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "numpy": np.__version__,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    revision, clean = _git_identity()
    if not clean:
        raise B0ModelError("ten-demonstration evidence requires a clean project tree")
    config = _config(args)
    _seed_everything(config.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise B0ModelError("the real ten-demonstration gate requires an available CUDA device")
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise B0ModelError("ten-demonstration evidence must be outside the Git checkout")
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse cohort output directory {output_root}")

    print("loading and decoding fixed ten-demonstration cohort", flush=True)
    cohort, episode_summaries, split_index, normalizer, dataset = _load_fixed_cohort(args, config)
    if normalizer.artifact_digest is None:
        raise B0ModelError("real cohort training requires a sealed normalization artifact")
    normalization_sha256 = _sha256_file(args.normalization)
    cohort_manifest = build_ten_demonstration_manifest(
        cohort,
        episode_summaries,
        config=config,
        project_git_revision=revision,
        source_identity=split_index.source_identity,
        normalization_artifact_digest=normalizer.artifact_digest,
        normalization_file_sha256=normalization_sha256,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "cohort-manifest.json"
    manifest_sha256 = _write_json_with_sidecar(manifest_path, cohort_manifest)
    cache_dir = str(args.cache_dir.expanduser().resolve()) if args.cache_dir is not None else None
    processor = load_pinned_qwen_processor(
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    print("building deterministic CPU processor cache", flush=True)
    batches, processor_summary = _cache_processor_batches(cohort, processor, config)
    del cohort
    gc.collect()

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    policy, exact_lora_names = _build_policy(
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        device=device,
    )
    result = run_ten_demonstration_optimization(
        policy,
        batches,
        device=device,
        exact_lora_parameter_names=exact_lora_names,
        config=config,
        progress_callback=_print_diagnostic,
    )
    curve_path = output_root / "loss-curve.jsonl"
    curve_sha256 = _write_curve_with_sidecar(curve_path, result.curve)
    checkpoint_summary = _save_and_verify_checkpoint(
        output_root / "trainable-state.pt",
        policy,
        batches,
        device,
        exact_lora_names=exact_lora_names,
        project_git_revision=revision,
        cohort_manifest_digest=str(cohort_manifest["digest"]),
    )
    post_reload = evaluate_ten_demonstration_cohort(
        policy,
        batches,
        device=device,
        epoch=result.epochs_completed,
        step=result.steps_completed,
        initial_loss=result.initial.l1_loss,
        config=config,
    )
    if post_reload.passed != result.passed:
        raise B0ModelError("checkpoint reload changed the cohort acceptance outcome")

    report = seal_contract(
        {
            "artifact_type": "eventground_wam.b0_ten_demonstration_overfit",
            "schema_version": 1,
            "status": "passed" if result.passed else "failed",
            "project": {
                "branch": "codex/week3-b0-model",
                "git_revision": revision,
                "tree_clean": clean,
            },
            "source": {
                **split_index.source_identity,
                "normalization_artifact_digest": normalizer.artifact_digest,
                "normalization_file_sha256": normalization_sha256,
                "training_sample_count": dataset.sample_count,
            },
            "model": {
                "repository_id": "Qwen/Qwen3.5-2B",
                "revision": QWEN_REVISION,
                "manifest_digest": QWEN_MANIFEST_DIGEST,
                "trainable_parameter_count": result.trainable_parameter_count,
                "trainable_parameter_names": list(result.trainable_parameter_names),
                "exact_lora_parameter_names": list(exact_lora_names),
            },
            "configuration": config.to_dict(),
            "cohort": {
                "manifest_digest": cohort_manifest["digest"],
                "manifest_file_sha256": manifest_sha256,
                "episode_indices": list(PINNED_TEN_DEMONSTRATION_EPISODE_INDICES),
                "task_indices": [summary["task_index"] for summary in episode_summaries],
                "sample_count": PINNED_TEN_DEMONSTRATION_SAMPLE_COUNT,
                "processor_cache": processor_summary,
            },
            "optimization": {
                "epochs_completed": result.epochs_completed,
                "steps_completed": result.steps_completed,
                "training_seconds": result.training_seconds,
                "initial_diagnostic": result.initial.to_dict(),
                "final_diagnostic": result.final.to_dict(),
                "post_reload_diagnostic": post_reload.to_dict(),
                "last_gradient_summary": dict(result.last_gradient_summary),
            },
            "acceptance": {
                "loss_ratio_at_most": config.loss_ratio_threshold,
                "normalized_action_mae_at_most": config.action_mae_threshold,
                "gripper_sign_accuracy_at_least": config.gripper_sign_accuracy_threshold,
                "loss_ratio_passed": (
                    result.final.loss_ratio_to_initial <= config.loss_ratio_threshold
                ),
                "normalized_action_mae_passed": (
                    result.final.normalized_action_mae <= config.action_mae_threshold
                ),
                "gripper_sign_accuracy_passed": (
                    result.final.gripper_sign_accuracy
                    >= config.gripper_sign_accuracy_threshold
                ),
                "passed": result.passed,
            },
            "checkpoint": checkpoint_summary,
            "artifacts": {
                "cohort_manifest": {
                    "relative_path": manifest_path.name,
                    "file_sha256": manifest_sha256,
                },
                "loss_curve": {
                    "relative_path": curve_path.name,
                    "file_sha256": curve_sha256,
                    "record_count": len(result.curve),
                },
                "checkpoint": checkpoint_summary,
            },
            "runtime": {
                **_runtime_versions(),
                "device_name": torch.cuda.get_device_name(device),
                "device_capability": list(torch.cuda.get_device_capability(device)),
                "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(device),
            },
        }
    )
    report_path = output_root / "overfit-report.json"
    _write_json_with_sidecar(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = _run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

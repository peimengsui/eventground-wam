#!/usr/bin/env python3
"""Overfit one predeclared real B0 batch and seal the locked acceptance evidence."""

from __future__ import annotations

import argparse
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
    B0DataConfig,
    B0DataLoader,
    B0StreamingDataset,
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
    FixedBatchDiagnostics,
    OneBatchOverfitConfig,
    build_fixed_batch_manifest,
    evaluate_fixed_batch,
    run_one_batch_optimization,
)
from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract
from eventground_wam.demonstration_reader import DemonstrationReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN_MANIFEST_DIGEST = "a6d4cddc5ab0dfb45fb34b3d91160fad15ea8aba8e18989eee29601a32859835"
CHECKPOINT_FORMAT = "eventground-wam-b0-one-batch-trainable-state-v1"


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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--evaluation-interval", type=int, default=20)
    parser.add_argument("--decoder-projector-learning-rate", type=float, default=1e-3)
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


def _config(args: argparse.Namespace) -> OneBatchOverfitConfig:
    return OneBatchOverfitConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        evaluation_interval=args.evaluation_interval,
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


def _fixed_batch(
    args: argparse.Namespace,
    config: OneBatchOverfitConfig,
) -> tuple[Any, Any, Any, Any]:
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
        raise B0ModelError(
            f"fixed-batch loader exposes {dataset.sample_count} samples, "
            f"expected {PINNED_TRAIN_SAMPLE_COUNT}"
        )
    iterator = B0DataLoader(dataset, batch_size=config.batch_size, drop_last=True).iter_batches()
    try:
        batch = next(iterator)
    finally:
        iterator.close()
    return batch, split_index, normalizer, dataset


def _device_batch(
    batch: Any,
    processor: Any,
    device: torch.device,
) -> tuple[dict[str, Tensor], Tensor, Tensor, dict[str, Any]]:
    encoding = QwenB0InputBuilder(processor)(
        batch.instructions,
        list(batch.front_rgb),
        list(batch.wrist_rgb),
    )
    inputs = {
        name: value.to(
            device=device,
            dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
        )
        for name, value in encoding.items()
    }
    proprioception = torch.from_numpy(batch.proprioception).to(device=device)
    targets = torch.from_numpy(batch.actions).to(device=device)
    summary = {
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "image_backend": processor.image_processor.backend,
        "tensor_shapes": {name: list(value.shape) for name, value in encoding.items()},
        "sequence_lengths": encoding["attention_mask"].sum(dim=1).tolist(),
        "image_grid_thw": encoding["image_grid_thw"].tolist(),
        "padding_side": processor.tokenizer.padding_side,
    }
    return inputs, proprioception, targets, summary


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


def _predict(
    policy: B0ActionPolicy,
    inputs: Mapping[str, Tensor],
    proprioception: Tensor,
) -> Tensor:
    policy.eval()
    with torch.no_grad():
        return policy(inputs, proprioception).normalized_actions.detach().clone()


def _save_and_verify_checkpoint(
    output: Path,
    policy: B0ActionPolicy,
    inputs: Mapping[str, Tensor],
    proprioception: Tensor,
    *,
    exact_lora_names: Sequence[str],
    project_git_revision: str,
    fixed_batch_digest: str,
) -> dict[str, Any]:
    policy.zero_grad(set_to_none=True)
    expected = _predict(policy, inputs, proprioception)
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
        "fixed_batch_digest": fixed_batch_digest,
        "state_dict": state_dict,
    }
    with output.open("xb") as handle:
        torch.save(checkpoint, handle)
    file_sha256 = _write_sha256_sidecar(output)

    parameters = dict(policy.named_parameters())
    with torch.no_grad():
        for name in state_dict:
            parameters[name].zero_()
    perturbed = _predict(policy, inputs, proprioception)
    perturbed_error = float((perturbed - expected).abs().max())
    if perturbed_error <= 1e-4:
        raise B0ModelError("zeroing the trained checkpoint tensors did not change predictions")

    restored = torch.load(output, map_location=proprioception.device, weights_only=True)
    if (
        restored.get("format") != CHECKPOINT_FORMAT
        or restored.get("project_git_revision") != project_git_revision
        or restored.get("qwen_revision") != QWEN_REVISION
        or restored.get("fixed_batch_digest") != fixed_batch_digest
    ):
        raise B0ModelError("saved one-batch checkpoint identity is invalid")
    load_b0_trainable_state_dict(
        policy,
        restored["state_dict"],
        allowed_exact_names=exact_lora_names,
    )
    reloaded = _predict(policy, inputs, proprioception)
    reload_error = float((reloaded - expected).abs().max())
    if reload_error > 1e-3:
        raise B0ModelError(f"trained checkpoint reload error is {reload_error}, expected <= 1e-3")
    return {
        "format": CHECKPOINT_FORMAT,
        "relative_path": output.name,
        "file_sha256": file_sha256,
        "tensor_count": len(state_dict),
        "perturbed_max_absolute_error": perturbed_error,
        "reload_max_absolute_error": reload_error,
    }


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


def _print_diagnostic(metrics: FixedBatchDiagnostics) -> None:
    print(
        "diagnostic "
        f"step={metrics.step} "
        f"l1={metrics.l1_loss:.8f} "
        f"ratio={metrics.loss_ratio_to_initial:.8f} "
        f"mae={metrics.normalized_action_mae:.8f} "
        f"passed={str(metrics.passed).lower()}",
        flush=True,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    revision, clean = _git_identity()
    if not clean:
        raise B0ModelError("one-batch overfit evidence requires a clean project tree")
    config = _config(args)
    _seed_everything(config.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise B0ModelError("the real one-batch overfit gate requires an available CUDA device")
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    else:
        raise B0ModelError("one-batch evidence must be written outside the Git checkout")
    output_root.mkdir(parents=True, exist_ok=False)

    batch, split_index, normalizer, dataset = _fixed_batch(args, config)
    if normalizer.artifact_digest is None:
        raise B0ModelError("real one-batch training requires a sealed normalization artifact")
    normalization_sha256 = _sha256_file(args.normalization)
    batch_manifest = build_fixed_batch_manifest(
        batch,
        config=config,
        project_git_revision=revision,
        source_identity=split_index.source_identity,
        normalization_artifact_digest=normalizer.artifact_digest,
        normalization_file_sha256=normalization_sha256,
    )
    batch_manifest_path = output_root / "fixed-batch-manifest.json"
    batch_manifest_sha256 = _write_json_with_sidecar(batch_manifest_path, batch_manifest)

    cache_dir = str(args.cache_dir.expanduser().resolve()) if args.cache_dir is not None else None
    processor = load_pinned_qwen_processor(
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    inputs, proprioception, targets, processor_summary = _device_batch(batch, processor, device)
    policy, exact_lora_names = _build_policy(
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        device=device,
    )
    result = run_one_batch_optimization(
        policy,
        inputs,
        proprioception,
        targets,
        exact_lora_parameter_names=exact_lora_names,
        config=config,
        progress_callback=_print_diagnostic,
    )

    curve_path = output_root / "loss-curve.jsonl"
    curve_sha256 = _write_curve_with_sidecar(curve_path, result.curve)
    checkpoint_summary = _save_and_verify_checkpoint(
        output_root / "trainable-state.pt",
        policy,
        inputs,
        proprioception,
        exact_lora_names=exact_lora_names,
        project_git_revision=revision,
        fixed_batch_digest=str(batch_manifest["digest"]),
    )
    post_reload = evaluate_fixed_batch(
        policy,
        inputs,
        proprioception,
        targets,
        step=result.steps_completed,
        initial_loss=result.initial.l1_loss,
        config=config,
    )
    if post_reload.passed != result.passed:
        raise B0ModelError("checkpoint reload changed the one-batch acceptance outcome")

    report = seal_contract(
        {
            "artifact_type": "eventground_wam.b0_one_batch_overfit",
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
            "fixed_batch": {
                "manifest_digest": batch_manifest["digest"],
                "manifest_file_sha256": batch_manifest_sha256,
                "sample_count": len(batch.identities),
                "sample_identity_digests": [identity.digest for identity in batch.identities],
                "processor": processor_summary,
            },
            "optimization": {
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
                "loss_ratio_passed": (
                    result.final.loss_ratio_to_initial <= config.loss_ratio_threshold
                ),
                "normalized_action_mae_passed": (
                    result.final.normalized_action_mae <= config.action_mae_threshold
                ),
                "passed": result.passed,
            },
            "checkpoint": checkpoint_summary,
            "artifacts": {
                "fixed_batch_manifest": {
                    "relative_path": batch_manifest_path.name,
                    "file_sha256": batch_manifest_sha256,
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

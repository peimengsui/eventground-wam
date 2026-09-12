#!/usr/bin/env python3
"""Run pinned Qwen processor or BF16 B0 model compatibility smoke checks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from eventground_wam.b0_model import (
    QWEN_IMAGE_TOKEN_ID,
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
    validate_b0_trainable_parameters,
)
from eventground_wam.demonstration_contracts import contract_digest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("processor", "model"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
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


def _synthetic_inputs(processor: Any) -> tuple[dict[str, Tensor], dict[str, Any]]:
    front = np.zeros((3, 256, 256), dtype=np.uint8)
    wrist = np.zeros((3, 256, 256), dtype=np.uint8)
    front[0] = 31
    wrist[1] = 47
    encoding = QwenB0InputBuilder(processor)(
        ["pick up the red block"],
        [front],
        [wrist],
    )
    summary = {
        "processor_class": type(processor).__name__,
        "image_processor_class": type(processor.image_processor).__name__,
        "image_backend": processor.image_processor.backend,
        "tensor_shapes": {name: list(value.shape) for name, value in encoding.items()},
        "image_grid_thw": encoding["image_grid_thw"].tolist(),
        "image_token_count": int((encoding["input_ids"] == QWEN_IMAGE_TOKEN_ID).sum()),
        "sequence_length": int(encoding["input_ids"].shape[1]),
        "padding_side": processor.tokenizer.padding_side,
    }
    return encoding, summary


def _processor_report(args: argparse.Namespace) -> dict[str, Any]:
    revision, clean = _git_identity()
    processor = load_pinned_qwen_processor(
        cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
        local_files_only=args.local_files_only,
    )
    _, processor_summary = _synthetic_inputs(processor)
    return _seal_report(
        {
            "artifact_type": "eventground_wam.b0_qwen_processor_smoke",
            "schema_version": 1,
            "project_git_revision": revision,
            "project_tree_clean": clean,
            "qwen_revision": QWEN_REVISION,
            "processor": processor_summary,
            "passed": True,
        }
    )


def _model_report(args: argparse.Namespace) -> dict[str, Any]:
    revision, clean = _git_identity()
    if not clean:
        raise B0ModelError("model smoke evidence requires a clean project tree")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise B0ModelError("the BF16 model smoke requires an available CUDA device")
    cache_dir = str(args.cache_dir) if args.cache_dir is not None else None
    processor = load_pinned_qwen_processor(
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    encoding, processor_summary = _synthetic_inputs(processor)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    full_model = load_pinned_qwen_model(
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    core_lora_names = install_qwen_upper_lora(full_model)
    exact_lora_names = tuple(
        f"conditioning_adapter.backbone.{name}" for name in core_lora_names
    )
    model_load_seconds = time.perf_counter() - load_started
    policy = B0ActionPolicy(
        QwenConditioningAdapter(
            full_model.model,
            trainable_backbone_parameter_names=core_lora_names,
        ),
        ActionQueryTransformer(),
    ).to(device=device, dtype=torch.bfloat16)
    policy.train()
    trainable_names = validate_b0_trainable_parameters(
        policy,
        allowed_exact_names=exact_lora_names,
    )
    device_inputs = {
        name: value.to(
            device=device,
            dtype=torch.bfloat16 if value.is_floating_point() else value.dtype,
        )
        for name, value in encoding.items()
    }
    proprioception = torch.linspace(
        -0.35,
        0.35,
        steps=8,
        device=device,
        dtype=torch.bfloat16,
    ).unsqueeze(0)
    targets = torch.zeros(1, 8, 7, device=device, dtype=torch.bfloat16)

    torch.cuda.synchronize(device)
    forward_started = time.perf_counter()
    output = policy(
        device_inputs,
        proprioception,
        normalized_action_targets=targets,
    )
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    if output.loss is None or not torch.isfinite(output.loss):
        raise B0ModelError("BF16 B0 forward did not produce a finite L1 loss")
    backward_started = time.perf_counter()
    output.loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started
    parameters = dict(policy.named_parameters())
    missing_gradients = [name for name in trainable_names if parameters[name].grad is None]
    nonfinite_gradients = [
        name
        for name in trainable_names
        if parameters[name].grad is not None
        and not torch.isfinite(parameters[name].grad).all()
    ]
    forbidden_gradients = [
        name
        for name, parameter in parameters.items()
        if name.startswith("conditioning_adapter.backbone.") and parameter.grad is not None
        and name not in exact_lora_names
    ]
    gradient_groups = {
        "qwen_upper_lora": exact_lora_names,
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
        for group, names in gradient_groups.items()
        if not names
        or not any(
            parameters[name].grad is not None
            and bool((parameters[name].grad.detach().abs() > 0).any())
            for name in names
        )
    ]
    if missing_gradients or nonfinite_gradients or forbidden_gradients or zero_signal_groups:
        raise B0ModelError(
            "gradient contract failed: "
            f"missing={missing_gradients}, nonfinite={nonfinite_gradients}, "
            f"forbidden={forbidden_gradients}, zero_groups={zero_signal_groups}"
        )
    first_actions = output.normalized_actions.detach().clone()
    policy.zero_grad(set_to_none=True)
    policy.eval()
    with torch.no_grad():
        repeated_actions = policy(device_inputs, proprioception).normalized_actions
        reloaded_actions, pre_load_error = _reload_trainable_state_and_forward(
            policy,
            device_inputs,
            proprioception,
            trainable_names,
            exact_lora_names,
            expected_actions=repeated_actions,
        )
        third_actions = policy(device_inputs, proprioception).normalized_actions
    repeat_error = float((repeated_actions - third_actions).abs().max())
    reload_error = float((reloaded_actions - repeated_actions).abs().max())
    if repeat_error != 0.0 or reload_error > 1e-3:
        raise B0ModelError(
            f"determinism/reload error exceeds limits: repeat={repeat_error}, reload={reload_error}"
        )
    if first_actions.shape != (1, 8, 7):
        raise B0ModelError(f"B0 action shape changed to {tuple(first_actions.shape)}")

    return _seal_report(
        {
            "artifact_type": "eventground_wam.b0_qwen_model_smoke",
            "schema_version": 1,
            "project_git_revision": revision,
            "project_tree_clean": clean,
            "qwen_revision": QWEN_REVISION,
            "device": {
                "name": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "torch_version": str(torch.__version__),
                "cuda_version": torch.version.cuda,
            },
            "processor": processor_summary,
            "conditioning": {
                "memory_shape": list(output.conditioning_memory.tokens.shape),
                "token_selection": "all-non-padding-final-hidden-state",
                "proprioception_tokens": 1,
                "future_tokens": 0,
            },
            "actions": {
                "shape": list(first_actions.shape),
                "dtype": str(first_actions.dtype),
                "l1_loss": float(output.loss.detach()),
                "repeat_max_absolute_error": repeat_error,
                "perturbed_pre_load_max_absolute_error": pre_load_error,
                "reload_max_absolute_error": reload_error,
            },
            "gradients": {
                "trainable_parameter_names": list(trainable_names),
                "trainable_parameter_count": sum(parameters[name].numel() for name in trainable_names),
                "missing_gradient_names": missing_gradients,
                "nonfinite_gradient_names": nonfinite_gradients,
                "frozen_backbone_gradient_names": forbidden_gradients,
                "zero_signal_groups": zero_signal_groups,
                "lora": {
                    "adapter_name": "b0_upper",
                    "layers": [19, 23],
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "rank": 16,
                    "alpha": 32,
                    "parameter_names": list(exact_lora_names),
                },
            },
            "timing_seconds": {
                "model_load": model_load_seconds,
                "forward": forward_seconds,
                "backward": backward_seconds,
            },
            "cuda_memory_bytes": {
                "peak_allocated": torch.cuda.max_memory_allocated(device),
                "peak_reserved": torch.cuda.max_memory_reserved(device),
            },
            "attention_backend": "pytorch-sdpa",
            "passed": True,
        }
    )


def _reload_trainable_state_and_forward(
    policy: B0ActionPolicy,
    inputs: Mapping[str, Tensor],
    proprioception: Tensor,
    trainable_names: Sequence[str],
    exact_lora_names: Sequence[str],
    *,
    expected_actions: Tensor,
) -> tuple[Tensor, float]:
    checkpoint = {
        "format": "eventground-wam-b0-trainable-state-v1",
        "qwen_revision": QWEN_REVISION,
        "state_dict": b0_trainable_state_dict(
            policy,
            allowed_exact_names=exact_lora_names,
        ),
    }
    serialized = io.BytesIO()
    torch.save(checkpoint, serialized)
    serialized.seek(0)
    restored = torch.load(serialized, map_location=proprioception.device, weights_only=True)
    if (
        restored.get("format") != "eventground-wam-b0-trainable-state-v1"
        or restored.get("qwen_revision") != QWEN_REVISION
    ):
        raise B0ModelError("serialized trainable checkpoint has the wrong identity")
    fresh_policy = B0ActionPolicy(
        QwenConditioningAdapter(
            policy.conditioning_adapter.backbone,
            config=policy.conditioning_adapter.config,
            trainable_backbone_parameter_names=(
                policy.conditioning_adapter.trainable_backbone_parameter_names
            ),
        ),
        ActionQueryTransformer(config=policy.action_decoder.config),
    ).to(device=proprioception.device, dtype=proprioception.dtype)
    fresh_exact_names = tuple(exact_lora_names)
    fresh_trainable_names = validate_b0_trainable_parameters(
        fresh_policy,
        allowed_exact_names=fresh_exact_names,
    )
    if tuple(trainable_names) != fresh_trainable_names:
        raise B0ModelError("fresh policy trainable names differ from the saved policy")
    fresh_parameters = dict(fresh_policy.named_parameters())
    with torch.no_grad():
        for name in fresh_trainable_names:
            fresh_parameters[name].zero_()
        perturbed_actions = fresh_policy(inputs, proprioception).normalized_actions.detach().clone()
    pre_load_error = float((perturbed_actions - expected_actions).abs().max())
    if pre_load_error <= 1e-4:
        raise B0ModelError("perturbing every trainable tensor did not change model actions")
    load_b0_trainable_state_dict(
        fresh_policy,
        restored["state_dict"],
        allowed_exact_names=fresh_exact_names,
    )
    fresh_policy.eval()
    with torch.no_grad():
        actions = fresh_policy(inputs, proprioception).normalized_actions.detach().clone()
    return actions, pre_load_error


def _seal_report(report: dict[str, Any]) -> dict[str, Any]:
    report["digest"] = contract_digest(report)
    return report


def _write_exclusive(path: Path, report: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    digest_path = output.with_suffix(output.suffix + ".sha256")
    conflicts = [candidate for candidate in (output, digest_path) if candidate.exists()]
    if conflicts:
        raise FileExistsError(f"refusing to overwrite B0 model smoke outputs: {conflicts}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(report), indent=2, sort_keys=True) + "\n"
    digest_created = False
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload)
        with digest_path.open("x", encoding="utf-8") as handle:
            digest_created = True
            handle.write(hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n")
    except BaseException:
        output.unlink(missing_ok=True)
        if digest_created:
            digest_path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = _processor_report(args) if args.mode == "processor" else _model_report(args)
    if args.output is not None:
        _write_exclusive(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

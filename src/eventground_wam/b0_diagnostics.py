"""Read-only diagnostics for the frozen B0 objective and demonstration source."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch


def noop_mask(actions: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """OpenVLA regeneration criterion; retain stationary gripper transitions."""
    values = np.asarray(actions)
    if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
        raise ValueError("require finite [frames,7] actions")
    unchanged = np.ones(len(values), dtype=bool)
    unchanged[1:] = values[1:, 6] == values[:-1, 6]
    return (np.linalg.norm(values[:, :6], axis=1) < threshold) & unchanged


def numeric_digest(*arrays: np.ndarray) -> str:
    """Canonical float32 numeric identity, independent of parquet encoding."""
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f4")
        digest.update(str(value.shape).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def parameter_group(name: str) -> str:
    if name.startswith("conditioning_adapter.backbone.language_model.") and ".lora_" in name:
        return "language_lora"
    if name.startswith("conditioning_adapter.backbone.visual.") and ".lora_" in name:
        return "vision_lora"
    if name.startswith("conditioning_adapter.proprioception_projector."):
        return "state_projector"
    if name.startswith("action_decoder."):
        return "action_head"
    raise ValueError(f"unclassified trainable parameter: {name}")


def gradient_statistics(
    names: Iterable[str], motion: Iterable[torch.Tensor | None],
    gripper: Iterable[torch.Tensor | None],
) -> dict[str, dict[str, Any]]:
    """Accumulate separate pre-clipping gradients in FP64, without flattening the model."""
    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for name, left, right in zip(names, motion, gripper, strict=True):
        group = parameter_group(name)
        values = []
        for tensor in (left, right):
            if tensor is not None and not torch.isfinite(tensor).all():
                raise ValueError(f"nonfinite gradient: {name}")
            values.append(None if tensor is None else tensor.detach().to(torch.float64))
        a, b = values
        pair = [0.0 if a is None else float(a.square().sum()),
                0.0 if b is None else float(b.square().sum()),
                0.0 if a is None or b is None else float((a * b).sum())]
        groups = [group, "all_trainable"]
        if group in {"language_lora", "vision_lora"}:
            groups.append("backbone_lora")
        for key in groups:
            for i in range(3):
                sums[key][i] += pair[i]
    result = {}
    for group, (a2, b2, dot) in sums.items():
        a, b = math.sqrt(a2), math.sqrt(b2)
        result[group] = {
            "motion_norm": a, "bce_norm": b, "dot": dot,
            "bce_motion_ratio": b / a if a else None,
            "cosine": dot / (a * b) if a and b else None,
            "combined_norm": math.sqrt(max(0.0, a2 + b2 + 2 * dot)),
            "motion_zero": a == 0, "bce_zero": b == 0,
        }
    return result


def summarize_gradients(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for group in rows[0]["groups"]:
        result[group] = {}
        for metric in ("motion_norm", "bce_norm", "bce_motion_ratio", "cosine", "combined_norm"):
            values = [r["groups"][group][metric] for r in rows
                      if r["groups"][group][metric] is not None]
            result[group][metric] = {
                "defined_batches": len(values),
                "median": float(np.median(values)) if values else None,
                "q25": float(np.quantile(values, .25)) if values else None,
                "q75": float(np.quantile(values, .75)) if values else None,
                "max": max(values) if values else None,
            }
        ratios = [r["groups"][group]["bce_motion_ratio"] for r in rows]
        count = sum(x is not None and x >= 2 for x in ratios)
        result[group]["ratio_at_least_two_batches"] = count
        result[group]["ratio_at_least_two_fraction"] = count / len(rows)
    return result


def tensor_tree_digest(value: Any) -> str:
    """Hash model/RNG tensors exactly, including BF16, without serialization metadata."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                digest.update(str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f"{type(item).__name__}:{len(item)}:".encode())
            for part in item:
                visit(part)
        else:
            digest.update(repr(item).encode() + b"\0")

    visit(value)
    return digest.hexdigest()

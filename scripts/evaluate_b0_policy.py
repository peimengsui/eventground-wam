#!/usr/bin/env python3
"""Run the accepted B0 checkpoint through the shared LIBERO episode evaluator."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from check_reference_policy import LIBERO_REVISION, verify_libero
from evaluate_reference_policy import (
    evaluate_task,
    git_revision,
    load_manifest,
)

from eventground_wam.b0_inference import B0InferencePolicy
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import sha256_file
from eventground_wam.evaluation_protocol import load_evaluation_protocol
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "evaluation" / "splits" / "libero-plus-v1" / "full-10K.json"
)
DEFAULT_REVIEW_EVIDENCE = (
    PROJECT_ROOT / "docs" / "evaluation" / "week3-b0-overfit-10demo-bce-v1.json"
)


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--manifest", type=expanded_path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-output-root", type=expanded_path, required=True)
    parser.add_argument("--normalization", type=expanded_path, required=True)
    parser.add_argument(
        "--review-evidence",
        type=expanded_path,
        default=DEFAULT_REVIEW_EVIDENCE,
    )
    parser.add_argument("--cache-dir", type=expanded_path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--libero-root", type=expanded_path, required=True)
    parser.add_argument("--output-dir", type=expanded_path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def select_manifest_task(
    manifest: dict[str, Any],
    *,
    suite: str,
    runtime_task_index: int,
) -> dict[str, Any]:
    if runtime_task_index < 0:
        raise ValueError("--task-id cannot be negative")
    matches = [
        task
        for task in manifest["tasks"]
        if task["suite"] == suite and task["runtime_task_index"] == runtime_task_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one manifest record for {suite}[{runtime_task_index}], "
            f"found {len(matches)}"
        )
    return matches[0]


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "torchvision": importlib.metadata.version("torchvision"),
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
        "cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_capability": list(torch.cuda.get_device_capability(device)),
    }


def _write_sealed_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite B0 rollout report {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite partial B0 rollout report {temporary}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
        digest = sha256_file(path)
        path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _seed_episode(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_evaluation_protocol()
    manifest = load_manifest(args.manifest, protocol)
    manifest_task = select_manifest_task(
        manifest,
        suite=args.suite,
        runtime_task_index=args.task_id,
    )
    selection = {
        "manifest": str(args.manifest),
        "manifest_file_sha256": sha256_file(args.manifest),
        "split_id": manifest["split_id"],
        "task": manifest_task,
    }
    if args.dry_run:
        return {"mode": "dry-run", "selection": selection}

    project_revision = git_revision(PROJECT_ROOT)
    libero = verify_libero(args.libero_root)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B0 rollout requires an available CUDA device")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    policy = B0InferencePolicy.from_checkpoint(
        checkpoint_root=args.checkpoint_output_root,
        normalization_path=args.normalization,
        review_evidence_path=args.review_evidence,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        device=args.device,
    )
    runtime = _runtime_metadata(device)
    if args.load_only:
        return {
            "mode": "load-only",
            "selection": selection,
            "project_git_revision": project_revision,
            "libero": libero,
            "checkpoint": policy.checkpoint.to_dict(),
            "runtime": runtime,
            "model_memory": policy.memory_summary(),
        }

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    from libero.libero import benchmark

    install_codec_independent_motion_blur()
    suite_class = benchmark.get_benchmark_dict().get(args.suite)
    if suite_class is None:
        raise RuntimeError(f"LIBERO runtime does not provide suite {args.suite!r}")
    suite = suite_class()
    runtime_task = suite.get_task(args.task_id)
    if runtime_task.name != manifest_task["task_name"]:
        raise RuntimeError("runtime LIBERO task differs from the selected manifest record")
    cohort_matches = policy.checkpoint.cohort_matches_for_task_name(runtime_task.name)
    if not cohort_matches:
        raise RuntimeError(
            "selected rollout task is not derived from an instruction in the accepted "
            "ten-demonstration training cohort"
        )
    selection["cohort_matches"] = list(cohort_matches)
    _seed_episode(manifest_task["episode_seed"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_task(
        ordinal=0,
        manifest_task=manifest_task,
        suite=suite,
        policy=policy,
        protocol=protocol,
        output_dir=args.output_dir,
        project_revision=project_revision,
        resume=args.resume,
        checkpoint_id=policy.checkpoint.checkpoint_id,
        checkpoint_revision=policy.checkpoint.checkpoint_revision,
    )
    latencies = np.asarray(result.policy_latencies_ms, dtype=np.float64)
    report = seal_contract(
        {
            "artifact_type": "eventground_wam.b0_closed_loop_rollout",
            "schema_version": 1,
            "status": "passed" if result.success else "failed",
            "project": {
                "git_revision": project_revision,
                "tree_clean": True,
            },
            "libero": {
                **libero,
                "expected_revision": LIBERO_REVISION,
            },
            "selection": selection,
            "checkpoint": policy.checkpoint.to_dict(),
            "execution": {
                "action_chunk_length": 8,
                "replan_interval_steps": 8,
                "continuous_action_decoding": (
                    "invert-train-q01-q99-affine-then-clip-to-simulator-range"
                ),
                "simulator_action_range": [-1.0, 1.0],
                "gripper_decoding": "nonnegative-logit-to-+1-negative-logit-to--1",
                "camera_transform": "rotate-180-before-qwen-processor",
            },
            "result": asdict(result),
            "latency_ms": {
                "sample_count": int(latencies.size),
                "median": float(np.median(latencies)) if latencies.size else None,
                "p95": float(np.percentile(latencies, 95)) if latencies.size else None,
                "maximum": float(latencies.max()) if latencies.size else None,
            },
            "inference": policy.inference_summary(),
            "runtime": {
                **runtime,
                "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(device),
            },
            "environment_lock": {
                "relative_path": "environments/b0/uv.lock",
                "file_sha256": sha256_file(PROJECT_ROOT / "environments/b0/uv.lock"),
            },
        }
    )
    report_path = args.output_dir / "rollout-report.json"
    _write_sealed_report(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = _run(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    if result.get("mode") in {"dry-run", "load-only"}:
        return 0
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

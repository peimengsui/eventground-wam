#!/usr/bin/env python3
"""Record one sealed-cohort B0 base task through the shared HDF5 evaluator."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from check_reference_policy import LIBERO_REVISION, verify_libero
from evaluate_b0_policy import (
    DEFAULT_REVIEW_EVIDENCE,
    _runtime_metadata,
    _seed_episode,
    _write_sealed_report,
    expanded_path,
)
from evaluate_reference_policy import evaluate_task, git_revision

from eventground_wam.b0_inference import B0InferencePolicy
from eventground_wam.b0_training_rollout import (
    BaseTrainingTaskSuite,
    resolve_base_task,
    select_cohort_episode,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import sha256_file
from eventground_wam.evaluation_protocol import (
    B0_TRAINING_ROLLOUT_PROTOCOL_PATH,
    derive_episode_seed,
    load_evaluation_protocol,
    protocol_sha256,
)
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-episode-index", type=int, required=True)
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
    parser.add_argument("--resume", action="store_true")
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_evaluation_protocol(B0_TRAINING_ROLLOUT_PROTOCOL_PATH)
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
    cohort_episode = select_cohort_episode(policy.checkpoint, args.cohort_episode_index)
    task_selection = resolve_base_task(
        args.libero_root,
        cohort_episode["instruction"],
    )
    episode_seed = derive_episode_seed(
        protocol,
        task_selection.suite,
        task_selection.runtime_task_index,
    )
    manifest_task = {
        "suite": task_selection.suite,
        "runtime_task_index": task_selection.runtime_task_index,
        "classification_task_id": task_selection.runtime_task_index + 1,
        "task_name": task_selection.task_name,
        "perturbation_category": "Training Cohort",
        "difficulty_level": None,
        "episode_seed": episode_seed,
    }
    selection = {
        "selection_rule": (
            "first success in the predeclared complete sealed ten-task cohort order"
        ),
        "cohort_episode": cohort_episode,
        "base_task": manifest_task,
        "policy_instruction": cohort_episode["instruction"],
        "tasks_info": {
            "path": str(task_selection.tasks_info_path),
            "file_sha256": sha256_file(task_selection.tasks_info_path),
        },
        "protocol": {
            "path": str(B0_TRAINING_ROLLOUT_PROTOCOL_PATH),
            "sha256": protocol_sha256(protocol),
        },
    }
    _seed_episode(episode_seed)

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    install_codec_independent_motion_blur()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate_task(
        ordinal=0,
        manifest_task=manifest_task,
        suite=BaseTrainingTaskSuite(task_selection),
        policy=policy,
        protocol=protocol,
        output_dir=args.output_dir,
        project_revision=project_revision,
        resume=args.resume,
        checkpoint_id=policy.checkpoint.checkpoint_id,
        checkpoint_revision=policy.checkpoint.checkpoint_revision,
        policy_instruction=cohort_episode["instruction"],
    )
    latencies = np.asarray(result.policy_latencies_ms, dtype=np.float64)
    report = seal_contract(
        {
            "artifact_type": "eventground_wam.b0_training_task_rollout",
            "schema_version": 1,
            "status": "passed" if result.success else "failed",
            "claim_scope": (
                "one unperturbed training task represented in the sealed cohort; "
                "not a generalization or cohort success-rate claim"
            ),
            "project": {"git_revision": project_revision, "tree_clean": True},
            "libero": {**libero, "expected_revision": LIBERO_REVISION},
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
                **_runtime_metadata(device),
                "peak_allocated_cuda_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_cuda_bytes": torch.cuda.max_memory_reserved(device),
            },
            "environment_lock": {
                "relative_path": "environments/b0/uv.lock",
                "file_sha256": sha256_file(PROJECT_ROOT / "environments/b0/uv.lock"),
            },
        }
    )
    _write_sealed_report(args.output_dir / "training-rollout-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    result = _run(_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

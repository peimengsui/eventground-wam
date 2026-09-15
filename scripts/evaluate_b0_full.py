#!/usr/bin/env python3
"""Evaluate a full-training B0 bundle on all clean-40 or frozen dev-210 tasks."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from check_reference_policy import verify_libero
from evaluate_b0_policy import _seed_episode
from evaluate_reference_policy import evaluate_task, git_revision, load_manifest, wilson_interval
from train_b0 import build_policy

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_inference import B0InferencePolicy
from eventground_wam.b0_model import QWEN_REVISION, QwenB0InputBuilder, load_b0_trainable_state_dict
from eventground_wam.b0_model import load_pinned_qwen_processor as load_processor
from eventground_wam.b0_training import (
    POLICY_FORMAT,
    TrainingContractError,
    atomic_json,
    read_sealed,
    sha256_file,
)
from eventground_wam.b0_training_rollout import (
    BASE_SUITES,
    BaseTaskSelection,
    BaseTrainingTaskSuite,
    _base_task_paths,
)
from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract
from eventground_wam.evaluation_protocol import (
    derive_episode_seed,
    load_evaluation_protocol,
    protocol_sha256,
)
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PROTOCOL = PROJECT_ROOT / "configs/evaluation/libero_clean_v1.json"


@dataclass(frozen=True)
class FullCheckpointIdentity:
    checkpoint_id: str
    checkpoint_revision: str
    training_manifest_digest: str
    training_report_digest: str
    selected_epoch: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_training_bundle(root: Path, normalization: Path) -> tuple[Any, Any, Any]:
    manifest = read_sealed(root / "training-manifest.json")
    report = read_sealed(root / "training-report.json")
    recipe = manifest["recipe"]
    if (manifest["artifact_type"] != "eventground_wam.b0_full_training_manifest"
            or manifest["mode"] != "production"
            or report["artifact_type"] != "eventground_wam.b0_full_training_report"
            or report["status"] != "complete"
            or report["identity_digest"] != manifest["digest"]
            or recipe["qwen_revision"] != QWEN_REVISION
            or recipe["objective"] != "continuous-l1-plus-gripper-bce-v1"
            or recipe["gripper_bce_weight"] != 1.0
            or recipe["lora_rank"] != 16
            or recipe["epochs"] != 5
            or report["epochs_completed"] != 5
            or report["samples_seen"] != 5 * 1_715_954
            or report["reload_validation_passed"] is not True):
        raise TrainingContractError("invalid or incomplete full-training acceptance bundle")
    checkpoint_hash = sha256_file(root / "B0.ckpt")
    if checkpoint_hash != report["checkpoint_sha256"]:
        raise TrainingContractError("full-training checkpoint SHA-256 mismatch")
    if sha256_file(normalization) != manifest["source"]["normalization_file_sha256"]:
        raise TrainingContractError("full-training normalizer hash mismatch")
    identity = FullCheckpointIdentity(
        checkpoint_id=f"eventground-wam/{recipe['recipe_id']}",
        checkpoint_revision=checkpoint_hash, training_manifest_digest=manifest["digest"],
        training_report_digest=report["digest"], selected_epoch=report["selected_epoch"],
    )
    return manifest, report, identity


def clean_instruction(task_name: str) -> str:
    # Same canonical filename rule as the pinned upstream grab_language_from_filename.
    if task_name[0].isupper():
        task_name = re.sub(r"^.*SCENE\d+_", "", task_name)
    return task_name.replace("_", " ")


def clean_manifest(libero_root: Path, protocol: Any) -> tuple[Any, dict[Any, Any]]:
    import hashlib

    tasks = []
    suites = {}
    for suite in BASE_SUITES:
        inventory = libero_root / "libero/libero/bddl_files" / suite / "tasks_info.txt"
        for index, (bddl, states) in enumerate(_base_task_paths(libero_root, suite)):
            selection = BaseTaskSelection(
                suite, index, bddl.stem, clean_instruction(bddl.stem), bddl, states, inventory,
            )
            suites[(suite, index)] = BaseTrainingTaskSuite(selection)
            tasks.append({
                "suite": suite, "runtime_task_index": index, "classification_task_id": index + 1,
                "task_name": bddl.stem, "instruction": selection.instruction,
                "perturbation_category": "Clean", "difficulty_level": None,
                "episode_seed": derive_episode_seed(protocol, suite, index),
                "bddl_sha256": sha256_file(bddl), "initial_states_file_sha256": sha256_file(states),
                "tasks_info_sha256": sha256_file(inventory),
            })
    return seal_contract({
        "artifact_type": "eventground_wam.clean_evaluation_manifest", "schema_version": 1,
        "split_id": "clean-40", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256(protocol), "task_count": 40,
        "task_records_sha256": hashlib.sha256(canonical_json_bytes(tasks)).hexdigest(),
        "tasks": tasks,
    }), suites


def run(args: Any) -> None:
    protocol = load_evaluation_protocol(CLEAN_PROTOCOL) if args.split == "clean-40" else (
        load_evaluation_protocol()
    )
    verify_libero(args.libero_root)
    if args.split == "clean-40":
        manifest, suites = clean_manifest(args.libero_root, protocol)
        frozen = read_sealed(PROJECT_ROOT / "evaluation/splits/libero-clean-v1/clean-40.json")
        if manifest != frozen:
            raise TrainingContractError("clean inventory differs from the predeclared frozen manifest")
    else:
        manifest = load_manifest(
            PROJECT_ROOT / "evaluation/splits/libero-plus-v1/dev-210.json", protocol,
        )
        suites = {}
    if args.manifest_only:
        atomic_json(args.output_root / "task-manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return
    revision = git_revision(PROJECT_ROOT)
    training, _report, identity = verify_training_bundle(args.training_root, args.normalization)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise TrainingContractError("full B0 evaluation requires CUDA")
    torch.cuda.set_device(device)
    model, exact_names = build_policy(training["recipe"], str(args.cache_dir), device)
    state = torch.load(args.training_root / "B0.ckpt", map_location="cpu", weights_only=True)
    if (state["format"] != POLICY_FORMAT or state["identity_digest"] != training["digest"]
            or state["qwen_revision"] != QWEN_REVISION):
        raise TrainingContractError("checkpoint tensor envelope changed")
    load_b0_trainable_state_dict(model, state["state_dict"], allowed_exact_names=exact_names)
    policy = B0InferencePolicy(
        policy=model,
        input_builder=QwenB0InputBuilder(load_processor(cache_dir=str(args.cache_dir),
                                                       local_files_only=True)),
        normalizer=load_b0_action_normalizer(args.normalization), device=device,
        checkpoint=identity,
    )
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    install_codec_independent_motion_blur()
    from libero.libero import benchmark

    args.output_root.mkdir(parents=True, exist_ok=True)
    saved_manifest = args.output_root / "task-manifest.json"
    if saved_manifest.exists() and json.loads(saved_manifest.read_text()) != manifest:
        raise TrainingContractError("saved task manifest differs from this evaluation")
    atomic_json(saved_manifest, manifest)
    results = []
    for ordinal, task in enumerate(manifest["tasks"]):
        key = (task["suite"], task["runtime_task_index"]) if args.split == "clean-40" else task["suite"]
        if key not in suites:
            suites[key] = benchmark.get_benchmark_dict()[task["suite"]]()
        _seed_episode(task["episode_seed"])
        result = evaluate_task(
            ordinal=ordinal, manifest_task=task, suite=suites[key], policy=policy,
            protocol=protocol, output_dir=args.output_root, project_revision=revision,
            resume=args.resume, checkpoint_id=identity.checkpoint_id,
            checkpoint_revision=identity.checkpoint_revision,
        )
        results.append(asdict(result))
        atomic_json(args.output_root / "progress.json", {"completed": len(results),
                                                         "task_count": manifest["task_count"],
                                                         "episodes": results})
    breakdowns = {}
    for name in ("suite", "perturbation_category", "difficulty_level"):
        groups = defaultdict(list)
        for result in results:
            groups[str(result[name])].append(result)
        breakdowns[name] = {
            label: {"count": len(group), "success_count": sum(x["success"] for x in group),
                    "success_rate": sum(x["success"] for x in group) / len(group),
                    "wilson_95_percent": wilson_interval(sum(x["success"] for x in group), len(group))}
            for label, group in groups.items()
        }
    successes = sum(result["success"] for result in results)
    atomic_json(args.output_root / "evaluation-report.json", seal_contract({
        "artifact_type": "eventground_wam.b0_full_evaluation", "status": "complete",
        "project_git_revision": revision, "checkpoint": identity.to_dict(),
        "protocol_id": protocol["protocol_id"], "protocol_sha256": protocol_sha256(protocol),
        "task_records_sha256": manifest["task_records_sha256"],
        "split_id": args.split, "selection_is_complete_manifest": len(results) == manifest["task_count"],
        "episode_count": len(results), "success_count": successes,
        "micro_success_rate": successes / len(results),
        "wilson_95_percent": wilson_interval(successes, len(results)),
        "breakdowns": breakdowns, "episodes": results, "action_decode": policy.inference_summary(),
        "b0_runtime_lock_sha256": sha256_file(PROJECT_ROOT / "environments/b0/uv.lock"),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("clean-40", "dev-210"), required=True)
    for name in ("training-root", "normalization", "cache-dir", "libero-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    run(args)


if __name__ == "__main__":
    main()

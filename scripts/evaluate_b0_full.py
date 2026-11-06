#!/usr/bin/env python3
"""Evaluate a full-training B0 bundle on all clean-40 or frozen dev-210 tasks."""

from __future__ import annotations

import argparse
import json
import os
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
from eventground_wam.clean_evaluation import clean_manifest
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.evaluation_protocol import (
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
    selected_epoch: int | None
    selected_step: int | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.selected_step is None:
            result.pop("selected_step")
        return result


def verify_training_bundle(root: Path, normalization: Path) -> tuple[Any, Any, Any]:
    manifest = read_sealed(root / "training-manifest.json")
    if manifest.get("artifact_type") == "eventground_wam.b0_fresh_v2_training_manifest":
        from eventground_wam.b0_fresh import verify_fresh_bundle

        manifest, report = verify_fresh_bundle(root, normalization)
        if report['selected_epoch'] != 2:
            raise TrainingContractError('full final evaluation requires the fixed epoch-2 endpoint')
        return manifest, report, FullCheckpointIdentity(
            checkpoint_id=f"eventground-wam/{manifest['recipe']['recipe_id']}",
            checkpoint_revision=report['checkpoint_sha256'], training_manifest_digest=manifest['digest'],
            training_report_digest=report['digest'], selected_epoch=2,
            selected_step=report['global_step'])
    report = read_sealed(root / "training-report.json")
    recipe = manifest["recipe"]
    oft = recipe.get("policy_family") == "qwen35-oft-b0-v1"
    if oft:
        selected = [item for item in report.get("validation_history", [])
                    if item["global_step"] == report.get("selected_step")]
        specific_valid = (
            report.get("artifact_type") == "eventground_wam.b0_oft_training_report"
            and report.get("policy_family") == recipe["policy_family"]
            and recipe["lora_rank"] == 32
            and 0 < report.get("global_step", 0) <= recipe["maximum_updates"]
            and 0 < report.get("samples_seen", 0) <= recipe["epochs"] * 1_715_954
            and len(selected) == 1
            and selected[0]["validation"]["sample_count"] == 212471
            and report.get("final_reload_validation", {}).get("sample_count") == 212471
        )
    else:
        specific_valid = (
            report["artifact_type"] == "eventground_wam.b0_full_training_report"
            and recipe["lora_rank"] == 16 and recipe["epochs"] == 5
            and report["epochs_completed"] == 5 and report["samples_seen"] == 5 * 1_715_954
        )
    if (manifest["artifact_type"] != "eventground_wam.b0_full_training_manifest"
            or manifest["mode"] != "production"
            or not specific_valid
            or report["status"] != "complete"
            or report["identity_digest"] != manifest["digest"]
            or recipe["qwen_revision"] != QWEN_REVISION
            or recipe["objective"] != "continuous-l1-plus-gripper-bce-v1"
            or recipe["gripper_bce_weight"] != 1.0
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
        training_report_digest=report["digest"], selected_epoch=report.get("selected_epoch"),
        selected_step=report.get("selected_step"),
    )
    return manifest, report, identity


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
    input_builder = QwenB0InputBuilder
    if training["recipe"].get("policy_family") == "qwen35-oft-b0-v1":
        from eventground_wam.b0_oft import OFTInputBuilder

        input_builder = OFTInputBuilder
    policy = B0InferencePolicy(
        policy=model,
        input_builder=input_builder(load_processor(cache_dir=str(args.cache_dir),
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

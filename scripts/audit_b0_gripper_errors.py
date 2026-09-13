#!/usr/bin/env python3
"""Audit residual gripper errors from a sealed B0 ten-demonstration checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import run_b0_cohort_overfit as cohort_runner
import torch

from eventground_wam.b0_gripper_audit import audit_b0_gripper_errors
from eventground_wam.b0_model import B0ModelError, B0ObjectiveConfig
from eventground_wam.b0_overfit import (
    TenDemonstrationOverfitConfig,
    build_ten_demonstration_manifest,
)
from eventground_wam.demonstration_contracts import seal_contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-minimum-accuracy", type=float, default=0.995)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_config(report: dict[str, Any]) -> TenDemonstrationOverfitConfig:
    configuration = report.get("configuration", {})
    names = {field.name for field in fields(TenDemonstrationOverfitConfig)}
    try:
        values = {name: configuration[name] for name in names}
    except KeyError as error:
        raise B0ModelError(f"source report omits configuration field {error.args[0]}") from error
    return TenDemonstrationOverfitConfig(**values)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0.0 < args.reviewed_minimum_accuracy <= 1.0:
        raise B0ModelError("reviewed minimum accuracy must be in (0,1]")
    output = args.output.expanduser().resolve()
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise FileExistsError(f"refusing to overwrite audit output {output}")
    checkpoint_root = args.checkpoint_output_root.expanduser().resolve()
    source_report_path = checkpoint_root / "overfit-report.json"
    source_manifest_path = checkpoint_root / "cohort-manifest.json"
    source_report = cohort_runner._read_sealed_json(
        source_report_path,
        artifact_type="eventground_wam.b0_ten_demonstration_overfit",
    )
    source_manifest = cohort_runner._read_sealed_json(
        source_manifest_path,
        artifact_type="eventground_wam.b0_ten_demonstration_manifest",
    )
    config = _source_config(source_report)
    cohort_runner._seed_everything(config.seed)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise B0ModelError("the real checkpoint audit requires an available CUDA device")

    print("loading and validating the fixed ten-demonstration cohort", flush=True)
    cohort, episode_summaries, split_index, normalizer, _dataset = (
        cohort_runner._load_fixed_cohort(args, config)
    )
    current_manifest = build_ten_demonstration_manifest(
        cohort,
        episode_summaries,
        config=config,
        project_git_revision=source_report["project"]["git_revision"],
        source_identity=split_index.source_identity,
        normalization_artifact_digest=str(normalizer.artifact_digest),
        normalization_file_sha256=_sha256_file(args.normalization.expanduser().resolve()),
    )
    if current_manifest != source_manifest:
        raise B0ModelError("decoded audit cohort differs from the sealed training cohort")

    cache_dir = str(args.cache_dir.expanduser().resolve()) if args.cache_dir is not None else None
    processor = cohort_runner.load_pinned_qwen_processor(
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    print("building deterministic processor cache", flush=True)
    batches, processor_summary = cohort_runner._cache_processor_batches(cohort, processor, config)
    torch.cuda.set_device(device)
    policy, exact_lora_names = cohort_runner._build_policy(
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        device=device,
        objective_config=B0ObjectiveConfig(
            objective=config.objective,
            gripper_bce_weight=config.gripper_bce_weight,
        ),
    )
    continuation = cohort_runner._load_verified_continuation(
        checkpoint_root,
        output_root=output.parent,
        current_manifest=current_manifest,
        policy=policy,
        exact_lora_names=exact_lora_names,
    )

    logits = []
    policy.eval()
    with torch.no_grad():
        for batch_number, batch in enumerate(batches, start=1):
            inputs = {
                name: value.to(device=device) for name, value in batch.backbone_inputs.items()
            }
            prediction = policy(inputs, batch.proprioception.to(device=device))
            if prediction.gripper_logits is None:
                raise B0ModelError("BCE checkpoint produced no gripper logits")
            logits.append(prediction.gripper_logits.float().cpu().numpy())
            if batch_number % 50 == 0:
                print(f"audit-inference batches={batch_number}/{len(batches)}", flush=True)
    audit = audit_b0_gripper_errors(cohort, np.concatenate(logits, axis=0))
    source_final = source_report["optimization"]["final_diagnostic"]
    residuals = audit["residual_errors"]
    if residuals["error_slot_count"] != source_final["gripper_sign_error_count"]:
        raise B0ModelError("audit error count differs from the sealed source report")
    if abs(residuals["accuracy"] - source_final["gripper_sign_accuracy"]) > 1e-12:
        raise B0ModelError("audit accuracy differs from the sealed source report")

    label_integrity = audit["label_integrity"]
    original_gates = source_report["acceptance"]
    low_exercise_risk = (
        residuals["accuracy"] >= args.reviewed_minimum_accuracy
        and label_integrity["nonbinary_target_count"] == 0
        and label_integrity["overlapping_chunk_label_contradiction_count"] == 0
        and label_integrity["exact_conditioning_conflicting_label_group_count"] == 0
        and original_gates["loss_ratio_passed"] is True
        and original_gates["normalized_action_mae_passed"] is True
    )
    report = seal_contract(
        {
            "artifact_type": "eventground_wam.b0_gripper_error_audit",
            "schema_version": 1,
            "status": "completed",
            "audit_project": {
                "git_revision": cohort_runner._git_identity()[0],
                "tree_clean": cohort_runner._git_identity()[1],
            },
            "source_run": {
                "external_root": str(checkpoint_root),
                "report_digest": source_report["digest"],
                "report_file_sha256": _sha256_file(source_report_path),
                "manifest_digest": source_manifest["digest"],
                "checkpoint_file_sha256": continuation["source_checkpoint_file_sha256"],
                "project_git_revision": source_report["project"]["git_revision"],
                "objective": config.objective,
                "gripper_bce_weight": config.gripper_bce_weight,
            },
            "processor_cache": processor_summary,
            "audit": audit,
            "reviewed_disposition": {
                "scope": "week-3-fixed-ten-demonstration-overfit-exercise-only",
                "basis": "post-hoc-engineering-acceptance-with-documented-residuals",
                "reviewed_minimum_gripper_sign_accuracy": args.reviewed_minimum_accuracy,
                "original_exact_memorization_gate_passed": original_gates["passed"],
                "continuous_action_gates_passed": (
                    original_gates["loss_ratio_passed"]
                    and original_gates["normalized_action_mae_passed"]
                ),
                "risk_for_advancing_to_next_development_step": (
                    "low" if low_exercise_risk else "not-low"
                ),
                "accepted_with_documented_residuals": low_exercise_risk,
                "does_not_establish_rollout_or_generalization_quality": True,
            },
        }
    )
    if report["audit_project"]["tree_clean"] is not True:
        raise B0ModelError("gripper audit evidence requires a clean project tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    file_sha256 = _sha256_file(output)
    with output.with_suffix(output.suffix + ".sha256").open("x", encoding="utf-8") as handle:
        handle.write(file_sha256 + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    report = _run(_parser().parse_args(argv))
    return 0 if report["reviewed_disposition"]["accepted_with_documented_residuals"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

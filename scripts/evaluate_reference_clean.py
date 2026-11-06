#!/usr/bin/env python3
"""Evaluate the installed OpenVLA reference on B0's unchanged clean-40 manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from check_reference_policy import (
    EVALUATOR_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    verify_checkpoint,
    verify_evaluator,
    verify_libero,
)
from evaluate_reference_policy import (
    OpenVLAOFTPolicy,
    collect_runtime_metadata,
    evaluate_task,
    git_revision,
    summarize_results,
    write_json,
)

from eventground_wam.clean_evaluation import clean_manifest
from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.episode_recorder import (
    read_episode_metadata,
    sha256_array,
    sha256_file,
    validate_episode_file,
)
from eventground_wam.evaluation_protocol import load_evaluation_protocol
from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur

ROOT = Path(__file__).resolve().parents[1]
PAIRED_METADATA_FIELDS = (
    "protocol_id", "protocol_sha256", "benchmark_revision", "suite", "runtime_task_index",
    "classification_task_id", "perturbation_category", "difficulty_level", "instruction",
    "bddl_sha256", "initial_state_sha256", "episode_seed",
)


def read_sealed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("digest") != contract_digest(value):
        raise ValueError(f"invalid sealed record: {path}")
    return value


def finalize_report(progress: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    """Seal complete saved metrics without loading a policy or rerunning episodes."""
    if (progress["episode_count"] != 40 or len(progress["episodes"]) != 40
            or not progress["selection_is_complete_manifest"]
            or progress["project_git_revision"] != selection["project_git_revision"]
            or progress["protocol_sha256"] != selection["protocol_sha256"]
            or progress["checkpoint_revision"] != selection["checkpoint_revision"]):
        raise ValueError("cannot finalize incomplete or mismatched reference progress")
    # torch.__version__ is a TorchVersion str subclass. JSON round-tripping retains
    # the recorded values while converting subclasses to the contract's exact types.
    progress = json.loads(json.dumps(progress, allow_nan=False))
    return seal_contract({
        **progress, "artifact_type": "eventground_wam.reference_clean40_evaluation",
        "status": "complete", "task_records_sha256": selection["task_records_sha256"],
        "selection_digest": selection["digest"], "b0_report_digest": selection["b0_report_digest"],
        "policy_profile": selection["policy_profile"],
        "comparison_scope": selection["comparison_scope"],
        "acceptance_gate": selection["acceptance_gate"],
    })


def match_metadata(left: dict[str, Any], right: dict[str, Any]) -> None:
    for field in PAIRED_METADATA_FIELDS:
        if left[field] != right[field]:
            raise ValueError(f"paired episode mismatch: {field}")


def validate_artifact(episode: dict[str, Any], root: Path, protocol: Any) -> Path:
    path = Path(episode["artifact"]).resolve()
    if path.parent != root.resolve() or sha256_file(path) != episode["artifact_sha256"]:
        raise ValueError(f"invalid episode artifact identity: {path}")
    summary = validate_episode_file(path, protocol)
    for field in ("suite", "runtime_task_index", "success", "termination_reason",
                  "policy_step_count"):
        if summary[field] != episode[field]:
            raise ValueError(f"episode report differs from HDF5: {field}")
    return path


def verify_b0(root: Path, manifest: Any, suites: Any, protocol: Any) -> dict[str, Any]:
    report = read_sealed(root / "evaluation-report.json")
    expected = read_sealed(ROOT / "docs/evaluation/week4-b0-clean40-v1.json")
    if report != expected or read_sealed(root / "task-manifest.json") != manifest:
        raise ValueError("B0 comparison must use the recorded, complete Week 4 clean-40 run")
    if len(report["episodes"]) != manifest["task_count"]:
        raise ValueError("B0 comparison episode count mismatch")
    for task, episode in zip(manifest["tasks"], report["episodes"]):
        path = validate_artifact(episode, root, protocol)
        metadata = read_episode_metadata(path)
        index = task["runtime_task_index"]
        states = suites[(task["suite"], index)].get_task_init_states(index)
        state = np.asarray(states[protocol["benchmark"]["initial_state"]["state_index"]])
        expected_metadata = {
            **task, "protocol_id": manifest["protocol_id"],
            "protocol_sha256": manifest["protocol_sha256"],
            "benchmark_revision": protocol["benchmark"]["revision"],
            "initial_state_sha256": sha256_array(state),
        }
        match_metadata(metadata, expected_metadata)
        if (metadata["checkpoint_revision"] != report["checkpoint"]["checkpoint_revision"]
                or metadata["project_git_revision"] != report["project_git_revision"]):
            raise ValueError("B0 artifact checkpoint/source identity mismatch")
    return report


def verify_pairs(b0: Any, reference: Any, args: Any, protocol: Any) -> dict[str, Any]:
    if len(reference["episodes"]) != len(b0["episodes"]):
        raise ValueError("paired evaluation is incomplete")
    checks = []
    outcomes = {"both_success": 0, "b0_only_success": 0,
                "reference_only_success": 0, "both_failure": 0}
    for left, right in zip(b0["episodes"], reference["episodes"]):
        left_path = validate_artifact(left, args.b0_output_root, protocol)
        right_path = validate_artifact(right, args.output_root, protocol)
        left_meta, right_meta = read_episode_metadata(left_path), read_episode_metadata(right_path)
        match_metadata(left_meta, right_meta)
        if (right_meta["checkpoint_id"] != MODEL_ID
                or right_meta["checkpoint_revision"] != MODEL_REVISION
                or right_meta["project_git_revision"] != reference["project_git_revision"]):
            raise ValueError("reference artifact checkpoint/source identity mismatch")
        # Compare observations/states through settling and at the first policy input.
        # Later observations differ because the policies execute different actions.
        input_matches = {}
        rows = protocol["simulation"]["settling_steps"] + 1
        with h5py.File(left_path, "r") as lh, h5py.File(right_path, "r") as rh:
            for key in ("agent_rgb", "wrist_rgb", "proprioception", "mujoco_state"):
                input_matches[key] = bool(np.array_equal(
                    lh[f"steps/{key}"][:rows], rh[f"steps/{key}"][:rows],
                ))
        label = ("both_success" if left["success"] and right["success"] else
                 "b0_only_success" if left["success"] else
                 "reference_only_success" if right["success"] else "both_failure")
        outcomes[label] += 1
        checks.append({
            "suite": left["suite"], "runtime_task_index": left["runtime_task_index"],
            "b0_artifact_sha256": left["artifact_sha256"],
            "reference_artifact_sha256": right["artifact_sha256"],
            "bddl_sha256": left_meta["bddl_sha256"],
            "initial_state_sha256": left_meta["initial_state_sha256"],
            "episode_seed": left_meta["episode_seed"], "metadata_match": True,
            "settling_and_first_policy_input_exact_match": input_matches,
            "b0_success": left["success"], "reference_success": right["success"],
        })
    return seal_contract({
        "artifact_type": "eventground_wam.clean40_paired_verification", "schema_version": 1,
        "b0_report_digest": b0["digest"], "reference_report_digest": reference["digest"],
        "protocol_sha256": reference["protocol_sha256"], "episode_pair_count": len(checks),
        "paired_outcomes": outcomes,
        "b0_minus_reference_percentage_points": 100 * (
            b0["success_count"] - reference["success_count"]) / len(checks),
        "all_metadata_matched": True,
        "all_initial_observations_exactly_matched": all(
            all(x["settling_and_first_policy_input_exact_match"].values()) for x in checks),
        "comparison_scope": "evaluation-matched-external-reference-training-not-matched",
        "acceptance_gate": "open-no-configuration-matched-comparator-established",
        "episode_checks": checks,
    })


def run(args: Any) -> None:
    protocol = load_evaluation_protocol(ROOT / "configs/evaluation/libero_clean_v1.json")
    verify_libero(args.libero_root)
    manifest, suites = clean_manifest(args.libero_root, protocol)
    frozen = read_sealed(ROOT / "evaluation/splits/libero-clean-v1/clean-40.json")
    if manifest != frozen:
        raise ValueError("clean inventory differs from B0's frozen manifest")
    revision = git_revision(ROOT)
    b0 = verify_b0(args.b0_output_root, manifest, suites, protocol)
    if args.verify_only or (args.output_root / "evaluation-report.json").exists():
        reference = read_sealed(args.output_root / "evaluation-report.json")
        if (reference["status"] != "complete"
                or reference["project_git_revision"] != revision
                or reference["task_records_sha256"] != manifest["task_records_sha256"]
                or reference["protocol_sha256"] != manifest["protocol_sha256"]):
            raise ValueError("reference report identity mismatch")
        verified = verify_pairs(b0, reference, args, protocol)
        write_json(args.output_root / "paired-verification.json", verified)
        print(json.dumps({k: v for k, v in verified.items() if k != "episode_checks"}, indent=2))
        return
    verify_evaluator(args.evaluator_root)
    checkpoint = verify_checkpoint(args.checkpoint_dir, full_hash=True)
    selection = seal_contract({
        "artifact_type": "eventground_wam.reference_clean40_selection", "schema_version": 1,
        "project_git_revision": revision, "task_manifest_digest": manifest["digest"],
        "task_records_sha256": manifest["task_records_sha256"],
        "protocol_sha256": manifest["protocol_sha256"], "b0_report_digest": b0["digest"],
        "checkpoint_id": MODEL_ID, "checkpoint_revision": MODEL_REVISION,
        "checkpoint_verification": checkpoint, "evaluator_revision": EVALUATOR_REVISION,
        "reference_runtime_lock_sha256": sha256_file(ROOT / "environments/reference-policy/uv.lock"),
        "policy_profile": load_evaluation_protocol()["reference_policy_profile"],
        "comparison_scope": "evaluation-matched-external-reference-training-not-matched",
        "acceptance_gate": "open-no-configuration-matched-comparator-established",
    })
    if args.preflight_only:
        print(json.dumps(selection, indent=2))
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    for filename, expected in (("selection.json", selection), ("task-manifest.json", manifest)):
        target = args.output_root / filename
        if target.exists() and read_sealed(target) != expected:
            raise ValueError(f"cannot resume changed evaluation identity: {target}")
        write_json(target, expected)
    policy = OpenVLAOFTPolicy(args.checkpoint_dir)
    runtime = collect_runtime_metadata()
    for module_name in ("mujoco", "robosuite", "h5py", "PIL"):
        module = __import__(module_name)
        runtime[module_name] = module.__version__
    install_codec_independent_motion_blur()
    results = []
    for ordinal, task in enumerate(manifest["tasks"]):
        seed = task["episode_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        result = evaluate_task(
            ordinal=ordinal, manifest_task=task,
            suite=suites[(task["suite"], task["runtime_task_index"])], policy=policy,
            protocol=protocol, output_dir=args.output_root, project_revision=revision, resume=True,
        )
        results.append(result)
        report = summarize_results(
            manifest=manifest, results=results, project_revision=revision, runtime_metadata=runtime,
            environment_lock_sha256=selection["reference_runtime_lock_sha256"],
            selected_task_count=manifest["task_count"],
        )
        write_json(args.output_root / "progress.json", report)
        print(f"[{len(results)}/40] {task['suite']} task={task['runtime_task_index']} "
              f"success={result.success} reason={result.termination_reason}", flush=True)
    report = finalize_report(report, selection)
    write_json(args.output_root / "evaluation-report.json", report)
    verified = verify_pairs(b0, report, args, protocol)
    write_json(args.output_root / "paired-verification.json", verified)
    print(json.dumps({"success_count": report["success_count"], "episode_count": 40,
                      "paired_outcomes": verified["paired_outcomes"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-dir", "evaluator-root", "libero-root", "b0-output-root", "output-root"):
        parser.add_argument(f"--{name}", type=lambda x: Path(x).expanduser().resolve(), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

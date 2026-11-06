"""Independent cross-file validation for a recorded B0 training-task rollout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.episode_recorder import (
    read_episode_metadata,
    sha256_array,
    sha256_file,
    validate_episode_file,
)
from eventground_wam.evaluation_protocol import protocol_sha256


class B0RolloutValidationError(ValueError):
    """Raised when a B0 report, episode, or source-provenance link disagrees."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise B0RolloutValidationError(message)


def _load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B0RolloutValidationError(f"could not load rollout report: {error}") from error
    _require(isinstance(report, dict), "rollout report must contain a JSON object")
    _require(report.get("digest") == contract_digest(report), "rollout report digest changed")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        recorded_sha256 = sidecar.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise B0RolloutValidationError(f"could not read rollout sidecar: {error}") from error
    _require(recorded_sha256 == sha256_file(path), "rollout report sidecar changed")
    return report


def _base_initial_states_path(bddl_path: Path, suite: str) -> Path:
    bddl_root = bddl_path.parent.parent
    if bddl_root.name != "bddl_files":
        raise B0RolloutValidationError("recorded BDDL path is not in a base task inventory")
    return bddl_root.parent / "init_files" / suite / f"{bddl_path.stem}.pruned_init"


def validate_b0_rollout_bundle(
    artifact_path: str | Path,
    report_path: str | Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Validate report, HDF5, source provenance, and action-row alignment."""

    artifact = Path(artifact_path).expanduser().resolve()
    report_file = Path(report_path).expanduser().resolve()
    report = _load_report(report_file)
    summary = validate_episode_file(artifact, protocol)
    metadata = read_episode_metadata(artifact)
    result = report.get("result", {})
    checkpoint = report.get("checkpoint", {})
    selection = report.get("selection", {})
    base_task = selection.get("base_task", {})

    _require(report.get("status") == "passed", "rollout report is not passed")
    _require(report.get("artifact_type") == "eventground_wam.b0_training_task_rollout", "unexpected rollout artifact type")
    _require(result.get("success") is True, "rollout report does not record success")
    _require(summary["success"] is True, "episode does not record success")
    _require(Path(result.get("artifact", "")).resolve() == artifact, "report points to a different episode")
    artifact_sha256 = sha256_file(artifact)
    _require(result.get("artifact_sha256") == artifact_sha256, "episode SHA-256 differs from report")
    _require(metadata["protocol_id"] == protocol["protocol_id"], "episode protocol ID changed")
    _require(metadata["protocol_sha256"] == protocol_sha256(protocol), "episode protocol hash changed")

    matching_fields = (
        "suite",
        "runtime_task_index",
        "perturbation_category",
        "difficulty_level",
        "success",
        "termination_reason",
    )
    for field in matching_fields:
        _require(result.get(field) == (metadata[field] if field in metadata else summary[field]), f"report and episode disagree on {field}")
    for field in ("suite", "runtime_task_index", "classification_task_id", "difficulty_level"):
        _require(base_task.get(field) == metadata[field], f"selection and episode disagree on {field}")
    _require(base_task.get("episode_seed") == metadata["episode_seed"], "selection episode seed changed")
    _require(selection.get("policy_instruction") == metadata["instruction"], "policy instruction changed")
    _require(checkpoint.get("checkpoint_id") == metadata["checkpoint_id"], "checkpoint ID changed")
    _require(checkpoint.get("checkpoint_revision") == metadata["checkpoint_revision"], "checkpoint revision changed")

    bddl_path = Path(metadata["bddl_path"])
    _require(bddl_path.is_file(), "recorded BDDL source is missing")
    _require(sha256_file(bddl_path) == metadata["bddl_sha256"], "recorded BDDL hash changed")
    tasks_info = selection.get("tasks_info", {})
    tasks_info_path = Path(tasks_info.get("path", ""))
    _require(tasks_info_path.is_file(), "recorded tasks_info inventory is missing")
    _require(sha256_file(tasks_info_path) == tasks_info.get("file_sha256"), "tasks_info inventory hash changed")
    initial_states_path = _base_initial_states_path(bddl_path, metadata["suite"])
    _require(initial_states_path.is_file(), "recorded base-task initial states are missing")
    initial_states = torch.load(initial_states_path, weights_only=False)
    state_index = protocol["benchmark"]["initial_state"]["state_index"]
    initial_state = np.asarray(initial_states[state_index])
    _require(sha256_array(initial_state) == metadata["initial_state_sha256"], "initial state hash changed")

    settling_steps = summary["settling_step_count"]
    with h5py.File(artifact, "r") as episode:
        steps = episode["steps"]
        chunk_indexes = steps["action_chunk_index"][settling_steps:]
        predicted = steps["predicted_action_chunk"][settling_steps:]
        executed = steps["executed_action"][settling_steps:]
        selected_predictions = predicted[np.arange(len(predicted)), chunk_indexes]
        _require(np.array_equal(selected_predictions, executed), "executed actions do not match their predicted chunk rows")
        _require(np.all(np.isin(predicted[:, :, 6], (-1.0, 1.0))), "predicted gripper commands are not binary")
        _require(np.all(np.isin(executed[:, 6], (-1.0, 1.0))), "executed gripper commands are not binary")
        latencies = steps["policy_latency_ms"][settling_steps:]
        finite_latencies = latencies[np.isfinite(latencies)]
        rewards = steps["reward"][:]
        done = steps["done"][:]

    recorded_latencies = np.asarray(result.get("policy_latencies_ms", []), dtype=np.float64)
    _require(np.array_equal(finite_latencies, recorded_latencies), "HDF5 and report latency samples differ")
    _require(bool(done[-1]) and not np.any(done[:-1]), "success signal is not terminal and unique")
    _require(float(rewards[-1]) > 0.0, "successful final transition has no positive reward")
    inference = report.get("inference", {})
    query_count = len(finite_latencies)
    chunk_length = protocol["simulation"]["action"]["chunk_length"]
    action_dimension = protocol["simulation"]["action"]["dimension"]
    _require(inference.get("prediction_count") == query_count, "prediction count differs from HDF5 query count")
    _require(inference.get("decoded_action_count") == query_count * chunk_length, "decoded action count differs from query count")
    _require(inference.get("continuous_value_count") == query_count * chunk_length * (action_dimension - 1), "continuous value count differs from decoded chunks")

    return {
        "status": "passed",
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256,
        "report": str(report_file),
        "report_file_sha256": sha256_file(report_file),
        "report_digest": report["digest"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256(protocol),
        "step_count": summary["step_count"],
        "settling_step_count": settling_steps,
        "policy_step_count": summary["policy_step_count"],
        "policy_query_count": query_count,
        "executed_action_rows_match_predictions": True,
        "binary_gripper_contract_passed": True,
        "terminal_success_signal_passed": True,
        "source_provenance_passed": True,
    }

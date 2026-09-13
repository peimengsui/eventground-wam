from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eventground_wam.b0_rollout_validation import (
    B0RolloutValidationError,
    validate_b0_rollout_bundle,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import (
    EpisodeMetadata,
    EpisodeRecorder,
    StepRecord,
    sha256_array,
    sha256_file,
)
from eventground_wam.evaluation_protocol import (
    B0_TRAINING_ROLLOUT_PROTOCOL_PATH,
    derive_episode_seed,
    load_evaluation_protocol,
    protocol_sha256,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    protocol = load_evaluation_protocol(B0_TRAINING_ROLLOUT_PROTOCOL_PATH)
    source_root = tmp_path / "libero"
    bddl_root = source_root / "bddl_files" / "libero_goal"
    init_root = source_root / "init_files" / "libero_goal"
    bddl_root.mkdir(parents=True)
    init_root.mkdir(parents=True)
    task_name = "put_the_bowl_on_the_plate"
    bddl_path = bddl_root / f"{task_name}.bddl"
    bddl_path.write_text("task\n", encoding="utf-8")
    tasks_info_path = bddl_root / "tasks_info.txt"
    tasks_info_path.write_text(
        f"libero/bddl_files/libero_goal/{task_name}.bddl\n",
        encoding="utf-8",
    )
    initial_states = np.arange(24, dtype=np.float64).reshape(2, 12)
    torch.save(initial_states, init_root / f"{task_name}.pruned_init")
    metadata = EpisodeMetadata(
        protocol_id=protocol["protocol_id"],
        protocol_sha256=protocol_sha256(protocol),
        project_git_revision="a" * 40,
        benchmark_revision=protocol["benchmark"]["revision"],
        checkpoint_id="b0-test",
        checkpoint_revision="b" * 64,
        suite="libero_goal",
        runtime_task_index=4,
        classification_task_id=5,
        perturbation_category="Training Cohort",
        difficulty_level=None,
        instruction="put the bowl on the plate",
        bddl_path=str(bddl_path),
        bddl_sha256=sha256_file(bddl_path),
        initial_state_sha256=sha256_array(initial_states[0]),
        episode_seed=derive_episode_seed(protocol, "libero_goal", 4),
    )
    artifact_path = tmp_path / "episode.hdf5"
    with EpisodeRecorder(artifact_path, protocol, metadata) as recorder:
        for step_index in range(13):
            policy_step = step_index >= 10
            chunk = np.zeros((8, 7), dtype=np.float64) if policy_step else None
            if chunk is not None:
                chunk[:, 6] = 1.0
            action = (
                chunk[step_index - 10]
                if chunk is not None
                else np.asarray(protocol["simulation"]["settling_action"])
            )
            done = step_index == 12
            recorder.append(
                StepRecord(
                    step_index=step_index,
                    phase="policy" if policy_step else "settling",
                    sim_time_seconds=step_index / 20,
                    agent_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                    wrist_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                    proprioception=np.zeros(8),
                    predicted_action_chunk=chunk,
                    executed_action=action,
                    action_chunk_index=step_index - 10 if policy_step else -1,
                    mujoco_state=np.zeros(12),
                    reward=1.0 if done else 0.0,
                    done=done,
                    policy_latency_ms=1.25 if step_index == 10 else None,
                )
            )
        recorder.finalize(success=True, termination_reason="environment_done")

    report = seal_contract(
        {
            "artifact_type": "eventground_wam.b0_training_task_rollout",
            "status": "passed",
            "result": {
                "artifact": str(artifact_path),
                "artifact_sha256": sha256_file(artifact_path),
                "suite": "libero_goal",
                "runtime_task_index": 4,
                "perturbation_category": "Training Cohort",
                "difficulty_level": None,
                "success": True,
                "termination_reason": "environment_done",
                "policy_latencies_ms": [1.25],
            },
            "checkpoint": {
                "checkpoint_id": "b0-test",
                "checkpoint_revision": "b" * 64,
            },
            "selection": {
                "policy_instruction": "put the bowl on the plate",
                "base_task": {
                    "suite": "libero_goal",
                    "runtime_task_index": 4,
                    "classification_task_id": 5,
                    "difficulty_level": None,
                    "episode_seed": derive_episode_seed(protocol, "libero_goal", 4),
                },
                "tasks_info": {
                    "path": str(tasks_info_path),
                    "file_sha256": sha256_file(tasks_info_path),
                },
            },
            "inference": {
                "prediction_count": 1,
                "decoded_action_count": 8,
                "continuous_value_count": 48,
            },
        }
    )
    report_path = tmp_path / "training-rollout-report.json"
    report_payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(report_payload, encoding="utf-8")
    report_path.with_suffix(".json.sha256").write_text(
        hashlib.sha256(report_payload.encode()).hexdigest() + "\n",
        encoding="utf-8",
    )
    return artifact_path, report_path, protocol


def test_b0_rollout_bundle_cross_validates_every_identity_chain(tmp_path: Path) -> None:
    artifact, report, protocol = _write_fixture(tmp_path)

    result = validate_b0_rollout_bundle(artifact, report, protocol)

    assert result["status"] == "passed"
    assert result["policy_step_count"] == 3
    assert result["policy_query_count"] == 1
    assert result["executed_action_rows_match_predictions"] is True
    assert result["source_provenance_passed"] is True


def test_b0_rollout_bundle_rejects_a_tampered_report(tmp_path: Path) -> None:
    artifact, report, protocol = _write_fixture(tmp_path)
    value = json.loads(report.read_text(encoding="utf-8"))
    value["result"]["success"] = False
    report.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(B0RolloutValidationError, match="digest changed"):
        validate_b0_rollout_bundle(artifact, report, protocol)

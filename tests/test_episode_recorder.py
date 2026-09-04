from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from eventground_wam.episode_recorder import (
    EpisodeMetadata,
    EpisodeRecorder,
    EpisodeValidationError,
    StepRecord,
    read_episode_metadata,
    validate_episode_file,
)
from eventground_wam.evaluation_protocol import (
    derive_episode_seed,
    load_evaluation_protocol,
    protocol_sha256,
)


@pytest.fixture()
def protocol() -> dict:
    return load_evaluation_protocol()


def episode_metadata(protocol: dict) -> EpisodeMetadata:
    return EpisodeMetadata(
        protocol_id=protocol["protocol_id"],
        protocol_sha256=protocol_sha256(protocol),
        project_git_revision="a" * 40,
        benchmark_revision=protocol["benchmark"]["revision"],
        checkpoint_id="test-zero-policy",
        checkpoint_revision="test-zero-policy-v1",
        suite="libero_spatial",
        runtime_task_index=0,
        classification_task_id=1,
        perturbation_category="Background Textures",
        difficulty_level=1,
        instruction="pick up the black bowl",
        bddl_path="/benchmark/task.bddl",
        bddl_sha256="b" * 64,
        initial_state_sha256="c" * 64,
        episode_seed=derive_episode_seed(protocol, "libero_spatial", 0),
    )


def step_record(
    protocol: dict,
    step_index: int,
    *,
    policy: bool,
    done: bool = False,
) -> StepRecord:
    phase = "policy" if policy else "settling"
    if policy:
        action = np.zeros(7, dtype=np.float32)
        predicted_chunk = np.zeros((8, 7), dtype=np.float32)
        chunk_index = step_index - protocol["simulation"]["settling_steps"]
        chunk_index %= protocol["simulation"]["action"]["chunk_length"]
        latency = 1.25 if chunk_index == 0 else None
    else:
        action = np.asarray(protocol["simulation"]["settling_action"], dtype=np.float32)
        predicted_chunk = None
        chunk_index = -1
        latency = None
    image = np.full((256, 256, 3), step_index, dtype=np.uint8)
    return StepRecord(
        step_index=step_index,
        phase=phase,
        sim_time_seconds=step_index / protocol["simulation"]["control_frequency_hz"],
        agent_rgb=image,
        wrist_rgb=image,
        proprioception=np.arange(8, dtype=np.float64),
        predicted_action_chunk=predicted_chunk,
        executed_action=action,
        action_chunk_index=chunk_index,
        mujoco_state=np.arange(12, dtype=np.float64),
        reward=1.0 if done else 0.0,
        done=done,
        policy_latency_ms=latency,
    )


def test_recorder_publishes_only_a_valid_complete_episode(
    tmp_path: Path, protocol: dict
) -> None:
    output_path = tmp_path / "episode.hdf5"
    recorder = EpisodeRecorder(
        output_path,
        protocol,
        episode_metadata(protocol),
        flush_interval=1,
    )
    settling_steps = protocol["simulation"]["settling_steps"]
    for step_index in range(settling_steps + 3):
        recorder.append(
            step_record(
                protocol,
                step_index,
                policy=step_index >= settling_steps,
                done=step_index == settling_steps + 2,
            )
        )

    finalized_path = recorder.finalize(success=True, termination_reason="environment_done")
    summary = validate_episode_file(finalized_path, protocol)

    assert finalized_path == output_path
    assert output_path.is_file()
    assert not Path(str(output_path) + ".partial").exists()
    assert summary["step_count"] == 13
    assert summary["settling_step_count"] == 10
    assert summary["policy_step_count"] == 3
    assert summary["success"] is True
    assert read_episode_metadata(output_path)["termination_reason"] == "environment_done"
    with h5py.File(output_path, "r") as artifact:
        assert artifact.attrs["status"] == "complete"
        assert artifact["steps/agent_rgb"].compression == "gzip"
        assert artifact["steps/mujoco_state"].dtype == np.float64
        assert np.isnan(artifact["steps/policy_latency_ms"][1])
        assert artifact["steps/policy_latency_ms"][10] == pytest.approx(1.25)


def test_context_exception_retains_an_aborted_partial_artifact(
    tmp_path: Path, protocol: dict
) -> None:
    output_path = tmp_path / "episode.hdf5"

    with (
        pytest.raises(RuntimeError, match="policy failed"),
        EpisodeRecorder(output_path, protocol, episode_metadata(protocol)) as recorder,
    ):
        recorder.append(step_record(protocol, 0, policy=False))
        raise RuntimeError("policy failed")

    partial_path = Path(str(output_path) + ".partial")
    assert not output_path.exists()
    assert partial_path.is_file()
    with h5py.File(partial_path, "r") as artifact:
        assert artifact.attrs["status"] == "aborted"
        assert artifact.attrs["step_count"] == 1
    metadata = read_episode_metadata(partial_path)
    assert metadata["success"] is False
    assert metadata["termination_reason"] == "exception:RuntimeError"
    summary = validate_episode_file(partial_path, protocol, allow_incomplete=True)
    assert summary["status"] == "aborted"
    assert summary["settling_step_count"] == 1


def test_recorder_rejects_step_index_gaps(tmp_path: Path, protocol: dict) -> None:
    output_path = tmp_path / "episode.hdf5"

    with (
        EpisodeRecorder(output_path, protocol, episode_metadata(protocol)) as recorder,
        pytest.raises(EpisodeValidationError, match="Expected step_index=0"),
    ):
        recorder.append(step_record(protocol, 1, policy=False))


def test_failed_final_validation_never_publishes_output(
    tmp_path: Path, protocol: dict
) -> None:
    output_path = tmp_path / "episode.hdf5"
    recorder = EpisodeRecorder(output_path, protocol, episode_metadata(protocol))
    recorder.append(step_record(protocol, 0, policy=False))

    with pytest.raises(EpisodeValidationError, match="every settling step"):
        recorder.finalize(success=False, termination_reason="test_short_episode")

    assert not output_path.exists()
    with h5py.File(Path(str(output_path) + ".partial"), "r") as artifact:
        assert artifact.attrs["status"] == "invalid"


def test_success_during_settling_can_stop_the_episode(
    tmp_path: Path, protocol: dict
) -> None:
    output_path = tmp_path / "already-successful.hdf5"
    recorder = EpisodeRecorder(output_path, protocol, episode_metadata(protocol))
    recorder.append(step_record(protocol, 0, policy=False, done=True))

    recorder.finalize(success=True, termination_reason="environment_done")

    summary = validate_episode_file(output_path, protocol)
    assert summary["settling_step_count"] == 1
    assert summary["policy_step_count"] == 0

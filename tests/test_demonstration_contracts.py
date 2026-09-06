from __future__ import annotations

import json

import pytest

from eventground_wam.demonstration_contracts import (
    DemonstrationContractError,
    EventCandidate,
    EventInterval,
    MultiHorizonSample,
    Provenance,
    SplitAssignment,
    TrajectoryRecord,
    canonical_json_bytes,
    contract_digest,
    seal_contract,
    validate_contract,
    validate_contract_sequence,
    validate_sample_references,
    validate_trajectory_reference,
)


@pytest.fixture()
def provenance() -> Provenance:
    return Provenance(
        source_episode_index=1_234,
        source_task_index=7,
        source_episode_length=100,
        project_git_revision="a" * 40,
        annotation_version="week2-v1",
    )


@pytest.fixture()
def trajectory(provenance: Provenance) -> TrajectoryRecord:
    return TrajectoryRecord.from_episode(provenance, "put the red mug in the cabinet")


@pytest.fixture()
def candidate(provenance: Provenance, trajectory: TrajectoryRecord) -> EventCandidate:
    return EventCandidate(
        provenance=provenance,
        trajectory_digest=trajectory.digest,
        frame_start=20,
        frame_end=25,
        event_type="contact",
        confidence=0.95,
        evidence=("front_rgb:object-hand overlap", "state:end-effector motion"),
        detector_name="contact-detector",
        detector_version="v1",
    )


@pytest.fixture()
def event(
    provenance: Provenance,
    trajectory: TrajectoryRecord,
    candidate: EventCandidate,
) -> EventInterval:
    return EventInterval(
        provenance=provenance,
        trajectory_digest=trajectory.digest,
        frame_start=20,
        frame_end=25,
        event_type="contact",
        confidence=0.93,
        evidence=("front_rgb:object-hand overlap",),
        candidate_digests=(candidate.digest,),
        ontology_tier="reliable",
        review_status="accepted",
    )


@pytest.fixture()
def split(provenance: Provenance, trajectory: TrajectoryRecord) -> SplitAssignment:
    return SplitAssignment(
        provenance=provenance,
        trajectory_digest=trajectory.digest,
        frame_start=0,
        frame_end=100,
        split="train",
        group_strategy="task-state-rgb-surrogate-v1",
        initial_condition_digest="b" * 64,
    )


def test_trajectory_contract_locks_source_shape_paths_and_absent_metadata(
    trajectory: TrajectoryRecord,
) -> None:
    record = trajectory.to_dict()

    assert record["parquet_path"] == "data/chunk-001/episode_001234.parquet"
    assert record["front_video_path"].endswith("episode_001234.mp4")
    assert record["fps"] == 20
    assert record["state_dimension"] == 8
    assert record["action_dimension"] == 7
    assert record["camera_keys"] == [
        "observation.images.front",
        "observation.images.wrist",
    ]
    assert record["provenance"]["source_initial_seed"] is None
    assert record["provenance"]["source_has_mujoco_state"] is False
    assert json.loads(canonical_json_bytes(record))["digest"] == trajectory.digest


def test_canonical_digest_is_order_independent_and_covers_nested_data() -> None:
    first = {"z": [1, {"value": "x"}], "a": True}
    second = {"a": True, "z": [1, {"value": "x"}]}

    assert contract_digest(first) == contract_digest(second)
    assert seal_contract(first)["digest"] == contract_digest(second)

    changed = {"z": [1, {"value": "y"}], "a": True}
    assert contract_digest(first) != contract_digest(changed)


def test_serialized_contract_rejects_extra_fields_and_digest_tampering(
    trajectory: TrajectoryRecord,
) -> None:
    extra = trajectory.to_dict()
    extra["unexpected"] = "silently accepting this would change the schema"
    with pytest.raises(DemonstrationContractError, match="keys differ"):
        validate_contract(extra)

    tampered = trajectory.to_dict()
    tampered["instruction"] = "a different task"
    with pytest.raises(DemonstrationContractError, match="digest mismatch"):
        validate_contract(tampered)


def test_object_relative_candidate_requires_object_evidence(
    provenance: Provenance,
    trajectory: TrajectoryRecord,
) -> None:
    with pytest.raises(DemonstrationContractError, match="object-relative"):
        EventCandidate(
            provenance=provenance,
            trajectory_digest=trajectory.digest,
            frame_start=20,
            frame_end=25,
            event_type="grasp_established",
            confidence=0.9,
            evidence=("action:gripper closed", "state:end-effector stationary"),
            detector_name="grasp-detector",
            detector_version="v1",
        )


def test_gripper_close_can_be_supported_by_state_and_action_only(
    provenance: Provenance,
    trajectory: TrajectoryRecord,
) -> None:
    proposal = EventCandidate(
        provenance=provenance,
        trajectory_digest=trajectory.digest,
        frame_start=10,
        frame_end=12,
        event_type="gripper_close",
        confidence=1.0,
        evidence=("action:gripper command transition", "state:gripper transition"),
        detector_name="gripper-transition",
        detector_version="v1",
    )

    validate_trajectory_reference(proposal, trajectory)


def test_foreign_key_validation_detects_wrong_trajectory(
    candidate: EventCandidate,
    provenance: Provenance,
) -> None:
    wrong_trajectory = TrajectoryRecord.from_episode(provenance, "a different instruction")

    with pytest.raises(DemonstrationContractError, match="does not identify"):
        validate_trajectory_reference(candidate, wrong_trajectory)


def test_fixed_horizon_sample_is_within_episode_and_matches_split(
    provenance: Provenance,
    trajectory: TrajectoryRecord,
    split: SplitAssignment,
) -> None:
    sample = MultiHorizonSample(
        provenance=provenance,
        trajectory_digest=trajectory.digest,
        split_assignment_digest=split.digest,
        frame_start=40,
        frame_end=49,
        anchor_frame_index=40,
        target_frame_index=48,
        horizon=8,
        split="train",
    )

    validate_trajectory_reference(sample, trajectory)
    validate_sample_references(sample, split)

    with pytest.raises(DemonstrationContractError, match="episode boundary"):
        MultiHorizonSample(
            provenance=provenance,
            trajectory_digest=trajectory.digest,
            split_assignment_digest=split.digest,
            frame_start=96,
            frame_end=105,
            anchor_frame_index=96,
            target_frame_index=104,
            horizon=8,
            split="train",
        )


def test_sample_references_reject_a_split_for_another_trajectory(
    provenance: Provenance,
    trajectory: TrajectoryRecord,
) -> None:
    other_split = SplitAssignment(
        provenance=provenance,
        trajectory_digest="c" * 64,
        frame_start=0,
        frame_end=100,
        split="train",
        group_strategy="task-state-rgb-surrogate-v1",
        initial_condition_digest="b" * 64,
    )
    sample = MultiHorizonSample(
        provenance=provenance,
        trajectory_digest=trajectory.digest,
        split_assignment_digest=other_split.digest,
        frame_start=40,
        frame_end=49,
        anchor_frame_index=40,
        target_frame_index=48,
        horizon=8,
        split="train",
    )

    with pytest.raises(DemonstrationContractError, match="different trajectories"):
        validate_sample_references(sample, other_split)


def test_next_event_sample_targets_event_start(
    provenance: Provenance,
    trajectory: TrajectoryRecord,
    split: SplitAssignment,
    event: EventInterval,
) -> None:
    sample = MultiHorizonSample(
        provenance=provenance,
        trajectory_digest=trajectory.digest,
        split_assignment_digest=split.digest,
        frame_start=5,
        frame_end=21,
        anchor_frame_index=5,
        target_frame_index=20,
        horizon="next-event",
        split="train",
        target_event_digest=event.digest,
    )

    validate_sample_references(sample, split, target_event=event)

    wrong_boundary = sample.to_dict()
    wrong_boundary["target_frame_index"] = 19
    wrong_boundary["frame_end"] = 20
    wrong_boundary["digest"] = contract_digest(wrong_boundary)
    with pytest.raises(DemonstrationContractError, match="first frame"):
        validate_sample_references(wrong_boundary, split, target_event=event)


def test_contract_sequence_rejects_duplicate_records(trajectory: TrajectoryRecord) -> None:
    serialized = trajectory.to_dict()

    with pytest.raises(DemonstrationContractError, match="duplicate digest"):
        validate_contract_sequence([serialized, serialized])


def test_provenance_rejects_seed_or_mujoco_claims_missing_from_source() -> None:
    with pytest.raises(DemonstrationContractError, match="source_initial_seed"):
        Provenance(
            source_episode_index=0,
            source_task_index=0,
            source_episode_length=20,
            project_git_revision="a" * 40,
            annotation_version="week2-v1",
            source_initial_seed=123,
        )

    with pytest.raises(DemonstrationContractError, match="source_has_mujoco_state"):
        Provenance(
            source_episode_index=0,
            source_task_index=0,
            source_episode_length=20,
            project_git_revision="a" * 40,
            annotation_version="week2-v1",
            source_has_mujoco_state=True,
        )

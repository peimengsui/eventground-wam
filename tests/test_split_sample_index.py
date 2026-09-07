from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import (
    EventInterval,
    Provenance,
    SplitAssignment,
    TrajectoryRecord,
    contract_digest,
    validate_contract_sequence,
    validate_sample_references,
    validate_trajectory_reference,
)
from eventground_wam.demonstration_reader import DecodedVideoFrame
from eventground_wam.split_sample_index import (
    GROUP_STRATEGY,
    EpisodeSampleSource,
    InitialConditionEvidence,
    SplitRatios,
    SplitSampleIndexError,
    assign_group_split,
    decoded_rgb_fingerprint,
    initial_condition_digest,
    iter_episode_samples,
    write_split_sample_indexes,
)

PROJECT_REVISION = "a" * 40


def _trajectory(
    episode_index: int,
    *,
    task_index: int = 3,
    length: int = 7,
) -> TrajectoryRecord:
    provenance = Provenance(
        source_episode_index=episode_index,
        source_task_index=task_index,
        source_episode_length=length,
        project_git_revision=PROJECT_REVISION,
        annotation_version="week2-events-v1",
    )
    return TrajectoryRecord.from_episode(provenance, f"instruction {task_index}")


def _evidence(
    trajectory: TrajectoryRecord,
    *,
    state: tuple[float, ...] | None = None,
    front_digest: str = "b" * 64,
    wrist_digest: str = "c" * 64,
) -> InitialConditionEvidence:
    provenance = trajectory.provenance
    return InitialConditionEvidence(
        source_episode_index=provenance.source_episode_index,
        source_task_index=provenance.source_task_index,
        initial_state=state or tuple(float(index) / 10 for index in range(8)),
        front_frame_sha256=front_digest,
        wrist_frame_sha256=wrist_digest,
    )


def _event(
    trajectory: TrajectoryRecord,
    frame_start: int,
    *,
    candidate_digit: str = "d",
    evidence_suffix: str = "transition",
    ontology_tier: str = "reliable",
    review_status: str = "accepted",
) -> EventInterval:
    return EventInterval(
        provenance=trajectory.provenance,
        trajectory_digest=trajectory.digest,
        frame_start=frame_start,
        frame_end=frame_start + 1,
        event_type="gripper_close",
        confidence=0.95,
        evidence=(f"action:{evidence_suffix}",),
        candidate_digests=(candidate_digit * 64,),
        ontology_tier=ontology_tier,
        review_status=review_status,
    )


def _assignment(trajectory: TrajectoryRecord) -> SplitAssignment:
    return SplitAssignment(
        provenance=trajectory.provenance,
        trajectory_digest=trajectory.digest,
        frame_start=0,
        frame_end=trajectory.frame_end,
        split="train",
        group_strategy=GROUP_STRATEGY,
        initial_condition_digest="e" * 64,
    )


def test_initial_condition_group_uses_task_float32_state_and_both_frames() -> None:
    trajectory = _trajectory(4)
    rounded = struct.unpack(">f", struct.pack(">f", 0.1))[0]
    first = _evidence(
        trajectory,
        state=(0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    same_group = InitialConditionEvidence(
        source_episode_index=99,
        source_task_index=trajectory.provenance.source_task_index,
        initial_state=(rounded, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        front_frame_sha256=first.front_frame_sha256,
        wrist_frame_sha256=first.wrist_frame_sha256,
    )
    different_task = InitialConditionEvidence(
        source_episode_index=99,
        source_task_index=4,
        initial_state=same_group.initial_state,
        front_frame_sha256=first.front_frame_sha256,
        wrist_frame_sha256=first.wrist_frame_sha256,
    )
    different_wrist = InitialConditionEvidence(
        source_episode_index=99,
        source_task_index=3,
        initial_state=same_group.initial_state,
        front_frame_sha256=first.front_frame_sha256,
        wrist_frame_sha256="f" * 64,
    )

    assert initial_condition_digest(first) == initial_condition_digest(same_group)
    assert initial_condition_digest(first) != initial_condition_digest(different_task)
    assert initial_condition_digest(first) != initial_condition_digest(different_wrist)


def test_group_split_is_repeatable_and_ratios_are_strict() -> None:
    digest = hashlib.sha256(b"group").hexdigest()

    assert assign_group_split(digest) == assign_group_split(digest)
    assert assign_group_split(digest, namespace="alternate") in {
        "train",
        "validation",
        "test",
    }
    with pytest.raises(SplitSampleIndexError, match="sum to one"):
        SplitRatios(train=0.7, validation=0.2, test=0.2)
    with pytest.raises(SplitSampleIndexError, match="between zero and one"):
        SplitRatios(train=0.9, validation=0.1, test=0.0)


class _FakeFormat:
    name = "rgb24"


class _FakePlane:
    line_size = 9

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __bytes__(self) -> bytes:
        return self._payload


class _FakeVideoFrame:
    width = 3
    height = 2
    format = _FakeFormat()

    def __init__(self, payload: bytes) -> None:
        self.planes = (_FakePlane(payload),)

    def reformat(self, *, format: str):
        assert format == "rgb24"
        return self


def test_decoded_rgb_fingerprint_covers_dimensions_and_pixels() -> None:
    frame = DecodedVideoFrame(
        frame_index=0,
        timestamp=0.0,
        width=3,
        height=2,
        pixel_format="yuv420p",
        payload=_FakeVideoFrame(bytes(range(18))),
    )
    changed = DecodedVideoFrame(
        frame_index=0,
        timestamp=0.0,
        width=3,
        height=2,
        pixel_format="yuv420p",
        payload=_FakeVideoFrame(bytes(range(17)) + b"x"),
    )

    assert decoded_rgb_fingerprint(frame) == decoded_rgb_fingerprint(frame)
    assert decoded_rgb_fingerprint(frame) != decoded_rgb_fingerprint(changed)


def test_multi_horizon_samples_are_complete_within_episode_and_target_next_events() -> None:
    trajectory = _trajectory(8, length=7)
    assignment = _assignment(trajectory)
    simultaneous = sorted(
        (
            _event(trajectory, 3, candidate_digit="d", evidence_suffix="first"),
            _event(trajectory, 3, candidate_digit="e", evidence_suffix="second"),
        ),
        key=lambda event: (event.frame_start, event.frame_end, event.event_type, event.digest),
    )
    last = _event(trajectory, 5, candidate_digit="f", evidence_suffix="last")
    events = (*simultaneous, last)

    samples = tuple(iter_episode_samples(trajectory, assignment, events))

    fixed = [sample for sample in samples if sample.horizon != "next-event"]
    next_event = [sample for sample in samples if sample.horizon == "next-event"]
    assert {horizon: sum(sample.horizon == horizon for sample in fixed) for horizon in (2, 4, 8, 16, 32)} == {
        2: 5,
        4: 3,
        8: 0,
        16: 0,
        32: 0,
    }
    assert len(next_event) == 8
    assert {(sample.anchor_frame_index, sample.target_frame_index) for sample in next_event} == {
        (0, 3),
        (1, 3),
        (2, 3),
        (3, 5),
        (4, 5),
    }
    assert all(0 <= sample.anchor_frame_index < sample.target_frame_index < 7 for sample in samples)


def test_weak_and_unreviewed_events_are_not_next_event_training_targets() -> None:
    trajectory = _trajectory(8, length=7)
    assignment = _assignment(trajectory)
    events = (
        _event(
            trajectory,
            2,
            candidate_digit="d",
            ontology_tier="weak",
            review_status="accepted",
        ),
        _event(
            trajectory,
            4,
            candidate_digit="e",
            review_status="unreviewed",
        ),
    )

    samples = tuple(iter_episode_samples(trajectory, assignment, events))

    assert all(sample.horizon != "next-event" for sample in samples)


def test_writer_is_deterministic_validates_foreign_keys_and_proves_no_overlap(
    tmp_path: Path,
) -> None:
    first_trajectory = _trajectory(1, length=7)
    second_trajectory = _trajectory(2, length=6)
    first_event = _event(first_trajectory, 3)
    excluded_event = _event(
        second_trajectory,
        2,
        candidate_digit="e",
        ontology_tier="weak",
        review_status="unreviewed",
    )
    sources = (
        EpisodeSampleSource(
            first_trajectory,
            _evidence(first_trajectory),
            (first_event,),
        ),
        EpisodeSampleSource(
            second_trajectory,
            _evidence(second_trajectory),
            (excluded_event,),
        ),
    )

    first_manifest = write_split_sample_indexes(
        sources,
        tmp_path / "first",
        project_git_revision=PROJECT_REVISION,
        project_tree_clean=True,
        annotation_version="week2-splits-v1",
        require_complete_source=False,
    )
    second_manifest = write_split_sample_indexes(
        sources,
        tmp_path / "second",
        project_git_revision=PROJECT_REVISION,
        project_tree_clean=True,
        annotation_version="week2-splits-v1",
        require_complete_source=False,
    )

    split_bytes = (tmp_path / "first/split-assignments.jsonl").read_bytes()
    sample_bytes = (tmp_path / "first/multi-horizon-samples.jsonl").read_bytes()
    assert split_bytes == (tmp_path / "second/split-assignments.jsonl").read_bytes()
    assert sample_bytes == (tmp_path / "second/multi-horizon-samples.jsonl").read_bytes()
    assert first_manifest == second_manifest
    assert first_manifest["digest"] == contract_digest(first_manifest)
    assert first_manifest["grouping"]["unique_group_count"] == 1
    assert first_manifest["grouping"]["cross_split_group_overlap_count"] == 0
    assert first_manifest["sampling"]["sample_counts_by_horizon"] == {
        "2": 9,
        "4": 5,
        "8": 0,
        "16": 0,
        "32": 0,
        "next-event": 3,
    }
    assert first_manifest["sampling"]["event_targets"] == {
        "eligibility": (
            "ontology_tier == reliable and review_status in {accepted, corrected}"
        ),
        "input_event_count": 2,
        "eligible_event_count": 1,
        "excluded_event_count": 1,
        "excluded_weak_event_count": 1,
        "excluded_unreviewed_event_count": 1,
    }
    assert first_manifest["outputs"]["split_assignments"]["sha256"] == hashlib.sha256(
        split_bytes
    ).hexdigest()
    assert first_manifest["outputs"]["multi_horizon_samples"][
        "sha256"
    ] == hashlib.sha256(sample_bytes).hexdigest()

    splits = [json.loads(line) for line in split_bytes.splitlines()]
    samples = [json.loads(line) for line in sample_bytes.splitlines()]
    trajectories = {
        first_trajectory.digest: first_trajectory,
        second_trajectory.digest: second_trajectory,
    }
    split_by_digest = {record["digest"]: record for record in splits}
    validate_contract_sequence(splits)
    validate_contract_sequence(samples)
    assert splits[0]["initial_condition_digest"] == splits[1]["initial_condition_digest"]
    assert splits[0]["split"] == splits[1]["split"]
    for split in splits:
        validate_trajectory_reference(split, trajectories[split["trajectory_digest"]])
    for sample in samples:
        split = split_by_digest[sample["split_assignment_digest"]]
        event = first_event if sample["horizon"] == "next-event" else None
        validate_trajectory_reference(sample, trajectories[sample["trajectory_digest"]])
        validate_sample_references(sample, split, target_event=event)


def test_writer_rejects_mismatched_identity_and_cleans_partial_outputs(
    tmp_path: Path,
) -> None:
    first = _trajectory(2)
    second = _trajectory(1)
    output_dir = tmp_path / "indexes"

    with pytest.raises(SplitSampleIndexError, match="strictly ordered"):
        write_split_sample_indexes(
            (
                EpisodeSampleSource(first, _evidence(first)),
                EpisodeSampleSource(second, _evidence(second)),
            ),
            output_dir,
            project_git_revision=PROJECT_REVISION,
            project_tree_clean=False,
            annotation_version="week2-splits-v1",
            require_complete_source=False,
        )

    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []


def test_writer_refuses_to_overwrite_any_member(tmp_path: Path) -> None:
    trajectory = _trajectory(1)
    output_dir = tmp_path / "indexes"
    output_dir.mkdir()
    split_path = output_dir / "split-assignments.jsonl"
    split_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        write_split_sample_indexes(
            (EpisodeSampleSource(trajectory, _evidence(trajectory)),),
            output_dir,
            project_git_revision=PROJECT_REVISION,
            project_tree_clean=False,
            annotation_version="week2-splits-v1",
            require_complete_source=False,
        )

    assert split_path.read_text(encoding="utf-8") == "existing\n"

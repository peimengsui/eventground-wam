from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import (
    EventCandidate,
    Provenance,
    TrajectoryRecord,
    contract_digest,
    validate_contract_sequence,
)
from eventground_wam.demonstration_reader import TrajectoryRow
from eventground_wam.event_extraction import (
    EVENT_OUTPUT_FILENAMES,
    EventExtractionConfig,
    EventExtractionError,
    extract_event_intervals,
    publish_event_extraction,
)


def _rows(
    apertures: list[float],
    commands: list[float],
    *,
    episode_index: int = 0,
    task_index: int = 0,
) -> tuple[TrajectoryRow, ...]:
    assert len(apertures) == len(commands)
    return tuple(
        TrajectoryRow(
            index=frame_index,
            episode_index=episode_index,
            frame_index=frame_index,
            task_index=task_index,
            timestamp=frame_index / 20,
            state=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, aperture / 2, -aperture / 2),
            action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, command),
        )
        for frame_index, (aperture, command) in enumerate(zip(apertures, commands, strict=True))
    )


def _trajectory(length: int, *, episode_index: int = 0) -> TrajectoryRecord:
    return TrajectoryRecord.from_episode(
        Provenance(
            source_episode_index=episode_index,
            source_task_index=0,
            source_episode_length=length,
            project_git_revision="a" * 40,
            annotation_version="week2-pilot-v1",
        ),
        "task 0",
    )


def _candidate(
    trajectory: TrajectoryRecord,
    *,
    frame_start: int = 2,
    frame_end: int = 8,
    confidence: float = 0.95,
    event_type: str = "gripper_close",
    detector_name: str = "gripper-close-telemetry",
) -> EventCandidate:
    evidence = (
        ("action:raw close transition", "state:raw aperture response")
        if event_type == "gripper_close"
        else ("front_rgb:reviewed object-relative boundary",)
    )
    return EventCandidate(
        provenance=trajectory.provenance,
        trajectory_digest=trajectory.digest,
        frame_start=frame_start,
        frame_end=frame_end,
        event_type=event_type,
        confidence=confidence,
        evidence=evidence,
        detector_name=detector_name,
        detector_version="v1",
    )


def _confirmed_signal() -> tuple[list[float], list[float]]:
    return (
        [0.08, 0.08, 0.08, 0.071, 0.052, 0.031, 0.012, 0.004, 0.004, 0.004],
        [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
    )


def test_confirmed_close_becomes_weak_unreviewed_interval_with_lineage() -> None:
    apertures, commands = _confirmed_signal()
    rows = _rows(apertures, commands)
    trajectory = _trajectory(len(rows))
    candidate = _candidate(trajectory)

    result = extract_event_intervals(rows, trajectory, [candidate])

    assert result.raw_candidates == (candidate,)
    assert len(result.intervals) == 1
    interval = result.intervals[0]
    assert (interval.frame_start, interval.frame_end) == (2, 8)
    assert interval.event_type == "gripper_close"
    assert interval.ontology_tier == "weak"
    assert interval.review_status == "unreviewed"
    assert interval.confidence == candidate.confidence
    assert interval.candidate_digests == (candidate.digest,)
    assert candidate.evidence[0] in interval.evidence
    assert any("centered-median" in evidence for evidence in interval.evidence)
    assert result.decisions[0].status == "admitted_weak"
    assert result.decisions[0].reason == "telemetry-close-filtered-pending-human-review"
    assert result.decisions[0].interval_digests == (interval.digest,)


def test_command_hysteresis_ignores_single_frame_chatter_and_marks_absence() -> None:
    rows = _rows([0.08] * 8, [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    trajectory = _trajectory(len(rows))

    result = extract_event_intervals(rows, trajectory, [])

    assert not result.intervals
    assert [(decision.status, decision.reason) for decision in result.decisions] == [
        ("absent", "no-persistent-close-command-or-raw-candidate")
    ]


def test_persistent_command_without_candidate_is_explicitly_ambiguous() -> None:
    apertures, commands = _confirmed_signal()
    result = extract_event_intervals(
        _rows(apertures, commands),
        _trajectory(len(apertures)),
        [],
    )

    assert not result.intervals
    assert result.decisions[0].status == "ambiguous"
    assert result.decisions[0].reason == "persistent-close-command-without-raw-candidate"


def test_close_run_below_minimum_duration_is_explicitly_ambiguous() -> None:
    rows = _rows(
        [0.08, 0.08, 0.07, 0.06, 0.05, 0.05, 0.05],
        [-1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0],
    )
    trajectory = _trajectory(len(rows))

    result = extract_event_intervals(rows, trajectory, [])

    assert not result.intervals
    assert result.decisions[0].status == "ambiguous"
    assert result.decisions[0].reason == "close-command-below-minimum-duration"


def test_smoothed_response_in_hysteresis_band_is_not_admitted() -> None:
    apertures = [0.08, 0.08, 0.08, 0.0799, 0.0797, 0.0795, 0.0793, 0.0793, 0.0793]
    commands = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    trajectory = _trajectory(len(apertures))

    result = extract_event_intervals(
        _rows(apertures, commands), trajectory, [_candidate(trajectory)]
    )

    assert not result.intervals
    assert result.decisions[0].status == "ambiguous"
    assert result.decisions[0].reason == "aperture-drop-in-hysteresis-band"


def test_low_confidence_and_unaligned_candidates_remain_raw_and_ambiguous() -> None:
    apertures, commands = _confirmed_signal()
    trajectory = _trajectory(len(apertures))
    low_confidence = _candidate(trajectory, confidence=0.8)

    low_result = extract_event_intervals(
        _rows(apertures, commands), trajectory, [low_confidence]
    )
    assert low_result.raw_candidates == (low_confidence,)
    assert not low_result.intervals
    assert low_result.decisions[0].reason == "candidate-confidence-below-admission-threshold"

    unaligned = _candidate(trajectory, frame_start=0, frame_end=6)
    unaligned_result = extract_event_intervals(
        _rows(apertures, commands), trajectory, [unaligned]
    )
    assert not unaligned_result.intervals
    assert {decision.reason for decision in unaligned_result.decisions} == {
        "persistent-close-command-without-raw-candidate",
        "raw-candidate-not-aligned-to-persistent-close-command",
    }


def test_object_relative_candidate_is_deferred_even_with_visual_evidence() -> None:
    rows = _rows([0.08] * 8, [-1.0] * 8)
    trajectory = _trajectory(len(rows))
    candidate = _candidate(
        trajectory,
        frame_start=2,
        frame_end=5,
        event_type="contact",
        detector_name="visual-contact",
    )

    result = extract_event_intervals(rows, trajectory, [candidate])

    assert result.raw_candidates == (candidate,)
    assert not result.intervals
    by_type = {decision.event_type: decision for decision in result.decisions}
    assert by_type["gripper_close"].status == "absent"
    assert by_type["contact"].status == "deferred"
    assert by_type["contact"].reason == "object-relative-detector-not-admitted"


def test_candidate_input_order_does_not_change_interval() -> None:
    apertures, commands = _confirmed_signal()
    trajectory = _trajectory(len(apertures))
    first = _candidate(trajectory, detector_name="detector-a")
    second = _candidate(trajectory, detector_name="detector-b")

    forward = extract_event_intervals(_rows(apertures, commands), trajectory, [first, second])
    reverse = extract_event_intervals(_rows(apertures, commands), trajectory, [second, first])

    assert [item.to_dict() for item in forward.intervals] == [
        item.to_dict() for item in reverse.intervals
    ]
    assert [item.to_dict() for item in forward.decisions] == [
        item.to_dict() for item in reverse.decisions
    ]


def test_interval_uses_derived_provenance_without_changing_raw_candidate() -> None:
    apertures, commands = _confirmed_signal()
    trajectory = _trajectory(len(apertures))
    candidate = _candidate(trajectory)
    derived = Provenance(
        source_episode_index=trajectory.provenance.source_episode_index,
        source_task_index=trajectory.provenance.source_task_index,
        source_episode_length=trajectory.provenance.source_episode_length,
        project_git_revision="b" * 40,
        annotation_version="week2-events-v1",
    )

    result = extract_event_intervals(
        _rows(apertures, commands),
        trajectory,
        [candidate],
        interval_provenance=derived,
    )

    assert result.raw_candidates == (candidate,)
    assert result.raw_candidates[0].provenance == trajectory.provenance
    assert result.intervals[0].provenance == derived


def test_duplicate_candidate_is_rejected() -> None:
    apertures, commands = _confirmed_signal()
    trajectory = _trajectory(len(apertures))
    candidate = _candidate(trajectory)

    with pytest.raises(EventExtractionError, match="duplicate candidate"):
        extract_event_intervals(
            _rows(apertures, commands), trajectory, [candidate, candidate]
        )


def test_atomic_publication_is_canonical_digest_sealed_and_non_overwriting(
    tmp_path: Path,
) -> None:
    apertures, commands = _confirmed_signal()
    trajectory = _trajectory(len(apertures))
    derived = Provenance(
        source_episode_index=trajectory.provenance.source_episode_index,
        source_task_index=trajectory.provenance.source_task_index,
        source_episode_length=trajectory.provenance.source_episode_length,
        project_git_revision="b" * 40,
        annotation_version="week2-events-v1",
    )
    result = extract_event_intervals(
        _rows(apertures, commands),
        trajectory,
        [_candidate(trajectory)],
        interval_provenance=derived,
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    manifest = publish_event_extraction(
        first_dir,
        [result],
        project_git_revision="b" * 40,
        project_tree_clean=True,
        annotation_version="week2-events-v1",
    )
    publish_event_extraction(
        second_dir,
        [result],
        project_git_revision="b" * 40,
        project_tree_clean=True,
        annotation_version="week2-events-v1",
    )

    assert manifest["digest"] == contract_digest(manifest)
    assert manifest["candidate_count"] == 1
    assert manifest["interval_count"] == 1
    assert manifest["decision_counts"]["admitted_weak"] == 1
    assert manifest["publishable"] is True
    for filename in EVENT_OUTPUT_FILENAMES:
        payload = (first_dir / filename).read_bytes()
        assert payload == (second_dir / filename).read_bytes()
        assert manifest["records_sha256"][filename] == hashlib.sha256(payload).hexdigest()
    interval_records = [
        json.loads(line)
        for line in (first_dir / "event-intervals.jsonl").read_text().splitlines()
    ]
    validate_contract_sequence(interval_records)
    assert not (tmp_path / "first.partial").exists()

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        publish_event_extraction(
            first_dir,
            [result],
            project_git_revision="b" * 40,
            project_tree_clean=True,
            annotation_version="week2-events-v1",
        )


def test_empty_publication_is_rejected_without_partial_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "empty"

    with pytest.raises(EventExtractionError, match="empty event extraction"):
        publish_event_extraction(
            output_dir,
            [],
            project_git_revision="b" * 40,
            project_tree_clean=False,
            annotation_version="week2-events-v1",
        )

    assert not output_dir.exists()
    assert not (tmp_path / "empty.partial").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"aperture_smoothing_window_frames": 2}, "positive and odd"),
        ({"command_hysteresis_frames": 0}, "must be positive"),
        (
            {"confirmed_aperture_drop": 0.001, "ambiguous_aperture_drop": 0.001},
            "must be below",
        ),
    ],
)
def test_invalid_extraction_config_is_rejected(
    kwargs: dict[str, float | int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        EventExtractionConfig(**kwargs)

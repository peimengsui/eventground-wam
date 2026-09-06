from __future__ import annotations

from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import Provenance, TrajectoryRecord
from eventground_wam.demonstration_reader import EpisodeRecord, TrajectoryRow
from eventground_wam.event_labeling import (
    EventLabelingError,
    GripperPolarityAudit,
    extract_gripper_close_candidates,
    review_item,
    select_task_stratified_episodes,
)


def _episode(episode_index: int, task_index: int, length: int) -> EpisodeRecord:
    root = Path("/dataset")
    return EpisodeRecord(
        episode_index=episode_index,
        task_index=task_index,
        instruction=f"task {task_index}",
        length=length,
        start_index=0,
        parquet_path=root / f"episode_{episode_index:06d}.parquet",
        video_paths=(
            ("observation.images.front", root / "front.mp4"),
            ("observation.images.wrist", root / "wrist.mp4"),
        ),
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


def _trajectory(length: int) -> TrajectoryRecord:
    provenance = Provenance(
        source_episode_index=0,
        source_task_index=0,
        source_episode_length=length,
        project_git_revision="a" * 40,
        annotation_version="week2-pilot-v1",
    )
    return TrajectoryRecord.from_episode(provenance, "task 0")


def test_task_stratification_selects_median_length_per_task() -> None:
    episodes = [
        _episode(0, 0, 30),
        _episode(1, 1, 40),
        _episode(2, 0, 10),
        _episode(3, 1, 20),
        _episode(4, 0, 20),
        _episode(5, 1, 30),
    ]

    selected = select_task_stratified_episodes(episodes)

    assert [(episode.task_index, episode.length) for episode in selected] == [(0, 20), (1, 30)]
    assert [episode.episode_index for episode in selected] == [4, 5]


def test_close_candidate_requires_command_persistence_and_aperture_response() -> None:
    apertures = [0.08, 0.08, 0.07, 0.05, 0.03, 0.01, 0.004, 0.004]
    commands = [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0]
    rows = _rows(apertures, commands)
    trajectory = _trajectory(len(rows))

    result = extract_gripper_close_candidates(rows, trajectory)

    assert result.summary.close_transition_count == 1
    assert result.summary.candidate_count == 1
    candidate = result.candidates[0]
    assert (candidate.frame_start, candidate.frame_end) == (1, 7)
    assert candidate.event_type == "gripper_close"
    assert candidate.confidence == 0.95
    assert candidate.evidence[0].startswith("action:")
    assert candidate.evidence[1].startswith("state:")


def test_short_or_weak_close_commands_are_retained_as_rejection_counts() -> None:
    short_rows = _rows(
        [0.08] * 8,
        [-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    )
    weak_rows = _rows(
        [0.08] * 8,
        [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
    )

    short = extract_gripper_close_candidates(short_rows, _trajectory(8))
    weak = extract_gripper_close_candidates(weak_rows, _trajectory(8))

    assert short.summary.rejected_short_command == 1
    assert weak.summary.rejected_weak_aperture_response == 1
    assert not short.candidates
    assert not weak.candidates


def test_polarity_audit_tracks_physical_response_to_both_commands() -> None:
    apertures = [
        0.08,
        0.08,
        0.06,
        0.04,
        0.02,
        0.01,
        0.004,
        0.004,
        0.02,
        0.04,
        0.06,
        0.075,
        0.08,
        0.08,
    ]
    commands = [-1.0] + [1.0] * 6 + [-1.0] * 7
    audit = GripperPolarityAudit()

    audit.update(_rows(apertures, commands))

    report = audit.to_dict()
    assert report["close_transition_count"] == 1
    assert report["close_response_rate"] == 1.0
    assert report["open_transition_count"] == 1
    assert report["open_response_rate"] == 1.0


def test_review_item_adds_bounded_context() -> None:
    rows = _rows(
        [0.08, 0.08, 0.07, 0.05, 0.03, 0.01, 0.004, 0.004],
        [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
    )
    trajectory = _trajectory(len(rows))
    candidate = extract_gripper_close_candidates(rows, trajectory).candidates[0]

    item = review_item(candidate, trajectory, context_frames=3)

    assert item["candidate_frame_range"] == [1, 7]
    assert item["review_frame_range"] == [0, 8]
    assert item["candidate_time_seconds"] == [0.05, 0.3]
    assert item["review_status"] == "pending"


def test_detector_rejects_unknown_gripper_encoding() -> None:
    rows = _rows([0.08, 0.08], [-1.0, 0.0])

    with pytest.raises(EventLabelingError, match="unsupported gripper command"):
        extract_gripper_close_candidates(rows, _trajectory(2))

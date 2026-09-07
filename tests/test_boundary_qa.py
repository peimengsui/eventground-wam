from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from eventground_wam.boundary_qa import (
    BOUNDARY_QA_REPORT_ARTIFACT_TYPE,
    BOUNDARY_REVIEW_ITEM_ARTIFACT_TYPE,
    BOUNDARY_REVIEW_OUTPUT_ARTIFACT_TYPE,
    BoundaryQAError,
    RGBFrame,
    aggregate_boundary_reviews,
    build_boundary_review_selection,
    pending_review_output,
    render_boundary_previews,
    validate_boundary_review_item,
    validate_review_output,
)
from eventground_wam.demonstration_contracts import (
    EventCandidate,
    Provenance,
    TrajectoryRecord,
    contract_digest,
)

PROJECT_REVISION = "a" * 40


def _trajectory(
    episode_index: int,
    *,
    task_index: int = 0,
    length: int = 100,
) -> TrajectoryRecord:
    provenance = Provenance(
        source_episode_index=episode_index,
        source_task_index=task_index,
        source_episode_length=length,
        project_git_revision=PROJECT_REVISION,
        annotation_version="week2-qa-v1",
    )
    return TrajectoryRecord.from_episode(provenance, f"task {task_index}")


def _event(
    trajectory: TrajectoryRecord,
    *,
    event_type: str = "gripper_close",
    frame_start: int = 20,
    frame_end: int = 25,
    discriminator: int = 0,
) -> EventCandidate:
    evidence = (
        ("state:gripper response" if event_type == "gripper_close" else "front_rgb:visual cue"),
    )
    return EventCandidate(
        provenance=trajectory.provenance,
        trajectory_digest=trajectory.digest,
        frame_start=frame_start,
        frame_end=frame_end,
        event_type=event_type,
        confidence=0.9,
        evidence=evidence,
        detector_name=f"qa-detector-{discriminator}",
        detector_version="v1",
    )


def _selection(
    trajectories: list[TrajectoryRecord],
    events: list[EventCandidate],
    *,
    target: int = 200,
    tile_size: int = 128,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    return build_boundary_review_selection(
        [trajectory.to_dict() for trajectory in trajectories],
        [event.to_dict() for event in events],
        target_episode_count=target,
        preview_tile_size=tile_size,
    )


def _completed_review(
    item: dict[str, Any],
    judgments: list[dict[str, Any]],
    *,
    missed_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "artifact_type": BOUNDARY_REVIEW_OUTPUT_ARTIFACT_TYPE,
        "schema_version": 1,
        "review_item_digest": item["digest"],
        "status": "completed",
        "reviewer_id": "reviewer-01",
        "reviewed_event_types": ["gripper_close"],
        "prediction_judgments": judgments,
        "missed_events": missed_events or [],
        "notes": "",
    }


def _judgment(
    event_digest: str,
    verdict: str,
    corrected_frame_range: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "event_digest": event_digest,
        "verdict": verdict,
        "corrected_frame_range": corrected_frame_range,
        "comment": "",
    }


def test_selection_fulfills_200_episode_target_deterministically() -> None:
    trajectories = [
        _trajectory(index, task_index=index % 4, length=60 + index % 15)
        for index in range(240)
    ]
    event_types = ("gripper_close", "contact", "success")
    events = [
        _event(
            trajectory,
            event_type=event_types[index % len(event_types)],
            discriminator=index,
        )
        for index, trajectory in enumerate(trajectories)
    ]

    items, manifest = _selection(trajectories, events)
    repeated_items, repeated_manifest = _selection(trajectories, events)

    assert items == repeated_items
    assert manifest == repeated_manifest
    assert len(items) == 200
    assert len({item["source_episode_index"] for item in items}) == 200
    assert manifest["requested_episode_count"] == 200
    assert manifest["available_eligible_episode_count"] == 240
    assert manifest["target_satisfied"] is True
    assert manifest["selection_shortfall"] == 0
    assert set(manifest["selected_task_counts"]) == {"0", "1", "2", "3"}
    assert set(manifest["selected_event_counts"]) == set(event_types)
    assert set(manifest["selected_length_stratum_counts"]) == {
        "short",
        "middle",
        "long",
    }
    assert all(item["artifact_type"] == BOUNDARY_REVIEW_ITEM_ARTIFACT_TYPE for item in items)


def test_selection_records_explicit_shortfall_when_fewer_than_200_are_eligible() -> None:
    trajectories = [_trajectory(index, task_index=index % 2) for index in range(5)]
    events = [_event(trajectory, discriminator=index) for index, trajectory in enumerate(trajectories)]

    items, manifest = _selection(trajectories, events)

    assert len(items) == 5
    assert manifest["target_satisfied"] is False
    assert manifest["selection_shortfall"] == 195
    assert manifest["human_quality_metrics_status"] == "unavailable_until_reviews_complete"


def test_selection_rejects_an_event_with_an_unknown_trajectory() -> None:
    trajectory = _trajectory(0)
    other = _trajectory(1)

    with pytest.raises(BoundaryQAError, match="unknown trajectory"):
        build_boundary_review_selection(
            [trajectory.to_dict()],
            [_event(other).to_dict()],
        )


def test_review_schema_requires_complete_judgments_for_reviewed_types() -> None:
    trajectory = _trajectory(0)
    events = [
        _event(trajectory, discriminator=0),
        _event(trajectory, frame_start=40, frame_end=45, discriminator=1),
    ]
    item = _selection([trajectory], events, target=1)[0][0]
    incomplete = _completed_review(
        item,
        [_judgment(item["predictions"][0]["event_digest"], "accepted")],
    )

    with pytest.raises(BoundaryQAError, match="must judge every prediction"):
        validate_review_output(incomplete, item)

    pending = pending_review_output(item)
    validate_boundary_review_item(item)
    validate_review_output(pending, item)


def test_completed_reviews_aggregate_precision_recall_boundaries_and_ambiguity() -> None:
    first = _trajectory(0)
    second = _trajectory(1)
    events = [
        _event(first, frame_start=10, frame_end=15, discriminator=0),
        _event(first, frame_start=20, frame_end=25, discriminator=1),
        _event(first, frame_start=30, frame_end=35, discriminator=2),
        _event(second, frame_start=40, frame_end=45, discriminator=3),
    ]
    items, _ = _selection([first, second], events, target=2)
    first_item, second_item = items
    first_review = _completed_review(
        first_item,
        [
            _judgment(first_item["predictions"][0]["event_digest"], "accepted"),
            _judgment(
                first_item["predictions"][1]["event_digest"],
                "corrected",
                [22, 24],
            ),
            _judgment(first_item["predictions"][2]["event_digest"], "rejected"),
        ],
        missed_events=[
            {"event_type": "gripper_close", "frame_range": [60, 65], "comment": "missed"}
        ],
    )
    second_review = _completed_review(
        second_item,
        [_judgment(second_item["predictions"][0]["event_digest"], "ambiguous")],
    )

    report = aggregate_boundary_reviews(items, [first_review, second_review])

    metrics = report["per_event"]["gripper_close"]
    assert report["artifact_type"] == BOUNDARY_QA_REPORT_ARTIFACT_TYPE
    assert report["status"] == "human_review_complete"
    assert report["human_quality_metrics_available"] is True
    assert metrics["true_positive_count"] == 2
    assert metrics["false_positive_count"] == 1
    assert metrics["false_negative_count"] == 1
    assert metrics["ambiguous_count"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["mean_absolute_start_boundary_error_frames"] == 1.0
    assert metrics["mean_absolute_end_boundary_error_frames"] == 0.5
    assert metrics["ambiguity_rate"] == 0.25
    assert metrics["reviewer_correction_count"] == 1
    assert report["digest"] == contract_digest(report)


def test_pending_reviews_never_produce_human_quality_metrics() -> None:
    trajectory = _trajectory(0)
    items, _ = _selection([trajectory], [_event(trajectory)], target=1)

    report = aggregate_boundary_reviews(items, [pending_review_output(items[0])])

    assert report["status"] == "pending_human_review"
    assert report["human_quality_metrics_available"] is False
    metrics = report["per_event"]["gripper_close"]
    assert metrics["true_positive_count"] is None
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["mean_absolute_start_boundary_error_frames"] is None
    assert metrics["ambiguity_rate"] is None


def test_renderer_writes_dual_camera_png_from_repository_relative_sources(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repository"
    trajectory = _trajectory(0, length=50)
    for relative in (trajectory.front_video_path, trajectory.wrist_video_path):
        source = repository_root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"test video placeholder")
    items, _ = _selection(
        [trajectory],
        [_event(trajectory, frame_start=20, frame_end=25)],
        target=1,
        tile_size=16,
    )
    requested_paths: list[Path] = []

    def fake_reader(path: Path, indices: list[int], tile_size: int) -> dict[int, RGBFrame]:
        requested_paths.append(path)
        return {
            index: RGBFrame(tile_size, tile_size, bytes([index % 256, 16, 32]) * tile_size**2)
            for index in indices
        }

    output = tmp_path / "qa-output"
    manifest = render_boundary_previews(
        items,
        repository_root=repository_root,
        output_directory=output,
        tile_size=16,
        frame_reader=fake_reader,
    )

    preview_path = output / items[0]["predictions"][0]["preview_path"]
    png = preview_path.read_bytes()
    frame_count = len(items[0]["predictions"][0]["sampled_frame_indices"])
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", png[16:24]) == (16 * frame_count, 32)
    assert len(requested_paths) == 2
    assert {path.name for path in requested_paths} == {"episode_000000.mp4"}
    assert manifest[0]["camera_rows"] == ["front", "wrist"]
    assert manifest[0]["overlay_legend"]["start_frame"] == "red_left_bar"

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        render_boundary_previews(
            items,
            repository_root=repository_root,
            output_directory=output,
            tile_size=16,
            frame_reader=fake_reader,
        )

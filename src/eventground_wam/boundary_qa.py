"""Deterministic boundary-review selection, previews, and human-QA metrics."""

from __future__ import annotations

import binascii
import re
import struct
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from eventground_wam.demonstration_contracts import (
    EVENT_CANDIDATE_ARTIFACT_TYPE,
    EVENT_INTERVAL_ARTIFACT_TYPE,
    EVENT_TYPES,
    TRAJECTORY_ARTIFACT_TYPE,
    contract_digest,
    seal_contract,
    validate_contract,
    validate_trajectory_reference,
)

BOUNDARY_REVIEW_ITEM_ARTIFACT_TYPE = "eventground_wam.boundary_review_item"
BOUNDARY_REVIEW_OUTPUT_ARTIFACT_TYPE = "eventground_wam.boundary_review_output"
BOUNDARY_QA_REPORT_ARTIFACT_TYPE = "eventground_wam.boundary_qa_report"
BOUNDARY_QA_SCHEMA_VERSION = 1
BOUNDARY_SELECTION_ALGORITHM = "greedy-task-event-length-coverage-v1"
LENGTH_STRATA = ("short", "middle", "long")
REVIEW_VERDICTS = ("accepted", "corrected", "rejected", "ambiguous")


class BoundaryQAError(ValueError):
    """Raised when boundary-QA inputs are incomplete or inconsistent."""


@dataclass(frozen=True)
class RGBFrame:
    """Packed RGB image used by the dependency-injectable preview renderer."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("RGB frame dimensions must be positive")
        if len(self.pixels) != self.width * self.height * 3:
            raise ValueError("RGB frame payload does not match its dimensions")


RGBFrameReader = Callable[[Path, Sequence[int], int], Mapping[int, RGBFrame]]


def build_boundary_review_selection(
    trajectories: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]],
    *,
    target_episode_count: int = 200,
    context_frames: int = 10,
    preview_tile_size: int = 128,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Select unique event-bearing episodes for balanced human boundary review.

    Selection greedily covers previously unseen task, event-type, and within-task
    trajectory-length strata, then balances the accumulated stratum loads. All
    tie-breaking uses the source episode index.
    """

    if target_episode_count < 1:
        raise ValueError("target_episode_count must be positive")
    if context_frames < 0:
        raise ValueError("context_frames cannot be negative")
    if preview_tile_size < 16:
        raise ValueError("preview_tile_size must be at least 16")

    trajectory_by_digest: dict[str, Mapping[str, Any]] = {}
    for trajectory in trajectories:
        validate_contract(trajectory)
        if trajectory["artifact_type"] != TRAJECTORY_ARTIFACT_TYPE:
            raise BoundaryQAError("trajectory input contains a non-trajectory contract")
        digest = str(trajectory["digest"])
        if digest in trajectory_by_digest:
            raise BoundaryQAError(f"duplicate trajectory digest: {digest}")
        trajectory_by_digest[digest] = trajectory
    if not trajectory_by_digest:
        raise BoundaryQAError("boundary review requires at least one trajectory")

    events_by_trajectory: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    event_digests: set[str] = set()
    supported_event_artifacts = {
        EVENT_CANDIDATE_ARTIFACT_TYPE,
        EVENT_INTERVAL_ARTIFACT_TYPE,
    }
    for event in events:
        validate_contract(event)
        if event["artifact_type"] not in supported_event_artifacts:
            raise BoundaryQAError("event input contains an unsupported contract type")
        event_digest = str(event["digest"])
        if event_digest in event_digests:
            raise BoundaryQAError(f"duplicate event digest: {event_digest}")
        event_digests.add(event_digest)
        trajectory_digest = str(event["trajectory_digest"])
        trajectory = trajectory_by_digest.get(trajectory_digest)
        if trajectory is None:
            raise BoundaryQAError(
                f"event {event_digest} references an unknown trajectory {trajectory_digest}"
            )
        validate_trajectory_reference(event, trajectory)
        events_by_trajectory[trajectory_digest].append(event)
    if not events_by_trajectory:
        raise BoundaryQAError("no event-bearing episodes are eligible for boundary review")

    eligible = [trajectory_by_digest[digest] for digest in events_by_trajectory]
    length_strata = _assign_length_strata(eligible)
    selected = _balanced_episode_selection(
        eligible,
        events_by_trajectory,
        length_strata,
        limit=min(target_episode_count, len(eligible)),
    )

    items: list[dict[str, Any]] = []
    for trajectory in sorted(
        selected,
        key=lambda item: int(item["provenance"]["source_episode_index"]),
    ):
        trajectory_digest = str(trajectory["digest"])
        episode_index = int(trajectory["provenance"]["source_episode_index"])
        episode_length = int(trajectory["provenance"]["source_episode_length"])
        predictions = []
        ordered_events = sorted(
            events_by_trajectory[trajectory_digest],
            key=lambda event: (
                int(event["frame_start"]),
                int(event["frame_end"]),
                str(event["event_type"]),
                str(event["digest"]),
            ),
        )
        for event in ordered_events:
            frame_start = int(event["frame_start"])
            frame_end = int(event["frame_end"])
            review_start = max(0, frame_start - context_frames)
            review_end = min(episode_length, frame_end + context_frames)
            preview_path = (
                f"previews/episode_{episode_index:06d}/event_{event['digest']}.png"
            )
            predictions.append(
                {
                    "event_digest": event["digest"],
                    "event_artifact_type": event["artifact_type"],
                    "event_type": event["event_type"],
                    "predicted_frame_range": [frame_start, frame_end],
                    "review_frame_range": [review_start, review_end],
                    "sampled_frame_indices": list(
                        _preview_frame_indices(
                            review_start,
                            review_end,
                            frame_start,
                            frame_end,
                        )
                    ),
                    "preview_path": preview_path,
                }
            )

        item = seal_contract(
            {
                "artifact_type": BOUNDARY_REVIEW_ITEM_ARTIFACT_TYPE,
                "schema_version": BOUNDARY_QA_SCHEMA_VERSION,
                "selection_algorithm": BOUNDARY_SELECTION_ALGORITHM,
                "trajectory_digest": trajectory_digest,
                "source_episode_index": episode_index,
                "source_task_index": trajectory["provenance"]["source_task_index"],
                "instruction": trajectory["instruction"],
                "episode_length": episode_length,
                "length_stratum": length_strata[trajectory_digest],
                "source_references": {
                    "front_video_path": trajectory["front_video_path"],
                    "wrist_video_path": trajectory["wrist_video_path"],
                },
                "predictions": predictions,
                "review_status": "pending",
            }
        )
        validate_boundary_review_item(item)
        items.append(item)

    available_count = len(eligible)
    selected_count = len(items)
    task_counts = Counter(int(item["source_task_index"]) for item in items)
    length_counts = Counter(str(item["length_stratum"]) for item in items)
    event_counts = Counter(
        str(prediction["event_type"])
        for item in items
        for prediction in item["predictions"]
    )
    manifest = seal_contract(
        {
            "artifact_type": "eventground_wam.boundary_review_selection",
            "schema_version": BOUNDARY_QA_SCHEMA_VERSION,
            "selection_algorithm": BOUNDARY_SELECTION_ALGORITHM,
            "requested_episode_count": target_episode_count,
            "available_eligible_episode_count": available_count,
            "selected_episode_count": selected_count,
            "selection_shortfall": max(0, target_episode_count - selected_count),
            "target_satisfied": selected_count == target_episode_count,
            "context_frames": context_frames,
            "preview_tile_size": preview_tile_size,
            "selected_episode_indices": [
                int(item["source_episode_index"]) for item in items
            ],
            "selected_task_counts": _string_key_counts(task_counts),
            "selected_event_counts": _string_key_counts(event_counts),
            "selected_length_stratum_counts": _string_key_counts(length_counts),
            "human_quality_metrics_status": "unavailable_until_reviews_complete",
        }
    )
    return tuple(items), manifest


def pending_review_output(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return a machine-readable review template for one selection item."""

    validate_boundary_review_item(item)
    return {
        "artifact_type": BOUNDARY_REVIEW_OUTPUT_ARTIFACT_TYPE,
        "schema_version": BOUNDARY_QA_SCHEMA_VERSION,
        "review_item_digest": item["digest"],
        "status": "pending",
        "reviewer_id": None,
        "reviewed_event_types": [],
        "prediction_judgments": [],
        "missed_events": [],
        "notes": "",
    }


def validate_boundary_review_item(item: Mapping[str, Any]) -> None:
    """Strictly validate a generated episode-level review input record."""

    keys = {
        "artifact_type",
        "schema_version",
        "selection_algorithm",
        "trajectory_digest",
        "source_episode_index",
        "source_task_index",
        "instruction",
        "episode_length",
        "length_stratum",
        "source_references",
        "predictions",
        "review_status",
        "digest",
    }
    _require_exact_keys(item, keys, "boundary review item")
    if item["artifact_type"] != BOUNDARY_REVIEW_ITEM_ARTIFACT_TYPE:
        raise BoundaryQAError("unsupported boundary review item artifact type")
    if item["schema_version"] != BOUNDARY_QA_SCHEMA_VERSION:
        raise BoundaryQAError("unsupported boundary review item schema version")
    if item["selection_algorithm"] != BOUNDARY_SELECTION_ALGORITHM:
        raise BoundaryQAError("unsupported boundary review selection algorithm")
    _require_sha256(item["trajectory_digest"], "trajectory_digest")
    episode_index = _require_int(item["source_episode_index"], "source_episode_index", 0)
    _require_int(item["source_task_index"], "source_task_index", 0)
    episode_length = _require_int(item["episode_length"], "episode_length", 1)
    _require_nonempty_string(item["instruction"], "instruction")
    if item["length_stratum"] not in LENGTH_STRATA:
        raise BoundaryQAError("unsupported length_stratum")
    if item["review_status"] != "pending":
        raise BoundaryQAError("generated boundary review items must be pending")

    references = item["source_references"]
    _require_exact_keys(
        references,
        {"front_video_path", "wrist_video_path"},
        "source_references",
    )
    expected_stem = f"episode_{episode_index:06d}.mp4"
    for name, value in references.items():
        path = _require_relative_path(value, f"source_references.{name}")
        if path.name != expected_stem:
            raise BoundaryQAError(f"source_references.{name} identifies the wrong episode")

    predictions = item["predictions"]
    if type(predictions) is not list or not predictions:
        raise BoundaryQAError("predictions must be a nonempty array")
    seen_events: set[str] = set()
    for index, prediction in enumerate(predictions):
        _require_exact_keys(
            prediction,
            {
                "event_digest",
                "event_artifact_type",
                "event_type",
                "predicted_frame_range",
                "review_frame_range",
                "sampled_frame_indices",
                "preview_path",
            },
            f"predictions[{index}]",
        )
        event_digest = _require_sha256(
            prediction["event_digest"], f"predictions[{index}].event_digest"
        )
        if event_digest in seen_events:
            raise BoundaryQAError("predictions contain a duplicate event digest")
        seen_events.add(event_digest)
        if prediction["event_artifact_type"] not in {
            EVENT_CANDIDATE_ARTIFACT_TYPE,
            EVENT_INTERVAL_ARTIFACT_TYPE,
        }:
            raise BoundaryQAError("prediction has an unsupported event artifact type")
        if prediction["event_type"] not in EVENT_TYPES:
            raise BoundaryQAError("prediction has an unsupported event type")
        predicted_start, predicted_end = _require_frame_range(
            prediction["predicted_frame_range"],
            f"predictions[{index}].predicted_frame_range",
            episode_length,
        )
        review_start, review_end = _require_frame_range(
            prediction["review_frame_range"],
            f"predictions[{index}].review_frame_range",
            episode_length,
        )
        if review_start > predicted_start or review_end < predicted_end:
            raise BoundaryQAError("review frame range does not contain the prediction")
        expected_samples = list(
            _preview_frame_indices(review_start, review_end, predicted_start, predicted_end)
        )
        if prediction["sampled_frame_indices"] != expected_samples:
            raise BoundaryQAError("sampled frame indices are not canonical")
        _require_relative_path(
            prediction["preview_path"], f"predictions[{index}].preview_path"
        )
    _require_sha256(item["digest"], "digest")
    if item["digest"] != contract_digest(item):
        raise BoundaryQAError("boundary review item digest mismatch")


def validate_review_output(
    review: Mapping[str, Any],
    item: Mapping[str, Any],
) -> None:
    """Validate a pending or completed review against its exact input item."""

    validate_boundary_review_item(item)
    keys = {
        "artifact_type",
        "schema_version",
        "review_item_digest",
        "status",
        "reviewer_id",
        "reviewed_event_types",
        "prediction_judgments",
        "missed_events",
        "notes",
    }
    _require_exact_keys(review, keys, "boundary review output")
    if review["artifact_type"] != BOUNDARY_REVIEW_OUTPUT_ARTIFACT_TYPE:
        raise BoundaryQAError("unsupported boundary review output artifact type")
    if review["schema_version"] != BOUNDARY_QA_SCHEMA_VERSION:
        raise BoundaryQAError("unsupported boundary review output schema version")
    if review["review_item_digest"] != item["digest"]:
        raise BoundaryQAError("review output does not identify the supplied review item")
    if type(review["notes"]) is not str:
        raise BoundaryQAError("review notes must be a string")
    if review["status"] == "pending":
        if review != pending_review_output(item):
            raise BoundaryQAError("pending review output differs from its canonical template")
        return
    if review["status"] != "completed":
        raise BoundaryQAError("review status must be pending or completed")
    _require_nonempty_string(review["reviewer_id"], "reviewer_id")

    reviewed_event_types = review["reviewed_event_types"]
    if type(reviewed_event_types) is not list or not reviewed_event_types:
        raise BoundaryQAError("completed review must name reviewed_event_types")
    if len(reviewed_event_types) != len(set(reviewed_event_types)):
        raise BoundaryQAError("reviewed_event_types must be unique")
    if reviewed_event_types != sorted(reviewed_event_types):
        raise BoundaryQAError("reviewed_event_types must be sorted")
    if any(event_type not in EVENT_TYPES for event_type in reviewed_event_types):
        raise BoundaryQAError("reviewed_event_types contains an unsupported event type")

    prediction_by_digest = {
        prediction["event_digest"]: prediction for prediction in item["predictions"]
    }
    expected_judgment_digests = {
        digest
        for digest, prediction in prediction_by_digest.items()
        if prediction["event_type"] in reviewed_event_types
    }
    judgments = review["prediction_judgments"]
    if type(judgments) is not list:
        raise BoundaryQAError("prediction_judgments must be an array")
    actual_judgment_digests: set[str] = set()
    for index, judgment in enumerate(judgments):
        _require_exact_keys(
            judgment,
            {"event_digest", "verdict", "corrected_frame_range", "comment"},
            f"prediction_judgments[{index}]",
        )
        digest = _require_sha256(
            judgment["event_digest"], f"prediction_judgments[{index}].event_digest"
        )
        if digest in actual_judgment_digests:
            raise BoundaryQAError("prediction_judgments contains a duplicate event")
        actual_judgment_digests.add(digest)
        prediction = prediction_by_digest.get(digest)
        if prediction is None:
            raise BoundaryQAError("prediction judgment identifies an unknown event")
        if judgment["verdict"] not in REVIEW_VERDICTS:
            raise BoundaryQAError("prediction judgment has an unsupported verdict")
        if type(judgment["comment"]) is not str:
            raise BoundaryQAError("prediction judgment comment must be a string")
        corrected_range = judgment["corrected_frame_range"]
        if judgment["verdict"] == "corrected":
            _require_frame_range(
                corrected_range,
                f"prediction_judgments[{index}].corrected_frame_range",
                int(item["episode_length"]),
            )
        elif corrected_range is not None:
            raise BoundaryQAError("only corrected judgments may set corrected_frame_range")
    if actual_judgment_digests != expected_judgment_digests:
        raise BoundaryQAError(
            "completed review must judge every prediction for each reviewed event type"
        )

    missed_events = review["missed_events"]
    if type(missed_events) is not list:
        raise BoundaryQAError("missed_events must be an array")
    seen_misses: set[tuple[str, int, int]] = set()
    for index, missed in enumerate(missed_events):
        _require_exact_keys(
            missed,
            {"event_type", "frame_range", "comment"},
            f"missed_events[{index}]",
        )
        if missed["event_type"] not in reviewed_event_types:
            raise BoundaryQAError("missed event type was not declared fully reviewed")
        start, end = _require_frame_range(
            missed["frame_range"],
            f"missed_events[{index}].frame_range",
            int(item["episode_length"]),
        )
        if type(missed["comment"]) is not str:
            raise BoundaryQAError("missed event comment must be a string")
        identity = (str(missed["event_type"]), start, end)
        if identity in seen_misses:
            raise BoundaryQAError("missed_events contains a duplicate interval")
        seen_misses.add(identity)


def aggregate_boundary_reviews(
    items: Iterable[Mapping[str, Any]],
    reviews: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate only completed human reviews into event-specific QA metrics."""

    item_by_digest: dict[str, Mapping[str, Any]] = {}
    for item in items:
        validate_boundary_review_item(item)
        digest = str(item["digest"])
        if digest in item_by_digest:
            raise BoundaryQAError(f"duplicate review item digest: {digest}")
        item_by_digest[digest] = item
    if not item_by_digest:
        raise BoundaryQAError("cannot aggregate an empty boundary review selection")

    completed_by_item: dict[str, Mapping[str, Any]] = {}
    reviewed_items: set[str] = set()
    pending_count = 0
    supplied_count = 0
    for review in reviews:
        supplied_count += 1
        digest = review.get("review_item_digest")
        item = item_by_digest.get(str(digest))
        if item is None:
            raise BoundaryQAError(f"review references an unknown item: {digest!r}")
        if str(digest) in reviewed_items:
            raise BoundaryQAError(f"multiple review records supplied for item {digest}")
        reviewed_items.add(str(digest))
        validate_review_output(review, item)
        if review["status"] == "pending":
            pending_count += 1
            continue
        completed_by_item[str(digest)] = review

    per_event: dict[str, dict[str, Any]] = {}
    for event_type in EVENT_TYPES:
        selected_episode_count = sum(
            any(
                prediction["event_type"] == event_type
                for prediction in item["predictions"]
            )
            for item in item_by_digest.values()
        )
        completed_pairs = [
            (item_by_digest[digest], review)
            for digest, review in completed_by_item.items()
            if event_type in review["reviewed_event_types"]
        ]
        reviewed_episode_count = len(completed_pairs)
        if not completed_pairs:
            per_event[event_type] = {
                "selected_prediction_episode_count": selected_episode_count,
                "reviewed_episode_count": 0,
                "review_coverage": 0.0,
                "true_positive_count": None,
                "false_positive_count": None,
                "false_negative_count": None,
                "ambiguous_count": None,
                "precision": None,
                "recall": None,
                "mean_absolute_start_boundary_error_frames": None,
                "mean_absolute_end_boundary_error_frames": None,
                "ambiguity_rate": None,
                "reviewer_correction_count": None,
                "human_quality_metrics_available": False,
            }
            continue

        accepted = corrected = rejected = ambiguous = false_negative = 0
        start_errors: list[int] = []
        end_errors: list[int] = []
        for item, review in completed_pairs:
            prediction_by_digest = {
                prediction["event_digest"]: prediction
                for prediction in item["predictions"]
            }
            for judgment in review["prediction_judgments"]:
                prediction = prediction_by_digest[judgment["event_digest"]]
                if prediction["event_type"] != event_type:
                    continue
                verdict = judgment["verdict"]
                if verdict == "accepted":
                    accepted += 1
                    start_errors.append(0)
                    end_errors.append(0)
                elif verdict == "corrected":
                    corrected += 1
                    predicted_start, predicted_end = prediction["predicted_frame_range"]
                    corrected_start, corrected_end = judgment["corrected_frame_range"]
                    start_errors.append(abs(corrected_start - predicted_start))
                    end_errors.append(abs(corrected_end - predicted_end))
                elif verdict == "rejected":
                    rejected += 1
                else:
                    ambiguous += 1
            false_negative += sum(
                missed["event_type"] == event_type for missed in review["missed_events"]
            )

        true_positive = accepted + corrected
        precision = _ratio(true_positive, true_positive + rejected)
        recall = _ratio(true_positive, true_positive + false_negative)
        judgment_count = true_positive + rejected + ambiguous
        per_event[event_type] = {
            "selected_prediction_episode_count": selected_episode_count,
            "reviewed_episode_count": reviewed_episode_count,
            "review_coverage": reviewed_episode_count / len(item_by_digest),
            "true_positive_count": true_positive,
            "false_positive_count": rejected,
            "false_negative_count": false_negative,
            "ambiguous_count": ambiguous,
            "precision": precision,
            "recall": recall,
            "mean_absolute_start_boundary_error_frames": _mean(start_errors),
            "mean_absolute_end_boundary_error_frames": _mean(end_errors),
            "ambiguity_rate": _ratio(ambiguous, judgment_count),
            "reviewer_correction_count": corrected,
            "human_quality_metrics_available": True,
        }

    completed_count = len(completed_by_item)
    if completed_count == 0:
        status = "pending_human_review"
        disclaimer = (
            "No completed human reviews were supplied; precision, recall, boundary error, "
            "ambiguity, and correction metrics are unavailable."
        )
    elif completed_count < len(item_by_digest):
        status = "human_review_in_progress"
        disclaimer = "Quality metrics include completed reviews only and are not final."
    else:
        status = "human_review_complete"
        disclaimer = "Quality metrics are computed from completed human reviews."
    return seal_contract(
        {
            "artifact_type": BOUNDARY_QA_REPORT_ARTIFACT_TYPE,
            "schema_version": BOUNDARY_QA_SCHEMA_VERSION,
            "status": status,
            "selected_episode_count": len(item_by_digest),
            "supplied_review_record_count": supplied_count,
            "completed_review_count": completed_count,
            "pending_review_count": pending_count,
            "missing_review_count": len(item_by_digest) - completed_count - pending_count,
            "review_completion": completed_count / len(item_by_digest),
            "human_quality_metrics_available": completed_count > 0,
            "quality_disclaimer": disclaimer,
            "per_event": per_event,
        }
    )


def render_boundary_previews(
    items: Iterable[Mapping[str, Any]],
    *,
    repository_root: Path,
    output_directory: Path,
    tile_size: int = 128,
    frame_reader: RGBFrameReader | None = None,
) -> tuple[dict[str, Any], ...]:
    """Render dual-camera PNG contact sheets for each proposed event boundary."""

    if tile_size < 16:
        raise ValueError("tile_size must be at least 16")
    root = repository_root.expanduser().resolve()
    output_root = output_directory.expanduser().resolve()
    read_frames = frame_reader or read_rgb_frames
    rendered: list[dict[str, Any]] = []
    for item in items:
        validate_boundary_review_item(item)
        source_paths = {
            camera: _resolve_existing_source(root, item["source_references"][f"{camera}_video_path"])
            for camera in ("front", "wrist")
        }
        requested_indices = sorted(
            {
                int(frame_index)
                for prediction in item["predictions"]
                for frame_index in prediction["sampled_frame_indices"]
            }
        )
        decoded = {
            camera: read_frames(path, requested_indices, tile_size)
            for camera, path in source_paths.items()
        }
        for camera, frames in decoded.items():
            missing = sorted(set(requested_indices) - set(frames))
            if missing:
                raise BoundaryQAError(
                    f"{camera} source is missing requested frames for episode "
                    f"{item['source_episode_index']}: {missing}"
                )
            for frame_index, frame in frames.items():
                if frame_index not in requested_indices:
                    raise BoundaryQAError(
                        f"{camera} decoder returned unrequested frame {frame_index}"
                    )
                if (frame.width, frame.height) != (tile_size, tile_size):
                    raise BoundaryQAError(
                        f"{camera} decoder returned {frame.width}x{frame.height}, "
                        f"expected {tile_size}x{tile_size}"
                    )

        for prediction in item["predictions"]:
            relative_preview = _require_relative_path(
                prediction["preview_path"], "prediction.preview_path"
            )
            preview_path = (output_root / relative_preview).resolve()
            try:
                preview_path.relative_to(output_root)
            except ValueError as error:
                raise BoundaryQAError("preview path escapes the output directory") from error
            if preview_path.exists():
                raise FileExistsError(f"preview renderer refuses to overwrite {preview_path}")
            frame_indices = prediction["sampled_frame_indices"]
            predicted_start, predicted_end = prediction["predicted_frame_range"]
            pixels = _compose_contact_sheet(
                decoded,
                frame_indices,
                predicted_start,
                predicted_end,
                tile_size,
            )
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            _write_rgb_png(
                preview_path,
                width=tile_size * len(frame_indices),
                height=tile_size * 2,
                pixels=pixels,
            )
            rendered.append(
                {
                    "review_item_digest": item["digest"],
                    "event_digest": prediction["event_digest"],
                    "preview_path": str(relative_preview),
                    "sampled_frame_indices": list(frame_indices),
                    "camera_rows": ["front", "wrist"],
                    "overlay_legend": {
                        "predicted_interval": "green_border",
                        "start_frame": "red_left_bar",
                        "end_frame": "blue_right_bar",
                    },
                }
            )
    return tuple(rendered)


def read_rgb_frames(
    path: Path,
    frame_indices: Sequence[int],
    tile_size: int,
) -> dict[int, RGBFrame]:
    """Decode selected video frames to packed RGB using the optional PyAV runtime."""

    try:
        import av
    except ImportError as error:  # pragma: no cover - exercised in the data runtime
        raise RuntimeError("boundary preview rendering requires PyAV") from error
    requested = set(frame_indices)
    if not requested:
        return {}
    if min(requested) < 0:
        raise ValueError("frame indices cannot be negative")
    result: dict[int, RGBFrame] = {}
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise BoundaryQAError(f"video contains no stream: {path}")
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index in requested:
                rgb_frame = frame.reformat(width=tile_size, height=tile_size, format="rgb24")
                plane = rgb_frame.planes[0]
                raw = bytes(plane)
                row_size = tile_size * 3
                pixels = b"".join(
                    raw[offset : offset + row_size]
                    for offset in range(0, plane.line_size * tile_size, plane.line_size)
                )
                result[frame_index] = RGBFrame(tile_size, tile_size, pixels)
                if len(result) == len(requested):
                    break
    return result


def _assign_length_strata(
    trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    by_task: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        by_task[int(trajectory["provenance"]["source_task_index"])].append(trajectory)
    result: dict[str, str] = {}
    for task_trajectories in by_task.values():
        ordered = sorted(
            task_trajectories,
            key=lambda trajectory: (
                int(trajectory["provenance"]["source_episode_length"]),
                int(trajectory["provenance"]["source_episode_index"]),
            ),
        )
        total = len(ordered)
        for rank, trajectory in enumerate(ordered):
            if total == 1:
                stratum = "middle"
            elif total == 2:
                stratum = ("short", "long")[rank]
            else:
                stratum = LENGTH_STRATA[min(2, rank * 3 // total)]
            result[str(trajectory["digest"])] = stratum
    return result


def _balanced_episode_selection(
    eligible: Sequence[Mapping[str, Any]],
    events_by_trajectory: Mapping[str, Sequence[Mapping[str, Any]]],
    length_strata: Mapping[str, str],
    *,
    limit: int,
) -> list[Mapping[str, Any]]:
    remaining = list(eligible)
    selected: list[Mapping[str, Any]] = []
    task_load: Counter[int] = Counter()
    event_load: Counter[str] = Counter()
    length_load: Counter[str] = Counter()
    while remaining and len(selected) < limit:
        def score(trajectory: Mapping[str, Any]) -> tuple[Any, ...]:
            digest = str(trajectory["digest"])
            task = int(trajectory["provenance"]["source_task_index"])
            event_types = sorted(
                {str(event["event_type"]) for event in events_by_trajectory[digest]}
            )
            length_stratum = length_strata[digest]
            novel_dimensions = (
                int(task_load[task] == 0)
                + sum(event_load[event_type] == 0 for event_type in event_types)
                + int(length_load[length_stratum] == 0)
            )
            mean_event_load = sum(event_load[event_type] for event_type in event_types) / len(
                event_types
            )
            return (
                -novel_dimensions,
                task_load[task],
                mean_event_load,
                length_load[length_stratum],
                int(trajectory["provenance"]["source_episode_index"]),
            )

        chosen = min(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        digest = str(chosen["digest"])
        task_load[int(chosen["provenance"]["source_task_index"])] += 1
        length_load[length_strata[digest]] += 1
        for event_type in {
            str(event["event_type"]) for event in events_by_trajectory[digest]
        }:
            event_load[event_type] += 1
    return selected


def _preview_frame_indices(
    review_start: int,
    review_end: int,
    event_start: int,
    event_end: int,
) -> tuple[int, ...]:
    values = (
        review_start,
        max(review_start, event_start - 1),
        event_start,
        event_end - 1,
        min(review_end - 1, event_end),
        review_end - 1,
    )
    return tuple(sorted(set(values)))


def _compose_contact_sheet(
    decoded: Mapping[str, Mapping[int, RGBFrame]],
    frame_indices: Sequence[int],
    event_start: int,
    event_end: int,
    tile_size: int,
) -> bytes:
    sheet_width = tile_size * len(frame_indices)
    sheet = bytearray(sheet_width * tile_size * 2 * 3)
    for camera_row, camera in enumerate(("front", "wrist")):
        for column, frame_index in enumerate(frame_indices):
            tile = bytearray(decoded[camera][frame_index].pixels)
            _paint_boundary_overlay(tile, tile_size, frame_index, event_start, event_end)
            for row in range(tile_size):
                source_start = row * tile_size * 3
                target_row = camera_row * tile_size + row
                target_start = (target_row * sheet_width + column * tile_size) * 3
                sheet[target_start : target_start + tile_size * 3] = tile[
                    source_start : source_start + tile_size * 3
                ]
    return bytes(sheet)


def _paint_boundary_overlay(
    pixels: bytearray,
    size: int,
    frame_index: int,
    event_start: int,
    event_end: int,
) -> None:
    border = max(2, size // 32)
    color = (32, 208, 96) if event_start <= frame_index < event_end else (128, 128, 128)
    _paint_rect(pixels, size, 0, 0, size, border, color)
    _paint_rect(pixels, size, 0, size - border, size, border, color)
    _paint_rect(pixels, size, 0, 0, border, size, color)
    _paint_rect(pixels, size, size - border, 0, border, size, color)
    if frame_index == event_start:
        _paint_rect(pixels, size, 0, 0, border * 2, size, (240, 48, 48))
    if frame_index == event_end - 1:
        _paint_rect(pixels, size, size - border * 2, 0, border * 2, size, (48, 96, 240))


def _paint_rect(
    pixels: bytearray,
    size: int,
    left: int,
    top: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    packed_color = bytes(color)
    for row in range(top, top + height):
        for column in range(left, left + width):
            offset = (row * size + column) * 3
            pixels[offset : offset + 3] = packed_color


def _write_rgb_png(path: Path, *, width: int, height: int, pixels: bytes) -> None:
    if len(pixels) != width * height * 3:
        raise BoundaryQAError("PNG payload size does not match its dimensions")
    scanlines = b"".join(
        b"\x00" + pixels[row * width * 3 : (row + 1) * width * 3]
        for row in range(height)
    )
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def _resolve_existing_source(repository_root: Path, relative_value: Any) -> Path:
    relative = _require_relative_path(relative_value, "source reference")
    path = (repository_root / relative).resolve()
    try:
        path.relative_to(repository_root)
    except ValueError as error:
        raise BoundaryQAError("source reference escapes the repository") from error
    if not path.is_file():
        raise FileNotFoundError(f"boundary preview source is missing: {path}")
    return path


def _require_exact_keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise BoundaryQAError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise BoundaryQAError(
            f"{name} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_int(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise BoundaryQAError(f"{name} must be an integer at least {minimum}")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise BoundaryQAError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise BoundaryQAError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_relative_path(value: Any, name: str) -> PurePosixPath:
    if type(value) is not str or not value:
        raise BoundaryQAError(f"{name} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value == ".":
        raise BoundaryQAError(f"{name} must be a safe relative path")
    return path


def _require_frame_range(value: Any, name: str, episode_length: int) -> tuple[int, int]:
    if type(value) is not list or len(value) != 2:
        raise BoundaryQAError(f"{name} must be a two-integer array")
    start = _require_int(value[0], f"{name}[0]", 0)
    end = _require_int(value[1], f"{name}[1]", 1)
    if start >= end or end > episode_length:
        raise BoundaryQAError(f"{name} must be a nonempty in-episode half-open range")
    return start, end


def _string_key_counts(counts: Mapping[Any, int]) -> dict[str, int]:
    return {str(key): counts[key] for key in sorted(counts, key=str)}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: Sequence[int]) -> float | None:
    return sum(values) / len(values) if values else None

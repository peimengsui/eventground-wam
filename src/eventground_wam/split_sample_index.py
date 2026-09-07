"""Leakage-resistant splits and streaming multi-horizon sample indexes.

The pinned demonstrations do not expose reset seeds or MuJoCo state.  This
module therefore groups episodes by a deterministic surrogate composed of the
task identity, initial float32 proprioception, and decoded RGB fingerprints for
both cameras.  Every member of a surrogate group is assigned to the same split.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from eventground_wam.demonstration_contracts import (
    CONTRACT_SCHEMA_VERSION,
    MULTI_HORIZON_SAMPLE_ARTIFACT_TYPE,
    SOURCE_DATASET_REPO_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_EPISODE_COUNT,
    SOURCE_FRAME_COUNT,
    SOURCE_MANIFEST_SHA256,
    SOURCE_STATE_DIMENSION,
    SOURCE_TASK_COUNT,
    SPLIT_ASSIGNMENT_ARTIFACT_TYPE,
    SUPPORTED_HORIZONS,
    EventInterval,
    MultiHorizonSample,
    Provenance,
    SplitAssignment,
    TrajectoryRecord,
    canonical_json_bytes,
    contract_digest,
    validate_trajectory_reference,
)
from eventground_wam.demonstration_reader import DecodedVideoFrame, SynchronizedStep

GROUP_STRATEGY = "task-state-rgb-surrogate-v1"
DEFAULT_SPLIT_NAMESPACE = "week2-splits-v1"
SPLIT_SAMPLE_INDEX_ARTIFACT_TYPE = "eventground_wam.split_sample_indexes"
SPLIT_NAMES = ("train", "validation", "test")
FIXED_HORIZONS = tuple(value for value in SUPPORTED_HORIZONS if type(value) is int)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_HASH_SPACE = 1 << 256


class SplitSampleIndexError(ValueError):
    """Raised when split or sample indexes cannot be generated safely."""


@dataclass(frozen=True)
class SplitRatios:
    """Explicit deterministic group-level split probabilities."""

    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise SplitSampleIndexError("split ratios must be finite floats")
        if any(value <= 0.0 or value >= 1.0 for value in values):
            raise SplitSampleIndexError("every split ratio must be between zero and one")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise SplitSampleIndexError("split ratios must sum to one")

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in SPLIT_NAMES}


@dataclass(frozen=True)
class InitialConditionEvidence:
    """The source identity and first-step signals used by the grouping surrogate."""

    source_episode_index: int
    source_task_index: int
    initial_state: tuple[float, ...]
    front_frame_sha256: str
    wrist_frame_sha256: str

    def __post_init__(self) -> None:
        if type(self.source_episode_index) is not int or self.source_episode_index < 0:
            raise SplitSampleIndexError("source_episode_index must be a nonnegative integer")
        if type(self.source_task_index) is not int or self.source_task_index < 0:
            raise SplitSampleIndexError("source_task_index must be a nonnegative integer")
        _state_float32_hex(self.initial_state)
        _require_sha256(self.front_frame_sha256, "front_frame_sha256")
        _require_sha256(self.wrist_frame_sha256, "wrist_frame_sha256")


@dataclass(frozen=True)
class EpisodeSampleSource:
    """One streamable unit needed to create split and sample records."""

    trajectory: TrajectoryRecord
    initial_condition: InitialConditionEvidence
    events: tuple[EventInterval, ...] = ()


def decoded_rgb_fingerprint(frame: DecodedVideoFrame) -> str:
    """Hash one decoded frame after canonical conversion to packed RGB24."""

    if frame.frame_index != 0:
        raise SplitSampleIndexError("initial-condition frame must have frame_index zero")
    if not math.isclose(frame.timestamp, 0.0, rel_tol=0.0, abs_tol=1e-6):
        raise SplitSampleIndexError("initial-condition frame must have timestamp zero")
    try:
        rgb = frame.payload.reformat(format="rgb24")
        plane = rgb.planes[0]
        line_size = int(plane.line_size)
        plane_bytes = bytes(plane)
    except Exception as error:
        raise SplitSampleIndexError("cannot convert decoded frame to packed RGB24") from error
    if (
        rgb.width != frame.width
        or rgb.height != frame.height
        or str(rgb.format.name) != "rgb24"
        or len(rgb.planes) != 1
        or line_size < frame.width * 3
        or len(plane_bytes) < line_size * frame.height
    ):
        raise SplitSampleIndexError(
            "decoded RGB frame does not expose the expected packed plane layout"
        )

    digest = hashlib.sha256()
    digest.update(b"eventground-wam-decoded-rgb24-v1\0")
    digest.update(struct.pack(">II", frame.width, frame.height))
    row_size = frame.width * 3
    for row_index in range(frame.height):
        row_start = row_index * line_size
        digest.update(plane_bytes[row_start : row_start + row_size])
    return digest.hexdigest()


def initial_condition_from_step(
    trajectory: TrajectoryRecord,
    step: SynchronizedStep,
) -> InitialConditionEvidence:
    """Create grouping evidence from the synchronized first step of an episode."""

    provenance = trajectory.provenance
    row = step.row
    if row.frame_index != 0 or not math.isclose(row.timestamp, 0.0, abs_tol=1e-6):
        raise SplitSampleIndexError("initial-condition telemetry must be frame zero at time zero")
    if (
        row.episode_index != provenance.source_episode_index
        or row.task_index != provenance.source_task_index
    ):
        raise SplitSampleIndexError("initial-condition telemetry does not match the trajectory")
    return InitialConditionEvidence(
        source_episode_index=row.episode_index,
        source_task_index=row.task_index,
        initial_state=row.state,
        front_frame_sha256=decoded_rgb_fingerprint(step.front_frame),
        wrist_frame_sha256=decoded_rgb_fingerprint(step.wrist_frame),
    )


def initial_condition_digest(evidence: InitialConditionEvidence) -> str:
    """Return the stable group identity for one initial-condition surrogate."""

    payload = {
        "group_strategy": GROUP_STRATEGY,
        "source_task_index": evidence.source_task_index,
        "initial_state_float32_be": _state_float32_hex(evidence.initial_state),
        "front_decoded_rgb24_sha256": evidence.front_frame_sha256,
        "wrist_decoded_rgb24_sha256": evidence.wrist_frame_sha256,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def assign_group_split(
    group_digest: str,
    *,
    ratios: SplitRatios | None = None,
    namespace: str = DEFAULT_SPLIT_NAMESPACE,
) -> str:
    """Assign a complete group by hashing its identity into explicit ratio buckets."""

    _require_sha256(group_digest, "group_digest")
    split_ratios = ratios or SplitRatios()
    if type(namespace) is not str or not namespace.strip():
        raise SplitSampleIndexError("split namespace must be a nonempty string")
    bucket_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "algorithm": "sha256-ratio-bucket-v1",
                "group_digest": group_digest,
                "namespace": namespace,
            }
        )
    ).digest()
    bucket = int.from_bytes(bucket_digest, "big")
    train_end = int(split_ratios.train * _HASH_SPACE)
    validation_end = int(
        (split_ratios.train + split_ratios.validation) * _HASH_SPACE
    )
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "test"


def iter_episode_samples(
    trajectory: TrajectoryRecord,
    split_assignment: SplitAssignment,
    events: Sequence[EventInterval],
) -> Iterator[MultiHorizonSample]:
    """Yield fixed and next-event samples without crossing this episode's boundary."""

    validate_trajectory_reference(split_assignment, trajectory)
    ordered_events = _validated_events(events, trajectory)
    eligible_events = tuple(event for event in ordered_events if _is_eligible_event(event))
    provenance = split_assignment.provenance
    episode_length = provenance.source_episode_length
    trajectory_digest = trajectory.digest
    split_assignment_digest = split_assignment.digest

    for horizon in FIXED_HORIZONS:
        for anchor in range(episode_length - horizon):
            sample = MultiHorizonSample(
                provenance=provenance,
                trajectory_digest=trajectory_digest,
                split_assignment_digest=split_assignment_digest,
                frame_start=anchor,
                frame_end=anchor + horizon + 1,
                anchor_frame_index=anchor,
                target_frame_index=anchor + horizon,
                horizon=horizon,
                split=split_assignment.split,
            )
            _validate_generated_sample(
                sample,
                trajectory,
                split_assignment,
                trajectory_digest=trajectory_digest,
                split_assignment_digest=split_assignment_digest,
            )
            yield sample

    previous_event_start = 0
    for event_start, simultaneous_events in _events_by_start(eligible_events):
        for anchor in range(previous_event_start, event_start):
            for event in simultaneous_events:
                sample = MultiHorizonSample(
                    provenance=provenance,
                    trajectory_digest=trajectory_digest,
                    split_assignment_digest=split_assignment_digest,
                    frame_start=anchor,
                    frame_end=event_start + 1,
                    anchor_frame_index=anchor,
                    target_frame_index=event_start,
                    horizon="next-event",
                    split=split_assignment.split,
                    target_event_digest=event.digest,
                )
                _validate_generated_sample(
                    sample,
                    trajectory,
                    split_assignment,
                    trajectory_digest=trajectory_digest,
                    split_assignment_digest=split_assignment_digest,
                    target_event=event,
                )
                yield sample
        previous_event_start = event_start


def write_split_sample_indexes(
    sources: Iterable[EpisodeSampleSource],
    output_dir: Path,
    *,
    project_git_revision: str,
    project_tree_clean: bool,
    annotation_version: str,
    ratios: SplitRatios | None = None,
    namespace: str = DEFAULT_SPLIT_NAMESPACE,
    require_complete_source: bool = True,
) -> dict[str, Any]:
    """Atomically write split/sample JSONL streams and a sealed validation manifest."""

    if type(project_tree_clean) is not bool:
        raise TypeError("project_tree_clean must be a bool")
    if type(require_complete_source) is not bool:
        raise TypeError("require_complete_source must be a bool")
    split_ratios = ratios or SplitRatios()
    # Validate the namespace even if the input stream is empty.
    assign_group_split("0" * 64, ratios=split_ratios, namespace=namespace)

    output_dir = output_dir.expanduser().resolve()
    split_path = output_dir / "split-assignments.jsonl"
    sample_path = output_dir / "multi-horizon-samples.jsonl"
    manifest_path = output_dir / "split-sample-manifest.json"
    final_paths = (split_path, sample_path, manifest_path)
    partial_paths = tuple(path.with_suffix(path.suffix + ".partial") for path in final_paths)
    conflicts = [path for path in (*final_paths, *partial_paths) if path.exists()]
    if conflicts:
        raise FileExistsError(f"split/sample generator refuses to overwrite outputs: {conflicts}")

    output_dir.mkdir(parents=True, exist_ok=True)
    split_hash = hashlib.sha256()
    sample_hash = hashlib.sha256()
    episode_count = 0
    frame_count = 0
    task_indices: set[int] = set()
    trajectory_digests: set[str] = set()
    event_digests: set[str] = set()
    eligible_event_count = 0
    excluded_weak_event_count = 0
    excluded_unreviewed_event_count = 0
    group_splits: dict[str, str] = {}
    episode_counts = {name: 0 for name in SPLIT_NAMES}
    group_sets = {name: set() for name in SPLIT_NAMES}
    sample_counts: dict[str, int] = {str(horizon): 0 for horizon in SUPPORTED_HORIZONS}
    samples_by_split = {name: 0 for name in SPLIT_NAMES}
    previous_episode_index: int | None = None

    try:
        with (
            partial_paths[0].open("x", encoding="utf-8") as split_handle,
            partial_paths[1].open("x", encoding="utf-8") as sample_handle,
        ):
            for source in sources:
                trajectory = source.trajectory
                trajectory_record = trajectory.to_dict()
                provenance = trajectory.provenance
                episode_index = provenance.source_episode_index
                if previous_episode_index is not None and episode_index <= previous_episode_index:
                    raise SplitSampleIndexError(
                        "episode sources must be strictly ordered by source_episode_index"
                    )
                if require_complete_source and episode_index != episode_count:
                    raise SplitSampleIndexError(
                        f"complete indexes expected episode {episode_count}, found {episode_index}"
                    )
                if trajectory.digest in trajectory_digests:
                    raise SplitSampleIndexError("duplicate trajectory digest in source stream")
                _validate_initial_condition(source.initial_condition, trajectory)
                group_digest = initial_condition_digest(source.initial_condition)
                split_name = assign_group_split(
                    group_digest,
                    ratios=split_ratios,
                    namespace=namespace,
                )
                previous_split = group_splits.setdefault(group_digest, split_name)
                if previous_split != split_name:
                    raise SplitSampleIndexError(
                        f"initial-condition group {group_digest} crosses splits"
                    )

                derived_provenance = Provenance(
                    source_episode_index=episode_index,
                    source_task_index=provenance.source_task_index,
                    source_episode_length=provenance.source_episode_length,
                    project_git_revision=project_git_revision,
                    annotation_version=annotation_version,
                )
                assignment = SplitAssignment(
                    provenance=derived_provenance,
                    trajectory_digest=trajectory.digest,
                    frame_start=0,
                    frame_end=provenance.source_episode_length,
                    split=split_name,
                    group_strategy=GROUP_STRATEGY,
                    initial_condition_digest=group_digest,
                )
                validate_trajectory_reference(assignment, trajectory)
                split_hash.update(_write_contract_line(split_handle, assignment.to_dict()))

                ordered_events = _validated_events(source.events, trajectory)
                for event in ordered_events:
                    if event.digest in event_digests:
                        raise SplitSampleIndexError("duplicate event digest in source stream")
                    event_digests.add(event.digest)
                    eligible_event_count += _is_eligible_event(event)
                    excluded_weak_event_count += event.ontology_tier != "reliable"
                    excluded_unreviewed_event_count += event.review_status not in {
                        "accepted",
                        "corrected",
                    }
                for sample in iter_episode_samples(trajectory, assignment, ordered_events):
                    sample_hash.update(_write_contract_line(sample_handle, sample.to_dict()))
                    sample_counts[str(sample.horizon)] += 1
                    samples_by_split[split_name] += 1

                episode_count += 1
                frame_count += provenance.source_episode_length
                task_indices.add(provenance.source_task_index)
                trajectory_digests.add(trajectory_record["digest"])
                episode_counts[split_name] += 1
                group_sets[split_name].add(group_digest)
                previous_episode_index = episode_index

        if episode_count == 0:
            raise SplitSampleIndexError("cannot publish indexes from an empty source stream")
        if require_complete_source and (
            episode_count != SOURCE_EPISODE_COUNT
            or frame_count != SOURCE_FRAME_COUNT
            or task_indices != set(range(SOURCE_TASK_COUNT))
        ):
            raise SplitSampleIndexError(
                "complete index totals do not match the pinned source: "
                f"episodes={episode_count}, frames={frame_count}, tasks={len(task_indices)}"
            )

        overlap = _cross_split_overlap(group_sets)
        if overlap:
            raise SplitSampleIndexError(
                f"initial-condition groups overlap across splits: {sorted(overlap)}"
            )
        manifest = {
            "artifact_type": SPLIT_SAMPLE_INDEX_ARTIFACT_TYPE,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "source": {
                "dataset_repo_id": SOURCE_DATASET_REPO_ID,
                "dataset_revision": SOURCE_DATASET_REVISION,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            },
            "generator": {
                "project_git_revision": project_git_revision,
                "project_tree_clean": project_tree_clean,
                "annotation_version": annotation_version,
            },
            "grouping": {
                "strategy": GROUP_STRATEGY,
                "components": [
                    "source_task_index",
                    "initial_state_float32_be",
                    "front_decoded_rgb24_sha256",
                    "wrist_decoded_rgb24_sha256",
                ],
                "split_algorithm": "sha256-ratio-bucket-v1",
                "namespace": namespace,
                "ratios": split_ratios.to_dict(),
                "unique_group_count": len(group_splits),
                "groups_by_split": {
                    name: len(group_sets[name]) for name in SPLIT_NAMES
                },
                "episodes_by_split": episode_counts,
                "cross_split_group_overlap_count": 0,
            },
            "sampling": {
                "horizons": list(SUPPORTED_HORIZONS),
                "fixed_horizon_policy": "all-anchors-with-complete-target-v1",
                "next_event_policy": "all-anchors-to-every-event-at-next-start-v1",
                "sample_counts_by_horizon": sample_counts,
                "sample_counts_by_split": samples_by_split,
                "event_targets": {
                    "eligibility": (
                        "ontology_tier == reliable and review_status in {accepted, corrected}"
                    ),
                    "input_event_count": len(event_digests),
                    "eligible_event_count": eligible_event_count,
                    "excluded_event_count": len(event_digests) - eligible_event_count,
                    "excluded_weak_event_count": excluded_weak_event_count,
                    "excluded_unreviewed_event_count": excluded_unreviewed_event_count,
                },
            },
            "outputs": {
                "split_assignments": {
                    "path": split_path.name,
                    "artifact_type": SPLIT_ASSIGNMENT_ARTIFACT_TYPE,
                    "sha256": split_hash.hexdigest(),
                    "record_count": episode_count,
                },
                "multi_horizon_samples": {
                    "path": sample_path.name,
                    "artifact_type": MULTI_HORIZON_SAMPLE_ARTIFACT_TYPE,
                    "sha256": sample_hash.hexdigest(),
                    "record_count": sum(sample_counts.values()),
                },
            },
            "episode_count": episode_count,
            "frame_count": frame_count,
            "task_count": len(task_indices),
            "complete_source": require_complete_source,
            "foreign_keys_validated": True,
            "publishable": project_tree_clean,
        }
        manifest["digest"] = contract_digest(manifest)
        partial_paths[2].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for partial, final in zip(partial_paths, final_paths, strict=True):
            partial.replace(final)
        return manifest
    except BaseException:
        for path in (*partial_paths, *final_paths):
            path.unlink(missing_ok=True)
        raise


def _state_float32_hex(state: Sequence[float]) -> list[str]:
    if len(state) != SOURCE_STATE_DIMENSION:
        raise SplitSampleIndexError(
            f"initial state must have {SOURCE_STATE_DIMENSION} components"
        )
    encoded: list[str] = []
    for index, value in enumerate(state):
        if type(value) is not float or not math.isfinite(value):
            raise SplitSampleIndexError(f"initial_state[{index}] must be a finite float")
        try:
            encoded.append(struct.pack(">f", value).hex())
        except OverflowError as error:
            raise SplitSampleIndexError(
                f"initial_state[{index}] cannot be represented as float32"
            ) from error
    return encoded


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise SplitSampleIndexError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_initial_condition(
    evidence: InitialConditionEvidence,
    trajectory: TrajectoryRecord,
) -> None:
    provenance = trajectory.provenance
    if (
        evidence.source_episode_index != provenance.source_episode_index
        or evidence.source_task_index != provenance.source_task_index
    ):
        raise SplitSampleIndexError("initial-condition evidence does not match its trajectory")


def _validated_events(
    events: Sequence[EventInterval],
    trajectory: TrajectoryRecord,
) -> tuple[EventInterval, ...]:
    ordered: list[EventInterval] = []
    previous_key: tuple[int, int, str, str] | None = None
    for event in events:
        if not isinstance(event, EventInterval):
            raise SplitSampleIndexError("episode events must be EventInterval contracts")
        validate_trajectory_reference(event, trajectory)
        key = (event.frame_start, event.frame_end, event.event_type, event.digest)
        if previous_key is not None and key <= previous_key:
            raise SplitSampleIndexError(
                "episode events must be strictly ordered by frame range, type, and digest"
            )
        ordered.append(event)
        previous_key = key
    return tuple(ordered)


def _events_by_start(
    events: Sequence[EventInterval],
) -> Iterator[tuple[int, tuple[EventInterval, ...]]]:
    index = 0
    while index < len(events):
        frame_start = events[index].frame_start
        end = index + 1
        while end < len(events) and events[end].frame_start == frame_start:
            end += 1
        yield frame_start, tuple(events[index:end])
        index = end


def _is_eligible_event(event: EventInterval) -> bool:
    return event.ontology_tier == "reliable" and event.review_status in {
        "accepted",
        "corrected",
    }


def _validate_generated_sample(
    sample: MultiHorizonSample,
    trajectory: TrajectoryRecord,
    split_assignment: SplitAssignment,
    *,
    trajectory_digest: str,
    split_assignment_digest: str,
    target_event: EventInterval | None = None,
) -> None:
    # MultiHorizonSample.__post_init__ has already validated the locked record
    # itself.  Keep this per-record foreign-key check constant-time: the full
    # source creates millions of samples, so repeatedly serializing all three
    # referenced contracts would make the streaming writer impractical.
    if sample.trajectory_digest != trajectory_digest:
        raise SplitSampleIndexError("sample trajectory foreign key is invalid")
    if sample.split_assignment_digest != split_assignment_digest:
        raise SplitSampleIndexError("sample split-assignment foreign key is invalid")
    if sample.split != split_assignment.split:
        raise SplitSampleIndexError("sample and split assignment disagree")
    source_identity = (
        "source_dataset_repo_id",
        "source_dataset_revision",
        "source_manifest_sha256",
        "source_episode_index",
        "source_task_index",
        "source_episode_length",
    )
    if any(
        getattr(sample.provenance, key) != getattr(split_assignment.provenance, key)
        for key in source_identity
    ):
        raise SplitSampleIndexError("sample and split provenance differ")
    if target_event is None:
        if sample.horizon == "next-event" or sample.target_event_digest is not None:
            raise SplitSampleIndexError("fixed-horizon sample has an event foreign key")
        return
    if sample.target_event_digest != target_event.digest:
        raise SplitSampleIndexError("sample event foreign key is invalid")
    if target_event.trajectory_digest != trajectory_digest:
        raise SplitSampleIndexError("sample and event reference different trajectories")
    if sample.target_frame_index != target_event.frame_start:
        raise SplitSampleIndexError("next-event target does not match the event boundary")


def _cross_split_overlap(group_sets: dict[str, set[str]]) -> set[str]:
    return (
        (group_sets["train"] & group_sets["validation"])
        | (group_sets["train"] & group_sets["test"])
        | (group_sets["validation"] & group_sets["test"])
    )


def _write_contract_line(handle: TextIO, record: dict[str, Any]) -> bytes:
    line = (
        json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    handle.write(line.decode("utf-8"))
    return line

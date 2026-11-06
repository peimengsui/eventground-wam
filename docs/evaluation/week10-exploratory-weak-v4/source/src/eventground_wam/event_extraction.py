"""Deterministic admission of raw event candidates into final intervals.

The source snapshot only supports a telemetry candidate detector for
``gripper_close``. This module therefore keeps every raw candidate, records an
explicit disposition for detector opportunities, and keeps admitted intervals
weak until human QA clears the ontology quality gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from eventground_wam.demonstration_contracts import (
    CONTRACT_SCHEMA_VERSION,
    EVENT_CANDIDATE_ARTIFACT_TYPE,
    EVENT_TYPES,
    OBJECT_EVIDENCE_PREFIXES,
    OBJECT_RELATIVE_EVENT_TYPES,
    SOURCE_DATASET_REPO_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_MANIFEST_SHA256,
    EventCandidate,
    EventInterval,
    Provenance,
    TrajectoryRecord,
    canonical_json_bytes,
    contract_digest,
    validate_contract,
    validate_trajectory_reference,
)
from eventground_wam.demonstration_reader import TrajectoryRow
from eventground_wam.event_labeling import (
    LIBERO_CLOSE_COMMAND,
    LIBERO_OPEN_COMMAND,
    EventLabelingError,
)

EVENT_EXTRACTION_ARTIFACT_TYPE = "eventground_wam.event_extraction"
EVENT_EXTRACTOR_NAME = "gripper-close-deterministic"
EVENT_EXTRACTOR_VERSION = "v1"
EVENT_OUTPUT_FILENAMES = (
    "event-candidates.jsonl",
    "event-intervals.jsonl",
    "event-decisions.jsonl",
)
DecisionStatus = Literal["admitted_weak", "absent", "ambiguous", "deferred"]


class EventExtractionError(ValueError):
    """Raised when candidates cannot be deterministically filtered."""


@dataclass(frozen=True)
class EventExtractionConfig:
    """Locked signal-processing and admission parameters for v1 extraction."""

    aperture_smoothing_window_frames: int = 3
    command_hysteresis_frames: int = 2
    minimum_close_duration_frames: int = 5
    response_window_frames: int = 5
    confirmed_aperture_drop: float = 0.001
    ambiguous_aperture_drop: float = 0.0005
    minimum_candidate_confidence: float = 0.9
    candidate_start_tolerance_frames: int = 1

    def __post_init__(self) -> None:
        if (
            self.aperture_smoothing_window_frames < 1
            or self.aperture_smoothing_window_frames % 2 == 0
        ):
            raise ValueError("aperture_smoothing_window_frames must be positive and odd")
        integer_fields = (
            "command_hysteresis_frames",
            "minimum_close_duration_frames",
            "response_window_frames",
        )
        if any(getattr(self, name) < 1 for name in integer_fields):
            raise ValueError("hysteresis, duration, and response windows must be positive")
        if self.minimum_close_duration_frames < self.response_window_frames:
            raise ValueError("minimum close duration must cover the response window")
        if self.candidate_start_tolerance_frames < 0:
            raise ValueError("candidate_start_tolerance_frames cannot be negative")
        thresholds = (
            self.confirmed_aperture_drop,
            self.ambiguous_aperture_drop,
        )
        if any(not math.isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("aperture-drop thresholds must be finite and positive")
        if self.ambiguous_aperture_drop >= self.confirmed_aperture_drop:
            raise ValueError("ambiguous threshold must be below the confirmed threshold")
        if (
            not math.isfinite(self.minimum_candidate_confidence)
            or not 0 <= self.minimum_candidate_confidence <= 1
        ):
            raise ValueError("minimum_candidate_confidence must be between zero and one")


@dataclass(frozen=True)
class EventDecision:
    """Explicit per-opportunity disposition, including absence and ambiguity."""

    source_episode_index: int
    source_task_index: int
    event_type: str
    status: DecisionStatus
    reason: str
    frame_start: int | None
    frame_end: int | None
    candidate_digests: tuple[str, ...]
    interval_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise EventExtractionError(f"unsupported decision event type: {self.event_type}")
        if self.status not in {"admitted_weak", "absent", "ambiguous", "deferred"}:
            raise EventExtractionError(f"unsupported decision status: {self.status}")
        if not self.reason.strip():
            raise EventExtractionError("decision reason must be nonempty")
        if (self.frame_start is None) != (self.frame_end is None):
            raise EventExtractionError("decision frame bounds must both be present or absent")
        if self.frame_start is not None and (
            self.frame_start < 0 or self.frame_end is None or self.frame_start >= self.frame_end
        ):
            raise EventExtractionError("decision frame range must be nonempty and half-open")
        if self.status == "admitted_weak" and not self.interval_digests:
            raise EventExtractionError("admitted-weak decisions must identify an interval")
        if self.status != "admitted_weak" and self.interval_digests:
            raise EventExtractionError("only admitted-weak decisions may identify an interval")
        if self.status == "absent" and (self.candidate_digests or self.frame_start is not None):
            raise EventExtractionError("absent decisions cannot identify candidates or frames")
        for digest in (*self.candidate_digests, *self.interval_digests):
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise EventExtractionError("decision references must be lowercase SHA-256 digests")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-native decision record."""

        return json.loads(canonical_json_bytes(asdict(self)))


@dataclass(frozen=True)
class EventExtractionResult:
    """Raw inputs, admitted intervals, and complete extraction dispositions."""

    raw_candidates: tuple[EventCandidate, ...]
    intervals: tuple[EventInterval, ...]
    decisions: tuple[EventDecision, ...]


@dataclass(frozen=True)
class _CloseRun:
    frame_start: int
    frame_end: int


def extract_event_intervals(
    rows: Sequence[TrajectoryRow],
    trajectory: TrajectoryRecord,
    candidates: Iterable[EventCandidate],
    *,
    config: EventExtractionConfig | None = None,
    interval_provenance: Provenance | None = None,
) -> EventExtractionResult:
    """Filter raw candidates with deterministic smoothing and hysteresis.

    Command transitions are debounced by requiring consecutive samples of the
    new binary command. Aperture response uses a centered median to reject
    single-frame proprioceptive spikes. Drops between the ambiguous and
    confirmed thresholds are explicitly retained as ambiguous decisions.
    """

    extraction_config = config or EventExtractionConfig()
    _validate_rows(rows, trajectory)
    derived_provenance = interval_provenance or trajectory.provenance
    _validate_derived_provenance(derived_provenance, trajectory)
    ordered_candidates = tuple(
        sorted(
            candidates,
            key=lambda candidate: (
                candidate.event_type,
                candidate.frame_start,
                candidate.frame_end,
                candidate.digest,
            ),
        )
    )
    _validate_candidates(ordered_candidates, trajectory)

    close_candidates = tuple(
        candidate
        for candidate in ordered_candidates
        if candidate.event_type == "gripper_close"
    )
    object_candidates = tuple(
        candidate
        for candidate in ordered_candidates
        if candidate.event_type in OBJECT_RELATIVE_EVENT_TYPES
    )
    commands = tuple(_gripper_command(row) for row in rows)
    apertures = tuple(abs(row.state[-2]) + abs(row.state[-1]) for row in rows)
    smoothed_apertures = _centered_median(
        apertures,
        extraction_config.aperture_smoothing_window_frames,
    )
    close_runs = _debounced_close_runs(
        commands,
        hysteresis_frames=extraction_config.command_hysteresis_frames,
    )

    intervals: list[EventInterval] = []
    decisions: list[EventDecision] = []
    consumed_candidates: set[str] = set()
    for close_run in close_runs:
        matching = tuple(
            candidate
            for candidate in close_candidates
            if abs(candidate.frame_start - close_run.frame_start)
            <= extraction_config.candidate_start_tolerance_frames
        )
        consumed_candidates.update(candidate.digest for candidate in matching)
        if (
            close_run.frame_end - close_run.frame_start
            < extraction_config.minimum_close_duration_frames
        ):
            decisions.append(
                _decision(
                    trajectory,
                    "ambiguous",
                    "close-command-below-minimum-duration",
                    close_run.frame_start,
                    close_run.frame_end,
                    candidates=matching,
                )
            )
            continue
        response_index = close_run.frame_start + extraction_config.response_window_frames
        decision_end = min(close_run.frame_end, response_index + 1)
        if not matching:
            decisions.append(
                _decision(
                    trajectory,
                    "ambiguous",
                    "persistent-close-command-without-raw-candidate",
                    close_run.frame_start,
                    decision_end,
                )
            )
            continue
        if response_index >= len(rows):
            decisions.append(
                _decision(
                    trajectory,
                    "ambiguous",
                    "incomplete-aperture-response-window",
                    close_run.frame_start,
                    close_run.frame_end,
                    candidates=matching,
                )
            )
            continue

        aperture_drop = (
            smoothed_apertures[close_run.frame_start] - smoothed_apertures[response_index]
        )
        if any(
            candidate.confidence < extraction_config.minimum_candidate_confidence
            for candidate in matching
        ):
            decisions.append(
                _decision(
                    trajectory,
                    "ambiguous",
                    "candidate-confidence-below-admission-threshold",
                    close_run.frame_start,
                    response_index + 1,
                    candidates=matching,
                )
            )
            continue
        if aperture_drop < extraction_config.confirmed_aperture_drop:
            reason = (
                "aperture-drop-in-hysteresis-band"
                if aperture_drop >= extraction_config.ambiguous_aperture_drop
                else "aperture-drop-below-ambiguous-threshold"
            )
            decisions.append(
                _decision(
                    trajectory,
                    "ambiguous",
                    reason,
                    close_run.frame_start,
                    response_index + 1,
                    candidates=matching,
                )
            )
            continue

        interval = EventInterval(
            provenance=derived_provenance,
            trajectory_digest=trajectory.digest,
            frame_start=close_run.frame_start,
            frame_end=response_index + 1,
            event_type="gripper_close",
            confidence=min(candidate.confidence for candidate in matching),
            evidence=_merged_close_evidence(
                matching,
                extraction_config,
                close_run,
                aperture_drop,
            ),
            candidate_digests=tuple(candidate.digest for candidate in matching),
            ontology_tier="weak",
            review_status="unreviewed",
        )
        intervals.append(interval)
        decisions.append(
            _decision(
                trajectory,
                "admitted_weak",
                "telemetry-close-filtered-pending-human-review",
                close_run.frame_start,
                response_index + 1,
                candidates=matching,
                intervals=(interval,),
            )
        )

    for candidate in close_candidates:
        if candidate.digest not in consumed_candidates:
            decisions.append(
                _decision(
                    trajectory,
                    "ambiguous",
                    "raw-candidate-not-aligned-to-persistent-close-command",
                    candidate.frame_start,
                    candidate.frame_end,
                    candidates=(candidate,),
                )
            )

    if not close_runs and not close_candidates:
        decisions.append(
            _decision(
                trajectory,
                "absent",
                "no-persistent-close-command-or-raw-candidate",
                None,
                None,
            )
        )

    for candidate in object_candidates:
        has_object_evidence = any(
            evidence.startswith(OBJECT_EVIDENCE_PREFIXES) for evidence in candidate.evidence
        )
        reason = (
            "object-relative-detector-not-admitted"
            if has_object_evidence
            else "object-relative-evidence-absent"
        )
        decisions.append(
            _decision(
                trajectory,
                "deferred",
                reason,
                candidate.frame_start,
                candidate.frame_end,
                candidates=(candidate,),
            )
        )

    ordered_intervals = tuple(
        sorted(intervals, key=lambda item: (item.frame_start, item.frame_end, item.digest))
    )
    ordered_decisions = tuple(
        sorted(
            decisions,
            key=lambda item: (
                item.event_type,
                -1 if item.frame_start is None else item.frame_start,
                -1 if item.frame_end is None else item.frame_end,
                item.status,
                item.candidate_digests,
            ),
        )
    )
    return EventExtractionResult(ordered_candidates, ordered_intervals, ordered_decisions)


def publish_event_extraction(
    output_dir: Path,
    results: Iterable[EventExtractionResult],
    *,
    project_git_revision: str,
    project_tree_clean: bool,
    annotation_version: str,
    config: EventExtractionConfig | None = None,
) -> dict[str, Any]:
    """Atomically publish canonical candidates, intervals, decisions, and manifest."""

    if type(project_tree_clean) is not bool:
        raise TypeError("project_tree_clean must be a bool")
    extraction_config = config or EventExtractionConfig()
    output_dir = output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(output_dir.name + ".partial")
    conflicts = [path for path in (output_dir, staging_dir) if path.exists()]
    if conflicts:
        raise FileExistsError(f"event extraction refuses to overwrite outputs: {conflicts}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    ordered_results = tuple(results)
    if not ordered_results:
        raise EventExtractionError("cannot publish an empty event extraction")
    candidates = tuple(
        sorted(
            (candidate for result in ordered_results for candidate in result.raw_candidates),
            key=_contract_sort_key,
        )
    )
    intervals = tuple(
        sorted(
            (interval for result in ordered_results for interval in result.intervals),
            key=_contract_sort_key,
        )
    )
    decisions = tuple(
        sorted(
            (decision for result in ordered_results for decision in result.decisions),
            key=lambda item: (
                item.source_episode_index,
                item.event_type,
                -1 if item.frame_start is None else item.frame_start,
                item.status,
                item.candidate_digests,
            ),
        )
    )
    _require_unique_candidates(candidates)
    _validate_publication_records(
        candidates,
        intervals,
        decisions,
        project_git_revision=project_git_revision,
        annotation_version=annotation_version,
    )
    staging_dir.mkdir()
    try:
        payloads = {
            "event-candidates.jsonl": _jsonl_payload(
                candidate.to_dict() for candidate in candidates
            ),
            "event-intervals.jsonl": _jsonl_payload(interval.to_dict() for interval in intervals),
            "event-decisions.jsonl": _jsonl_payload(decision.to_dict() for decision in decisions),
        }
        for filename, payload in payloads.items():
            (staging_dir / filename).write_bytes(payload)

        decision_counts = {
            status: sum(decision.status == status for decision in decisions)
            for status in ("admitted_weak", "absent", "ambiguous", "deferred")
        }
        manifest: dict[str, Any] = {
            "artifact_type": EVENT_EXTRACTION_ARTIFACT_TYPE,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "source": {
                "dataset_repo_id": SOURCE_DATASET_REPO_ID,
                "dataset_revision": SOURCE_DATASET_REVISION,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            },
            "generator": {
                "name": EVENT_EXTRACTOR_NAME,
                "version": EVENT_EXTRACTOR_VERSION,
                "project_git_revision": project_git_revision,
                "annotation_version": annotation_version,
                "config": asdict(extraction_config),
            },
            "episode_count": len(
                {decision.source_episode_index for decision in decisions}
            ),
            "candidate_count": len(candidates),
            "interval_count": len(intervals),
            "decision_count": len(decisions),
            "decision_counts": decision_counts,
            "records_sha256": {
                filename: hashlib.sha256(payload).hexdigest()
                for filename, payload in payloads.items()
            },
            "publishable": project_tree_clean,
        }
        manifest["digest"] = contract_digest(manifest)
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging_dir.replace(output_dir)
        return manifest
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _validate_rows(rows: Sequence[TrajectoryRow], trajectory: TrajectoryRecord) -> None:
    if not rows:
        raise EventExtractionError("event extraction requires nonempty trajectory rows")
    provenance = trajectory.provenance
    if len(rows) != provenance.source_episode_length:
        raise EventExtractionError("trajectory row count differs from provenance")
    for frame_index, row in enumerate(rows):
        if (
            row.frame_index != frame_index
            or row.episode_index != provenance.source_episode_index
            or row.task_index != provenance.source_task_index
        ):
            raise EventExtractionError("trajectory rows are not contiguous within one episode")


def _validate_derived_provenance(
    provenance: Provenance,
    trajectory: TrajectoryRecord,
) -> None:
    source = trajectory.provenance
    if (
        provenance.source_episode_index != source.source_episode_index
        or provenance.source_task_index != source.source_task_index
        or provenance.source_episode_length != source.source_episode_length
        or provenance.source_dataset_repo_id != source.source_dataset_repo_id
        or provenance.source_dataset_revision != source.source_dataset_revision
        or provenance.source_manifest_sha256 != source.source_manifest_sha256
    ):
        raise EventExtractionError(
            "interval provenance must retain the trajectory source identity"
        )


def _validate_candidates(
    candidates: Sequence[EventCandidate], trajectory: TrajectoryRecord
) -> None:
    _require_unique_candidates(candidates)
    for candidate in candidates:
        validate_trajectory_reference(candidate, trajectory)


def _require_unique_candidates(candidates: Sequence[EventCandidate]) -> None:
    seen: set[str] = set()
    for candidate in candidates:
        record = candidate.to_dict()
        validate_contract(record)
        if record["artifact_type"] != EVENT_CANDIDATE_ARTIFACT_TYPE:
            raise EventExtractionError("raw candidate stream contains a non-candidate record")
        if candidate.digest in seen:
            raise EventExtractionError(f"duplicate candidate digest: {candidate.digest}")
        seen.add(candidate.digest)


def _validate_publication_records(
    candidates: Sequence[EventCandidate],
    intervals: Sequence[EventInterval],
    decisions: Sequence[EventDecision],
    *,
    project_git_revision: str,
    annotation_version: str,
) -> None:
    candidate_by_digest = {candidate.digest: candidate for candidate in candidates}
    interval_by_digest: dict[str, EventInterval] = {}
    for interval in intervals:
        validate_contract(interval.to_dict())
        if (
            interval.provenance.project_git_revision != project_git_revision
            or interval.provenance.annotation_version != annotation_version
        ):
            raise EventExtractionError(
                "final interval provenance does not match the extraction publication"
            )
        if interval.digest in interval_by_digest:
            raise EventExtractionError(f"duplicate interval digest: {interval.digest}")
        interval_by_digest[interval.digest] = interval
        for candidate_digest in interval.candidate_digests:
            candidate = candidate_by_digest.get(candidate_digest)
            if candidate is None:
                raise EventExtractionError(
                    f"interval references unknown candidate: {candidate_digest}"
                )
            if (
                candidate.event_type != interval.event_type
                or candidate.trajectory_digest != interval.trajectory_digest
            ):
                raise EventExtractionError("interval and candidate event identity differ")

    referenced_candidates: list[str] = []
    referenced_intervals: list[str] = []
    for decision in decisions:
        for candidate_digest in decision.candidate_digests:
            candidate = candidate_by_digest.get(candidate_digest)
            if candidate is None:
                raise EventExtractionError(
                    f"decision references unknown candidate: {candidate_digest}"
                )
            if (
                candidate.provenance.source_episode_index != decision.source_episode_index
                or candidate.event_type != decision.event_type
            ):
                raise EventExtractionError("decision and candidate event identity differ")
            referenced_candidates.append(candidate_digest)
        for interval_digest in decision.interval_digests:
            interval = interval_by_digest.get(interval_digest)
            if interval is None:
                raise EventExtractionError(
                    f"decision references unknown interval: {interval_digest}"
                )
            if (
                interval.provenance.source_episode_index != decision.source_episode_index
                or interval.event_type != decision.event_type
            ):
                raise EventExtractionError("decision and interval event identity differ")
            referenced_intervals.append(interval_digest)
    if sorted(referenced_candidates) != sorted(candidate_by_digest):
        raise EventExtractionError("each raw candidate must appear in exactly one decision")
    if sorted(referenced_intervals) != sorted(interval_by_digest):
        raise EventExtractionError("each final interval must appear in exactly one decision")


def _gripper_command(row: TrajectoryRow) -> float:
    command = row.action[-1]
    if command not in {LIBERO_OPEN_COMMAND, LIBERO_CLOSE_COMMAND}:
        raise EventLabelingError(
            f"episode {row.episode_index} frame {row.frame_index} has "
            f"unsupported gripper command {command}"
        )
    return command


def _centered_median(values: Sequence[float], window: int) -> tuple[float, ...]:
    radius = window // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        sample = sorted(values[max(0, index - radius) : min(len(values), index + radius + 1)])
        middle = len(sample) // 2
        if len(sample) % 2:
            smoothed.append(sample[middle])
        else:
            smoothed.append((sample[middle - 1] + sample[middle]) / 2)
    return tuple(smoothed)


def _debounced_close_runs(
    commands: Sequence[float], *, hysteresis_frames: int
) -> tuple[_CloseRun, ...]:
    if not commands:
        return ()
    stable = commands[0]
    pending_value: float | None = None
    pending_start = 0
    pending_count = 0
    transitions: list[tuple[int, float]] = []
    for frame_index, command in enumerate(commands[1:], start=1):
        if command == stable:
            pending_value = None
            pending_count = 0
            continue
        if command != pending_value:
            pending_value = command
            pending_start = frame_index
            pending_count = 1
        else:
            pending_count += 1
        if pending_count == hysteresis_frames:
            stable = command
            transitions.append((pending_start, stable))
            pending_value = None
            pending_count = 0

    runs: list[_CloseRun] = []
    close_start = 0 if commands[0] == LIBERO_CLOSE_COMMAND else None
    for frame_index, state in transitions:
        if state == LIBERO_CLOSE_COMMAND:
            close_start = frame_index
        elif close_start is not None:
            runs.append(_CloseRun(close_start, frame_index))
            close_start = None
    if close_start is not None:
        runs.append(_CloseRun(close_start, len(commands)))
    return tuple(runs)


def _merged_close_evidence(
    candidates: Sequence[EventCandidate],
    config: EventExtractionConfig,
    close_run: _CloseRun,
    aperture_drop: float,
) -> tuple[str, ...]:
    evidence = [item for candidate in candidates for item in candidate.evidence]
    evidence.extend(
        (
            (
                "action:binary close command passed "
                f"{config.command_hysteresis_frames}-frame hysteresis and persisted "
                f"{close_run.frame_end - close_run.frame_start} frames"
            ),
            (
                "state:centered-median aperture "
                f"window={config.aperture_smoothing_window_frames} dropped "
                f"{aperture_drop:.8f} over {config.response_window_frames} frames"
            ),
        )
    )
    return tuple(dict.fromkeys(evidence))


def _decision(
    trajectory: TrajectoryRecord,
    status: DecisionStatus,
    reason: str,
    frame_start: int | None,
    frame_end: int | None,
    *,
    candidates: Sequence[EventCandidate] = (),
    intervals: Sequence[EventInterval] = (),
) -> EventDecision:
    return EventDecision(
        source_episode_index=trajectory.provenance.source_episode_index,
        source_task_index=trajectory.provenance.source_task_index,
        event_type=(candidates[0].event_type if candidates else "gripper_close"),
        status=status,
        reason=reason,
        frame_start=frame_start,
        frame_end=frame_end,
        candidate_digests=tuple(candidate.digest for candidate in candidates),
        interval_digests=tuple(interval.digest for interval in intervals),
    )


def _contract_sort_key(record: EventCandidate | EventInterval) -> tuple[Any, ...]:
    return (
        record.provenance.source_episode_index,
        record.event_type,
        record.frame_start,
        record.frame_end,
        record.digest,
    )


def _jsonl_payload(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)

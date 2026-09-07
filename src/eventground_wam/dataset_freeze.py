"""Independent validation and sealing for a complete Week 2 dataset build."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from eventground_wam.demonstration_contracts import (
    EVENT_CANDIDATE_ARTIFACT_TYPE,
    EVENT_INTERVAL_ARTIFACT_TYPE,
    MULTI_HORIZON_SAMPLE_ARTIFACT_TYPE,
    SOURCE_DATASET_REPO_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_EPISODE_COUNT,
    SOURCE_FRAME_COUNT,
    SOURCE_MANIFEST_SHA256,
    SOURCE_TASK_COUNT,
    SPLIT_ASSIGNMENT_ARTIFACT_TYPE,
    TRAJECTORY_ARTIFACT_TYPE,
    contract_digest,
    seal_contract,
    validate_contract,
    validate_trajectory_reference,
)
from eventground_wam.event_extraction import EventDecision

DATASET_FREEZE_ARTIFACT_TYPE = "eventground_wam.dataset_freeze"
DATASET_FREEZE_SCHEMA_VERSION = 1
SOURCE_IDENTITY_KEYS = (
    "source_dataset_repo_id",
    "source_dataset_revision",
    "source_manifest_sha256",
    "source_episode_index",
    "source_task_index",
    "source_episode_length",
)
DETERMINISTIC_FILES = (
    "candidates/selection.json",
    "candidates/trajectory-contracts.jsonl",
    "candidates/event-candidates.jsonl",
    "candidates/review-items.jsonl",
    "events/event-candidates.jsonl",
    "events/event-intervals.jsonl",
    "events/event-decisions.jsonl",
    "events/manifest.json",
    "splits/split-assignments.jsonl",
    "splits/multi-horizon-samples.jsonl",
    "splits/split-sample-manifest.json",
)


class DatasetFreezeError(ValueError):
    """Raised when a candidate dataset version cannot be frozen safely."""


def validate_and_seal_dataset_freeze(
    primary_root: Path,
    replica_root: Path,
    *,
    dataset_version: str,
    project_git_revision: str,
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate full primary artifacts, compare a replica, and seal a freeze manifest."""

    started = time.perf_counter()
    if not dataset_version.strip():
        raise DatasetFreezeError("dataset_version must be nonempty")
    _require_revision(project_git_revision)
    primary_root = primary_root.expanduser().resolve()
    replica_root = replica_root.expanduser().resolve()
    if primary_root == replica_root:
        raise DatasetFreezeError("primary and replica roots must differ")

    primary = _validate_primary(primary_root, project_git_revision)
    deterministic_files = _compare_replica(primary_root, replica_root)
    stage_benchmarks = _validate_benchmark(benchmark)
    validation_seconds = time.perf_counter() - started

    event_targets = primary["split_manifest"]["sampling"]["event_targets"]
    manifest = seal_contract(
        {
            "artifact_type": DATASET_FREEZE_ARTIFACT_TYPE,
            "schema_version": DATASET_FREEZE_SCHEMA_VERSION,
            "dataset_version": dataset_version,
            "status": "frozen",
            "source": {
                "dataset_repo_id": SOURCE_DATASET_REPO_ID,
                "dataset_revision": SOURCE_DATASET_REVISION,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            },
            "generator": {
                "project_git_revision": project_git_revision,
                "project_tree_clean": True,
                "candidate_annotation_version": primary["candidate_report"]["generator"][
                    "annotation_version"
                ],
                "event_annotation_version": primary["event_manifest"]["generator"][
                    "annotation_version"
                ],
                "split_annotation_version": primary["split_manifest"]["generator"][
                    "annotation_version"
                ],
            },
            "scope": {
                "complete_source": True,
                "episode_count": SOURCE_EPISODE_COUNT,
                "frame_count": SOURCE_FRAME_COUNT,
                "task_count": SOURCE_TASK_COUNT,
            },
            "events": {
                "candidate_count": primary["event_manifest"]["candidate_count"],
                "interval_count": primary["event_manifest"]["interval_count"],
                "decision_count": primary["event_manifest"]["decision_count"],
                "decision_counts": primary["event_manifest"]["decision_counts"],
                "reliable_reviewed_interval_count": event_targets[
                    "eligible_event_count"
                ],
                "ontology_status": "weak_only_human_quality_gate_pending",
            },
            "splits": primary["split_manifest"]["grouping"],
            "samples": {
                **primary["split_manifest"]["sampling"],
                "record_count": primary["sample_validation"]["record_count"],
            },
            "validation": {
                "trajectory_contracts_valid": True,
                "event_contracts_and_lineage_valid": True,
                "split_contracts_and_isolation_valid": True,
                "all_sample_contracts_and_foreign_keys_valid": True,
                "manifest_and_file_digests_valid": True,
                "replica_byte_identical": True,
                "deterministic_file_count": len(deterministic_files),
                "validation_seconds": validation_seconds,
            },
            "determinism": {
                "method": "independent-complete-regeneration-byte-comparison-v1",
                "replica_byte_identical": True,
                "files": deterministic_files,
            },
            "benchmark": stage_benchmarks,
            "cache_decision": {
                "decoded_image_cache_enabled": False,
                "reason": (
                    "the frozen indexes retain source references and initial-frame digests; "
                    "no decoded-image cache is required for correctness"
                ),
            },
            "artifacts": primary["artifacts"],
            "publishable": True,
            "event_conditioned_training_ready": event_targets["eligible_event_count"] > 0,
            "quality_gate": "pending_human_boundary_review",
        }
    )
    return manifest


def _validate_primary(root: Path, project_revision: str) -> dict[str, Any]:
    candidate_dir = root / "candidates"
    event_dir = root / "events"
    split_dir = root / "splits"
    candidate_report = _load_json(candidate_dir / "feasibility-report.json")
    _validate_candidate_report(candidate_report, project_revision)

    trajectories, trajectory_sha, trajectory_bytes = _load_contract_stream(
        candidate_dir / "trajectory-contracts.jsonl",
        TRAJECTORY_ARTIFACT_TYPE,
    )
    if len(trajectories) != SOURCE_EPISODE_COUNT:
        raise DatasetFreezeError("trajectory stream does not cover the complete source")
    trajectory_by_digest: dict[str, Mapping[str, Any]] = {}
    trajectory_by_episode: dict[int, Mapping[str, Any]] = {}
    frame_count = 0
    task_indices: set[int] = set()
    for expected_episode, trajectory in enumerate(trajectories):
        provenance = trajectory["provenance"]
        episode_index = int(provenance["source_episode_index"])
        if episode_index != expected_episode:
            raise DatasetFreezeError("trajectory stream is not canonical and contiguous")
        trajectory_by_digest[str(trajectory["digest"])] = trajectory
        trajectory_by_episode[episode_index] = trajectory
        frame_count += int(provenance["source_episode_length"])
        task_indices.add(int(provenance["source_task_index"]))
    if frame_count != SOURCE_FRAME_COUNT or task_indices != set(range(SOURCE_TASK_COUNT)):
        raise DatasetFreezeError("trajectory stream totals differ from the pinned source")

    candidate_input_path = candidate_dir / "event-candidates.jsonl"
    candidate_input_sha, candidate_input_bytes = _sha256_file(candidate_input_path)
    event_manifest = _load_sealed_manifest(event_dir / "manifest.json")
    _validate_stage_manifest(event_manifest, project_revision, "event extraction")
    if int(event_manifest["episode_count"]) != SOURCE_EPISODE_COUNT:
        raise DatasetFreezeError("event extraction is not complete-source")

    candidates, candidate_sha, candidate_bytes = _load_contract_stream(
        event_dir / "event-candidates.jsonl",
        EVENT_CANDIDATE_ARTIFACT_TYPE,
    )
    if candidate_sha != candidate_input_sha or candidate_bytes != candidate_input_bytes:
        raise DatasetFreezeError("event extraction did not preserve raw candidate bytes")
    if candidate_sha != event_manifest["records_sha256"]["event-candidates.jsonl"]:
        raise DatasetFreezeError("event candidate SHA-256 differs from its manifest")
    candidate_by_digest: dict[str, Mapping[str, Any]] = {}
    for candidate in candidates:
        trajectory = trajectory_by_digest.get(str(candidate["trajectory_digest"]))
        if trajectory is None:
            raise DatasetFreezeError("event candidate references an unknown trajectory")
        validate_trajectory_reference(candidate, trajectory)
        candidate_by_digest[str(candidate["digest"])] = candidate
    if len(candidate_by_digest) != len(candidates):
        raise DatasetFreezeError("event candidates contain duplicate digests")

    intervals, interval_sha, interval_bytes = _load_contract_stream(
        event_dir / "event-intervals.jsonl",
        EVENT_INTERVAL_ARTIFACT_TYPE,
    )
    if interval_sha != event_manifest["records_sha256"]["event-intervals.jsonl"]:
        raise DatasetFreezeError("event interval SHA-256 differs from its manifest")
    interval_by_digest: dict[str, Mapping[str, Any]] = {}
    for interval in intervals:
        trajectory = trajectory_by_digest.get(str(interval["trajectory_digest"]))
        if trajectory is None:
            raise DatasetFreezeError("event interval references an unknown trajectory")
        validate_trajectory_reference(interval, trajectory)
        for candidate_digest in interval["candidate_digests"]:
            candidate = candidate_by_digest.get(str(candidate_digest))
            if candidate is None:
                raise DatasetFreezeError("event interval references an unknown candidate")
            if (
                candidate["trajectory_digest"] != interval["trajectory_digest"]
                or candidate["event_type"] != interval["event_type"]
            ):
                raise DatasetFreezeError("event interval and candidate lineage disagree")
        interval_by_digest[str(interval["digest"])] = interval
    if len(interval_by_digest) != len(intervals):
        raise DatasetFreezeError("event intervals contain duplicate digests")

    decisions = _validate_decisions(
        event_dir / "event-decisions.jsonl",
        trajectory_by_episode,
        candidate_by_digest,
        interval_by_digest,
        event_manifest,
    )
    split_manifest = _load_sealed_manifest(split_dir / "split-sample-manifest.json")
    _validate_stage_manifest(split_manifest, project_revision, "split/sample indexing")
    if not split_manifest.get("complete_source"):
        raise DatasetFreezeError("split/sample manifest is not complete-source")

    assignments, split_sha, split_bytes = _load_contract_stream(
        split_dir / "split-assignments.jsonl",
        SPLIT_ASSIGNMENT_ARTIFACT_TYPE,
    )
    assignment_by_digest: dict[str, Mapping[str, Any]] = {}
    assignment_episodes: set[int] = set()
    group_split: dict[str, str] = {}
    groups_by_split: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for assignment in assignments:
        trajectory = trajectory_by_digest.get(str(assignment["trajectory_digest"]))
        if trajectory is None:
            raise DatasetFreezeError("split assignment references an unknown trajectory")
        validate_trajectory_reference(assignment, trajectory)
        episode_index = int(assignment["provenance"]["source_episode_index"])
        if episode_index in assignment_episodes:
            raise DatasetFreezeError("multiple split assignments exist for one episode")
        assignment_episodes.add(episode_index)
        digest = str(assignment["digest"])
        assignment_by_digest[digest] = assignment
        group = str(assignment["initial_condition_digest"])
        split = str(assignment["split"])
        previous = group_split.setdefault(group, split)
        if previous != split:
            raise DatasetFreezeError("an initial-condition group crosses splits")
        groups_by_split[split].add(group)
    if assignment_episodes != set(range(SOURCE_EPISODE_COUNT)):
        raise DatasetFreezeError("split assignments do not cover every source episode")
    if any(
        groups_by_split[left] & groups_by_split[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise DatasetFreezeError("split group overlap detected")
    split_output = split_manifest["outputs"]["split_assignments"]
    if split_sha != split_output["sha256"] or len(assignments) != split_output["record_count"]:
        raise DatasetFreezeError("split assignment output differs from its manifest")

    sample_validation = _validate_sample_stream(
        split_dir / "multi-horizon-samples.jsonl",
        assignment_by_digest,
        trajectory_by_digest,
        interval_by_digest,
        split_manifest,
    )
    if len(candidates) != event_manifest["candidate_count"]:
        raise DatasetFreezeError("candidate count differs from event manifest")
    if len(intervals) != event_manifest["interval_count"]:
        raise DatasetFreezeError("interval count differs from event manifest")
    if decisions["record_count"] != event_manifest["decision_count"]:
        raise DatasetFreezeError("decision count differs from event manifest")

    artifacts = {
        "trajectory_contracts": _artifact_entry(
            "candidates/trajectory-contracts.jsonl",
            trajectory_sha,
            trajectory_bytes,
            len(trajectories),
        ),
        "event_candidates": _artifact_entry(
            "events/event-candidates.jsonl",
            candidate_sha,
            candidate_bytes,
            len(candidates),
        ),
        "event_intervals": _artifact_entry(
            "events/event-intervals.jsonl",
            interval_sha,
            interval_bytes,
            len(intervals),
        ),
        "event_decisions": _artifact_entry(
            "events/event-decisions.jsonl",
            decisions["sha256"],
            decisions["byte_count"],
            decisions["record_count"],
        ),
        "split_assignments": _artifact_entry(
            "splits/split-assignments.jsonl",
            split_sha,
            split_bytes,
            len(assignments),
        ),
        "multi_horizon_samples": _artifact_entry(
            "splits/multi-horizon-samples.jsonl",
            sample_validation["sha256"],
            sample_validation["byte_count"],
            sample_validation["record_count"],
        ),
    }
    return {
        "candidate_report": candidate_report,
        "event_manifest": event_manifest,
        "split_manifest": split_manifest,
        "sample_validation": sample_validation,
        "artifacts": artifacts,
    }


def _validate_sample_stream(
    path: Path,
    assignment_by_digest: Mapping[str, Mapping[str, Any]],
    trajectory_by_digest: Mapping[str, Mapping[str, Any]],
    interval_by_digest: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    horizon_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for line_number, line, sample in _iter_jsonl(path):
        digest.update(line)
        byte_count += len(line)
        try:
            validate_contract(sample)
        except ValueError as error:
            raise DatasetFreezeError(f"{path}:{line_number}: invalid sample: {error}") from error
        if sample["artifact_type"] != MULTI_HORIZON_SAMPLE_ARTIFACT_TYPE:
            raise DatasetFreezeError(f"{path}:{line_number}: unexpected artifact type")
        assignment = assignment_by_digest.get(str(sample["split_assignment_digest"]))
        if assignment is None:
            raise DatasetFreezeError(f"{path}:{line_number}: unknown split assignment")
        trajectory = trajectory_by_digest.get(str(sample["trajectory_digest"]))
        if trajectory is None:
            raise DatasetFreezeError(f"{path}:{line_number}: unknown trajectory")
        if (
            sample["trajectory_digest"] != assignment["trajectory_digest"]
            or sample["split"] != assignment["split"]
        ):
            raise DatasetFreezeError(f"{path}:{line_number}: sample split lineage disagrees")
        if any(
            sample["provenance"][key] != assignment["provenance"][key]
            for key in SOURCE_IDENTITY_KEYS
        ):
            raise DatasetFreezeError(f"{path}:{line_number}: sample provenance disagrees")
        horizon = sample["horizon"]
        if horizon == "next-event":
            event = interval_by_digest.get(str(sample["target_event_digest"]))
            if event is None:
                raise DatasetFreezeError(f"{path}:{line_number}: unknown target event")
            if (
                event["trajectory_digest"] != sample["trajectory_digest"]
                or event["frame_start"] != sample["target_frame_index"]
                or event["ontology_tier"] != "reliable"
                or event["review_status"] not in {"accepted", "corrected"}
            ):
                raise DatasetFreezeError(f"{path}:{line_number}: ineligible target event")
        elif sample["target_event_digest"] is not None:
            raise DatasetFreezeError(f"{path}:{line_number}: fixed sample has target event")
        count += 1
        horizon_counts[str(horizon)] += 1
        split_counts[str(sample["split"])] += 1

    output = manifest["outputs"]["multi_horizon_samples"]
    expected_horizons = manifest["sampling"]["sample_counts_by_horizon"]
    expected_splits = manifest["sampling"]["sample_counts_by_split"]
    actual_sha = digest.hexdigest()
    if actual_sha != output["sha256"] or count != output["record_count"]:
        raise DatasetFreezeError("sample output SHA-256 or count differs from its manifest")
    normalized_horizons = {
        str(key): horizon_counts[str(key)] for key in expected_horizons
    }
    normalized_splits = {str(key): split_counts[str(key)] for key in expected_splits}
    if set(horizon_counts) - set(normalized_horizons):
        raise DatasetFreezeError("sample output contains an unexpected horizon")
    if set(split_counts) - set(normalized_splits):
        raise DatasetFreezeError("sample output contains an unexpected split")
    if normalized_horizons != {str(key): value for key, value in expected_horizons.items()}:
        raise DatasetFreezeError("sample horizon counts differ from the manifest")
    if normalized_splits != {str(key): value for key, value in expected_splits.items()}:
        raise DatasetFreezeError("sample split counts differ from the manifest")
    return {
        "sha256": actual_sha,
        "record_count": count,
        "byte_count": byte_count,
    }


def _validate_decisions(
    path: Path,
    trajectory_by_episode: Mapping[int, Mapping[str, Any]],
    candidate_by_digest: Mapping[str, Mapping[str, Any]],
    interval_by_digest: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    status_counts: Counter[str] = Counter()
    candidate_refs: list[str] = []
    interval_refs: list[str] = []
    decision_episodes: set[int] = set()
    for line_number, line, raw in _iter_jsonl(path):
        digest.update(line)
        byte_count += len(line)
        try:
            decision = EventDecision(
                source_episode_index=raw["source_episode_index"],
                source_task_index=raw["source_task_index"],
                event_type=raw["event_type"],
                status=raw["status"],
                reason=raw["reason"],
                frame_start=raw["frame_start"],
                frame_end=raw["frame_end"],
                candidate_digests=tuple(raw["candidate_digests"]),
                interval_digests=tuple(raw["interval_digests"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetFreezeError(f"{path}:{line_number}: invalid decision") from error
        if decision.to_dict() != raw:
            raise DatasetFreezeError(f"{path}:{line_number}: decision fields are not canonical")
        trajectory = trajectory_by_episode.get(decision.source_episode_index)
        if trajectory is None or (
            trajectory["provenance"]["source_task_index"] != decision.source_task_index
        ):
            raise DatasetFreezeError(f"{path}:{line_number}: decision source is unknown")
        for candidate_digest in decision.candidate_digests:
            candidate = candidate_by_digest.get(candidate_digest)
            if candidate is None:
                raise DatasetFreezeError(f"{path}:{line_number}: unknown candidate reference")
            if candidate["provenance"]["source_episode_index"] != decision.source_episode_index:
                raise DatasetFreezeError(f"{path}:{line_number}: candidate source disagrees")
            candidate_refs.append(candidate_digest)
        for interval_digest in decision.interval_digests:
            interval = interval_by_digest.get(interval_digest)
            if interval is None:
                raise DatasetFreezeError(f"{path}:{line_number}: unknown interval reference")
            if interval["provenance"]["source_episode_index"] != decision.source_episode_index:
                raise DatasetFreezeError(f"{path}:{line_number}: interval source disagrees")
            interval_refs.append(interval_digest)
        record_count += 1
        status_counts[decision.status] += 1
        decision_episodes.add(decision.source_episode_index)
    actual_sha = digest.hexdigest()
    if actual_sha != manifest["records_sha256"]["event-decisions.jsonl"]:
        raise DatasetFreezeError("event decision SHA-256 differs from its manifest")
    expected_statuses = manifest["decision_counts"]
    normalized_statuses = {str(key): status_counts[str(key)] for key in expected_statuses}
    if set(status_counts) - set(normalized_statuses) or normalized_statuses != expected_statuses:
        raise DatasetFreezeError("event decision counts differ from the manifest")
    if sorted(candidate_refs) != sorted(candidate_by_digest):
        raise DatasetFreezeError("event decisions do not reference each candidate exactly once")
    if sorted(interval_refs) != sorted(interval_by_digest):
        raise DatasetFreezeError("event decisions do not reference each interval exactly once")
    if decision_episodes != set(range(SOURCE_EPISODE_COUNT)):
        raise DatasetFreezeError("event decisions do not cover every source episode")
    return {
        "sha256": actual_sha,
        "record_count": record_count,
        "byte_count": byte_count,
    }


def _compare_replica(primary_root: Path, replica_root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative in DETERMINISTIC_FILES:
        primary_sha, primary_bytes = _sha256_file(primary_root / relative)
        replica_sha, replica_bytes = _sha256_file(replica_root / relative)
        if primary_sha != replica_sha or primary_bytes != replica_bytes:
            raise DatasetFreezeError(f"replica differs from primary at {relative}")
        files[relative] = {
            "sha256": primary_sha,
            "byte_count": primary_bytes,
        }
    return files


def _validate_candidate_report(report: Mapping[str, Any], revision: str) -> None:
    _require_source(report.get("source"), "candidate report")
    generator = report.get("generator")
    selection = report.get("selection")
    if not isinstance(generator, Mapping) or not isinstance(selection, Mapping):
        raise DatasetFreezeError("candidate report is missing generator or selection")
    if (
        report.get("status") != "full_source_complete"
        or report.get("publishable") is not True
        or generator.get("project_git_revision") != revision
        or generator.get("project_tree_clean") is not True
        or selection.get("complete_source") is not True
        or selection.get("selected_episode_count") != SOURCE_EPISODE_COUNT
        or selection.get("selected_frame_count") != SOURCE_FRAME_COUNT
        or selection.get("selected_task_count") != SOURCE_TASK_COUNT
    ):
        raise DatasetFreezeError("candidate report is not a clean complete-source run")


def _validate_stage_manifest(
    manifest: Mapping[str, Any], revision: str, name: str
) -> None:
    _require_source(manifest.get("source"), name)
    generator = manifest.get("generator")
    if not isinstance(generator, Mapping):
        raise DatasetFreezeError(f"{name} manifest is missing generator metadata")
    tree_clean = generator.get("project_tree_clean", manifest.get("publishable"))
    if (
        generator.get("project_git_revision") != revision
        or tree_clean is not True
        or manifest.get("publishable") is not True
    ):
        raise DatasetFreezeError(f"{name} was not generated from the clean frozen revision")


def _require_source(value: Any, name: str) -> None:
    if not isinstance(value, Mapping) or (
        value.get("dataset_repo_id") != SOURCE_DATASET_REPO_ID
        or value.get("dataset_revision") != SOURCE_DATASET_REVISION
        or value.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
    ):
        raise DatasetFreezeError(f"{name} source identity differs from the pinned dataset")


def _validate_benchmark(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    stages = benchmark.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != {"primary", "replica"}:
        raise DatasetFreezeError("benchmark must contain primary and replica stages")
    normalized: dict[str, Any] = {}
    for run_name in ("primary", "replica"):
        run = stages[run_name]
        if not isinstance(run, Mapping):
            raise DatasetFreezeError(f"benchmark {run_name} must be an object")
        normalized[run_name] = {}
        for stage in ("candidate_generation", "event_extraction", "split_sample_indexing"):
            value = run.get(stage)
            if not isinstance(value, Mapping):
                raise DatasetFreezeError(f"benchmark is missing {run_name}.{stage}")
            seconds = value.get("wall_seconds")
            if type(seconds) not in {int, float} or seconds <= 0:
                raise DatasetFreezeError(f"benchmark {run_name}.{stage} is not positive")
            normalized[run_name][stage] = dict(value)
    return normalized


def _load_contract_stream(
    path: Path, artifact_type: str
) -> tuple[list[dict[str, Any]], str, int]:
    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    byte_count = 0
    seen: set[str] = set()
    for line_number, line, record in _iter_jsonl(path):
        digest.update(line)
        byte_count += len(line)
        try:
            validate_contract(record)
        except (TypeError, ValueError) as error:
            raise DatasetFreezeError(f"{path}:{line_number}: invalid contract: {error}") from error
        if record["artifact_type"] != artifact_type:
            raise DatasetFreezeError(f"{path}:{line_number}: unexpected artifact type")
        record_digest = str(record["digest"])
        if record_digest in seen:
            raise DatasetFreezeError(f"{path}:{line_number}: duplicate contract digest")
        seen.add(record_digest)
        records.append(record)
    return records, digest.hexdigest(), byte_count


def _iter_jsonl(path: Path) -> Iterator[tuple[int, bytes, dict[str, Any]]]:
    try:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise DatasetFreezeError(f"{path}:{line_number}: blank JSONL line")
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetFreezeError(f"{path}:{line_number}: invalid JSON") from error
                if not isinstance(value, dict):
                    raise DatasetFreezeError(f"{path}:{line_number}: record must be an object")
                yield line_number, line, value
    except OSError as error:
        raise DatasetFreezeError(f"cannot read required freeze artifact {path}") from error


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetFreezeError(f"cannot read valid JSON from {path}") from error
    if not isinstance(value, dict):
        raise DatasetFreezeError(f"{path} must contain a JSON object")
    return value


def _load_sealed_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("digest") != contract_digest(manifest):
        raise DatasetFreezeError(f"manifest digest mismatch: {path}")
    return manifest


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise DatasetFreezeError(f"cannot hash required freeze artifact {path}") from error
    return digest.hexdigest(), byte_count


def _artifact_entry(path: str, sha256: str, byte_count: int, record_count: int) -> dict[str, Any]:
    return {
        "path": path,
        "sha256": sha256,
        "byte_count": byte_count,
        "record_count": record_count,
    }


def _require_revision(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise DatasetFreezeError("project_git_revision must be a full lowercase Git revision")

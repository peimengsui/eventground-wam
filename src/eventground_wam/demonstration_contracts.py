"""Versioned, JSON-serializable contracts for derived demonstration artifacts.

Frame ranges in this module are always half-open: ``[frame_start, frame_end)``.
The contracts intentionally contain references to source files rather than image
or tensor payloads so a reader can stream the pinned LeRobot snapshot in place.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, ClassVar

CONTRACT_SCHEMA_VERSION = 1

TRAJECTORY_ARTIFACT_TYPE = "eventground_wam.trajectory"
EVENT_CANDIDATE_ARTIFACT_TYPE = "eventground_wam.event_candidate"
EVENT_INTERVAL_ARTIFACT_TYPE = "eventground_wam.event_interval"
SPLIT_ASSIGNMENT_ARTIFACT_TYPE = "eventground_wam.split_assignment"
MULTI_HORIZON_SAMPLE_ARTIFACT_TYPE = "eventground_wam.multi_horizon_sample"
ARTIFACT_TYPES = (
    TRAJECTORY_ARTIFACT_TYPE,
    EVENT_CANDIDATE_ARTIFACT_TYPE,
    EVENT_INTERVAL_ARTIFACT_TYPE,
    SPLIT_ASSIGNMENT_ARTIFACT_TYPE,
    MULTI_HORIZON_SAMPLE_ARTIFACT_TYPE,
)

SOURCE_DATASET_REPO_ID = "Sylvest/libero_plus_lerobot"
SOURCE_DATASET_REVISION = "22c57433fef692b5b9ecc0795344daac7fa867a5"
SOURCE_MANIFEST_SHA256 = "e673a1d3e58a1756879b4f30ce38474aa3407f97232ee5184fd7e1e6ed901510"
SOURCE_EPISODE_COUNT = 14_347
SOURCE_TASK_COUNT = 40
SOURCE_FPS = 20
SOURCE_STATE_DIMENSION = 8
SOURCE_ACTION_DIMENSION = 7
SOURCE_IMAGE_SHAPE = (256, 256, 3)
SOURCE_CAMERA_KEYS = ("observation.images.front", "observation.images.wrist")
SUPPORTED_HORIZONS = (2, 4, 8, 16, 32, "next-event")

EVENT_TYPES = (
    "reach",
    "contact",
    "gripper_close",
    "grasp_established",
    "lift",
    "transport",
    "target_region_entry",
    "release",
    "success",
)
OBJECT_RELATIVE_EVENT_TYPES = frozenset(EVENT_TYPES) - {"gripper_close"}
EVIDENCE_PREFIXES = (
    "state:",
    "action:",
    "front_rgb:",
    "wrist_rgb:",
    "simulator_sidecar:",
    "human_review:",
)
OBJECT_EVIDENCE_PREFIXES = (
    "front_rgb:",
    "wrist_rgb:",
    "simulator_sidecar:",
    "human_review:",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")
_VERSION_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")


class DemonstrationContractError(ValueError):
    """Raised when a demonstration artifact violates its locked contract."""


def _normalized_json(value: Any, *, path: str = "$") -> Any:
    """Return a JSON-native copy while rejecting lossy or unstable values."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise DemonstrationContractError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise DemonstrationContractError(f"{path} contains a non-string object key")
            normalized[key] = _normalized_json(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _normalized_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise DemonstrationContractError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Serialize a mapping with the canonical encoding used for contract digests."""

    normalized = _normalized_json(value)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def contract_digest(record: Mapping[str, Any]) -> str:
    """Return a SHA-256 over a record excluding its top-level ``digest`` field."""

    payload = dict(record)
    payload.pop("digest", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def seal_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a JSON-native copy with its canonical top-level digest attached."""

    if "digest" in record:
        raise DemonstrationContractError("seal_contract expects an unsealed record")
    sealed = _normalized_json(record)
    sealed["digest"] = contract_digest(sealed)
    return sealed


@dataclass(frozen=True)
class Provenance:
    """Source and generator identity shared by every contract record."""

    source_episode_index: int
    source_task_index: int
    source_episode_length: int
    project_git_revision: str
    annotation_version: str
    source_dataset_repo_id: str = SOURCE_DATASET_REPO_ID
    source_dataset_revision: str = SOURCE_DATASET_REVISION
    source_manifest_sha256: str = SOURCE_MANIFEST_SHA256
    source_initial_seed: int | None = None
    source_has_mujoco_state: bool = False

    def __post_init__(self) -> None:
        _validate_provenance(_normalized_json(asdict(self)))


class _ContractRecord:
    ARTIFACT_TYPE: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the validated canonical wire representation of this record."""

        payload = _normalized_json(asdict(self))
        record = seal_contract(
            {
                "artifact_type": self.ARTIFACT_TYPE,
                "schema_version": CONTRACT_SCHEMA_VERSION,
                **payload,
            }
        )
        validate_contract(record)
        return record

    @property
    def digest(self) -> str:
        """Return the content digest used as this record's stable identifier."""

        return self.to_dict()["digest"]


@dataclass(frozen=True)
class TrajectoryRecord(_ContractRecord):
    """One source episode and the paths needed to stream all of its modalities."""

    ARTIFACT_TYPE: ClassVar[str] = TRAJECTORY_ARTIFACT_TYPE

    provenance: Provenance
    instruction: str
    frame_start: int
    frame_end: int
    parquet_path: str
    front_video_path: str
    wrist_video_path: str
    fps: int = SOURCE_FPS
    state_dimension: int = SOURCE_STATE_DIMENSION
    action_dimension: int = SOURCE_ACTION_DIMENSION
    image_shape: tuple[int, int, int] = SOURCE_IMAGE_SHAPE
    camera_keys: tuple[str, str] = SOURCE_CAMERA_KEYS

    def __post_init__(self) -> None:
        self.to_dict()

    @classmethod
    def from_episode(
        cls,
        provenance: Provenance,
        instruction: str,
    ) -> TrajectoryRecord:
        """Build a trajectory using the pinned LeRobot v2.1 path templates."""

        episode_index = provenance.source_episode_index
        chunk_index = episode_index // 1_000
        return cls(
            provenance=provenance,
            instruction=instruction,
            frame_start=0,
            frame_end=provenance.source_episode_length,
            parquet_path=f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
            front_video_path=(
                f"videos/chunk-{chunk_index:03d}/observation.images.front/"
                f"episode_{episode_index:06d}.mp4"
            ),
            wrist_video_path=(
                f"videos/chunk-{chunk_index:03d}/observation.images.wrist/"
                f"episode_{episode_index:06d}.mp4"
            ),
        )


@dataclass(frozen=True)
class EventCandidate(_ContractRecord):
    """A raw detector proposal retained before filtering or ontology admission."""

    ARTIFACT_TYPE: ClassVar[str] = EVENT_CANDIDATE_ARTIFACT_TYPE

    provenance: Provenance
    trajectory_digest: str
    frame_start: int
    frame_end: int
    event_type: str
    confidence: float
    evidence: tuple[str, ...]
    detector_name: str
    detector_version: str

    def __post_init__(self) -> None:
        self.to_dict()


@dataclass(frozen=True)
class EventInterval(_ContractRecord):
    """A filtered event interval used as a training annotation."""

    ARTIFACT_TYPE: ClassVar[str] = EVENT_INTERVAL_ARTIFACT_TYPE

    provenance: Provenance
    trajectory_digest: str
    frame_start: int
    frame_end: int
    event_type: str
    confidence: float
    evidence: tuple[str, ...]
    candidate_digests: tuple[str, ...]
    ontology_tier: str
    review_status: str

    def __post_init__(self) -> None:
        self.to_dict()


@dataclass(frozen=True)
class SplitAssignment(_ContractRecord):
    """An episode-level, leakage-resistant split assignment."""

    ARTIFACT_TYPE: ClassVar[str] = SPLIT_ASSIGNMENT_ARTIFACT_TYPE

    provenance: Provenance
    trajectory_digest: str
    frame_start: int
    frame_end: int
    split: str
    group_strategy: str
    initial_condition_digest: str

    def __post_init__(self) -> None:
        self.to_dict()


@dataclass(frozen=True)
class MultiHorizonSample(_ContractRecord):
    """One within-episode future target at a fixed or next-event horizon."""

    ARTIFACT_TYPE: ClassVar[str] = MULTI_HORIZON_SAMPLE_ARTIFACT_TYPE

    provenance: Provenance
    trajectory_digest: str
    split_assignment_digest: str
    frame_start: int
    frame_end: int
    anchor_frame_index: int
    target_frame_index: int
    horizon: int | str
    split: str
    target_event_digest: str | None = None

    def __post_init__(self) -> None:
        self.to_dict()


_PROVENANCE_KEYS = {
    "source_episode_index",
    "source_task_index",
    "source_episode_length",
    "project_git_revision",
    "annotation_version",
    "source_dataset_repo_id",
    "source_dataset_revision",
    "source_manifest_sha256",
    "source_initial_seed",
    "source_has_mujoco_state",
}
_COMMON_KEYS = {"artifact_type", "schema_version", "digest", "provenance"}
_RECORD_KEYS = {
    TRAJECTORY_ARTIFACT_TYPE: _COMMON_KEYS
    | {
        "instruction",
        "frame_start",
        "frame_end",
        "parquet_path",
        "front_video_path",
        "wrist_video_path",
        "fps",
        "state_dimension",
        "action_dimension",
        "image_shape",
        "camera_keys",
    },
    EVENT_CANDIDATE_ARTIFACT_TYPE: _COMMON_KEYS
    | {
        "trajectory_digest",
        "frame_start",
        "frame_end",
        "event_type",
        "confidence",
        "evidence",
        "detector_name",
        "detector_version",
    },
    EVENT_INTERVAL_ARTIFACT_TYPE: _COMMON_KEYS
    | {
        "trajectory_digest",
        "frame_start",
        "frame_end",
        "event_type",
        "confidence",
        "evidence",
        "candidate_digests",
        "ontology_tier",
        "review_status",
    },
    SPLIT_ASSIGNMENT_ARTIFACT_TYPE: _COMMON_KEYS
    | {
        "trajectory_digest",
        "frame_start",
        "frame_end",
        "split",
        "group_strategy",
        "initial_condition_digest",
    },
    MULTI_HORIZON_SAMPLE_ARTIFACT_TYPE: _COMMON_KEYS
    | {
        "trajectory_digest",
        "split_assignment_digest",
        "frame_start",
        "frame_end",
        "anchor_frame_index",
        "target_frame_index",
        "horizon",
        "split",
        "target_event_digest",
    },
}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DemonstrationContractError(f"{name} keys differ; missing={missing}, extra={extra}")


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise DemonstrationContractError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise DemonstrationContractError(f"{name} must be at least {minimum}")
    return value


def _require_nonempty_string(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise DemonstrationContractError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise DemonstrationContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_version(value: Any, name: str) -> str:
    if type(value) is not str or _VERSION_RE.fullmatch(value) is None:
        raise DemonstrationContractError(
            f"{name} must contain only lowercase letters, digits, '.', '_', or '-'"
        )
    return value


def _validate_provenance(provenance: Any) -> Mapping[str, Any]:
    if not isinstance(provenance, Mapping):
        raise DemonstrationContractError("provenance must be an object")
    _require_exact_keys(provenance, _PROVENANCE_KEYS, "provenance")
    episode_index = _require_int(provenance["source_episode_index"], "source_episode_index", minimum=0)
    if episode_index >= SOURCE_EPISODE_COUNT:
        raise DemonstrationContractError("source_episode_index is outside the pinned snapshot")
    task_index = _require_int(provenance["source_task_index"], "source_task_index", minimum=0)
    if task_index >= SOURCE_TASK_COUNT:
        raise DemonstrationContractError("source_task_index is outside the pinned snapshot")
    _require_int(provenance["source_episode_length"], "source_episode_length", minimum=1)
    revision = provenance["project_git_revision"]
    if type(revision) is not str or _GIT_REVISION_RE.fullmatch(revision) is None:
        raise DemonstrationContractError("project_git_revision must be a lowercase 40-hex Git SHA")
    _require_version(provenance["annotation_version"], "annotation_version")

    expected = {
        "source_dataset_repo_id": SOURCE_DATASET_REPO_ID,
        "source_dataset_revision": SOURCE_DATASET_REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_initial_seed": None,
    }
    for key, expected_value in expected.items():
        if provenance[key] != expected_value:
            raise DemonstrationContractError(
                f"provenance {key} must be {expected_value!r} for the pinned source"
            )
    if provenance["source_has_mujoco_state"] is not False:
        raise DemonstrationContractError(
            "provenance source_has_mujoco_state must be False for the pinned source"
        )
    return provenance


def _validate_frame_range(record: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    frame_start = _require_int(record["frame_start"], "frame_start", minimum=0)
    frame_end = _require_int(record["frame_end"], "frame_end", minimum=1)
    if frame_start >= frame_end:
        raise DemonstrationContractError("frame range must be nonempty and half-open")
    if frame_end > provenance["source_episode_length"]:
        raise DemonstrationContractError("frame range crosses the source episode boundary")


def _validate_relative_path(value: Any, name: str) -> str:
    text = _require_nonempty_string(value, name)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text in {"", "."}:
        raise DemonstrationContractError(f"{name} must be a safe repository-relative path")
    return text


def _validate_evidence(record: Mapping[str, Any]) -> None:
    evidence = record["evidence"]
    if type(evidence) is not list or not evidence:
        raise DemonstrationContractError("evidence must be a nonempty array")
    if any(type(item) is not str or not item.strip() for item in evidence):
        raise DemonstrationContractError("evidence entries must be nonempty strings")
    if len(evidence) != len(set(evidence)):
        raise DemonstrationContractError("evidence entries must be unique")
    if any(not item.startswith(EVIDENCE_PREFIXES) for item in evidence):
        raise DemonstrationContractError("evidence contains an unsupported modality prefix")
    if record["event_type"] in OBJECT_RELATIVE_EVENT_TYPES and not any(
        item.startswith(OBJECT_EVIDENCE_PREFIXES) for item in evidence
    ):
        raise DemonstrationContractError(
            "object-relative events require RGB, simulator-sidecar, or human-review evidence"
        )


def _validate_event_common(record: Mapping[str, Any]) -> None:
    _require_sha256(record["trajectory_digest"], "trajectory_digest")
    if record["event_type"] not in EVENT_TYPES:
        raise DemonstrationContractError(f"unsupported event_type: {record['event_type']!r}")
    confidence = record["confidence"]
    if type(confidence) is not float or not math.isfinite(confidence):
        raise DemonstrationContractError("confidence must be a finite float")
    if not 0.0 <= confidence <= 1.0:
        raise DemonstrationContractError("confidence must be between 0 and 1")
    _validate_evidence(record)


def _expected_source_paths(episode_index: int) -> tuple[str, str, str]:
    chunk_index = episode_index // 1_000
    stem = f"episode_{episode_index:06d}"
    return (
        f"data/chunk-{chunk_index:03d}/{stem}.parquet",
        f"videos/chunk-{chunk_index:03d}/observation.images.front/{stem}.mp4",
        f"videos/chunk-{chunk_index:03d}/observation.images.wrist/{stem}.mp4",
    )


def _validate_trajectory(record: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    if record["frame_start"] != 0 or record["frame_end"] != provenance["source_episode_length"]:
        raise DemonstrationContractError("trajectory frame range must cover its complete episode")
    _require_nonempty_string(record["instruction"], "instruction")
    paths = tuple(
        _validate_relative_path(record[key], key)
        for key in ("parquet_path", "front_video_path", "wrist_video_path")
    )
    if paths != _expected_source_paths(provenance["source_episode_index"]):
        raise DemonstrationContractError("trajectory paths do not match the pinned LeRobot layout")
    locked_values = {
        "fps": SOURCE_FPS,
        "state_dimension": SOURCE_STATE_DIMENSION,
        "action_dimension": SOURCE_ACTION_DIMENSION,
        "image_shape": list(SOURCE_IMAGE_SHAPE),
        "camera_keys": list(SOURCE_CAMERA_KEYS),
    }
    for key, expected in locked_values.items():
        if record[key] != expected:
            raise DemonstrationContractError(f"trajectory {key} must be {expected!r}")


def _validate_event_candidate(record: Mapping[str, Any]) -> None:
    _validate_event_common(record)
    _require_version(record["detector_name"], "detector_name")
    _require_version(record["detector_version"], "detector_version")


def _validate_event_interval(record: Mapping[str, Any]) -> None:
    _validate_event_common(record)
    candidates = record["candidate_digests"]
    if type(candidates) is not list or not candidates:
        raise DemonstrationContractError("candidate_digests must be a nonempty array")
    for index, digest in enumerate(candidates):
        _require_sha256(digest, f"candidate_digests[{index}]")
    if len(candidates) != len(set(candidates)):
        raise DemonstrationContractError("candidate_digests must be unique")
    if record["ontology_tier"] not in {"reliable", "weak"}:
        raise DemonstrationContractError("ontology_tier must be 'reliable' or 'weak'")
    if record["review_status"] not in {"unreviewed", "accepted", "corrected"}:
        raise DemonstrationContractError("unsupported review_status")


def _validate_split(record: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    _require_sha256(record["trajectory_digest"], "trajectory_digest")
    _require_sha256(record["initial_condition_digest"], "initial_condition_digest")
    if record["split"] not in {"train", "validation", "test"}:
        raise DemonstrationContractError("split must be train, validation, or test")
    if record["group_strategy"] != "task-state-rgb-surrogate-v1":
        raise DemonstrationContractError("unsupported initial-condition grouping strategy")
    if record["frame_start"] != 0 or record["frame_end"] != provenance["source_episode_length"]:
        raise DemonstrationContractError("split assignments must cover a complete episode")


def _validate_sample(record: Mapping[str, Any]) -> None:
    _require_sha256(record["trajectory_digest"], "trajectory_digest")
    _require_sha256(record["split_assignment_digest"], "split_assignment_digest")
    if record["split"] not in {"train", "validation", "test"}:
        raise DemonstrationContractError("split must be train, validation, or test")
    anchor = _require_int(record["anchor_frame_index"], "anchor_frame_index", minimum=0)
    target = _require_int(record["target_frame_index"], "target_frame_index", minimum=0)
    if target <= anchor:
        raise DemonstrationContractError("target_frame_index must follow anchor_frame_index")
    if record["frame_start"] != anchor or record["frame_end"] != target + 1:
        raise DemonstrationContractError("sample frame range must be [anchor, target + 1)")
    horizon = record["horizon"]
    if horizon not in SUPPORTED_HORIZONS or type(horizon) not in {int, str}:
        raise DemonstrationContractError(f"unsupported horizon: {horizon!r}")
    target_event = record["target_event_digest"]
    if horizon == "next-event":
        _require_sha256(target_event, "target_event_digest")
    else:
        if target - anchor != horizon:
            raise DemonstrationContractError("fixed target distance does not match horizon")
        if target_event is not None:
            raise DemonstrationContractError("fixed-horizon samples cannot name a target event")


def validate_contract(record: Mapping[str, Any]) -> None:
    """Strictly validate any supported serialized contract record and its digest."""

    if not isinstance(record, Mapping):
        raise DemonstrationContractError("contract record must be an object")
    artifact_type = record.get("artifact_type")
    if artifact_type not in _RECORD_KEYS:
        raise DemonstrationContractError(f"unsupported artifact_type: {artifact_type!r}")
    _require_exact_keys(record, _RECORD_KEYS[artifact_type], str(artifact_type))
    if record["schema_version"] != CONTRACT_SCHEMA_VERSION or type(record["schema_version"]) is not int:
        raise DemonstrationContractError(
            f"schema_version must be integer {CONTRACT_SCHEMA_VERSION}"
        )
    provenance = _validate_provenance(record["provenance"])
    _validate_frame_range(record, provenance)
    _require_sha256(record["digest"], "digest")
    expected_digest = contract_digest(record)
    if record["digest"] != expected_digest:
        raise DemonstrationContractError(
            f"contract digest mismatch: expected {expected_digest}, found {record['digest']}"
        )

    validators = {
        TRAJECTORY_ARTIFACT_TYPE: lambda item: _validate_trajectory(item, provenance),
        EVENT_CANDIDATE_ARTIFACT_TYPE: _validate_event_candidate,
        EVENT_INTERVAL_ARTIFACT_TYPE: _validate_event_interval,
        SPLIT_ASSIGNMENT_ARTIFACT_TYPE: lambda item: _validate_split(item, provenance),
        MULTI_HORIZON_SAMPLE_ARTIFACT_TYPE: _validate_sample,
    }
    validators[artifact_type](record)


def validate_trajectory_reference(
    record: Mapping[str, Any] | _ContractRecord,
    trajectory: Mapping[str, Any] | TrajectoryRecord,
) -> None:
    """Validate a record's episode identity, bounds, and trajectory foreign key."""

    serialized_record = record.to_dict() if isinstance(record, _ContractRecord) else record
    serialized_trajectory = (
        trajectory.to_dict() if isinstance(trajectory, TrajectoryRecord) else trajectory
    )
    validate_contract(serialized_record)
    validate_contract(serialized_trajectory)
    if serialized_trajectory["artifact_type"] != TRAJECTORY_ARTIFACT_TYPE:
        raise DemonstrationContractError("trajectory reference target has the wrong artifact type")
    if serialized_record["artifact_type"] == TRAJECTORY_ARTIFACT_TYPE:
        raise DemonstrationContractError("a trajectory cannot reference itself as a derived record")
    if serialized_record["trajectory_digest"] != serialized_trajectory["digest"]:
        raise DemonstrationContractError("trajectory_digest does not identify the supplied trajectory")
    source_identity_keys = (
        "source_dataset_repo_id",
        "source_dataset_revision",
        "source_manifest_sha256",
        "source_episode_index",
        "source_task_index",
        "source_episode_length",
    )
    for key in source_identity_keys:
        if serialized_record["provenance"][key] != serialized_trajectory["provenance"][key]:
            raise DemonstrationContractError(f"trajectory provenance differs at {key}")


def validate_sample_references(
    sample: Mapping[str, Any] | MultiHorizonSample,
    split_assignment: Mapping[str, Any] | SplitAssignment,
    *,
    target_event: Mapping[str, Any] | EventInterval | None = None,
) -> None:
    """Validate a sample's split and optional next-event foreign keys."""

    serialized_sample = sample.to_dict() if isinstance(sample, MultiHorizonSample) else sample
    serialized_split = (
        split_assignment.to_dict()
        if isinstance(split_assignment, SplitAssignment)
        else split_assignment
    )
    validate_contract(serialized_sample)
    validate_contract(serialized_split)
    if serialized_sample["artifact_type"] != MULTI_HORIZON_SAMPLE_ARTIFACT_TYPE:
        raise DemonstrationContractError("sample reference source has the wrong artifact type")
    if serialized_split["artifact_type"] != SPLIT_ASSIGNMENT_ARTIFACT_TYPE:
        raise DemonstrationContractError("split reference target has the wrong artifact type")
    if serialized_sample["split_assignment_digest"] != serialized_split["digest"]:
        raise DemonstrationContractError("split_assignment_digest does not identify the supplied split")
    if serialized_sample["split"] != serialized_split["split"]:
        raise DemonstrationContractError("sample and split assignment disagree")
    if serialized_sample["trajectory_digest"] != serialized_split["trajectory_digest"]:
        raise DemonstrationContractError("sample and split assignment reference different trajectories")

    source_identity_keys = (
        "source_dataset_repo_id",
        "source_dataset_revision",
        "source_manifest_sha256",
        "source_episode_index",
        "source_task_index",
        "source_episode_length",
    )
    for key in source_identity_keys:
        if serialized_sample["provenance"][key] != serialized_split["provenance"][key]:
            raise DemonstrationContractError(f"sample and split provenance differ at {key}")

    if serialized_sample["horizon"] == "next-event":
        if target_event is None:
            raise DemonstrationContractError("next-event sample requires the target event record")
        serialized_event = (
            target_event.to_dict() if isinstance(target_event, EventInterval) else target_event
        )
        validate_contract(serialized_event)
        if serialized_event["artifact_type"] != EVENT_INTERVAL_ARTIFACT_TYPE:
            raise DemonstrationContractError("target event has the wrong artifact type")
        if serialized_sample["target_event_digest"] != serialized_event["digest"]:
            raise DemonstrationContractError("target_event_digest does not identify the event")
        if serialized_sample["trajectory_digest"] != serialized_event["trajectory_digest"]:
            raise DemonstrationContractError("sample and target event reference different trajectories")
        for key in source_identity_keys:
            if serialized_sample["provenance"][key] != serialized_event["provenance"][key]:
                raise DemonstrationContractError(f"sample and event provenance differ at {key}")
        if serialized_sample["target_frame_index"] != serialized_event["frame_start"]:
            raise DemonstrationContractError("next-event sample must target the event's first frame")
    elif target_event is not None:
        raise DemonstrationContractError("fixed-horizon sample must not supply a target event")


def validate_contract_sequence(records: Iterable[Mapping[str, Any]]) -> None:
    """Validate records and reject duplicate content identities in one artifact stream."""

    seen: set[str] = set()
    for index, record in enumerate(records):
        try:
            validate_contract(record)
        except DemonstrationContractError as error:
            raise DemonstrationContractError(f"record {index}: {error}") from error
        if record["digest"] in seen:
            raise DemonstrationContractError(f"record {index}: duplicate digest {record['digest']}")
        seen.add(record["digest"])

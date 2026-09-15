"""Deterministic B0 action normalization and dual-camera data streaming.

The B0 target at anchor ``t`` is the eight-action slice ``action[t:t + 8]``.
This is intentionally derived from frozen episode split assignments instead of
the Week 2 fixed-horizon sample stream: a fixed target at ``t + 8`` has a
different, nine-frame boundary contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eventground_wam.demonstration_contracts import (
    SOURCE_ACTION_DIMENSION,
    SOURCE_DATASET_REPO_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_EPISODE_COUNT,
    SOURCE_IMAGE_SHAPE,
    SOURCE_MANIFEST_SHA256,
    SPLIT_ASSIGNMENT_ARTIFACT_TYPE,
    canonical_json_bytes,
    contract_digest,
    validate_contract,
)
from eventground_wam.demonstration_reader import (
    DecodedVideoFrame,
    DemonstrationReader,
    EpisodeRecord,
    SynchronizedStep,
)
from eventground_wam.split_sample_index import SPLIT_NAMES

B0_ACTION_CHUNK_LENGTH = 8
B0_ACTION_STATISTICS_ARTIFACT_TYPE = "eventground_wam.b0_action_statistics"
B0_ACTION_STATISTICS_SCHEMA_VERSION = 1
B0_ACTION_NORMALIZATION_ID = "train-q01-q99-affine-gripper-passthrough-v1"
B0_SAMPLE_IDENTITY_ARTIFACT_TYPE = "eventground_wam.b0_sample_identity"
B0_SAMPLE_IDENTITY_SCHEMA_VERSION = 1
B0_DATASET_VERSION = "week2-dataset-v1"

PINNED_FREEZE_MANIFEST_SHA256 = (
    "53d80156d9f6eb0451572b29ab7fcf38add08405ffa6ab26c7400899aad47ae2"
)
PINNED_FREEZE_MANIFEST_DIGEST = (
    "f7e3c4e419f8d2167c78eecb58f8e19e2c748e4aff98edcbe12ffb506b86f6d1"
)
PINNED_SPLIT_MANIFEST_SHA256 = (
    "28e6b7a0ffb5b61c6776f5c16196e65f37c1cc62e2f1e895eb92217efbc9a6da"
)
PINNED_SPLIT_ASSIGNMENTS_SHA256 = (
    "71b8fc2a3ce04c84cd41e3c9f052f1a677a0fa2d546201f8540f57eeb3c46758"
)
PINNED_SPLIT_EPISODE_COUNTS = {
    "train": 11_521,
    "validation": 1_396,
    "test": 1_430,
}
PINNED_TRAIN_ACTION_COUNT = 1_796_601
PINNED_TRAIN_SAMPLE_COUNT = 1_715_954

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}\Z")


class B0DataError(ValueError):
    """Raised when the B0 data or normalization contract is violated."""


@dataclass(frozen=True)
class B0SplitAssignment:
    """The frozen fields needed to join one source episode to a split."""

    episode_index: int
    task_index: int
    episode_length: int
    split: str
    digest: str
    trajectory_digest: str
    initial_condition_digest: str


@dataclass(frozen=True)
class B0SplitIndex:
    """Validated complete episode-level split assignments."""

    assignments: tuple[B0SplitAssignment, ...]
    dataset_version: str = B0_DATASET_VERSION
    freeze_manifest_sha256: str = PINNED_FREEZE_MANIFEST_SHA256
    split_manifest_sha256: str = PINNED_SPLIT_MANIFEST_SHA256
    split_assignments_sha256: str = PINNED_SPLIT_ASSIGNMENTS_SHA256

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        expected_episode_count: int | None = None,
        dataset_version: str = B0_DATASET_VERSION,
        freeze_manifest_sha256: str = PINNED_FREEZE_MANIFEST_SHA256,
        split_manifest_sha256: str = PINNED_SPLIT_MANIFEST_SHA256,
        split_assignments_sha256: str = PINNED_SPLIT_ASSIGNMENTS_SHA256,
    ) -> B0SplitIndex:
        """Build an index while rejecting malformed, duplicate, or leaky records."""

        _require_sha256(freeze_manifest_sha256, "freeze_manifest_sha256")
        _require_sha256(split_manifest_sha256, "split_manifest_sha256")
        _require_sha256(split_assignments_sha256, "split_assignments_sha256")
        assignments: list[B0SplitAssignment] = []
        seen_digests: set[str] = set()
        group_splits: dict[str, str] = {}
        for position, record in enumerate(records):
            try:
                validate_contract(record)
            except ValueError as error:
                raise B0DataError(f"split record {position} is invalid: {error}") from error
            if record["artifact_type"] != SPLIT_ASSIGNMENT_ARTIFACT_TYPE:
                raise B0DataError(f"split record {position} has the wrong artifact type")
            provenance = record["provenance"]
            episode_index = int(provenance["source_episode_index"])
            if episode_index != position:
                raise B0DataError(
                    "split assignments must be canonical and complete: "
                    f"expected episode {position}, found {episode_index}"
                )
            digest = str(record["digest"])
            if digest in seen_digests:
                raise B0DataError(f"duplicate split assignment digest {digest}")
            seen_digests.add(digest)
            group_digest = str(record["initial_condition_digest"])
            split = str(record["split"])
            previous_split = group_splits.setdefault(group_digest, split)
            if previous_split != split:
                raise B0DataError(
                    f"initial-condition group {group_digest} crosses frozen splits"
                )
            assignments.append(
                B0SplitAssignment(
                    episode_index=episode_index,
                    task_index=int(provenance["source_task_index"]),
                    episode_length=int(provenance["source_episode_length"]),
                    split=split,
                    digest=digest,
                    trajectory_digest=str(record["trajectory_digest"]),
                    initial_condition_digest=group_digest,
                )
            )
        if not assignments:
            raise B0DataError("split assignment stream is empty")
        if expected_episode_count is not None and len(assignments) != expected_episode_count:
            raise B0DataError(
                f"split stream has {len(assignments)} episodes, expected {expected_episode_count}"
            )
        return cls(
            assignments=tuple(assignments),
            dataset_version=dataset_version,
            freeze_manifest_sha256=freeze_manifest_sha256,
            split_manifest_sha256=split_manifest_sha256,
            split_assignments_sha256=split_assignments_sha256,
        )

    @property
    def episode_counts(self) -> dict[str, int]:
        counts = Counter(assignment.split for assignment in self.assignments)
        return {name: counts[name] for name in SPLIT_NAMES}

    @property
    def source_identity(self) -> dict[str, str]:
        return {
            "dataset_repo_id": SOURCE_DATASET_REPO_ID,
            "dataset_revision": SOURCE_DATASET_REVISION,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "dataset_version": self.dataset_version,
            "freeze_manifest_sha256": self.freeze_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "split_assignments_sha256": self.split_assignments_sha256,
        }


def load_frozen_b0_split_index(frozen_root: str | Path) -> B0SplitIndex:
    """Load and revalidate the pinned Week 2 freeze and its primary split stream."""

    root = Path(frozen_root).expanduser().resolve()
    freeze_path = root / "dataset-freeze-manifest.json"
    split_manifest_path = root / "primary/splits/split-sample-manifest.json"
    if _sha256_file(freeze_path) != PINNED_FREEZE_MANIFEST_SHA256:
        raise B0DataError("dataset-freeze-manifest.json does not match week2-dataset-v1")
    freeze_manifest = _load_json_object(freeze_path)
    if (
        freeze_manifest.get("digest") != PINNED_FREEZE_MANIFEST_DIGEST
        or contract_digest(freeze_manifest) != PINNED_FREEZE_MANIFEST_DIGEST
        or freeze_manifest.get("artifact_type") != "eventground_wam.dataset_freeze"
        or freeze_manifest.get("dataset_version") != B0_DATASET_VERSION
        or freeze_manifest.get("status") != "frozen"
        or freeze_manifest.get("publishable") is not True
    ):
        raise B0DataError("dataset freeze manifest violates the locked B0 source contract")
    _validate_source_mapping(freeze_manifest.get("source"), "dataset freeze source")

    if _sha256_file(split_manifest_path) != PINNED_SPLIT_MANIFEST_SHA256:
        raise B0DataError("split-sample-manifest.json differs from the pinned Week 2 manifest")
    split_manifest = _load_json_object(split_manifest_path)
    if split_manifest.get("digest") != contract_digest(split_manifest):
        raise B0DataError("split/sample manifest digest is invalid")
    if (
        split_manifest.get("artifact_type") != "eventground_wam.split_sample_indexes"
        or split_manifest.get("complete_source") is not True
        or split_manifest.get("foreign_keys_validated") is not True
        or split_manifest.get("publishable") is not True
    ):
        raise B0DataError("split/sample manifest is not a complete publishable source")
    _validate_source_mapping(split_manifest.get("source"), "split manifest source")
    if split_manifest.get("grouping", {}).get("cross_split_group_overlap_count") != 0:
        raise B0DataError("split/sample manifest reports cross-split group overlap")

    output = split_manifest.get("outputs", {}).get("split_assignments")
    if not isinstance(output, Mapping):
        raise B0DataError("split/sample manifest has no split assignment output")
    relative_path = _safe_relative_path(output.get("path"), "split assignment path")
    assignment_path = split_manifest_path.parent.joinpath(*relative_path.parts)
    assignment_sha256 = _sha256_file(assignment_path)
    if (
        output.get("artifact_type") != SPLIT_ASSIGNMENT_ARTIFACT_TYPE
        or output.get("record_count") != SOURCE_EPISODE_COUNT
        or output.get("sha256") != PINNED_SPLIT_ASSIGNMENTS_SHA256
        or assignment_sha256 != PINNED_SPLIT_ASSIGNMENTS_SHA256
    ):
        raise B0DataError("split assignment stream differs from the pinned Week 2 output")

    index = B0SplitIndex.from_records(
        _iter_jsonl_objects(assignment_path),
        expected_episode_count=SOURCE_EPISODE_COUNT,
    )
    if index.episode_counts != PINNED_SPLIT_EPISODE_COUNTS:
        raise B0DataError(
            f"split episode counts are {index.episode_counts}, expected {PINNED_SPLIT_EPISODE_COUNTS}"
        )
    return index


@dataclass(frozen=True)
class ActionNormalizer:
    """Invertible affine action transform with an unchanged gripper channel."""

    center: tuple[float, ...]
    scale: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    artifact_digest: str | None = None
    source_identity: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        vectors = {
            "center": self.center,
            "scale": self.scale,
            "lower": self.lower,
            "upper": self.upper,
        }
        for name, values in vectors.items():
            if len(values) != SOURCE_ACTION_DIMENSION:
                raise B0DataError(f"normalization {name} must have 7 values")
            if not all(math.isfinite(float(value)) for value in values):
                raise B0DataError(f"normalization {name} contains non-finite values")
        if any(value <= 0.0 for value in self.scale):
            raise B0DataError("normalization scales must be positive")
        if self.center[6] != 0.0 or self.scale[6] != 1.0:
            raise B0DataError("the gripper channel must be an identity transform")
        if self.lower[6] != -1.0 or self.upper[6] != 1.0:
            raise B0DataError("the gripper bounds must encode -1=open and +1=close")
        if self.artifact_digest is not None:
            _require_sha256(self.artifact_digest, "artifact_digest")

    @classmethod
    def identity(cls) -> ActionNormalizer:
        """Return an identity transform for synthetic tests and diagnostics."""

        return cls(
            center=(0.0,) * SOURCE_ACTION_DIMENSION,
            scale=(1.0,) * SOURCE_ACTION_DIMENSION,
            lower=(-1.0,) * SOURCE_ACTION_DIMENSION,
            upper=(1.0,) * SOURCE_ACTION_DIMENSION,
        )

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> ActionNormalizer:
        """Validate a sealed statistics artifact and construct its transform."""

        expected_keys = {
            "artifact_type",
            "schema_version",
            "normalization_id",
            "source",
            "generator",
            "split",
            "algorithm",
            "statistics",
            "validation",
            "digest",
        }
        if set(artifact) != expected_keys:
            raise B0DataError("action statistics artifact has unexpected fields")
        if (
            artifact["artifact_type"] != B0_ACTION_STATISTICS_ARTIFACT_TYPE
            or artifact["schema_version"] != B0_ACTION_STATISTICS_SCHEMA_VERSION
            or artifact["normalization_id"] != B0_ACTION_NORMALIZATION_ID
        ):
            raise B0DataError("unsupported B0 action statistics artifact")
        if artifact["digest"] != contract_digest(artifact):
            raise B0DataError("action statistics artifact digest is invalid")
        source = artifact["source"]
        _validate_b0_source_identity(source)
        algorithm = artifact["algorithm"]
        if algorithm != {
            "continuous_dimensions": [0, 1, 2, 3, 4, 5],
            "constant_dimension_scale": 1.0,
            "gripper_dimension": 6,
            "gripper_transform": "identity:-1=open,+1=close",
            "lower_quantile": 0.01,
            "normalized_clipping": False,
            "quantile_interpolation": "linear-index-(n-1)-v1",
            "upper_quantile": 0.99,
        }:
            raise B0DataError("action statistics artifact has a different algorithm")
        statistics = artifact["statistics"]
        if not isinstance(statistics, Mapping):
            raise B0DataError("action statistics must be an object")
        return cls(
            center=_float_tuple(statistics.get("center"), "statistics.center"),
            scale=_float_tuple(statistics.get("scale"), "statistics.scale"),
            lower=_float_tuple(statistics.get("lower_q01"), "statistics.lower_q01"),
            upper=_float_tuple(statistics.get("upper_q99"), "statistics.upper_q99"),
            artifact_digest=str(artifact["digest"]),
            source_identity=dict(source),
        )

    def normalize(self, actions: Any) -> NDArray[np.float64]:
        """Normalize finite actions without clipping, preserving invertibility."""

        values = _action_array(actions)
        normalized = (values - np.asarray(self.center)) / np.asarray(self.scale)
        normalized[..., 6] = values[..., 6]
        return normalized

    def denormalize(self, normalized_actions: Any) -> NDArray[np.float64]:
        """Invert normalized actions and preserve gripper polarity exactly."""

        values = _action_array(normalized_actions)
        actions = values * np.asarray(self.scale) + np.asarray(self.center)
        actions[..., 6] = values[..., 6]
        return actions


def fit_b0_action_statistics(
    reader: DemonstrationReader,
    split_index: B0SplitIndex,
    *,
    project_git_revision: str,
    project_tree_clean: bool,
) -> dict[str, Any]:
    """Fit exact q01/q99 statistics from every training action exactly once."""

    if not _GIT_REVISION_RE.fullmatch(project_git_revision):
        raise B0DataError("project_git_revision must be a full lowercase Git SHA")
    if project_tree_clean is not True:
        raise B0DataError("action statistics may only be sealed from a clean source tree")
    catalog = _validated_episode_catalog(reader, split_index)
    assignment_by_episode = {
        assignment.episode_index: assignment for assignment in split_index.assignments
    }
    train_episodes = tuple(
        episode
        for episode in catalog
        if assignment_by_episode[episode.episode_index].split == "train"
    )
    action_count = sum(episode.length for episode in train_episodes)
    continuous_actions = np.empty((action_count, 6), dtype=np.float32)
    gripper_counts = {-1: 0, 1: 0}
    write_offset = 0
    for episode in train_episodes:
        episode_start = write_offset
        for row in reader.iter_episode_rows(episode):
            action = np.asarray(row.action, dtype=np.float32)
            gripper = float(action[6])
            if gripper not in {-1.0, 1.0}:
                raise B0DataError(
                    f"episode {episode.episode_index} frame {row.frame_index} has unsupported "
                    f"gripper command {gripper}; expected -1=open or +1=close"
                )
            continuous_actions[write_offset] = action[:6]
            gripper_counts[int(gripper)] += 1
            write_offset += 1
        if write_offset - episode_start != episode.length:
            raise B0DataError(f"episode {episode.episode_index} action count changed while fitting")
    if write_offset != action_count:
        raise B0DataError(f"fitted {write_offset} actions, expected {action_count}")
    if not np.isfinite(continuous_actions).all():
        raise B0DataError("training actions contain non-finite continuous values")

    lower = [_linear_quantile(continuous_actions[:, index], 0.01) for index in range(6)]
    upper = [_linear_quantile(continuous_actions[:, index], 0.99) for index in range(6)]
    center: list[float] = []
    scale: list[float] = []
    for low, high in zip(lower, upper, strict=True):
        center.append((low + high) / 2.0)
        half_range = (high - low) / 2.0
        scale.append(half_range if half_range > 0.0 else 1.0)
    lower.append(-1.0)
    upper.append(1.0)
    center.append(0.0)
    scale.append(1.0)

    sample_count = sum(
        max(0, episode.length - B0_ACTION_CHUNK_LENGTH + 1) for episode in train_episodes
    )
    raw_min = continuous_actions.min(axis=0).astype(np.float64).tolist() + [-1.0]
    raw_max = continuous_actions.max(axis=0).astype(np.float64).tolist() + [1.0]
    raw_mean = continuous_actions.mean(axis=0, dtype=np.float64).tolist() + [
        (gripper_counts[1] - gripper_counts[-1]) / action_count
    ]
    raw_std = continuous_actions.std(axis=0, dtype=np.float64).tolist() + [
        math.sqrt(max(0.0, 1.0 - raw_mean[6] ** 2))
    ]
    artifact: dict[str, Any] = {
        "artifact_type": B0_ACTION_STATISTICS_ARTIFACT_TYPE,
        "schema_version": B0_ACTION_STATISTICS_SCHEMA_VERSION,
        "normalization_id": B0_ACTION_NORMALIZATION_ID,
        "source": split_index.source_identity,
        "generator": {
            "project_git_revision": project_git_revision,
            "project_tree_clean": True,
        },
        "split": {
            "name": "train",
            "episode_count": len(train_episodes),
            "action_count": action_count,
            "valid_action_chunk_count": sample_count,
            "action_chunk_length": B0_ACTION_CHUNK_LENGTH,
        },
        "algorithm": {
            "continuous_dimensions": [0, 1, 2, 3, 4, 5],
            "constant_dimension_scale": 1.0,
            "gripper_dimension": 6,
            "gripper_transform": "identity:-1=open,+1=close",
            "lower_quantile": 0.01,
            "normalized_clipping": False,
            "quantile_interpolation": "linear-index-(n-1)-v1",
            "upper_quantile": 0.99,
        },
        "statistics": {
            "center": center,
            "scale": scale,
            "lower_q01": lower,
            "upper_q99": upper,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "raw_mean": raw_mean,
            "raw_std_population": raw_std,
            "gripper_counts": {"open_-1": gripper_counts[-1], "close_+1": gripper_counts[1]},
        },
        "validation": {
            "statistics_split": "train-only",
            "actions_counted_once_not_from_overlapping_chunks": True,
            "video_decoding_used": False,
            "normalized_clipping": False,
            "gripper_values_exactly_negative_or_positive_one": True,
        },
    }
    artifact["digest"] = contract_digest(artifact)
    ActionNormalizer.from_artifact(artifact)
    return artifact


def load_b0_action_normalizer(path: str | Path) -> ActionNormalizer:
    """Load a validated action-normalization artifact."""

    return ActionNormalizer.from_artifact(_load_json_object(Path(path).expanduser().resolve()))


def write_b0_action_statistics(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    """Write a validated statistics artifact atomically without overwriting."""

    ActionNormalizer.from_artifact(artifact)
    output_path = Path(path).expanduser().resolve()
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    conflicts = [candidate for candidate in (output_path, partial_path) if candidate.exists()]
    if conflicts:
        raise FileExistsError(f"refusing to overwrite action statistics output: {conflicts}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        partial_path.write_text(
            json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        partial_path.replace(output_path)
    except BaseException:
        partial_path.unlink(missing_ok=True)
        raise
    return output_path


def validate_b0_action_roundtrip(
    reader: DemonstrationReader,
    split_index: B0SplitIndex,
    normalizer: ActionNormalizer,
    *,
    absolute_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Stream every source action and verify normalization inversion by split."""

    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise B0DataError("absolute_tolerance must be finite and nonnegative")
    catalog = _validated_episode_catalog(reader, split_index)
    assignment_by_episode = {
        assignment.episode_index: assignment for assignment in split_index.assignments
    }
    action_counts = {name: 0 for name in SPLIT_NAMES}
    max_errors = {name: 0.0 for name in SPLIT_NAMES}
    gripper_counts = {name: {-1: 0, 1: 0} for name in SPLIT_NAMES}
    for episode in catalog:
        split = assignment_by_episode[episode.episode_index].split
        for row in reader.iter_episode_rows(episode):
            action = np.asarray(row.action, dtype=np.float64)
            _validate_gripper_commands(
                action[6:7],
                context=f"episode {episode.episode_index} frame {row.frame_index}",
            )
            reconstructed = normalizer.denormalize(normalizer.normalize(action))
            error = float(np.max(np.abs(reconstructed - action)))
            max_errors[split] = max(max_errors[split], error)
            if error > absolute_tolerance:
                raise B0DataError(
                    f"normalization round trip error {error} exceeds {absolute_tolerance} "
                    f"in {split} episode {episode.episode_index} frame {row.frame_index}"
                )
            if reconstructed[6] != action[6]:
                raise B0DataError("gripper command changed during normalization round trip")
            action_counts[split] += 1
            gripper_counts[split][int(action[6])] += 1
    return {
        "absolute_tolerance": absolute_tolerance,
        "action_counts_by_split": action_counts,
        "max_absolute_error_by_split": max_errors,
        "gripper_counts_by_split": {
            split: {
                "open_-1": gripper_counts[split][-1],
                "close_+1": gripper_counts[split][1],
            }
            for split in SPLIT_NAMES
        },
        "all_actions_passed": True,
        "gripper_polarity_preserved": True,
    }


@dataclass(frozen=True)
class B0DataConfig:
    """Every data transform that precedes Qwen processor/model preprocessing."""

    split: str = "train"
    seed: int = 0
    shuffle_episodes: bool = True
    action_chunk_length: int = B0_ACTION_CHUNK_LENGTH
    camera_orientation: str = "identity"
    rgb_conversion: str = "pyav-rgb24"
    image_layout: str = "CHW"
    resize_crop: str = "none-source-256x256"

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise B0DataError(f"split must be one of {SPLIT_NAMES}")
        if type(self.seed) is not int:
            raise B0DataError("seed must be an integer")
        if type(self.shuffle_episodes) is not bool:
            raise B0DataError("shuffle_episodes must be a bool")
        if self.action_chunk_length != B0_ACTION_CHUNK_LENGTH:
            raise B0DataError("B0 action chunks must contain exactly eight actions")
        if self.camera_orientation not in {"identity", "rotate-180"}:
            raise B0DataError("camera_orientation must be identity or rotate-180")
        if self.rgb_conversion != "pyav-rgb24":
            raise B0DataError("B0 RGB conversion must be explicit PyAV rgb24")
        if self.image_layout != "CHW":
            raise B0DataError("B0 images must use CHW layout")
        if self.resize_crop != "none-source-256x256":
            raise B0DataError("B0 loader must not resize or crop source images")


@dataclass(frozen=True)
class B0SampleIdentity:
    """Stable identity for one B0 action-chunk example."""

    split: str
    episode_index: int
    task_index: int
    anchor_frame_index: int
    action_frame_end_exclusive: int
    split_assignment_digest: str

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "artifact_type": B0_SAMPLE_IDENTITY_ARTIFACT_TYPE,
            "schema_version": B0_SAMPLE_IDENTITY_SCHEMA_VERSION,
            **asdict(self),
        }
        payload["digest"] = contract_digest(payload)
        return payload

    @property
    def digest(self) -> str:
        return str(self.to_dict()["digest"])


@dataclass(frozen=True)
class B0Sample:
    """One model-ready NumPy sample before Qwen-specific processing."""

    identity: B0SampleIdentity
    instruction: str
    front_rgb: NDArray[np.uint8]
    wrist_rgb: NDArray[np.uint8]
    proprioception: NDArray[np.float32]
    actions: NDArray[np.float32]


@dataclass(frozen=True)
class B0Batch:
    """A collated batch with explicit camera axes and target shape."""

    identities: tuple[B0SampleIdentity, ...]
    instructions: tuple[str, ...]
    front_rgb: NDArray[np.uint8]
    wrist_rgb: NDArray[np.uint8]
    proprioception: NDArray[np.float32]
    actions: NDArray[np.float32]


RGBFrameConverter = Callable[[DecodedVideoFrame], NDArray[np.uint8]]


class B0StreamingDataset:
    """Repeatable episode-shuffled B0 stream with bounded decode memory."""

    def __init__(
        self,
        reader: DemonstrationReader,
        split_index: B0SplitIndex,
        normalizer: ActionNormalizer,
        *,
        config: B0DataConfig | None = None,
        rgb_frame_converter: RGBFrameConverter | None = None,
    ) -> None:
        self.reader = reader
        self.split_index = split_index
        self.normalizer = normalizer
        self.config = config or B0DataConfig()
        self._rgb_frame_converter = rgb_frame_converter or decoded_rgb_chw
        self._catalog = _validated_episode_catalog(reader, split_index)
        assignment_by_episode = {
            assignment.episode_index: assignment for assignment in split_index.assignments
        }
        self._episodes = tuple(
            episode
            for episode in self._catalog
            if assignment_by_episode[episode.episode_index].split == self.config.split
        )
        self._episode_by_index = {episode.episode_index: episode for episode in self._episodes}
        self._assignment_by_episode = assignment_by_episode
        if normalizer.source_identity is not None and dict(normalizer.source_identity) != (
            split_index.source_identity
        ):
            raise B0DataError("action normalizer and split index name different frozen sources")

    @property
    def sample_count(self) -> int:
        return sum(
            max(0, episode.length - self.config.action_chunk_length + 1)
            for episode in self._episodes
        )

    @property
    def episodes(self) -> tuple[EpisodeRecord, ...]:
        """Return selected descriptors without opening any episode payload."""

        return self._episodes

    def iter_sample_identities(
        self,
        *,
        epoch: int = 0,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> Iterator[B0SampleIdentity]:
        """Yield ordered identities without opening Parquet or video payloads."""

        for episode in self._ordered_shard_episodes(epoch, shard_index, shard_count):
            assignment = self._assignment_by_episode[episode.episode_index]
            for anchor in range(episode.length - self.config.action_chunk_length + 1):
                yield self._identity(episode, assignment, anchor)

    def ordered_identity_digest(
        self,
        *,
        epoch: int = 0,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> str:
        """Hash configuration plus every ordered sample identity."""

        digest = hashlib.sha256()
        digest.update(
            canonical_json_bytes(
                {
                    "algorithm": "ordered-b0-sample-identities-v1",
                    "config": asdict(self.config),
                    "epoch": epoch,
                    "shard_count": shard_count,
                    "shard_index": shard_index,
                }
            )
            + b"\n"
        )
        for identity in self.iter_sample_identities(
            epoch=epoch,
            shard_index=shard_index,
            shard_count=shard_count,
        ):
            digest.update(canonical_json_bytes(identity.to_dict()) + b"\n")
        return digest.hexdigest()

    def iter_samples(
        self,
        *,
        epoch: int = 0,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> Iterator[B0Sample]:
        """Decode synchronized episodes and yield complete eight-action chunks."""

        for episode in self._ordered_shard_episodes(epoch, shard_index, shard_count):
            yield from self.iter_episode_samples(episode)

    def ordered_episodes(self, epoch: int = 0) -> tuple[EpisodeRecord, ...]:
        """Expose the frozen deterministic epoch order without opening payloads."""

        return self._ordered_shard_episodes(epoch, 0, 1)

    def iter_episode_samples(self, episode: EpisodeRecord) -> Iterator[B0Sample]:
        """Decode one selected episode, retaining the same streaming sample contract."""

        if self._episode_by_index.get(episode.episode_index) != episode:
            raise B0DataError("episode does not belong to this dataset split")
        assignment = self._assignment_by_episode[episode.episode_index]
        window: deque[
            tuple[
                SynchronizedStep,
                NDArray[np.uint8],
                NDArray[np.uint8],
            ]
        ] = deque()
        for step in self.reader.iter_episode_steps(episode):
            front = self._transform_rgb(self._rgb_frame_converter(step.front_frame))
            wrist = self._transform_rgb(self._rgb_frame_converter(step.wrist_frame))
            window.append((step, front, wrist))
            if len(window) < self.config.action_chunk_length:
                continue
            anchor_step, anchor_front, anchor_wrist = window[0]
            actions = np.asarray(
                [item[0].row.action for item in window],
                dtype=np.float32,
            )
            _validate_gripper_commands(
                actions[..., 6],
                context=f"episode {episode.episode_index} action chunk",
            )
            normalized_actions = self.normalizer.normalize(actions).astype(
                np.float32,
                copy=False,
            )
            anchor = anchor_step.row.frame_index
            yield B0Sample(
                identity=self._identity(episode, assignment, anchor),
                instruction=episode.instruction,
                front_rgb=anchor_front,
                wrist_rgb=anchor_wrist,
                proprioception=np.asarray(anchor_step.row.state, dtype=np.float32),
                actions=normalized_actions,
            )
            window.popleft()

    def _ordered_shard_episodes(
        self,
        epoch: int,
        shard_index: int,
        shard_count: int,
    ) -> tuple[EpisodeRecord, ...]:
        if type(epoch) is not int or epoch < 0:
            raise B0DataError("epoch must be a nonnegative integer")
        if (
            type(shard_count) is not int
            or shard_count <= 0
            or type(shard_index) is not int
            or not 0 <= shard_index < shard_count
        ):
            raise B0DataError("shard_index must identify one of shard_count positive shards")
        ordered = self._episodes
        if self.config.shuffle_episodes:
            ordered = tuple(
                sorted(
                    ordered,
                    key=lambda episode: (
                        _episode_shuffle_key(
                            seed=self.config.seed,
                            epoch=epoch,
                            episode_index=episode.episode_index,
                            split_assignment_digest=self._assignment_by_episode[
                                episode.episode_index
                            ].digest,
                        ),
                        episode.episode_index,
                    ),
                )
            )
        return tuple(
            episode
            for position, episode in enumerate(ordered)
            if position % shard_count == shard_index
        )

    def _identity(
        self,
        episode: EpisodeRecord,
        assignment: B0SplitAssignment,
        anchor: int,
    ) -> B0SampleIdentity:
        return B0SampleIdentity(
            split=self.config.split,
            episode_index=episode.episode_index,
            task_index=episode.task_index,
            anchor_frame_index=anchor,
            action_frame_end_exclusive=anchor + self.config.action_chunk_length,
            split_assignment_digest=assignment.digest,
        )

    def _transform_rgb(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        expected_shape = (
            SOURCE_IMAGE_SHAPE[2],
            SOURCE_IMAGE_SHAPE[0],
            SOURCE_IMAGE_SHAPE[1],
        )
        if image.dtype != np.uint8 or image.shape != expected_shape:
            raise B0DataError(
                f"RGB converter returned {image.dtype} {image.shape}, expected uint8 {expected_shape}"
            )
        if self.config.camera_orientation == "rotate-180":
            image = image[:, ::-1, ::-1]
        return np.ascontiguousarray(image)


class B0DataLoader:
    """Small framework-neutral batch loader over :class:`B0StreamingDataset`."""

    def __init__(
        self,
        dataset: B0StreamingDataset,
        *,
        batch_size: int,
        drop_last: bool = False,
    ) -> None:
        if type(batch_size) is not int or batch_size <= 0:
            raise B0DataError("batch_size must be a positive integer")
        if type(drop_last) is not bool:
            raise B0DataError("drop_last must be a bool")
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def iter_batches(
        self,
        *,
        epoch: int = 0,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> Iterator[B0Batch]:
        pending: list[B0Sample] = []
        for sample in self.dataset.iter_samples(
            epoch=epoch,
            shard_index=shard_index,
            shard_count=shard_count,
        ):
            pending.append(sample)
            if len(pending) == self.batch_size:
                yield collate_b0_samples(pending)
                pending.clear()
        if pending and not self.drop_last:
            yield collate_b0_samples(pending)


def collate_b0_samples(samples: Sequence[B0Sample]) -> B0Batch:
    """Stack a non-empty sequence after checking its locked shapes and dtypes."""

    if not samples:
        raise B0DataError("cannot collate an empty B0 batch")
    for sample in samples:
        if sample.front_rgb.shape != (3, 256, 256) or sample.front_rgb.dtype != np.uint8:
            raise B0DataError("front RGB sample must be uint8 [3,256,256]")
        if sample.wrist_rgb.shape != (3, 256, 256) or sample.wrist_rgb.dtype != np.uint8:
            raise B0DataError("wrist RGB sample must be uint8 [3,256,256]")
        if sample.proprioception.shape != (8,) or sample.proprioception.dtype != np.float32:
            raise B0DataError("proprioception sample must be float32 [8]")
        if sample.actions.shape != (8, 7) or sample.actions.dtype != np.float32:
            raise B0DataError("action target must be float32 [8,7]")
    return B0Batch(
        identities=tuple(sample.identity for sample in samples),
        instructions=tuple(sample.instruction for sample in samples),
        front_rgb=np.stack([sample.front_rgb for sample in samples]),
        wrist_rgb=np.stack([sample.wrist_rgb for sample in samples]),
        proprioception=np.stack([sample.proprioception for sample in samples]),
        actions=np.stack([sample.actions for sample in samples]),
    )


def decoded_rgb_chw(frame: DecodedVideoFrame) -> NDArray[np.uint8]:
    """Convert one validated PyAV payload to contiguous RGB uint8 CHW."""

    try:
        image = frame.payload.to_ndarray(format="rgb24")
    except Exception as error:
        raise B0DataError("cannot convert decoded frame to rgb24") from error
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.shape != SOURCE_IMAGE_SHAPE:
        raise B0DataError(
            f"decoded rgb24 is {array.dtype} {array.shape}, expected uint8 {SOURCE_IMAGE_SHAPE}"
        )
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def _validated_episode_catalog(
    reader: DemonstrationReader,
    split_index: B0SplitIndex,
) -> tuple[EpisodeRecord, ...]:
    assignment_by_episode = {
        assignment.episode_index: assignment for assignment in split_index.assignments
    }
    if len(assignment_by_episode) != len(split_index.assignments):
        raise B0DataError("multiple split assignments exist for one episode")
    catalog: list[EpisodeRecord] = []
    for episode in reader.iter_episodes():
        assignment = assignment_by_episode.get(episode.episode_index)
        if assignment is None:
            raise B0DataError(f"episode {episode.episode_index} has no split assignment")
        if (
            assignment.task_index != episode.task_index
            or assignment.episode_length != episode.length
        ):
            raise B0DataError(
                f"episode {episode.episode_index} metadata differs from its split provenance"
            )
        catalog.append(episode)
    if len(catalog) != len(split_index.assignments):
        raise B0DataError(
            "split assignments and source metadata do not cover the same episode set"
        )
    return tuple(catalog)


def _episode_shuffle_key(
    *,
    seed: int,
    epoch: int,
    episode_index: int,
    split_assignment_digest: str,
) -> bytes:
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "algorithm": "sha256-episode-order-v1",
                "epoch": epoch,
                "episode_index": episode_index,
                "seed": seed,
                "split_assignment_digest": split_assignment_digest,
            }
        )
    ).digest()


def _linear_quantile(values: NDArray[np.float32], quantile: float) -> float:
    if values.ndim != 1 or values.size == 0:
        raise B0DataError("quantile input must be a non-empty vector")
    ordered = np.sort(values)
    position = (ordered.size - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    fraction = position - lower_index
    low = float(ordered[lower_index])
    high = float(ordered[upper_index])
    return low + (high - low) * fraction


def _action_array(value: Any) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise B0DataError("actions must be numeric") from error
    if array.ndim == 0 or array.shape[-1] != SOURCE_ACTION_DIMENSION:
        raise B0DataError("actions must have final dimension 7")
    if not np.isfinite(array).all():
        raise B0DataError("actions must be finite")
    return array


def _validate_gripper_commands(values: Any, *, context: str) -> None:
    array = np.asarray(values)
    if not np.isin(array, (-1.0, 1.0)).all():
        raise B0DataError(f"{context} contains a gripper value other than -1=open or +1=close")


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != SOURCE_ACTION_DIMENSION:
        raise B0DataError(f"{name} must be a seven-element array")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise B0DataError(f"{name} must contain numeric values") from error
    if not all(math.isfinite(item) for item in result):
        raise B0DataError(f"{name} contains non-finite values")
    return result


def _validate_source_mapping(value: Any, name: str) -> None:
    expected = {
        "dataset_repo_id": SOURCE_DATASET_REPO_ID,
        "dataset_revision": SOURCE_DATASET_REVISION,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
    }
    if not isinstance(value, Mapping) or any(value.get(key) != item for key, item in expected.items()):
        raise B0DataError(f"{name} differs from the pinned source identity")


def _validate_b0_source_identity(value: Any) -> None:
    _validate_source_mapping(value, "action statistics source")
    if not isinstance(value, Mapping):
        raise B0DataError("action statistics source must be an object")
    expected = {
        "dataset_version": B0_DATASET_VERSION,
        "freeze_manifest_sha256": PINNED_FREEZE_MANIFEST_SHA256,
        "split_manifest_sha256": PINNED_SPLIT_MANIFEST_SHA256,
        "split_assignments_sha256": PINNED_SPLIT_ASSIGNMENTS_SHA256,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise B0DataError("action statistics source differs from the frozen split identity")


def _safe_relative_path(value: Any, name: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise B0DataError(f"{name} must be a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise B0DataError(f"{name} is unsafe: {value!r}")
    return path


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise B0DataError(f"cannot read valid JSON object from {path}") from error
    if not isinstance(value, dict):
        raise B0DataError(f"{path} must contain a JSON object")
    return value


def _iter_jsonl_objects(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise B0DataError(f"cannot open {path}") from error
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise B0DataError(f"invalid JSON in {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise B0DataError(f"{path}:{line_number} must contain an object")
            yield value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise B0DataError(f"cannot hash required file {path}") from error
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise B0DataError(f"{name} must be a lowercase SHA-256 digest")

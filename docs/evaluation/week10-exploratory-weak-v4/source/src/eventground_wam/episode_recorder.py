"""Crash-safe, protocol-aware HDF5 recording for LIBERO evaluation episodes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

import h5py
import numpy as np

from eventground_wam.evaluation_protocol import (
    derive_episode_seed,
    protocol_sha256,
    validate_evaluation_protocol,
)

EPISODE_SCHEMA_VERSION = 1
ARTIFACT_TYPE = "eventground_wam_episode"
PHASE_ENCODING = {"settling": 0, "policy": 1}
_SHA256_HEX_LENGTH = 64


class EpisodeValidationError(ValueError):
    """Raised when episode metadata or an HDF5 artifact violates the protocol."""


@dataclass(frozen=True)
class EpisodeMetadata:
    """Immutable episode identity plus the outcome filled in during finalization."""

    protocol_id: str
    protocol_sha256: str
    project_git_revision: str
    benchmark_revision: str
    checkpoint_id: str
    checkpoint_revision: str
    suite: str
    runtime_task_index: int
    classification_task_id: int
    perturbation_category: str
    difficulty_level: int | None
    instruction: str
    bddl_path: str
    bddl_sha256: str
    initial_state_sha256: str
    episode_seed: int
    success: bool | None = None
    termination_reason: str = "recording"

    def with_outcome(self, success: bool, termination_reason: str) -> EpisodeMetadata:
        """Return a copy containing a terminal outcome."""

        return replace(
            self,
            success=success,
            termination_reason=termination_reason,
        )


@dataclass(frozen=True)
class StepRecord:
    """One control transition, with the observation and state captured before action."""

    step_index: int
    phase: str
    sim_time_seconds: float
    agent_rgb: np.ndarray
    wrist_rgb: np.ndarray
    proprioception: np.ndarray
    predicted_action_chunk: np.ndarray | None
    executed_action: np.ndarray
    action_chunk_index: int
    mujoco_state: np.ndarray
    reward: float
    done: bool
    policy_latency_ms: float | None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _decode_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading the whole artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash array dtype, shape, and C-order bytes for an unambiguous state identity."""

    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _validate_metadata(
    metadata: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    complete: bool,
) -> None:
    required = set(protocol["recording"]["required_episode_fields"])
    missing = required - set(metadata)
    if missing:
        raise EpisodeValidationError(f"Missing episode metadata fields: {sorted(missing)!r}")

    expected_values = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256(protocol),
        "benchmark_revision": protocol["benchmark"]["revision"],
    }
    for name, expected in expected_values.items():
        if metadata[name] != expected:
            raise EpisodeValidationError(
                f"Expected episode {name}={expected!r}, found {metadata[name]!r}"
            )

    suite = metadata["suite"]
    suites = protocol["benchmark"]["suites"]
    if suite not in suites:
        raise EpisodeValidationError(f"Unknown episode suite: {suite!r}")
    runtime_task_index = metadata["runtime_task_index"]
    if not isinstance(runtime_task_index, int) or isinstance(runtime_task_index, bool):
        raise EpisodeValidationError("runtime_task_index must be an integer")
    if not 0 <= runtime_task_index < suites[suite]["task_count"]:
        raise EpisodeValidationError("runtime_task_index is outside the episode suite")
    if metadata["classification_task_id"] != runtime_task_index + 1:
        raise EpisodeValidationError(
            "classification_task_id must be the one-based runtime_task_index"
        )
    if not isinstance(metadata["classification_task_id"], int) or isinstance(
        metadata["classification_task_id"], bool
    ):
        raise EpisodeValidationError("classification_task_id must be an integer")

    categories = protocol["benchmark"]["classification"]["categories"]
    if metadata["perturbation_category"] not in categories:
        raise EpisodeValidationError("Unknown episode perturbation_category")
    difficulty = metadata["difficulty_level"]
    difficulties = protocol["benchmark"]["classification"]["difficulty_levels"]
    if difficulty is not None:
        if not isinstance(difficulty, int) or isinstance(difficulty, bool):
            raise EpisodeValidationError("difficulty_level must be an integer or null")
        if difficulty not in difficulties:
            raise EpisodeValidationError("Unknown episode difficulty_level")

    trial_index = 0
    if protocol["protocol_id"] == "week4-clean-comparison-v1":
        trial_index = metadata["trial_index"]
        if type(trial_index) is not int or not 0 <= trial_index < 50:
            raise EpisodeValidationError("trial_index must be an integer in [0,50)")
        expected_pair = f"{suite}/task-{runtime_task_index:02d}/state-{trial_index:02d}"
        expected_comparison = {
            "comparison_id": protocol["protocol_id"],
            "acceptance_rule_digest": protocol["comparison"]["acceptance_rule_digest"],
            "pair_id": expected_pair,
            "initial_state_index": trial_index,
        }
        for name, expected in expected_comparison.items():
            if metadata[name] != expected or (name == "initial_state_index"
                                             and type(metadata[name]) is not int):
                raise EpisodeValidationError(f"Invalid comparison metadata: {name}")
    expected_seed = derive_episode_seed(protocol, suite, runtime_task_index, trial_index)
    if metadata["episode_seed"] != expected_seed:
        raise EpisodeValidationError(
            f"Expected episode_seed={expected_seed}, found {metadata['episode_seed']!r}"
        )

    for name in (
        "project_git_revision",
        "checkpoint_id",
        "checkpoint_revision",
        "instruction",
        "bddl_path",
    ):
        if not isinstance(metadata[name], str) or not metadata[name].strip():
            raise EpisodeValidationError(f"Episode {name} must be a nonempty string")
    for name in ("bddl_sha256", "initial_state_sha256"):
        if not _is_sha256(metadata[name]):
            raise EpisodeValidationError(f"Episode {name} must be a lowercase SHA-256 digest")

    if complete:
        if not isinstance(metadata["success"], bool):
            raise EpisodeValidationError("A complete episode must have a boolean success value")
        reason = metadata["termination_reason"]
        if not isinstance(reason, str) or not reason.strip() or reason == "recording":
            raise EpisodeValidationError(
                "A complete episode must have a terminal termination_reason"
            )
    elif metadata["success"] is not None and not isinstance(metadata["success"], bool):
        raise EpisodeValidationError("Episode success must be boolean or null")


def _normalized_step(record: StepRecord, protocol: Mapping[str, Any]) -> dict[str, Any]:
    height, width = protocol["observation"]["capture_resolution"]
    action_dimension = protocol["simulation"]["action"]["dimension"]
    action_chunk_length = protocol["simulation"]["action"]["chunk_length"]
    proprioception_dimension = protocol["observation"]["proprioception"]["dimension"]

    if record.phase not in PHASE_ENCODING:
        raise EpisodeValidationError(f"Unknown episode phase: {record.phase!r}")
    if not isinstance(record.step_index, int) or isinstance(record.step_index, bool):
        raise EpisodeValidationError("step_index must be an integer")

    agent_rgb = np.asarray(record.agent_rgb)
    wrist_rgb = np.asarray(record.wrist_rgb)
    expected_image_shape = (height, width, 3)
    for name, image in (("agent_rgb", agent_rgb), ("wrist_rgb", wrist_rgb)):
        if image.shape != expected_image_shape or image.dtype != np.uint8:
            raise EpisodeValidationError(
                f"{name} must have shape {expected_image_shape} and dtype uint8"
            )

    proprioception = np.asarray(record.proprioception, dtype=np.float64)
    executed_action = np.asarray(record.executed_action, dtype=np.float64)
    mujoco_state = np.asarray(record.mujoco_state, dtype=np.float64).reshape(-1)
    if proprioception.shape != (proprioception_dimension,):
        raise EpisodeValidationError(
            f"proprioception must have shape ({proprioception_dimension},)"
        )
    if executed_action.shape != (action_dimension,):
        raise EpisodeValidationError(f"executed_action must have shape ({action_dimension},)")
    if not mujoco_state.size:
        raise EpisodeValidationError("mujoco_state must contain at least one value")

    if record.predicted_action_chunk is None:
        predicted_action_chunk = np.full(
            (action_chunk_length, action_dimension), np.nan, dtype=np.float64
        )
    else:
        predicted_action_chunk = np.asarray(record.predicted_action_chunk, dtype=np.float64)
        if predicted_action_chunk.shape != (action_chunk_length, action_dimension):
            raise EpisodeValidationError(
                "predicted_action_chunk must have shape "
                f"({action_chunk_length}, {action_dimension})"
            )

    numeric_values = (
        proprioception,
        executed_action,
        mujoco_state,
        np.asarray([record.sim_time_seconds, record.reward], dtype=np.float64),
    )
    if any(not np.all(np.isfinite(value)) for value in numeric_values):
        raise EpisodeValidationError("Step numeric values must be finite")

    action_minimum, action_maximum = protocol["simulation"]["action"]["range"]
    if np.any(executed_action < action_minimum) or np.any(executed_action > action_maximum):
        raise EpisodeValidationError("executed_action is outside the protocol action range")

    if record.phase == "settling":
        expected_action = np.asarray(
            protocol["simulation"]["settling_action"], dtype=np.float64
        )
        if not np.array_equal(executed_action, expected_action):
            raise EpisodeValidationError("Settling step does not use the locked settling action")
        if record.action_chunk_index != -1:
            raise EpisodeValidationError("Settling action_chunk_index must be -1")
        if not np.all(np.isnan(predicted_action_chunk)):
            raise EpisodeValidationError("Settling predicted_action_chunk must be absent")
        if record.policy_latency_ms is not None:
            raise EpisodeValidationError("Settling policy_latency_ms must be null")
        latency = np.nan
    else:
        if not 0 <= record.action_chunk_index < action_chunk_length:
            raise EpisodeValidationError("Policy action_chunk_index is outside the action chunk")
        if not np.all(np.isfinite(predicted_action_chunk)):
            raise EpisodeValidationError("Policy predicted_action_chunk must be finite")
        if np.any(predicted_action_chunk < action_minimum) or np.any(
            predicted_action_chunk > action_maximum
        ):
            raise EpisodeValidationError(
                "predicted_action_chunk is outside the protocol action range"
            )
        if record.action_chunk_index == 0:
            if record.policy_latency_ms is None or not np.isfinite(record.policy_latency_ms):
                raise EpisodeValidationError("Policy query steps require finite latency")
            if record.policy_latency_ms < 0:
                raise EpisodeValidationError("policy_latency_ms cannot be negative")
            latency = float(record.policy_latency_ms)
        else:
            if record.policy_latency_ms is not None:
                raise EpisodeValidationError(
                    "Non-query policy steps must use null policy_latency_ms"
                )
            latency = np.nan

    return {
        "step_index": np.int64(record.step_index),
        "phase": np.uint8(PHASE_ENCODING[record.phase]),
        "sim_time_seconds": np.float64(record.sim_time_seconds),
        "agent_rgb": np.ascontiguousarray(agent_rgb),
        "wrist_rgb": np.ascontiguousarray(wrist_rgb),
        "proprioception": np.ascontiguousarray(proprioception),
        "predicted_action_chunk": np.ascontiguousarray(predicted_action_chunk),
        "executed_action": np.ascontiguousarray(executed_action),
        "action_chunk_index": np.int32(record.action_chunk_index),
        "mujoco_state": np.ascontiguousarray(mujoco_state),
        "reward": np.float64(record.reward),
        "done": np.bool_(record.done),
        "policy_latency_ms": np.float64(latency),
    }


class EpisodeRecorder:
    """Stream aligned control-step records to a partial HDF5 file, then publish atomically."""

    def __init__(
        self,
        output_path: str | Path,
        protocol: Mapping[str, Any],
        metadata: EpisodeMetadata,
        *,
        flush_interval: int = 16,
    ) -> None:
        validate_evaluation_protocol(protocol)
        if flush_interval < 1:
            raise ValueError("flush_interval must be at least one")
        self.output_path = Path(output_path).expanduser().resolve()
        if self.output_path.suffix not in {".h5", ".hdf5"}:
            raise ValueError("Episode output must use an .h5 or .hdf5 suffix")
        self.partial_path = self.output_path.with_name(self.output_path.name + ".partial")
        if self.output_path.exists():
            raise FileExistsError(f"Refusing to replace completed episode: {self.output_path}")
        if self.partial_path.exists():
            raise FileExistsError(f"Incomplete episode already exists: {self.partial_path}")

        self.protocol = deepcopy(protocol)
        self.metadata = metadata
        _validate_metadata(asdict(metadata), self.protocol, complete=False)
        self.flush_interval = flush_interval
        self.step_count = 0
        self._episode_done = False
        self._closed = False
        self._finalized = False

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.partial_path, "x", libver="latest")
        self._file.attrs.update(
            {
                "artifact_type": ARTIFACT_TYPE,
                "schema_version": EPISODE_SCHEMA_VERSION,
                "status": "recording",
                "created_at_utc": _utc_now(),
                "step_count": 0,
            }
        )
        text_dtype = h5py.string_dtype(encoding="utf-8")
        provenance = self._file.create_group("provenance")
        provenance.create_dataset(
            "evaluation_protocol_json",
            data=_canonical_json(self.protocol),
            dtype=text_dtype,
        )
        episode = self._file.create_group("episode")
        self._metadata_dataset = episode.create_dataset(
            "metadata_json",
            data=_canonical_json(asdict(metadata)),
            dtype=text_dtype,
        )
        self._steps = self._file.create_group("steps")
        self._steps.attrs["phase_encoding_json"] = _canonical_json(PHASE_ENCODING)
        self._create_datasets()
        self._file.flush()

    def _create_datasets(self) -> None:
        height, width = self.protocol["observation"]["capture_resolution"]
        action_dimension = self.protocol["simulation"]["action"]["dimension"]
        action_chunk_length = self.protocol["simulation"]["action"]["chunk_length"]
        proprioception_dimension = self.protocol["observation"]["proprioception"][
            "dimension"
        ]
        scalar_dtypes = {
            "step_index": np.int64,
            "phase": np.uint8,
            "sim_time_seconds": np.float64,
            "action_chunk_index": np.int32,
            "reward": np.float64,
            "done": np.bool_,
            "policy_latency_ms": np.float64,
        }
        for name, dtype in scalar_dtypes.items():
            self._steps.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=(64,),
                dtype=dtype,
            )

        self._steps.create_dataset(
            "agent_rgb",
            shape=(0, height, width, 3),
            maxshape=(None, height, width, 3),
            chunks=(1, height, width, 3),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        self._steps.create_dataset(
            "wrist_rgb",
            shape=(0, height, width, 3),
            maxshape=(None, height, width, 3),
            chunks=(1, height, width, 3),
            dtype=np.uint8,
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        for name, trailing_shape in {
            "proprioception": (proprioception_dimension,),
            "predicted_action_chunk": (action_chunk_length, action_dimension),
            "executed_action": (action_dimension,),
        }.items():
            self._steps.create_dataset(
                name,
                shape=(0, *trailing_shape),
                maxshape=(None, *trailing_shape),
                chunks=(64, *trailing_shape),
                dtype=np.float64,
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )

    def append(self, record: StepRecord) -> None:
        """Validate and append one complete, row-aligned control transition."""

        if self._closed:
            raise RuntimeError("Cannot append to a closed episode recorder")
        if self._episode_done:
            raise EpisodeValidationError("Cannot append after environment done")
        if record.step_index != self.step_count:
            raise EpisodeValidationError(
                f"Expected step_index={self.step_count}, found {record.step_index}"
            )
        expected_time = record.step_index / self.protocol["simulation"][
            "control_frequency_hz"
        ]
        if not np.isclose(record.sim_time_seconds, expected_time, rtol=0.0, atol=1e-12):
            raise EpisodeValidationError(
                f"Expected sim_time_seconds={expected_time}, found {record.sim_time_seconds}"
            )
        settling_steps = self.protocol["simulation"]["settling_steps"]
        expected_phase = "settling" if self.step_count < settling_steps else "policy"
        if record.phase != expected_phase:
            raise EpisodeValidationError(
                f"Expected phase={expected_phase!r} at step {self.step_count}"
            )
        if expected_phase == "policy":
            expected_chunk_index = (self.step_count - settling_steps) % self.protocol[
                "simulation"
            ]["action"]["chunk_length"]
            if record.action_chunk_index != expected_chunk_index:
                raise EpisodeValidationError(
                    f"Expected action_chunk_index={expected_chunk_index}, "
                    f"found {record.action_chunk_index}"
                )
        normalized = _normalized_step(record, self.protocol)

        if "mujoco_state" not in self._steps:
            state_dimension = normalized["mujoco_state"].size
            self._steps.create_dataset(
                "mujoco_state",
                shape=(0, state_dimension),
                maxshape=(None, state_dimension),
                chunks=(64, state_dimension),
                dtype=np.float64,
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
        elif normalized["mujoco_state"].shape != self._steps["mujoco_state"].shape[1:]:
            raise EpisodeValidationError("mujoco_state dimension changed within the episode")

        row = self.step_count
        for name in self.protocol["recording"]["required_step_fields"]:
            dataset = self._steps[name]
            dataset.resize(row + 1, axis=0)
            dataset[row] = normalized[name]
        self.step_count += 1
        self._episode_done = bool(record.done)
        self._file.attrs["step_count"] = self.step_count
        if self.step_count % self.flush_interval == 0:
            self._file.flush()

    def finalize(self, *, success: bool, termination_reason: str) -> Path:
        """Close, validate, and atomically promote the partial file to its final path."""

        if self._closed:
            raise RuntimeError("Episode recorder is already closed")
        if not isinstance(success, bool):
            raise TypeError("success must be boolean")
        if not isinstance(termination_reason, str) or not termination_reason.strip():
            raise ValueError("termination_reason must be a nonempty string")
        if success != self._episode_done:
            raise EpisodeValidationError(
                "Episode success must match whether the environment returned done"
            )

        self.metadata = self.metadata.with_outcome(success, termination_reason)
        metadata_dict = asdict(self.metadata)
        _validate_metadata(metadata_dict, self.protocol, complete=True)
        self._metadata_dataset[()] = _canonical_json(metadata_dict)
        self._file.attrs["status"] = "complete"
        self._file.attrs["completed_at_utc"] = _utc_now()
        self._file.attrs["step_count"] = self.step_count
        self._file.flush()
        self._file.close()
        self._closed = True

        try:
            validate_episode_file(self.partial_path, self.protocol)
        except Exception:
            with h5py.File(self.partial_path, "r+") as artifact:
                artifact.attrs["status"] = "invalid"
            raise
        os.replace(self.partial_path, self.output_path)
        self._finalized = True
        return self.output_path

    def abort(self, termination_reason: str) -> Path:
        """Close gracefully while deliberately retaining the partial artifact."""

        if self._closed:
            return self.partial_path
        reason = termination_reason.strip() or "aborted"
        self.metadata = self.metadata.with_outcome(False, reason)
        self._metadata_dataset[()] = _canonical_json(asdict(self.metadata))
        self._file.attrs["status"] = "aborted"
        self._file.attrs["aborted_at_utc"] = _utc_now()
        self._file.attrs["step_count"] = self.step_count
        self._file.flush()
        self._file.close()
        self._closed = True
        return self.partial_path

    def __enter__(self) -> EpisodeRecorder:  # noqa: PYI034
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._closed:
            if exception is None:
                self.abort("context_exited_without_finalize")
            else:
                exception_name = exception_type.__name__ if exception_type else "unknown"
                self.abort(f"exception:{exception_name}")


def read_episode_metadata(path: str | Path) -> dict[str, Any]:
    """Read the canonical metadata document from an episode artifact."""

    with h5py.File(Path(path).expanduser().resolve(), "r") as artifact:
        try:
            raw_metadata = artifact["episode/metadata_json"][()]
        except KeyError as error:
            raise EpisodeValidationError("Episode artifact has no metadata_json") from error
    try:
        metadata = json.loads(_decode_text(raw_metadata))
    except json.JSONDecodeError as error:
        raise EpisodeValidationError("Episode metadata_json is invalid") from error
    if not isinstance(metadata, dict):
        raise EpisodeValidationError("Episode metadata_json must contain an object")
    return metadata


def validate_episode_file(
    path: str | Path,
    protocol: Mapping[str, Any] | None = None,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate a recorded artifact and return a compact machine-readable summary."""

    artifact_path = Path(path).expanduser().resolve()
    try:
        artifact = h5py.File(artifact_path, "r")
    except OSError as error:
        raise EpisodeValidationError(f"Could not open episode artifact: {error}") from error

    with artifact:
        if _decode_text(artifact.attrs.get("artifact_type", "")) != ARTIFACT_TYPE:
            raise EpisodeValidationError("Unexpected episode artifact_type")
        if artifact.attrs.get("schema_version") != EPISODE_SCHEMA_VERSION:
            raise EpisodeValidationError("Unexpected episode schema_version")
        status = _decode_text(artifact.attrs.get("status", ""))
        allowed_statuses = {"complete"}
        if allow_incomplete:
            allowed_statuses.update({"recording", "aborted"})
        if status not in allowed_statuses:
            raise EpisodeValidationError(f"Episode status is not complete: {status!r}")

        try:
            embedded_protocol = json.loads(
                _decode_text(artifact["provenance/evaluation_protocol_json"][()])
            )
            metadata = json.loads(_decode_text(artifact["episode/metadata_json"][()]))
        except (KeyError, json.JSONDecodeError) as error:
            raise EpisodeValidationError("Episode provenance or metadata is invalid") from error
        if not isinstance(embedded_protocol, dict) or not isinstance(metadata, dict):
            raise EpisodeValidationError("Episode provenance and metadata must be objects")
        validate_evaluation_protocol(embedded_protocol)
        if protocol is not None:
            validate_evaluation_protocol(protocol)
            if protocol_sha256(protocol) != protocol_sha256(embedded_protocol):
                raise EpisodeValidationError("Embedded evaluation protocol does not match")
        active_protocol = embedded_protocol
        _validate_metadata(metadata, active_protocol, complete=status == "complete")

        try:
            steps = artifact["steps"]
        except KeyError as error:
            raise EpisodeValidationError("Episode artifact has no steps group") from error
        required_step_fields = active_protocol["recording"]["required_step_fields"]
        missing_fields = set(required_step_fields) - set(steps)
        if missing_fields:
            raise EpisodeValidationError(
                f"Missing episode step fields: {sorted(missing_fields)!r}"
            )
        lengths = {name: len(steps[name]) for name in required_step_fields}
        if len(set(lengths.values())) != 1:
            raise EpisodeValidationError(f"Episode step datasets are misaligned: {lengths!r}")
        step_count = next(iter(lengths.values()))
        if int(artifact.attrs.get("step_count", -1)) != step_count:
            raise EpisodeValidationError("Episode step_count attribute is inconsistent")
        if step_count == 0:
            raise EpisodeValidationError("Episode contains no control steps")

        height, width = active_protocol["observation"]["capture_resolution"]
        action_dimension = active_protocol["simulation"]["action"]["dimension"]
        action_chunk_length = active_protocol["simulation"]["action"]["chunk_length"]
        proprioception_dimension = active_protocol["observation"]["proprioception"][
            "dimension"
        ]
        expected_shapes = {
            "agent_rgb": (step_count, height, width, 3),
            "wrist_rgb": (step_count, height, width, 3),
            "proprioception": (step_count, proprioception_dimension),
            "predicted_action_chunk": (
                step_count,
                action_chunk_length,
                action_dimension,
            ),
            "executed_action": (step_count, action_dimension),
        }
        for name, expected_shape in expected_shapes.items():
            if steps[name].shape != expected_shape:
                raise EpisodeValidationError(
                    f"Expected {name} shape {expected_shape}, found {steps[name].shape}"
                )
        for name in (
            "step_index",
            "phase",
            "sim_time_seconds",
            "action_chunk_index",
            "reward",
            "done",
            "policy_latency_ms",
        ):
            if steps[name].shape != (step_count,):
                raise EpisodeValidationError(f"Expected scalar step dataset for {name}")
        if steps["mujoco_state"].ndim != 2 or steps["mujoco_state"].shape[1] == 0:
            raise EpisodeValidationError("mujoco_state must have a stable nonzero dimension")
        if steps["agent_rgb"].dtype != np.uint8 or steps["wrist_rgb"].dtype != np.uint8:
            raise EpisodeValidationError("RGB datasets must use uint8")
        expected_dtypes = {
            "step_index": np.dtype(np.int64),
            "phase": np.dtype(np.uint8),
            "sim_time_seconds": np.dtype(np.float64),
            "proprioception": np.dtype(np.float64),
            "predicted_action_chunk": np.dtype(np.float64),
            "executed_action": np.dtype(np.float64),
            "action_chunk_index": np.dtype(np.int32),
            "mujoco_state": np.dtype(np.float64),
            "reward": np.dtype(np.float64),
            "done": np.dtype(np.bool_),
            "policy_latency_ms": np.dtype(np.float64),
        }
        for name, expected_dtype in expected_dtypes.items():
            if steps[name].dtype != expected_dtype:
                raise EpisodeValidationError(
                    f"Expected {name} dtype {expected_dtype}, found {steps[name].dtype}"
                )
        phase_encoding = _decode_text(steps.attrs.get("phase_encoding_json", ""))
        if phase_encoding != _canonical_json(PHASE_ENCODING):
            raise EpisodeValidationError("Episode phase encoding is missing or changed")

        step_indexes = steps["step_index"][:]
        phases = steps["phase"][:]
        sim_times = steps["sim_time_seconds"][:]
        action_indexes = steps["action_chunk_index"][:]
        executed_actions = steps["executed_action"][:]
        predicted_chunks = steps["predicted_action_chunk"][:]
        rewards = steps["reward"][:]
        done = steps["done"][:]
        latencies = steps["policy_latency_ms"][:]
        settling_steps = active_protocol["simulation"]["settling_steps"]
        stopped_during_settling = (
            step_count < settling_steps
            and status == "complete"
            and metadata["success"] is True
            and bool(done[-1])
        )
        incomplete_during_settling = (
            step_count < settling_steps and status in {"recording", "aborted"}
        )
        if (
            step_count < settling_steps
            and not stopped_during_settling
            and not incomplete_during_settling
        ):
            raise EpisodeValidationError("Episode does not contain every settling step")
        recorded_settling_steps = min(step_count, settling_steps)
        if not np.array_equal(step_indexes, np.arange(step_count)):
            raise EpisodeValidationError("Episode step_index is not contiguous from zero")
        expected_times = np.arange(step_count) / active_protocol["simulation"][
            "control_frequency_hz"
        ]
        if not np.allclose(sim_times, expected_times, rtol=0.0, atol=1e-12):
            raise EpisodeValidationError("Episode sim_time_seconds drifted from control time")
        if not np.all(
            phases[:recorded_settling_steps] == PHASE_ENCODING["settling"]
        ):
            raise EpisodeValidationError("Episode settling phase is incomplete")
        if not np.all(phases[recorded_settling_steps:] == PHASE_ENCODING["policy"]):
            raise EpisodeValidationError("Episode policy phase is not contiguous")
        expected_settling_action = np.asarray(
            active_protocol["simulation"]["settling_action"], dtype=np.float64
        )
        if not np.all(
            executed_actions[:recorded_settling_steps] == expected_settling_action
        ):
            raise EpisodeValidationError("Episode settling actions changed")
        if not np.all(action_indexes[:recorded_settling_steps] == -1):
            raise EpisodeValidationError("Settling action_chunk_index must be -1")
        if not np.all(np.isnan(predicted_chunks[:recorded_settling_steps])):
            raise EpisodeValidationError("Settling predicted_action_chunk must be absent")
        if not np.all(np.isnan(latencies[:recorded_settling_steps])):
            raise EpisodeValidationError("Settling policy latency must be absent")

        policy_step_count = step_count - recorded_settling_steps
        expected_chunk_indexes = np.arange(policy_step_count) % action_chunk_length
        if not np.array_equal(
            action_indexes[recorded_settling_steps:], expected_chunk_indexes
        ):
            raise EpisodeValidationError("Policy action_chunk_index sequence is invalid")
        if not np.all(np.isfinite(predicted_chunks[recorded_settling_steps:])):
            raise EpisodeValidationError("Policy predicted action chunks must be finite")
        query_rows = expected_chunk_indexes == 0
        policy_latencies = latencies[recorded_settling_steps:]
        if not np.all(np.isfinite(policy_latencies[query_rows])):
            raise EpisodeValidationError("Policy query latency is missing")
        if np.any(policy_latencies[query_rows] < 0):
            raise EpisodeValidationError("Policy query latency cannot be negative")
        if not np.all(np.isnan(policy_latencies[~query_rows])):
            raise EpisodeValidationError("Non-query policy latency must be absent")

        finite_datasets = (
            steps["proprioception"][:],
            executed_actions,
            steps["mujoco_state"][:],
            rewards,
        )
        if any(not np.all(np.isfinite(values)) for values in finite_datasets):
            raise EpisodeValidationError("Episode contains non-finite recorded values")
        action_minimum, action_maximum = active_protocol["simulation"]["action"]["range"]
        if np.any(executed_actions < action_minimum) or np.any(
            executed_actions > action_maximum
        ):
            raise EpisodeValidationError("Episode contains an out-of-range executed action")
        if np.any(predicted_chunks[recorded_settling_steps:] < action_minimum) or np.any(
            predicted_chunks[recorded_settling_steps:] > action_maximum
        ):
            raise EpisodeValidationError("Episode contains an out-of-range predicted action")

        max_policy_steps = active_protocol["benchmark"]["suites"][metadata["suite"]][
            "max_policy_steps"
        ]
        if policy_step_count > max_policy_steps:
            raise EpisodeValidationError("Episode exceeds the suite policy-step budget")
        success = bool(metadata["success"])
        if status == "complete" and success != bool(np.any(done)):
            raise EpisodeValidationError("Episode success does not match environment done")
        if np.any(done[:-1]):
            raise EpisodeValidationError("Episode continued after environment success")

        return {
            "path": str(artifact_path),
            "status": status,
            "protocol_id": metadata["protocol_id"],
            "suite": metadata["suite"],
            "runtime_task_index": metadata["runtime_task_index"],
            "step_count": step_count,
            "settling_step_count": recorded_settling_steps,
            "policy_step_count": policy_step_count,
            "success": metadata["success"],
            "termination_reason": metadata["termination_reason"],
        }

"""Streaming validation for the pinned LIBERO-Plus LeRobot v2.1 dataset."""

from __future__ import annotations

import json
import math
import operator
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

PINNED_DATASET_REVISION = "22c57433fef692b5b9ecc0795344daac7fa867a5"
PINNED_DATASET_INFO = {
    "codebase_version": "v2.1",
    "robot_type": "panda",
    "total_episodes": 14_347,
    "total_frames": 2_238_036,
    "total_tasks": 40,
    "total_videos": 28_694,
    "total_chunks": 15,
    "chunks_size": 1_000,
    "fps": 20,
}
CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
)
REQUIRED_PARQUET_COLUMNS = (
    "action",
    "episode_index",
    "frame_index",
    "index",
    "observation.state",
    "task_index",
    "timestamp",
)
EXPECTED_PARQUET_SCHEMA = (
    ("observation.state", "fixed_size_list<element: float>[8]"),
    ("action", "fixed_size_list<element: float>[7]"),
    ("timestamp", "float"),
    ("frame_index", "int64"),
    ("episode_index", "int64"),
    ("index", "int64"),
    ("task_index", "int64"),
)
EXPECTED_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
EXPECTED_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)

ParquetRowReader = Callable[
    [Path, tuple[str, ...]],
    Iterable[Mapping[str, Any]],
]


class DemonstrationValidationError(ValueError):
    """Raised when demonstration metadata or episode contents violate the contract."""


@dataclass(frozen=True)
class DatasetInfo:
    """Validated fields needed to stream the pinned dataset."""

    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    data_path: str
    video_path: str


@dataclass(frozen=True)
class TaskRecord:
    """One task in ``meta/tasks.jsonl``."""

    task_index: int
    instruction: str


@dataclass(frozen=True)
class EpisodeRecord:
    """Resolved metadata and source paths for one episode."""

    episode_index: int
    task_index: int
    instruction: str
    length: int
    start_index: int
    parquet_path: Path
    video_paths: tuple[tuple[str, Path], ...]


@dataclass(frozen=True)
class TrajectoryRow:
    """One validated, row-aligned control sample."""

    index: int
    episode_index: int
    frame_index: int
    task_index: int
    timestamp: float
    state: tuple[float, ...]
    action: tuple[float, ...]


@dataclass(frozen=True)
class VideoStreamInfo:
    """Properties obtained by decoding one video stream."""

    frame_count: int
    width: int
    height: int
    fps: float
    codec: str


@dataclass(frozen=True)
class DatasetValidationSummary:
    """Compact result of a complete streaming validation pass."""

    episode_count: int
    frame_count: int
    task_count: int
    video_count: int
    decoded_video_count: int


@dataclass(frozen=True)
class EpisodeValidationSummary:
    """Compact result of validating one episode's payloads."""

    episode_index: int
    frame_count: int
    video_count: int
    decoded_video_count: int


VideoProbe = Callable[[Path], VideoStreamInfo]


def iter_parquet_rows(
    path: Path,
    columns: tuple[str, ...] = REQUIRED_PARQUET_COLUMNS,
) -> Iterator[Mapping[str, Any]]:
    """Yield Parquet rows in bounded batches using the optional PyArrow runtime."""

    try:
        from pyarrow import parquet
    except ImportError as error:  # pragma: no cover - depends on the data runtime
        raise RuntimeError(
            "Streaming Parquet rows requires pyarrow; install it in the dataset runtime "
            "or inject a parquet_row_reader"
        ) from error

    try:
        parquet_file = parquet.ParquetFile(path)
        actual_schema = tuple(
            (field.name, str(field.type)) for field in parquet_file.schema_arrow
        )
        if actual_schema != EXPECTED_PARQUET_SCHEMA:
            raise DemonstrationValidationError(
                f"{path} has unexpected Arrow schema: {actual_schema!r}"
            )
        batches = parquet_file.iter_batches(
            batch_size=1_024,
            columns=list(columns),
            use_threads=False,
        )
        for batch in batches:
            values = batch.to_pydict()
            for row_offset in range(batch.num_rows):
                yield {name: values[name][row_offset] for name in columns}
    except DemonstrationValidationError:
        raise
    except Exception as error:
        raise DemonstrationValidationError(f"cannot stream required columns from {path}") from error


def probe_video(path: Path) -> VideoStreamInfo:
    """Decode a complete video without retaining frames and report stream properties."""

    try:
        import av
    except ImportError as error:  # pragma: no cover - depends on the data runtime
        raise RuntimeError(
            "Video decoding requires PyAV; install it in the dataset runtime or inject a "
            "video_probe"
        ) from error

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise DemonstrationValidationError(f"video contains no stream: {path}")
        stream = container.streams.video[0]
        codec = str(stream.codec_context.name)
        rate = stream.average_rate
        if rate is None:
            raise DemonstrationValidationError(f"video has no average frame rate: {path}")

        frame_count = 0
        width = 0
        height = 0
        for frame in container.decode(video=0):
            if frame_count == 0:
                width, height = frame.width, frame.height
            elif (frame.width, frame.height) != (width, height):
                raise DemonstrationValidationError(f"video dimensions change mid-stream: {path}")
            frame_count += 1

    return VideoStreamInfo(
        frame_count=frame_count,
        width=width,
        height=height,
        fps=float(rate),
        codec=codec,
    )


class DemonstrationReader:
    """Stream and validate one local LeRobot v2.1 repository snapshot.

    Metadata and the small task table are retained in memory. Episode metadata,
    Parquet rows, and decoded video frames are consumed incrementally. Exhaust an
    iterator to run its final count checks; :meth:`validate` always does so.
    """

    def __init__(
        self,
        repository_root: str | Path,
        *,
        parquet_row_reader: ParquetRowReader | None = None,
        video_probe: VideoProbe | None = None,
        strict_pinned_metadata: bool = True,
        timestamp_tolerance_seconds: float = 1e-5,
    ) -> None:
        self.repository_root = Path(repository_root).expanduser().resolve()
        if timestamp_tolerance_seconds < 0 or not math.isfinite(timestamp_tolerance_seconds):
            raise ValueError("timestamp_tolerance_seconds must be finite and non-negative")
        self.timestamp_tolerance_seconds = timestamp_tolerance_seconds
        self._parquet_row_reader = parquet_row_reader or iter_parquet_rows
        self._video_probe = video_probe or probe_video
        self.info = self._load_info(strict_pinned_metadata=strict_pinned_metadata)
        self.tasks = self._load_tasks()
        self._task_by_instruction = {task.instruction: task for task in self.tasks}

    def iter_episodes(self) -> Iterator[EpisodeRecord]:
        """Yield episode descriptors in canonical index order."""

        path = self.repository_root / "meta/episodes.jsonl"
        episode_count = 0
        frame_count = 0
        for raw in _read_jsonl(path):
            episode_index = _metadata_int(raw, "episode_index", path)
            if episode_index != episode_count:
                raise DemonstrationValidationError(
                    f"{path}: expected episode_index {episode_count}, found {episode_index}"
                )
            length = _metadata_int(raw, "length", path)
            if length <= 0:
                raise DemonstrationValidationError(
                    f"{path}: episode {episode_index} length must be positive"
                )
            episode_tasks = raw.get("tasks")
            if (
                not isinstance(episode_tasks, list)
                or len(episode_tasks) != 1
                or not isinstance(episode_tasks[0], str)
                or not episode_tasks[0]
            ):
                raise DemonstrationValidationError(
                    f"{path}: episode {episode_index} must name exactly one non-empty task"
                )
            instruction = episode_tasks[0]
            task = self._task_by_instruction.get(instruction)
            if task is None:
                raise DemonstrationValidationError(
                    f"{path}: episode {episode_index} names an unknown task {instruction!r}"
                )

            chunk_index = episode_index // self.info.chunks_size
            parquet_path = self._resolve_source_path(
                self.info.data_path,
                episode_chunk=chunk_index,
                episode_index=episode_index,
            )
            video_paths = tuple(
                (
                    camera_key,
                    self._resolve_source_path(
                        self.info.video_path,
                        episode_chunk=chunk_index,
                        episode_index=episode_index,
                        video_key=camera_key,
                    ),
                )
                for camera_key in CAMERA_KEYS
            )
            yield EpisodeRecord(
                episode_index=episode_index,
                task_index=task.task_index,
                instruction=instruction,
                length=length,
                start_index=frame_count,
                parquet_path=parquet_path,
                video_paths=video_paths,
            )
            episode_count += 1
            frame_count += length

        if episode_count != self.info.total_episodes:
            raise DemonstrationValidationError(
                f"{path}: found {episode_count} episodes, expected {self.info.total_episodes}"
            )
        if frame_count != self.info.total_frames:
            raise DemonstrationValidationError(
                f"{path}: episode lengths sum to {frame_count}, expected {self.info.total_frames}"
            )

    def iter_episode_rows(self, episode: EpisodeRecord) -> Iterator[TrajectoryRow]:
        """Yield one episode's validated Parquet rows without materializing the shard."""

        row_count = 0
        rows = self._parquet_row_reader(episode.parquet_path, REQUIRED_PARQUET_COLUMNS)
        for row_count, raw in enumerate(rows, start=1):
            frame_offset = row_count - 1
            if row_count > episode.length:
                raise DemonstrationValidationError(
                    f"episode {episode.episode_index} has more than {episode.length} Parquet rows"
                )
            yield self._validate_row(raw, episode, frame_offset)

        if row_count != episode.length:
            raise DemonstrationValidationError(
                f"episode {episode.episode_index} has {row_count} Parquet rows, "
                f"expected {episode.length}"
            )

    def validate(self, *, decode_videos: bool = False) -> DatasetValidationSummary:
        """Run a complete streaming validation pass over metadata and every episode."""

        episode_count = 0
        frame_count = 0
        decoded_video_count = 0
        for episode in self.iter_episodes():
            episode_summary = self.validate_episode(episode, decode_videos=decode_videos)
            frame_count += episode_summary.frame_count
            decoded_video_count += episode_summary.decoded_video_count
            episode_count += 1

        return DatasetValidationSummary(
            episode_count=episode_count,
            frame_count=frame_count,
            task_count=len(self.tasks),
            video_count=episode_count * len(CAMERA_KEYS),
            decoded_video_count=decoded_video_count,
        )

    def validate_episode(
        self,
        episode: EpisodeRecord,
        *,
        decode_videos: bool = False,
    ) -> EpisodeValidationSummary:
        """Validate one descriptor's rows and, optionally, both decoded videos."""

        self._validate_source_files(episode)
        frame_count = sum(1 for _ in self.iter_episode_rows(episode))
        decoded_video_count = 0
        if decode_videos:
            for camera_key, path in episode.video_paths:
                self._validate_video(episode, camera_key, path)
                decoded_video_count += 1
        return EpisodeValidationSummary(
            episode_index=episode.episode_index,
            frame_count=frame_count,
            video_count=len(episode.video_paths),
            decoded_video_count=decoded_video_count,
        )

    @staticmethod
    def _validate_source_files(episode: EpisodeRecord) -> None:
        source_paths = (episode.parquet_path, *(item[1] for item in episode.video_paths))
        for source_path in source_paths:
            if not source_path.is_file():
                raise DemonstrationValidationError(
                    f"episode {episode.episode_index} source file is missing: {source_path}"
                )

    def _load_info(self, *, strict_pinned_metadata: bool) -> DatasetInfo:
        path = self.repository_root / "meta/info.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DemonstrationValidationError(f"cannot read valid JSON from {path}") from error
        if not isinstance(raw, dict):
            raise DemonstrationValidationError(f"{path} must contain a JSON object")

        invariant_values = {
            "codebase_version": "v2.1",
            "robot_type": "panda",
            "fps": 20,
            "data_path": EXPECTED_DATA_PATH,
            "video_path": EXPECTED_VIDEO_PATH,
        }
        for name, expected in invariant_values.items():
            if raw.get(name) != expected:
                raise DemonstrationValidationError(
                    f"{path}: {name} is {raw.get(name)!r}, expected {expected!r}"
                )
        if strict_pinned_metadata:
            for name, expected in PINNED_DATASET_INFO.items():
                if raw.get(name) != expected:
                    raise DemonstrationValidationError(
                        f"{path}: {name} is {raw.get(name)!r}, expected pinned value {expected!r}"
                    )

        numeric_names = (
            "total_episodes",
            "total_frames",
            "total_tasks",
            "total_videos",
            "total_chunks",
            "chunks_size",
            "fps",
        )
        values = {name: _metadata_int(raw, name, path) for name in numeric_names}
        if any(value <= 0 for value in values.values()):
            raise DemonstrationValidationError(f"{path}: dataset counts and fps must be positive")
        expected_video_count = values["total_episodes"] * len(CAMERA_KEYS)
        if values["total_videos"] != expected_video_count:
            raise DemonstrationValidationError(
                f"{path}: total_videos must be {expected_video_count} for two camera streams"
            )
        expected_chunks = math.ceil(values["total_episodes"] / values["chunks_size"])
        if values["total_chunks"] != expected_chunks:
            raise DemonstrationValidationError(
                f"{path}: total_chunks is {values['total_chunks']}, expected {expected_chunks}"
            )
        if raw.get("splits") != {"train": f"0:{values['total_episodes']}"}:
            raise DemonstrationValidationError(f"{path}: unexpected train split declaration")
        _validate_features(raw.get("features"), path)

        return DatasetInfo(
            total_episodes=values["total_episodes"],
            total_frames=values["total_frames"],
            total_tasks=values["total_tasks"],
            total_videos=values["total_videos"],
            total_chunks=values["total_chunks"],
            chunks_size=values["chunks_size"],
            fps=values["fps"],
            data_path=str(raw["data_path"]),
            video_path=str(raw["video_path"]),
        )

    def _load_tasks(self) -> tuple[TaskRecord, ...]:
        path = self.repository_root / "meta/tasks.jsonl"
        tasks: list[TaskRecord] = []
        instructions: set[str] = set()
        for raw in _read_jsonl(path):
            task_index = _metadata_int(raw, "task_index", path)
            if task_index != len(tasks):
                raise DemonstrationValidationError(
                    f"{path}: expected task_index {len(tasks)}, found {task_index}"
                )
            instruction = raw.get("task")
            if not isinstance(instruction, str) or not instruction:
                raise DemonstrationValidationError(
                    f"{path}: task {task_index} must have a non-empty instruction"
                )
            if instruction in instructions:
                raise DemonstrationValidationError(
                    f"{path}: duplicate task instruction {instruction!r}"
                )
            tasks.append(TaskRecord(task_index=task_index, instruction=instruction))
            instructions.add(instruction)
        if len(tasks) != self.info.total_tasks:
            raise DemonstrationValidationError(
                f"{path}: found {len(tasks)} tasks, expected {self.info.total_tasks}"
            )
        return tuple(tasks)

    def _resolve_source_path(self, template: str, **values: Any) -> Path:
        try:
            relative = template.format(**values)
        except (KeyError, ValueError) as error:
            raise DemonstrationValidationError(
                f"invalid source path template {template!r}"
            ) from error
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or relative in {"", "."}:
            raise DemonstrationValidationError(f"unsafe repository path {relative!r}")
        return self.repository_root.joinpath(*path.parts)

    def _validate_row(
        self,
        raw: Mapping[str, Any],
        episode: EpisodeRecord,
        frame_offset: int,
    ) -> TrajectoryRow:
        if not isinstance(raw, Mapping):
            raise DemonstrationValidationError(
                f"episode {episode.episode_index} frame {frame_offset} is not a row mapping"
            )
        missing = set(REQUIRED_PARQUET_COLUMNS) - set(raw)
        if missing:
            raise DemonstrationValidationError(
                f"episode {episode.episode_index} frame {frame_offset} is missing columns "
                f"{sorted(missing)}"
            )

        identities = {
            "index": episode.start_index + frame_offset,
            "episode_index": episode.episode_index,
            "frame_index": frame_offset,
            "task_index": episode.task_index,
        }
        parsed_identities: dict[str, int] = {}
        for name, expected in identities.items():
            actual = _row_int(raw[name], name, episode.episode_index, frame_offset)
            if actual != expected:
                raise DemonstrationValidationError(
                    f"episode {episode.episode_index} frame {frame_offset}: "
                    f"{name} is {actual}, expected {expected}"
                )
            parsed_identities[name] = actual

        try:
            timestamp = float(raw["timestamp"])
        except (TypeError, ValueError) as error:
            raise DemonstrationValidationError(
                f"episode {episode.episode_index} frame {frame_offset}: timestamp is not numeric"
            ) from error
        expected_timestamp = frame_offset / self.info.fps
        if not math.isfinite(timestamp) or not math.isclose(
            timestamp,
            expected_timestamp,
            rel_tol=0.0,
            abs_tol=self.timestamp_tolerance_seconds,
        ):
            raise DemonstrationValidationError(
                f"episode {episode.episode_index} frame {frame_offset}: timestamp is "
                f"{timestamp!r}, expected {expected_timestamp:.6f} at {self.info.fps} Hz"
            )
        state = _finite_vector(
            raw["observation.state"],
            length=8,
            name="observation.state",
            episode_index=episode.episode_index,
            frame_index=frame_offset,
        )
        action = _finite_vector(
            raw["action"],
            length=7,
            name="action",
            episode_index=episode.episode_index,
            frame_index=frame_offset,
        )
        return TrajectoryRow(
            index=parsed_identities["index"],
            episode_index=parsed_identities["episode_index"],
            frame_index=parsed_identities["frame_index"],
            task_index=parsed_identities["task_index"],
            timestamp=timestamp,
            state=state,
            action=action,
        )

    def _validate_video(
        self,
        episode: EpisodeRecord,
        camera_key: str,
        path: Path,
    ) -> None:
        stream = self._video_probe(path)
        if stream.frame_count != episode.length:
            raise DemonstrationValidationError(
                f"episode {episode.episode_index} {camera_key} has {stream.frame_count} frames, "
                f"expected {episode.length}"
            )
        if (stream.width, stream.height) != (256, 256):
            raise DemonstrationValidationError(
                f"episode {episode.episode_index} {camera_key} dimensions are "
                f"{stream.width}x{stream.height}, expected 256x256"
            )
        if not math.isclose(stream.fps, self.info.fps, rel_tol=0.0, abs_tol=1e-6):
            raise DemonstrationValidationError(
                f"episode {episode.episode_index} {camera_key} fps is {stream.fps}, "
                f"expected {self.info.fps}"
            )
        if stream.codec not in {"av1", "libdav1d"}:
            raise DemonstrationValidationError(
                f"episode {episode.episode_index} {camera_key} codec is {stream.codec!r}"
            )


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise DemonstrationValidationError(f"cannot open {path}") from error
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise DemonstrationValidationError(
                    f"invalid JSON in {path} at line {line_number}"
                ) from error
            if not isinstance(value, dict):
                raise DemonstrationValidationError(
                    f"{path} line {line_number} must contain a JSON object"
                )
            yield value


def _metadata_int(value: Mapping[str, Any], name: str, path: Path) -> int:
    item = value.get(name)
    if isinstance(item, bool) or not isinstance(item, int):
        raise DemonstrationValidationError(f"{path}: {name} must be an integer")
    return item


def _row_int(value: Any, name: str, episode_index: int, frame_index: int) -> int:
    if isinstance(value, bool):
        raise DemonstrationValidationError(
            f"episode {episode_index} frame {frame_index}: {name} must be an integer"
        )
    try:
        return operator.index(value)
    except TypeError as error:
        raise DemonstrationValidationError(
            f"episode {episode_index} frame {frame_index}: {name} must be an integer"
        ) from error


def _finite_vector(
    value: Any,
    *,
    length: int,
    name: str,
    episode_index: int,
    frame_index: int,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        vector: Sequence[Any] = ()
    else:
        try:
            vector = tuple(value)
        except TypeError:
            vector = ()
    if len(vector) != length:
        raise DemonstrationValidationError(
            f"episode {episode_index} frame {frame_index}: {name} shape is "
            f"[{len(vector)}], expected [{length}]"
        )
    try:
        converted = tuple(float(item) for item in vector)
    except (TypeError, ValueError) as error:
        raise DemonstrationValidationError(
            f"episode {episode_index} frame {frame_index}: {name} must be numeric"
        ) from error
    if not all(math.isfinite(item) for item in converted):
        raise DemonstrationValidationError(
            f"episode {episode_index} frame {frame_index}: {name} contains non-finite values"
        )
    return converted


def _validate_features(value: Any, path: Path) -> None:
    if not isinstance(value, Mapping):
        raise DemonstrationValidationError(f"{path}: features must be an object")
    expected_features = {
        "observation.state": ("float32", [8]),
        "action": ("float32", [7]),
        "timestamp": ("float32", [1]),
        "frame_index": ("int64", [1]),
        "episode_index": ("int64", [1]),
        "index": ("int64", [1]),
        "task_index": ("int64", [1]),
    }
    for name, (dtype, shape) in expected_features.items():
        feature = value.get(name)
        if not isinstance(feature, Mapping):
            raise DemonstrationValidationError(f"{path}: missing feature declaration {name}")
        if feature.get("dtype") != dtype or feature.get("shape") != shape:
            raise DemonstrationValidationError(
                f"{path}: unexpected feature declaration for {name}: {dict(feature)}"
            )
    for camera_key in CAMERA_KEYS:
        feature = value.get(camera_key)
        if not isinstance(feature, Mapping):
            raise DemonstrationValidationError(
                f"{path}: missing feature declaration {camera_key}"
            )
        video_info = feature.get("info")
        if (
            feature.get("dtype") != "video"
            or feature.get("shape") != [256, 256, 3]
            or not isinstance(video_info, Mapping)
            or video_info.get("video.width") != 256
            or video_info.get("video.height") != 256
            or video_info.get("video.channels") != 3
            or video_info.get("video.fps") != 20
            or video_info.get("video.codec") != "av1"
        ):
            raise DemonstrationValidationError(
                f"{path}: unexpected feature declaration for {camera_key}: {dict(feature)}"
            )

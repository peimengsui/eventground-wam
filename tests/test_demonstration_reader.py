from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from eventground_wam.demonstration_reader import (
    CAMERA_KEYS,
    REQUIRED_PARQUET_COLUMNS,
    DatasetValidationSummary,
    DecodedVideoFrame,
    DemonstrationReader,
    DemonstrationValidationError,
    EpisodeValidationSummary,
    SynchronizedStep,
    VideoStreamInfo,
)


def _info() -> dict[str, Any]:
    scalar_features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    video_feature = {
        "dtype": "video",
        "shape": [256, 256, 3],
        "names": ["height", "width", "channel"],
        "info": {
            "video.height": 256,
            "video.width": 256,
            "video.codec": "av1",
            "video.fps": 20,
            "video.channels": 3,
        },
    }
    return {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": 2,
        "total_frames": 3,
        "total_tasks": 2,
        "total_videos": 4,
        "total_chunks": 1,
        "chunks_size": 1_000,
        "fps": 20,
        "splits": {"train": "0:2"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "observation.images.front": deepcopy(video_feature),
            "observation.images.wrist": deepcopy(video_feature),
            "observation.state": {
                "dtype": "float32",
                "shape": [8],
                "names": [f"state_{index}" for index in range(8)],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": [f"action_{index}" for index in range(7)],
            },
            **scalar_features,
        },
    }


def _row(
    *,
    index: int,
    episode_index: int,
    frame_index: int,
    task_index: int,
) -> dict[str, Any]:
    return {
        "index": index,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "task_index": task_index,
        "timestamp": frame_index / 20,
        "observation.state": [float(value) for value in range(8)],
        "action": [float(value) for value in range(7)],
    }


@pytest.fixture()
def repository(tmp_path: Path) -> tuple[Path, dict[Path, list[dict[str, Any]]]]:
    root = tmp_path / "repo"
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(_info()), encoding="utf-8")
    tasks = [
        {"task_index": 0, "task": "put the mug on the plate"},
        {"task_index": 1, "task": "put the cheese in the bowl"},
    ]
    episodes = [
        {"episode_index": 0, "tasks": [tasks[0]["task"]], "length": 2},
        {"episode_index": 1, "tasks": [tasks[1]["task"]], "length": 1},
    ]
    (root / "meta/tasks.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in tasks),
        encoding="utf-8",
    )
    (root / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in episodes),
        encoding="utf-8",
    )

    row_data: dict[Path, list[dict[str, Any]]] = {}
    lengths = (2, 1)
    global_index = 0
    for episode_index, length in enumerate(lengths):
        data_path = root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.touch()
        row_data[data_path] = [
            _row(
                index=global_index + frame_index,
                episode_index=episode_index,
                frame_index=frame_index,
                task_index=episode_index,
            )
            for frame_index in range(length)
        ]
        global_index += length
        for camera_key in CAMERA_KEYS:
            video_path = (
                root
                / "videos/chunk-000"
                / camera_key
                / f"episode_{episode_index:06d}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.touch()
    return root, row_data


def test_complete_validation_streams_rows_in_deterministic_order(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
) -> None:
    root, row_data = repository
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def row_reader(path: Path, columns: tuple[str, ...]):
        calls.append((path, columns))
        yield from row_data[path]

    reader = DemonstrationReader(
        root,
        parquet_row_reader=row_reader,
        strict_pinned_metadata=False,
    )

    summary = reader.validate()

    assert summary == DatasetValidationSummary(
        episode_count=2,
        frame_count=3,
        task_count=2,
        video_count=4,
        decoded_video_count=0,
    )
    assert [call[0].name for call in calls] == [
        "episode_000000.parquet",
        "episode_000001.parquet",
    ]
    assert all(call[1] == REQUIRED_PARQUET_COLUMNS for call in calls)
    episodes = list(reader.iter_episodes())
    assert [(item.episode_index, item.task_index, item.start_index) for item in episodes] == [
        (0, 0, 0),
        (1, 1, 2),
    ]


def test_video_decoding_is_optional_and_validates_every_stream(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
) -> None:
    root, row_data = repository
    probed: list[Path] = []

    def video_probe(path: Path) -> VideoStreamInfo:
        probed.append(path)
        episode_index = int(path.stem.removeprefix("episode_"))
        return VideoStreamInfo(
            frame_count=(2, 1)[episode_index],
            width=256,
            height=256,
            fps=20.0,
            codec="libdav1d",
        )

    reader = DemonstrationReader(
        root,
        parquet_row_reader=lambda path, columns: iter(row_data[path]),
        video_probe=video_probe,
        strict_pinned_metadata=False,
    )
    assert reader.validate(decode_videos=False).decoded_video_count == 0
    assert probed == []

    summary = reader.validate(decode_videos=True)

    assert summary.decoded_video_count == 4
    assert [path.parent.name for path in probed] == [
        "observation.images.front",
        "observation.images.wrist",
        "observation.images.front",
        "observation.images.wrist",
    ]


def test_one_episode_can_be_validated_after_exhausting_all_metadata(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
) -> None:
    root, row_data = repository
    opened_parquet: list[Path] = []

    def row_reader(path: Path, columns: tuple[str, ...]):
        opened_parquet.append(path)
        return iter(row_data[path])

    reader = DemonstrationReader(
        root,
        parquet_row_reader=row_reader,
        strict_pinned_metadata=False,
    )
    episodes = list(reader.iter_episodes())

    summary = reader.validate_episode(episodes[1])

    assert summary == EpisodeValidationSummary(
        episode_index=1,
        frame_count=1,
        video_count=2,
        decoded_video_count=0,
    )
    assert [path.name for path in opened_parquet] == ["episode_000001.parquet"]


def test_episode_steps_stream_aligned_rows_and_both_cameras(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
) -> None:
    root, row_data = repository

    def frame_reader(path: Path):
        episode_index = int(path.stem.removeprefix("episode_"))
        for frame_index in range((2, 1)[episode_index]):
            yield DecodedVideoFrame(
                frame_index=frame_index,
                timestamp=frame_index / 20,
                width=256,
                height=256,
                pixel_format="yuv420p",
                payload=(path.parent.name, frame_index),
            )

    reader = DemonstrationReader(
        root,
        parquet_row_reader=lambda path, columns: iter(row_data[path]),
        video_frame_reader=frame_reader,
        strict_pinned_metadata=False,
    )
    first_episode = next(reader.iter_episodes())

    steps = list(reader.iter_episode_steps(first_episode))

    assert all(isinstance(step, SynchronizedStep) for step in steps)
    assert [step.row.frame_index for step in steps] == [0, 1]
    assert [step.front_frame.payload for step in steps] == [
        ("observation.images.front", 0),
        ("observation.images.front", 1),
    ]
    assert [step.wrist_frame.payload for step in steps] == [
        ("observation.images.wrist", 0),
        ("observation.images.wrist", 1),
    ]


def test_episode_steps_reject_mismatched_modality_counts(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
) -> None:
    root, row_data = repository

    def frame_reader(path: Path):
        frame_count = 1 if path.parent.name == "observation.images.wrist" else 2
        for frame_index in range(frame_count):
            yield DecodedVideoFrame(
                frame_index=frame_index,
                timestamp=frame_index / 20,
                width=256,
                height=256,
                pixel_format="yuv420p",
                payload=None,
            )

    reader = DemonstrationReader(
        root,
        parquet_row_reader=lambda path, columns: iter(row_data[path]),
        video_frame_reader=frame_reader,
        strict_pinned_metadata=False,
    )
    first_episode = next(reader.iter_episodes())

    with pytest.raises(DemonstrationValidationError, match="different frame counts"):
        list(reader.iter_episode_steps(first_episode))


def test_episode_steps_reject_video_timestamp_drift(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
) -> None:
    root, row_data = repository

    def frame_reader(path: Path):
        for frame_index in range(2):
            timestamp = frame_index / 20
            if path.parent.name == "observation.images.front" and frame_index == 1:
                timestamp += 0.001
            yield DecodedVideoFrame(
                frame_index=frame_index,
                timestamp=timestamp,
                width=256,
                height=256,
                pixel_format="yuv420p",
                payload=None,
            )

    reader = DemonstrationReader(
        root,
        parquet_row_reader=lambda path, columns: iter(row_data[path]),
        video_frame_reader=frame_reader,
        strict_pinned_metadata=False,
    )
    first_episode = next(reader.iter_episodes())

    with pytest.raises(DemonstrationValidationError, match="not aligned"):
        list(reader.iter_episode_steps(first_episode))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("frame_index", 4, "frame_index is 4, expected 0"),
        ("episode_index", 1, "episode_index is 1, expected 0"),
        ("task_index", 1, "task_index is 1, expected 0"),
        ("index", 10, "index is 10, expected 0"),
        ("timestamp", 0.05, "timestamp is 0.05"),
        ("observation.state", [0.0] * 7, "observation.state shape is [7]"),
        ("action", [0.0] * 8, "action shape is [8]"),
    ],
)
def test_row_alignment_and_shapes_are_enforced(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
    field: str,
    bad_value: Any,
    message: str,
) -> None:
    root, row_data = repository
    first_path = root / "data/chunk-000/episode_000000.parquet"
    row_data[first_path][0][field] = bad_value
    reader = DemonstrationReader(
        root,
        parquet_row_reader=lambda path, columns: iter(row_data[path]),
        strict_pinned_metadata=False,
    )

    with pytest.raises(DemonstrationValidationError, match=r"^episode 0 frame 0") as caught:
        reader.validate()

    assert message in str(caught.value)


def test_default_timestamp_tolerance_rejects_drift_above_observed_float_error(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
) -> None:
    root, row_data = repository
    first_path = root / "data/chunk-000/episode_000000.parquet"
    row_data[first_path][1]["timestamp"] = 0.05002
    reader = DemonstrationReader(
        root,
        parquet_row_reader=lambda path, columns: iter(row_data[path]),
        strict_pinned_metadata=False,
    )

    with pytest.raises(DemonstrationValidationError, match="expected 0.050000 at 20 Hz"):
        reader.validate()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda episodes: episodes[1].update(episode_index=3), "expected episode_index 1"),
        (lambda episodes: episodes[0].update(tasks=["unknown task"]), "names an unknown task"),
        (lambda episodes: episodes[0].update(length=3), "episode lengths sum"),
    ],
)
def test_episode_metadata_order_counts_and_task_identity_are_enforced(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
    mutation,
    message: str,
) -> None:
    root, row_data = repository
    episodes_path = root / "meta/episodes.jsonl"
    episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    mutation(episodes)
    episodes_path.write_text(
        "".join(json.dumps(item) + "\n" for item in episodes),
        encoding="utf-8",
    )
    reader = DemonstrationReader(
        root,
        parquet_row_reader=lambda path, columns: iter(row_data[path]),
        strict_pinned_metadata=False,
    )

    with pytest.raises(DemonstrationValidationError, match=message):
        list(reader.iter_episodes())


def test_missing_video_is_rejected_before_parquet_is_opened(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
) -> None:
    root, _ = repository
    missing = root / "videos/chunk-000/observation.images.front/episode_000000.mp4"
    missing.unlink()
    row_reader_called = False

    def row_reader(path: Path, columns: tuple[str, ...]):
        nonlocal row_reader_called
        row_reader_called = True
        return iter(())

    reader = DemonstrationReader(
        root,
        parquet_row_reader=row_reader,
        strict_pinned_metadata=False,
    )

    with pytest.raises(DemonstrationValidationError, match="source file is missing"):
        reader.validate()

    assert row_reader_called is False


def test_strict_mode_rejects_metadata_that_is_not_the_pinned_snapshot(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
) -> None:
    root, _ = repository

    with pytest.raises(DemonstrationValidationError, match="expected pinned value 14347"):
        DemonstrationReader(root)


def test_decoded_video_frame_count_must_match_episode_length(
    repository: tuple[Path, dict[Path, list[dict[str, Any]]]],
) -> None:
    root, row_data = repository
    reader = DemonstrationReader(
        root,
        parquet_row_reader=lambda path, columns: iter(row_data[path]),
        video_probe=lambda path: VideoStreamInfo(1, 256, 256, 20.0, "av1"),
        strict_pinned_metadata=False,
    )

    with pytest.raises(DemonstrationValidationError, match="has 1 frames, expected 2"):
        reader.validate(decode_videos=True)

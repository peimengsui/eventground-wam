"""Validate the pinned official LIBERO-Plus LeRobot demonstration snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from demonstration_dataset import (
    DATASET_REPO_ID,
    DATASET_REVISION,
    EXPECTED_DATASET_INFO,
    EXPECTED_FILE_COUNT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_METADATA_SHA256,
    EXPECTED_TOTAL_BYTES,
    active_snapshot_path,
    default_data_root,
    load_source_manifest,
    read_jsonl,
    sha256_file,
    snapshot_dir,
    source_manifest_path,
    source_record_path,
)

REQUIRED_PARQUET_COLUMNS = {
    "action",
    "episode_index",
    "frame_index",
    "index",
    "observation.state",
    "task_index",
    "timestamp",
}


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=expanded_path, default=default_data_root())
    parser.add_argument(
        "--sample-episodes",
        type=int,
        default=3,
        help="number of evenly spaced episodes to inspect (default: 3)",
    )
    parser.add_argument(
        "--full-sha256",
        action="store_true",
        help="hash every LFS payload (slow; size and sample hashes are always checked)",
    )
    return parser.parse_args()


def validate_source_record(data_root: Path) -> None:
    path = source_record_path(data_root)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "expected_file_count": EXPECTED_FILE_COUNT,
        "expected_total_bytes": EXPECTED_TOTAL_BYTES,
        "source_manifest_sha256": EXPECTED_MANIFEST_SHA256,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"source record {key} is {record.get(key)!r}, expected {value!r}")


def validate_local_files(data_root: Path, full_sha256: bool) -> dict[str, int]:
    repository_root = snapshot_dir(data_root)
    source_files = load_source_manifest(source_manifest_path(data_root))
    checked_bytes = 0
    checksummed_bytes = 0
    checksummed_files = 0
    missing: list[str] = []
    wrong_size: list[str] = []
    wrong_hash: list[str] = []

    sample_paths = set(EXPECTED_METADATA_SHA256)
    sample_paths.update(
        {
            "data/chunk-000/episode_000000.parquet",
            "videos/chunk-000/observation.images.front/episode_000000.mp4",
            "videos/chunk-000/observation.images.wrist/episode_000000.mp4",
        }
    )
    for source_file in source_files:
        local_path = repository_root / source_file.path
        if not local_path.is_file():
            missing.append(source_file.path)
            continue
        actual_size = local_path.stat().st_size
        if actual_size != source_file.size:
            wrong_size.append(f"{source_file.path}: {actual_size} != {source_file.size}")
            continue
        checked_bytes += actual_size
        expected_hash = EXPECTED_METADATA_SHA256.get(source_file.path) or source_file.sha256
        should_hash = source_file.path in sample_paths or (full_sha256 and bool(source_file.sha256))
        if should_hash and expected_hash:
            actual_hash = sha256_file(local_path)
            checksummed_bytes += actual_size
            checksummed_files += 1
            if actual_hash != expected_hash:
                wrong_hash.append(f"{source_file.path}: {actual_hash} != {expected_hash}")

    problems: list[str] = []
    if missing:
        problems.append(f"missing {len(missing)} file(s), first: {missing[:3]}")
    if wrong_size:
        problems.append(f"wrong size for {len(wrong_size)} file(s), first: {wrong_size[:3]}")
    if wrong_hash:
        problems.append(f"wrong SHA-256 for {len(wrong_hash)} file(s), first: {wrong_hash[:3]}")
    if checked_bytes != EXPECTED_TOTAL_BYTES:
        problems.append(f"checked {checked_bytes} bytes, expected {EXPECTED_TOTAL_BYTES}")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "files": len(source_files),
        "bytes": checked_bytes,
        "checksummed_files": checksummed_files,
        "checksummed_bytes": checksummed_bytes,
    }


def validate_metadata(repository_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = json.loads((repository_root / "meta/info.json").read_text(encoding="utf-8"))
    for key, expected_value in EXPECTED_DATASET_INFO.items():
        if info.get(key) != expected_value:
            raise ValueError(
                f"meta/info.json {key} is {info.get(key)!r}, expected {expected_value!r}"
            )

    expected_features = {
        "observation.images.front": ("video", [256, 256, 3]),
        "observation.images.wrist": ("video", [256, 256, 3]),
        "observation.state": ("float32", [8]),
        "action": ("float32", [7]),
    }
    features = info.get("features", {})
    for name, (expected_dtype, expected_shape) in expected_features.items():
        feature = features.get(name, {})
        if feature.get("dtype") != expected_dtype or feature.get("shape") != expected_shape:
            raise ValueError(f"unexpected feature declaration for {name}: {feature}")
    for camera_name in ("observation.images.front", "observation.images.wrist"):
        video_info = features[camera_name].get("info", {})
        if video_info.get("video.fps") != 20 or video_info.get("video.codec") != "av1":
            raise ValueError(f"unexpected video declaration for {camera_name}: {video_info}")

    tasks = list(read_jsonl(repository_root / "meta/tasks.jsonl"))
    task_indices = [task.get("task_index") for task in tasks]
    if task_indices != list(range(EXPECTED_DATASET_INFO["total_tasks"])):
        raise ValueError("task indices are not the expected contiguous range")
    if any(not isinstance(task.get("task"), str) or not task["task"] for task in tasks):
        raise ValueError("every task must have a non-empty instruction")

    episodes = list(read_jsonl(repository_root / "meta/episodes.jsonl"))
    if len(episodes) != EXPECTED_DATASET_INFO["total_episodes"]:
        raise ValueError(f"found {len(episodes)} episode records")
    if [episode.get("episode_index") for episode in episodes] != list(range(len(episodes))):
        raise ValueError("episode indices are not the expected contiguous range")
    total_frames = sum(int(episode.get("length", -1)) for episode in episodes)
    if total_frames != EXPECTED_DATASET_INFO["total_frames"]:
        raise ValueError(f"episode lengths sum to {total_frames}")
    return info, episodes


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("--sample-episodes must be positive")
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    return sorted({round(position * (total - 1) / (count - 1)) for position in range(count)})


def validate_episode_samples(
    repository_root: Path,
    info: dict[str, Any],
    episodes: list[dict[str, Any]],
    sample_count: int,
) -> list[int]:
    import av
    from pyarrow import parquet

    sampled_indices = evenly_spaced_indices(len(episodes), sample_count)
    for episode_index in sampled_indices:
        episode = episodes[episode_index]
        chunk_index = episode_index // int(info["chunks_size"])
        parquet_path = repository_root / info["data_path"].format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )
        parquet_file = parquet.ParquetFile(parquet_path)
        if parquet_file.metadata.num_rows != episode["length"]:
            raise ValueError(
                f"episode {episode_index} has {parquet_file.metadata.num_rows} Parquet rows, "
                f"expected {episode['length']}"
            )
        column_names = set(parquet_file.schema_arrow.names)
        if not REQUIRED_PARQUET_COLUMNS.issubset(column_names):
            raise ValueError(
                f"episode {episode_index} is missing Parquet columns: "
                f"{sorted(REQUIRED_PARQUET_COLUMNS - column_names)}"
            )

        for camera_name in ("observation.images.front", "observation.images.wrist"):
            video_path = repository_root / info["video_path"].format(
                episode_chunk=chunk_index,
                episode_index=episode_index,
                video_key=camera_name,
            )
            with av.open(str(video_path)) as container:
                if not container.streams.video:
                    raise ValueError(f"episode {episode_index} {camera_name} has no video stream")
                stream = container.streams.video[0]
                if stream.codec_context.name not in {"av1", "libdav1d"}:
                    raise ValueError(
                        f"episode {episode_index} {camera_name} codec is {stream.codec_context.name}"
                    )
                frame = next(container.decode(video=0), None)
                if frame is None or (frame.width, frame.height) != (256, 256):
                    raise ValueError(
                        f"episode {episode_index} {camera_name} first frame is invalid"
                    )
    return sampled_indices


def main() -> None:
    args = parse_args()
    repository_root = snapshot_dir(args.data_root)
    active_path = active_snapshot_path(args.data_root)
    if not repository_root.is_dir():
        raise SystemExit(f"snapshot is missing: {repository_root}")
    if not active_path.is_symlink() or active_path.resolve() != repository_root.resolve():
        raise SystemExit(f"active snapshot link is missing or incorrect: {active_path}")

    validate_source_record(args.data_root)
    file_summary = validate_local_files(args.data_root, args.full_sha256)
    info, episodes = validate_metadata(repository_root)
    sampled_indices = validate_episode_samples(
        repository_root,
        info,
        episodes,
        args.sample_episodes,
    )
    summary = {
        "status": "ok",
        "dataset_repo_id": DATASET_REPO_ID,
        "dataset_revision": DATASET_REVISION,
        "snapshot": str(repository_root),
        "files": file_summary["files"],
        "bytes": file_summary["bytes"],
        "checksummed_files": file_summary["checksummed_files"],
        "checksummed_bytes": file_summary["checksummed_bytes"],
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "tasks": info["total_tasks"],
        "sampled_episodes": sampled_indices,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

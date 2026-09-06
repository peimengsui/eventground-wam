#!/usr/bin/env python3
"""Stream and validate the pinned LIBERO-Plus demonstration dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from eventground_wam.demonstration_contracts import SOURCE_DATASET_REPO_ID
from eventground_wam.demonstration_reader import (
    PINNED_DATASET_REVISION,
    DemonstrationReader,
)


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def default_snapshot_path() -> Path:
    data_root = expanded_path(
        os.environ.get(
            "EVENTGROUND_LIBERO_DATA_ROOT",
            "~/data/eventground-wam/libero-plus",
        )
    )
    return data_root / "datasets" / "libero-plus-lerobot" / "current"


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Select deterministic validation episodes, including both endpoints."""

    if total < 1:
        raise ValueError("total must be positive")
    if count < 1:
        raise ValueError("--sample-episodes must be positive")
    if count >= total:
        return list(range(total))
    if count == 1:
        return [0]
    return sorted({round(position * (total - 1) / (count - 1)) for position in range(count)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=expanded_path, default=default_snapshot_path())
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--sample-episodes",
        type=int,
        default=3,
        metavar="N",
        help="fully validate N evenly spaced episodes (default: 3)",
    )
    scope.add_argument(
        "--all-episodes",
        action="store_true",
        help="stream and validate every Parquet row",
    )
    parser.add_argument(
        "--decode-videos",
        action="store_true",
        help="fully decode both videos for every selected episode",
    )
    return parser.parse_args()


def validate_selection(
    reader: DemonstrationReader,
    *,
    sample_episodes: int,
    all_episodes: bool,
    decode_videos: bool,
) -> dict[str, Any]:
    if all_episodes:
        selected_indices: set[int] | None = None
        selected_output: list[int] | str = "all"
    else:
        selected = evenly_spaced_indices(reader.info.total_episodes, sample_episodes)
        selected_indices = set(selected)
        selected_output = selected

    validated_episodes = 0
    validated_frames = 0
    decoded_videos = 0
    decoded_video_frames = 0
    for episode in reader.iter_episodes():
        if selected_indices is not None and episode.episode_index not in selected_indices:
            continue
        episode_summary = reader.validate_episode(episode, decode_videos=decode_videos)
        validated_episodes += 1
        validated_frames += episode_summary.frame_count
        decoded_videos += episode_summary.decoded_video_count
        decoded_video_frames += episode_summary.frame_count * episode_summary.decoded_video_count

    expected_selected = reader.info.total_episodes if selected_indices is None else len(selected_indices)
    if validated_episodes != expected_selected:
        raise RuntimeError(
            f"validated {validated_episodes} selected episodes, expected {expected_selected}"
        )

    return {
        "status": "ok",
        "dataset_repo_id": SOURCE_DATASET_REPO_ID,
        "dataset_revision": PINNED_DATASET_REVISION,
        "snapshot": str(reader.repository_root),
        "metadata_episodes": reader.info.total_episodes,
        "metadata_frames": reader.info.total_frames,
        "metadata_tasks": reader.info.total_tasks,
        "source_video_files": reader.info.total_videos,
        "selected_episode_indices": selected_output,
        "validated_episodes": validated_episodes,
        "validated_parquet_frames": validated_frames,
        "decoded_videos": decoded_videos,
        "decoded_video_frames": decoded_video_frames,
        "timestamp_tolerance_seconds": reader.timestamp_tolerance_seconds,
    }


def main() -> None:
    args = parse_args()
    reader = DemonstrationReader(args.snapshot)
    summary = validate_selection(
        reader,
        sample_episodes=args.sample_episodes,
        all_episodes=args.all_episodes,
        decode_videos=args.decode_videos,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

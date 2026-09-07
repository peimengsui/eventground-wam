#!/usr/bin/env python3
"""Generate deterministic final event intervals from raw pilot candidates."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eventground_wam.artifact_provenance import project_state
from eventground_wam.demonstration_contracts import (
    EVENT_CANDIDATE_ARTIFACT_TYPE,
    TRAJECTORY_ARTIFACT_TYPE,
    EventCandidate,
    Provenance,
    TrajectoryRecord,
    validate_contract,
    validate_contract_sequence,
)
from eventground_wam.demonstration_reader import DemonstrationReader
from eventground_wam.event_extraction import (
    EventExtractionConfig,
    extract_event_intervals,
    publish_event_extraction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_VERSION = "week2-events-v1"


def expanded_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def default_data_root() -> Path:
    return expanded_path(
        os.environ.get(
            "EVENTGROUND_LIBERO_DATA_ROOT",
            "~/data/eventground-wam/libero-plus",
        )
    )


def parse_args() -> argparse.Namespace:
    data_root = default_data_root()
    pilot_dir = data_root / "derived" / "event-labeling" / "week2-pilot-v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=expanded_path,
        default=data_root / "datasets" / "libero-plus-lerobot" / "current",
    )
    parser.add_argument(
        "--trajectory-contracts",
        type=expanded_path,
        default=pilot_dir / "trajectory-contracts.jsonl",
    )
    parser.add_argument(
        "--event-candidates",
        type=expanded_path,
        default=pilot_dir / "event-candidates.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=expanded_path,
        default=data_root / "derived" / "event-extraction" / DEFAULT_ANNOTATION_VERSION,
    )
    parser.add_argument("--annotation-version", default=DEFAULT_ANNOTATION_VERSION)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow non-publishable output from an uncommitted implementation",
    )
    return parser.parse_args()


def load_contract_records(path: Path, artifact_type: str) -> list[dict[str, Any]]:
    """Load and strictly validate one JSONL contract stream."""

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL lines are forbidden")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: contract must be a JSON object")
            validate_contract(record)
            if record["artifact_type"] != artifact_type:
                raise ValueError(
                    f"{path}:{line_number}: expected {artifact_type}, "
                    f"found {record['artifact_type']}"
                )
            records.append(record)
    validate_contract_sequence(records)
    return records


def trajectory_from_record(record: Mapping[str, Any]) -> TrajectoryRecord:
    provenance = Provenance(**record["provenance"])
    return TrajectoryRecord(
        provenance=provenance,
        instruction=record["instruction"],
        frame_start=record["frame_start"],
        frame_end=record["frame_end"],
        parquet_path=record["parquet_path"],
        front_video_path=record["front_video_path"],
        wrist_video_path=record["wrist_video_path"],
        fps=record["fps"],
        state_dimension=record["state_dimension"],
        action_dimension=record["action_dimension"],
        image_shape=tuple(record["image_shape"]),
        camera_keys=tuple(record["camera_keys"]),
    )


def candidate_from_record(record: Mapping[str, Any]) -> EventCandidate:
    return EventCandidate(
        provenance=Provenance(**record["provenance"]),
        trajectory_digest=record["trajectory_digest"],
        frame_start=record["frame_start"],
        frame_end=record["frame_end"],
        event_type=record["event_type"],
        confidence=record["confidence"],
        evidence=tuple(record["evidence"]),
        detector_name=record["detector_name"],
        detector_version=record["detector_version"],
    )


def run_extraction(
    *,
    snapshot: Path,
    trajectory_contracts_path: Path,
    event_candidates_path: Path,
    output_dir: Path,
    project_git_revision: str,
    project_tree_clean: bool,
    annotation_version: str,
) -> dict[str, Any]:
    """Load pilot inputs, stream telemetry, and publish final event artifacts."""

    trajectory_records = load_contract_records(
        trajectory_contracts_path, TRAJECTORY_ARTIFACT_TYPE
    )
    candidate_records = load_contract_records(
        event_candidates_path, EVENT_CANDIDATE_ARTIFACT_TYPE
    )
    trajectories = tuple(trajectory_from_record(record) for record in trajectory_records)
    if not trajectories:
        raise ValueError("event extraction requires at least one trajectory contract")
    episode_indices = [item.provenance.source_episode_index for item in trajectories]
    if episode_indices != sorted(set(episode_indices)):
        raise ValueError("trajectory contracts must have unique, increasing episode indices")

    candidates_by_episode: dict[int, list[EventCandidate]] = defaultdict(list)
    for record in candidate_records:
        candidate = candidate_from_record(record)
        candidates_by_episode[candidate.provenance.source_episode_index].append(candidate)
    unknown_candidate_episodes = set(candidates_by_episode) - set(episode_indices)
    if unknown_candidate_episodes:
        raise ValueError(
            "candidate records reference episodes without trajectories: "
            f"{sorted(unknown_candidate_episodes)}"
        )

    reader = DemonstrationReader(snapshot)
    episodes_by_index = {episode.episode_index: episode for episode in reader.iter_episodes()}
    config = EventExtractionConfig()
    results = []
    for trajectory in trajectories:
        episode_index = trajectory.provenance.source_episode_index
        try:
            episode = episodes_by_index[episode_index]
        except KeyError as error:
            raise ValueError(f"source episode {episode_index} is missing") from error
        rows = tuple(reader.iter_episode_rows(episode))
        interval_provenance = Provenance(
            source_episode_index=trajectory.provenance.source_episode_index,
            source_task_index=trajectory.provenance.source_task_index,
            source_episode_length=trajectory.provenance.source_episode_length,
            project_git_revision=project_git_revision,
            annotation_version=annotation_version,
        )
        results.append(
            extract_event_intervals(
                rows,
                trajectory,
                candidates_by_episode[episode_index],
                config=config,
                interval_provenance=interval_provenance,
            )
        )

    return publish_event_extraction(
        output_dir,
        results,
        project_git_revision=project_git_revision,
        project_tree_clean=project_tree_clean,
        annotation_version=annotation_version,
        config=config,
    )


def main() -> None:
    args = parse_args()
    revision, clean = project_state(PROJECT_ROOT)
    if not clean and not args.allow_dirty:
        raise SystemExit(
            "project tree is dirty; commit the extractor or pass --allow-dirty for "
            "explicitly non-publishable output"
        )
    manifest = run_extraction(
        snapshot=args.snapshot,
        trajectory_contracts_path=args.trajectory_contracts,
        event_candidates_path=args.event_candidates,
        output_dir=args.output_dir,
        project_git_revision=revision,
        project_tree_clean=clean,
        annotation_version=args.annotation_version,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate leakage-resistant split and multi-horizon sample indexes."""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eventground_wam.artifact_provenance import project_state
from eventground_wam.demonstration_contracts import (
    EVENT_INTERVAL_ARTIFACT_TYPE,
    TRAJECTORY_ARTIFACT_TYPE,
    EventInterval,
    Provenance,
    TrajectoryRecord,
    validate_contract,
)
from eventground_wam.demonstration_reader import DemonstrationReader, EpisodeRecord
from eventground_wam.split_sample_index import (
    DEFAULT_SPLIT_NAMESPACE,
    EpisodeSampleSource,
    SplitRatios,
    initial_condition_from_step,
    write_split_sample_indexes,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATION_VERSION = "week2-splits-v1"


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=expanded_path,
        default=data_root / "datasets" / "libero-plus-lerobot" / "current",
    )
    parser.add_argument(
        "--trajectory-index",
        type=expanded_path,
        required=True,
        help="trajectory JSONL; a subset such as the 40-episode pilot is supported",
    )
    parser.add_argument(
        "--event-index",
        type=expanded_path,
        help="optional event-interval JSONL; it may be a superset of the trajectories",
    )
    parser.add_argument(
        "--output-dir",
        type=expanded_path,
        default=data_root / "derived" / "splits" / DEFAULT_ANNOTATION_VERSION,
    )
    parser.add_argument("--annotation-version", default=DEFAULT_ANNOTATION_VERSION)
    parser.add_argument("--namespace", default=DEFAULT_SPLIT_NAMESPACE)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--fingerprint-workers",
        type=int,
        default=1,
        help="bounded parallel readers for initial telemetry and camera frames (default: 1)",
    )
    parser.add_argument(
        "--allow-partial-source",
        action="store_true",
        help="generate a non-full index, for example from the 40-episode pilot",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="allow non-publishable output from an uncommitted implementation",
    )
    return parser.parse_args()


def iter_episode_sources(
    *,
    reader: DemonstrationReader,
    trajectory_index: Path,
    event_index: Path | None,
    fingerprint_workers: int = 1,
) -> Iterator[EpisodeSampleSource]:
    """Merge ordered contracts with source metadata and decode only initial frames."""

    if type(fingerprint_workers) is not int or fingerprint_workers < 1:
        raise ValueError("fingerprint_workers must be a positive integer")
    prepared = _iter_prepared_sources(
        reader=reader,
        trajectory_index=trajectory_index,
        event_index=event_index,
    )
    if fingerprint_workers == 1:
        for source in prepared:
            yield _fingerprint_source(reader, source)
        return
    yield from _bounded_parallel_fingerprints(
        reader,
        prepared,
        worker_count=fingerprint_workers,
    )


@dataclass(frozen=True)
class _PreparedSource:
    trajectory: TrajectoryRecord
    episode: EpisodeRecord
    events: tuple[EventInterval, ...]


def _iter_prepared_sources(
    *,
    reader: DemonstrationReader,
    trajectory_index: Path,
    event_index: Path | None,
) -> Iterator[_PreparedSource]:
    episode_iterator = iter(reader.iter_episodes())
    episode = next(episode_iterator, None)
    event_cursor = _EventCursor(event_index)
    for raw in _read_jsonl(trajectory_index):
        trajectory = _trajectory_from_record(raw)
        target_episode_index = trajectory.provenance.source_episode_index
        while episode is not None and episode.episode_index < target_episode_index:
            episode = next(episode_iterator, None)
        if episode is None or episode.episode_index != target_episode_index:
            raise ValueError(
                f"trajectory episode {target_episode_index} is absent from source metadata"
            )
        _validate_episode_descriptor(trajectory, episode, reader.repository_root)
        yield _PreparedSource(
            trajectory=trajectory,
            episode=episode,
            events=event_cursor.for_episode(target_episode_index),
        )


def _fingerprint_source(
    reader: DemonstrationReader,
    source: _PreparedSource,
) -> EpisodeSampleSource:
    step_iterator = reader.iter_episode_steps(source.episode)
    try:
        first_step = next(step_iterator)
    except StopIteration as error:
        raise ValueError(
            f"episode {source.episode.episode_index} has no initial step"
        ) from error
    finally:
        close = getattr(step_iterator, "close", None)
        if close is not None:
            close()
    return EpisodeSampleSource(
        trajectory=source.trajectory,
        initial_condition=initial_condition_from_step(source.trajectory, first_step),
        events=source.events,
    )


def _bounded_parallel_fingerprints(
    reader: DemonstrationReader,
    prepared: Iterable[_PreparedSource],
    *,
    worker_count: int,
) -> Iterator[EpisodeSampleSource]:
    """Read a bounded number of episodes concurrently and preserve input order."""

    source_iterator = iter(prepared)
    pending: deque[Future[EpisodeSampleSource]] = deque()
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="initial-fingerprint",
    ) as executor:
        for _ in range(worker_count):
            source = next(source_iterator, None)
            if source is None:
                break
            pending.append(executor.submit(_fingerprint_source, reader, source))
        while pending:
            yield pending.popleft().result()
            source = next(source_iterator, None)
            if source is not None:
                pending.append(executor.submit(_fingerprint_source, reader, source))


class _EventCursor:
    """Streaming event-index cursor that permits an event superset."""

    def __init__(self, path: Path | None) -> None:
        self._records = iter(_read_jsonl(path)) if path is not None else iter(())
        self._previous_key: tuple[int, int, int, str, str] | None = None
        self._next = self._advance()

    def for_episode(self, episode_index: int) -> tuple[EventInterval, ...]:
        while (
            self._next is not None
            and self._next.provenance.source_episode_index < episode_index
        ):
            self._next = self._advance()
        events: list[EventInterval] = []
        while (
            self._next is not None
            and self._next.provenance.source_episode_index == episode_index
        ):
            events.append(self._next)
            self._next = self._advance()
        return tuple(events)

    def _advance(self) -> EventInterval | None:
        raw = next(self._records, None)
        if raw is None:
            return None
        event = _event_from_record(raw)
        key = (
            event.provenance.source_episode_index,
            event.frame_start,
            event.frame_end,
            event.event_type,
            event.digest,
        )
        if self._previous_key is not None and key <= self._previous_key:
            raise ValueError("event index is not in canonical episode/boundary order")
        self._previous_key = key
        return event


def _read_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"{path}:{line_number}: blank JSONL record")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number}: record must be an object")
                yield value
    except OSError as error:
        raise ValueError(f"cannot read JSONL input {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}:{error.lineno}: invalid JSON") from error


def _provenance_from_record(raw: Mapping[str, Any]) -> Provenance:
    provenance = raw["provenance"]
    if not isinstance(provenance, Mapping):
        raise TypeError("contract provenance must be an object")
    return Provenance(**dict(provenance))


def _trajectory_from_record(raw: Mapping[str, Any]) -> TrajectoryRecord:
    validate_contract(raw)
    if raw["artifact_type"] != TRAJECTORY_ARTIFACT_TYPE:
        raise ValueError("trajectory index contains a non-trajectory contract")
    return TrajectoryRecord(
        provenance=_provenance_from_record(raw),
        instruction=raw["instruction"],
        frame_start=raw["frame_start"],
        frame_end=raw["frame_end"],
        parquet_path=raw["parquet_path"],
        front_video_path=raw["front_video_path"],
        wrist_video_path=raw["wrist_video_path"],
        fps=raw["fps"],
        state_dimension=raw["state_dimension"],
        action_dimension=raw["action_dimension"],
        image_shape=tuple(raw["image_shape"]),
        camera_keys=tuple(raw["camera_keys"]),
    )


def _event_from_record(raw: Mapping[str, Any]) -> EventInterval:
    validate_contract(raw)
    if raw["artifact_type"] != EVENT_INTERVAL_ARTIFACT_TYPE:
        raise ValueError("event index contains a non-event-interval contract")
    return EventInterval(
        provenance=_provenance_from_record(raw),
        trajectory_digest=raw["trajectory_digest"],
        frame_start=raw["frame_start"],
        frame_end=raw["frame_end"],
        event_type=raw["event_type"],
        confidence=raw["confidence"],
        evidence=tuple(raw["evidence"]),
        candidate_digests=tuple(raw["candidate_digests"]),
        ontology_tier=raw["ontology_tier"],
        review_status=raw["review_status"],
    )


def _validate_episode_descriptor(
    trajectory: TrajectoryRecord,
    episode: EpisodeRecord,
    repository_root: Path,
) -> None:
    provenance = trajectory.provenance
    if (
        episode.task_index != provenance.source_task_index
        or episode.length != provenance.source_episode_length
    ):
        raise ValueError(
            f"trajectory identity does not match source episode {episode.episode_index}"
        )
    expected = TrajectoryRecord.from_episode(trajectory.provenance, episode.instruction)
    if expected.digest != trajectory.digest:
        raise ValueError(
            f"trajectory contract does not match source episode {episode.episode_index}"
        )
    source_paths = (
        episode.parquet_path,
        *(path for _, path in episode.video_paths),
    )
    contract_paths = (
        trajectory.parquet_path,
        trajectory.front_video_path,
        trajectory.wrist_video_path,
    )
    for source_path, contract_path in zip(source_paths, contract_paths, strict=True):
        candidate = source_path.expanduser()
        if not candidate.is_absolute():
            candidate = repository_root / candidate
        try:
            relative = candidate.relative_to(repository_root).as_posix()
        except ValueError as error:
            raise ValueError(
                f"source path escapes the snapshot for episode {episode.episode_index}"
            ) from error
        if relative != contract_path:
            raise ValueError(
                f"trajectory path does not match source episode {episode.episode_index}"
            )


def main() -> None:
    args = parse_args()
    revision, clean = project_state(PROJECT_ROOT)
    if not clean and not args.allow_dirty:
        raise SystemExit(
            "project tree is dirty; commit the generator or pass --allow-dirty for "
            "explicitly non-publishable output"
        )
    ratios = SplitRatios(
        train=args.train_ratio,
        validation=args.validation_ratio,
        test=args.test_ratio,
    )
    reader = DemonstrationReader(args.snapshot)
    manifest = write_split_sample_indexes(
        iter_episode_sources(
            reader=reader,
            trajectory_index=args.trajectory_index,
            event_index=args.event_index,
            fingerprint_workers=args.fingerprint_workers,
        ),
        args.output_dir,
        project_git_revision=revision,
        project_tree_clean=clean,
        annotation_version=args.annotation_version,
        ratios=ratios,
        namespace=args.namespace,
        require_complete_source=not args.allow_partial_source,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

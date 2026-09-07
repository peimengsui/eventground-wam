from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_split_sample_indexes import iter_episode_sources

from eventground_wam.demonstration_contracts import (
    EventInterval,
    Provenance,
    TrajectoryRecord,
)
from eventground_wam.demonstration_reader import (
    CAMERA_KEYS,
    DecodedVideoFrame,
    EpisodeRecord,
    SynchronizedStep,
    TrajectoryRow,
)

PROJECT_REVISION = "a" * 40


def _trajectory(episode_index: int) -> TrajectoryRecord:
    return TrajectoryRecord.from_episode(
        Provenance(
            source_episode_index=episode_index,
            source_task_index=3,
            source_episode_length=7,
            project_git_revision=PROJECT_REVISION,
            annotation_version="week2-events-v1",
        ),
        "instruction 3",
    )


def _event(trajectory: TrajectoryRecord) -> EventInterval:
    return EventInterval(
        provenance=trajectory.provenance,
        trajectory_digest=trajectory.digest,
        frame_start=3,
        frame_end=4,
        event_type="gripper_close",
        confidence=0.95,
        evidence=("action:transition",),
        candidate_digests=(
            hashlib.sha256(
                str(trajectory.provenance.source_episode_index).encode()
            ).hexdigest(),
        ),
        ontology_tier="reliable",
        review_status="unreviewed",
    )


class _FakeFormat:
    name = "rgb24"


class _FakePlane:
    line_size = 9

    def __init__(self, fill: int) -> None:
        self._payload = bytes([fill]) * 18

    def __bytes__(self) -> bytes:
        return self._payload


class _FakePayload:
    width = 3
    height = 2
    format = _FakeFormat()

    def __init__(self, fill: int) -> None:
        self.planes = (_FakePlane(fill),)

    def reformat(self, *, format: str):
        assert format == "rgb24"
        return self


class _FakeReader:
    def __init__(self, root: Path) -> None:
        self.repository_root = root

    def iter_episodes(self):
        for episode_index in range(3):
            trajectory = _trajectory(episode_index)
            yield EpisodeRecord(
                episode_index=episode_index,
                task_index=3,
                instruction="instruction 3",
                length=7,
                start_index=episode_index * 7,
                parquet_path=self.repository_root / trajectory.parquet_path,
                video_paths=(
                    (CAMERA_KEYS[0], self.repository_root / trajectory.front_video_path),
                    (CAMERA_KEYS[1], self.repository_root / trajectory.wrist_video_path),
                ),
            )

    def iter_episode_steps(self, episode: EpisodeRecord):
        row = TrajectoryRow(
            index=episode.start_index,
            episode_index=episode.episode_index,
            frame_index=0,
            task_index=episode.task_index,
            timestamp=0.0,
            state=tuple(float(index) for index in range(8)),
            action=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        )
        frames = tuple(
            DecodedVideoFrame(
                frame_index=0,
                timestamp=0.0,
                width=3,
                height=2,
                pixel_format="yuv420p",
                payload=_FakePayload(fill),
            )
            for fill in (1, 2)
        )
        return iter((SynchronizedStep(row, frames[0], frames[1]),))


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_cli_source_merge_supports_pilot_trajectory_subset_and_event_superset(
    tmp_path: Path,
) -> None:
    # The trajectory input selects only episode 2 while metadata and events are
    # supersets.  This is the mode used for a 40-episode pilot validation.
    trajectory = _trajectory(2)
    trajectory_index = tmp_path / "trajectories.jsonl"
    event_index = tmp_path / "events.jsonl"
    _write_jsonl(trajectory_index, [trajectory.to_dict()])
    _write_jsonl(
        event_index,
        [_event(_trajectory(0)).to_dict(), _event(trajectory).to_dict()],
    )

    sources = tuple(
        iter_episode_sources(
            reader=_FakeReader(tmp_path),  # type: ignore[arg-type]
            trajectory_index=trajectory_index,
            event_index=event_index,
            fingerprint_workers=2,
        )
    )

    assert len(sources) == 1
    assert sources[0].trajectory.digest == trajectory.digest
    assert [event.digest for event in sources[0].events] == [_event(trajectory).digest]
    assert sources[0].initial_condition.source_episode_index == 2
    assert sources[0].initial_condition.front_frame_sha256 != (
        sources[0].initial_condition.wrist_frame_sha256
    )

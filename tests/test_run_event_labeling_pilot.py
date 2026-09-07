from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_event_labeling_pilot as pilot_cli

from eventground_wam.demonstration_reader import CAMERA_KEYS, EpisodeRecord


def _episode(index: int, task: int, length: int) -> EpisodeRecord:
    root = Path("/dataset")
    return EpisodeRecord(
        episode_index=index,
        task_index=task,
        instruction=f"task {task}",
        length=length,
        start_index=0,
        parquet_path=root / f"episode_{index:06d}.parquet",
        video_paths=tuple(
            (camera, root / camera / f"episode_{index:06d}.mp4")
            for camera in CAMERA_KEYS
        ),
    )


def test_full_source_scope_requires_pinned_totals_and_canonical_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_cli, "SOURCE_EPISODE_COUNT", 2)
    monkeypatch.setattr(pilot_cli, "SOURCE_FRAME_COUNT", 30)
    monkeypatch.setattr(pilot_cli, "SOURCE_TASK_COUNT", 2)
    episodes = (_episode(0, 0, 10), _episode(1, 1, 20))

    assert pilot_cli.select_episode_scope(
        episodes,
        episodes_per_task=1,
        all_episodes=True,
    ) == episodes

    with pytest.raises(ValueError, match="canonical contiguous order"):
        pilot_cli.select_episode_scope(
            tuple(reversed(episodes)),
            episodes_per_task=1,
            all_episodes=True,
        )


def test_full_source_scope_rejects_incomplete_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot_cli, "SOURCE_EPISODE_COUNT", 2)
    monkeypatch.setattr(pilot_cli, "SOURCE_FRAME_COUNT", 30)
    monkeypatch.setattr(pilot_cli, "SOURCE_TASK_COUNT", 2)

    with pytest.raises(ValueError, match="pinned dataset totals"):
        pilot_cli.select_episode_scope(
            (_episode(0, 0, 10),),
            episodes_per_task=1,
            all_episodes=True,
        )

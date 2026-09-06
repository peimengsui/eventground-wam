from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import (
    SOURCE_DATASET_REVISION,
    contract_digest,
    validate_contract_sequence,
)
from eventground_wam.demonstration_reader import CAMERA_KEYS, EpisodeRecord
from eventground_wam.trajectory_index import (
    TRAJECTORY_INDEX_ARTIFACT_TYPE,
    TrajectoryIndexError,
    trajectory_contract_for_episode,
    write_trajectory_index,
)

PROJECT_REVISION = "a" * 40


def _episode(
    root: Path,
    episode_index: int,
    *,
    task_index: int = 3,
    length: int = 17,
) -> EpisodeRecord:
    chunk_index = episode_index // 1_000
    return EpisodeRecord(
        episode_index=episode_index,
        task_index=task_index,
        instruction=f"instruction {task_index}",
        length=length,
        start_index=0,
        parquet_path=(
            root / f"data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
        ),
        video_paths=tuple(
            (
                camera_key,
                root
                / f"videos/chunk-{chunk_index:03d}"
                / camera_key
                / f"episode_{episode_index:06d}.mp4",
            )
            for camera_key in CAMERA_KEYS
        ),
    )


def test_write_trajectory_index_seals_complete_canonical_stream(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    episodes = [
        _episode(repository_root, 0, task_index=3, length=17),
        _episode(repository_root, 1_001, task_index=4, length=23),
    ]
    output_path = tmp_path / "derived" / "trajectory-contracts.jsonl"

    manifest = write_trajectory_index(
        episodes,
        repository_root,
        output_path,
        project_git_revision=PROJECT_REVISION,
        project_tree_clean=True,
        annotation_version="week2-v1",
        require_complete_source=False,
    )

    records_bytes = output_path.read_bytes()
    records = [json.loads(line) for line in records_bytes.splitlines()]
    validate_contract_sequence(records)
    assert [record["provenance"]["source_episode_index"] for record in records] == [
        0,
        1_001,
    ]
    assert records[1]["parquet_path"] == "data/chunk-001/episode_001001.parquet"
    assert manifest["artifact_type"] == TRAJECTORY_INDEX_ARTIFACT_TYPE
    assert manifest["source"]["dataset_revision"] == SOURCE_DATASET_REVISION
    assert manifest["records_sha256"] == hashlib.sha256(records_bytes).hexdigest()
    assert manifest["record_count"] == 2
    assert manifest["frame_count"] == 40
    assert manifest["task_count"] == 2
    assert manifest["complete_source"] is False
    assert manifest["publishable"] is True
    assert manifest["digest"] == contract_digest(manifest)
    assert json.loads(
        output_path.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
    ) == manifest


def test_trajectory_contract_rejects_path_outside_repository(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    episode = _episode(repository_root, 0)
    escaped = EpisodeRecord(
        **{
            **episode.__dict__,
            "parquet_path": tmp_path / "outside.parquet",
        }
    )

    with pytest.raises(TrajectoryIndexError, match="escapes the repository"):
        trajectory_contract_for_episode(
            escaped,
            repository_root,
            project_git_revision=PROJECT_REVISION,
            annotation_version="week2-v1",
        )


def test_trajectory_index_refuses_to_overwrite_outputs(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    output_path = tmp_path / "trajectory-contracts.jsonl"
    output_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        write_trajectory_index(
            [_episode(repository_root, 0)],
            repository_root,
            output_path,
            project_git_revision=PROJECT_REVISION,
            project_tree_clean=False,
            annotation_version="week2-v1",
            require_complete_source=False,
        )

    assert output_path.read_text(encoding="utf-8") == "existing\n"


def test_trajectory_index_cleans_partial_stream_after_failure(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    valid = _episode(repository_root, 0)
    invalid = _episode(repository_root, 1)
    invalid = EpisodeRecord(
        **{
            **invalid.__dict__,
            "parquet_path": repository_root / "wrong.parquet",
        }
    )
    output_path = tmp_path / "trajectory-contracts.jsonl"

    with pytest.raises(TrajectoryIndexError, match="source path is"):
        write_trajectory_index(
            [valid, invalid],
            repository_root,
            output_path,
            project_git_revision=PROJECT_REVISION,
            project_tree_clean=False,
            annotation_version="week2-v1",
            require_complete_source=False,
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".jsonl.partial").exists()
    assert not output_path.with_suffix(".jsonl.manifest.json").exists()


def test_trajectory_index_rejects_empty_input_without_artifacts(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    output_path = tmp_path / "trajectory-contracts.jsonl"

    with pytest.raises(TrajectoryIndexError, match="empty"):
        write_trajectory_index(
            [],
            repository_root,
            output_path,
            project_git_revision=PROJECT_REVISION,
            project_tree_clean=False,
            annotation_version="week2-v1",
            require_complete_source=False,
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".jsonl.partial").exists()


def test_complete_trajectory_index_rejects_truncated_source(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    output_path = tmp_path / "trajectory-contracts.jsonl"

    with pytest.raises(TrajectoryIndexError, match="totals do not match"):
        write_trajectory_index(
            [_episode(repository_root, 0)],
            repository_root,
            output_path,
            project_git_revision=PROJECT_REVISION,
            project_tree_clean=False,
            annotation_version="week2-v1",
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".jsonl.partial").exists()

"""Deterministic generation of the complete trajectory-contract index."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

from eventground_wam.demonstration_contracts import (
    CONTRACT_SCHEMA_VERSION,
    SOURCE_CAMERA_KEYS,
    SOURCE_DATASET_REPO_ID,
    SOURCE_DATASET_REVISION,
    SOURCE_EPISODE_COUNT,
    SOURCE_FRAME_COUNT,
    SOURCE_MANIFEST_SHA256,
    SOURCE_TASK_COUNT,
    TRAJECTORY_ARTIFACT_TYPE,
    Provenance,
    TrajectoryRecord,
    contract_digest,
)
from eventground_wam.demonstration_reader import EpisodeRecord

TRAJECTORY_INDEX_ARTIFACT_TYPE = "eventground_wam.trajectory_index"


class TrajectoryIndexError(ValueError):
    """Raised when a trajectory index cannot be generated safely."""


def trajectory_contract_for_episode(
    episode: EpisodeRecord,
    repository_root: Path,
    *,
    project_git_revision: str,
    annotation_version: str,
) -> TrajectoryRecord:
    """Map one reader descriptor to its exact pinned trajectory contract."""

    provenance = Provenance(
        source_episode_index=episode.episode_index,
        source_task_index=episode.task_index,
        source_episode_length=episode.length,
        project_git_revision=project_git_revision,
        annotation_version=annotation_version,
    )
    trajectory = TrajectoryRecord.from_episode(provenance, episode.instruction)
    expected_paths = (
        trajectory.parquet_path,
        trajectory.front_video_path,
        trajectory.wrist_video_path,
    )
    actual_paths = (
        episode.parquet_path,
        *(path for _, path in episode.video_paths),
    )
    actual_camera_keys = tuple(key for key, _ in episode.video_paths)
    if actual_camera_keys != SOURCE_CAMERA_KEYS:
        raise TrajectoryIndexError(
            f"episode {episode.episode_index} camera keys are {actual_camera_keys}, "
            f"expected {SOURCE_CAMERA_KEYS}"
        )
    root = repository_root.expanduser()
    if not root.is_absolute():
        root = root.resolve()
    for expected, actual in zip(expected_paths, actual_paths, strict=True):
        candidate = actual.expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            relative = candidate.relative_to(root).as_posix()
        except ValueError as error:
            raise TrajectoryIndexError(
                f"episode {episode.episode_index} source path escapes the repository: {actual}"
            ) from error
        if relative != expected:
            raise TrajectoryIndexError(
                f"episode {episode.episode_index} source path is {relative}, expected {expected}"
            )
    return trajectory


def write_trajectory_index(
    episodes: Iterable[EpisodeRecord],
    repository_root: Path,
    output_path: Path,
    *,
    project_git_revision: str,
    project_tree_clean: bool,
    annotation_version: str,
    require_complete_source: bool = True,
) -> dict[str, Any]:
    """Atomically write canonical JSONL records and a digest-sealed manifest."""

    if type(project_tree_clean) is not bool:
        raise TypeError("project_tree_clean must be a bool")
    if type(require_complete_source) is not bool:
        raise TypeError("require_complete_source must be a bool")

    output_path = output_path.expanduser().resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    manifest_partial_path = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    conflicts = [
        path
        for path in (output_path, manifest_path, partial_path, manifest_partial_path)
        if path.exists()
    ]
    if conflicts:
        raise FileExistsError(f"trajectory index refuses to overwrite outputs: {conflicts}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_digest = hashlib.sha256()
    record_count = 0
    frame_count = 0
    task_indices: set[int] = set()
    previous_episode_index: int | None = None
    try:
        with partial_path.open("x", encoding="utf-8") as handle:
            for episode in episodes:
                if previous_episode_index is not None and episode.episode_index <= previous_episode_index:
                    raise TrajectoryIndexError(
                        "episode descriptors must be strictly ordered by episode_index"
                    )
                if require_complete_source and episode.episode_index != record_count:
                    raise TrajectoryIndexError(
                        f"complete index expected episode {record_count}, "
                        f"found {episode.episode_index}"
                    )
                trajectory = trajectory_contract_for_episode(
                    episode,
                    repository_root,
                    project_git_revision=project_git_revision,
                    annotation_version=annotation_version,
                )
                line = _write_contract_line(handle, trajectory)
                records_digest.update(line)
                record_count += 1
                frame_count += episode.length
                task_indices.add(episode.task_index)
                previous_episode_index = episode.episode_index

        if record_count == 0:
            raise TrajectoryIndexError("cannot publish an empty trajectory index")
        if require_complete_source and (
            record_count != SOURCE_EPISODE_COUNT
            or frame_count != SOURCE_FRAME_COUNT
            or task_indices != set(range(SOURCE_TASK_COUNT))
        ):
            raise TrajectoryIndexError(
                "complete index totals do not match the pinned source: "
                f"records={record_count}, frames={frame_count}, tasks={len(task_indices)}"
            )
        partial_path.replace(output_path)
        manifest = {
            "artifact_type": TRAJECTORY_INDEX_ARTIFACT_TYPE,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "records_artifact_type": TRAJECTORY_ARTIFACT_TYPE,
            "source": {
                "dataset_repo_id": SOURCE_DATASET_REPO_ID,
                "dataset_revision": SOURCE_DATASET_REVISION,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            },
            "generator": {
                "project_git_revision": project_git_revision,
                "project_tree_clean": project_tree_clean,
                "annotation_version": annotation_version,
            },
            "records_path": output_path.name,
            "records_sha256": records_digest.hexdigest(),
            "complete_source": require_complete_source,
            "record_count": record_count,
            "frame_count": frame_count,
            "task_count": len(task_indices),
            "publishable": project_tree_clean,
        }
        manifest["digest"] = contract_digest(manifest)
        with manifest_partial_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_partial_path.replace(manifest_path)
        return manifest
    except BaseException:
        for path in (partial_path, manifest_partial_path):
            path.unlink(missing_ok=True)
        # Neither member of the index is useful alone. If publishing the
        # manifest fails after the records rename, remove the new orphan.
        if output_path.exists() and not manifest_path.exists():
            output_path.unlink()
        raise


def _write_contract_line(handle: TextIO, trajectory: TrajectoryRecord) -> bytes:
    line = (
        json.dumps(
            trajectory.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    handle.write(line.decode("utf-8"))
    return line

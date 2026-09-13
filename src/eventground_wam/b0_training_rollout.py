"""Resolve canonical LIBERO base tasks for the Week 3 B0 rollout gate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from eventground_wam.b0_inference import VerifiedB0Checkpoint

BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class BaseTaskSelection:
    suite: str
    runtime_task_index: int
    task_name: str
    instruction: str
    bddl_path: Path
    initial_states_path: Path
    tasks_info_path: Path


class BaseTrainingTaskSuite:
    """Minimal suite boundary for one canonical task in a ten-task LIBERO suite."""

    def __init__(self, selection: BaseTaskSelection) -> None:
        self.selection = selection

    def _check_index(self, index: int) -> None:
        if index != self.selection.runtime_task_index:
            raise IndexError(
                f"suite contains only selected task index {self.selection.runtime_task_index}"
            )

    def get_task(self, index: int) -> SimpleNamespace:
        self._check_index(index)
        return SimpleNamespace(
            name=self.selection.task_name,
            language=self.selection.instruction,
        )

    def get_task_bddl_file_path(self, index: int) -> str:
        self._check_index(index)
        return str(self.selection.bddl_path)

    def get_task_init_states(self, index: int) -> Any:
        self._check_index(index)
        return torch.load(self.selection.initial_states_path, weights_only=False)


def _task_slug(instruction: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", instruction.lower()).strip("_")


def _base_task_paths(libero_root: Path, suite: str) -> list[tuple[Path, Path]]:
    if suite not in BASE_SUITES:
        raise ValueError(f"unsupported base suite: {suite!r}")
    tasks_info_path = (
        libero_root / "libero" / "libero" / "bddl_files" / suite / "tasks_info.txt"
    )
    try:
        task_lines = [
            line.strip()
            for line in tasks_info_path.read_text(encoding="utf-8").splitlines()
        ]
    except OSError as error:
        raise RuntimeError(
            f"could not read base task inventory {tasks_info_path}: {error}"
        ) from error
    task_lines = [line for line in task_lines if line]
    if len(task_lines) != 10 or len(set(task_lines)) != 10:
        raise RuntimeError(f"expected ten unique base tasks in {tasks_info_path}")

    paths: list[tuple[Path, Path]] = []
    expected_prefix = f"libero/bddl_files/{suite}/"
    for relative in task_lines:
        if not relative.startswith(expected_prefix) or not relative.endswith(".bddl"):
            raise RuntimeError(f"invalid base task inventory entry: {relative!r}")
        bddl_path = libero_root / "libero" / relative
        init_path = (
            libero_root
            / "libero"
            / "libero"
            / "init_files"
            / suite
            / f"{bddl_path.stem}.pruned_init"
        )
        if not bddl_path.is_file() or not init_path.is_file():
            raise RuntimeError(f"base task files are incomplete for {bddl_path.stem!r}")
        paths.append((bddl_path.resolve(), init_path.resolve()))
    return paths


def select_cohort_episode(
    checkpoint: VerifiedB0Checkpoint,
    episode_index: int,
) -> dict[str, Any]:
    """Select exactly one episode from the accepted checkpoint's sealed cohort."""

    matches = [
        {
            "episode_index": cohort_episode_index,
            "source_task_index": source_task_index,
            "instruction": instruction,
        }
        for cohort_episode_index, source_task_index, instruction in checkpoint.cohort_episodes
        if cohort_episode_index == episode_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected cohort episode index {episode_index} exactly once, found {len(matches)}"
        )
    return matches[0]


def resolve_base_task(
    libero_root: Path,
    instruction: str,
) -> BaseTaskSelection:
    """Resolve a cohort instruction uniquely against canonical base-suite inventories."""

    slug = _task_slug(instruction)
    matches: list[BaseTaskSelection] = []
    for suite in BASE_SUITES:
        tasks_info_path = (
            libero_root / "libero" / "libero" / "bddl_files" / suite / "tasks_info.txt"
        ).resolve()
        for task_index, (bddl_path, init_path) in enumerate(
            _base_task_paths(libero_root, suite)
        ):
            task_name = bddl_path.stem
            normalized_name = task_name.lower()
            if normalized_name != slug and not normalized_name.endswith(f"_{slug}"):
                continue
            matches.append(
                BaseTaskSelection(
                    suite=suite,
                    runtime_task_index=task_index,
                    task_name=task_name,
                    instruction=instruction,
                    bddl_path=bddl_path,
                    initial_states_path=init_path,
                    tasks_info_path=tasks_info_path,
                )
            )
    if len(matches) != 1:
        raise RuntimeError(
            f"expected instruction {instruction!r} to identify one base task, "
            f"found {len(matches)}"
        )
    return matches[0]

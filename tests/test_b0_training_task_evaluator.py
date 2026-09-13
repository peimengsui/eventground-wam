from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from eventground_wam.b0_training_rollout import (
    BASE_SUITES,
    BaseTrainingTaskSuite,
    resolve_base_task,
    select_cohort_episode,
)


def _write_base_inventory(root: Path, suite: str, names: list[str]) -> None:
    bddl_root = root / "libero" / "libero" / "bddl_files" / suite
    init_root = root / "libero" / "libero" / "init_files" / suite
    bddl_root.mkdir(parents=True)
    init_root.mkdir(parents=True)
    lines = []
    for name in names:
        (bddl_root / f"{name}.bddl").write_text("task\n", encoding="utf-8")
        (init_root / f"{name}.pruned_init").write_bytes(b"states")
        lines.append(f"libero/bddl_files/{suite}/{name}.bddl")
    (bddl_root / "tasks_info.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture()
def libero_root(tmp_path: Path) -> Path:
    fillers = [f"filler_task_{index}" for index in range(10)]
    for suite in BASE_SUITES:
        names = list(fillers)
        if suite == "libero_goal":
            names[4] = "put_the_bowl_on_the_plate"
        _write_base_inventory(tmp_path, suite, names)
    return tmp_path


def test_resolve_base_task_uses_canonical_ten_task_inventory(libero_root: Path) -> None:
    selection = resolve_base_task(libero_root, "put the bowl on the plate")

    assert selection.suite == "libero_goal"
    assert selection.runtime_task_index == 4
    assert selection.task_name == "put_the_bowl_on_the_plate"
    assert selection.bddl_path.name == "put_the_bowl_on_the_plate.bddl"


def test_base_suite_rejects_any_unselected_runtime_index(libero_root: Path) -> None:
    selection = resolve_base_task(libero_root, "put the bowl on the plate")
    suite = BaseTrainingTaskSuite(selection)

    assert suite.get_task(4).language == "put the bowl on the plate"
    with pytest.raises(IndexError, match="only selected task index 4"):
        suite.get_task(3)


def test_select_cohort_episode_requires_a_unique_episode_index() -> None:
    checkpoint = SimpleNamespace(
        cohort_episodes=((9, 9, "put the bowl on the plate"),)
    )

    assert select_cohort_episode(checkpoint, 9) == {
        "episode_index": 9,
        "source_task_index": 9,
        "instruction": "put the bowl on the plate",
    }
    with pytest.raises(ValueError, match="found 0"):
        select_cohort_episode(checkpoint, 8)

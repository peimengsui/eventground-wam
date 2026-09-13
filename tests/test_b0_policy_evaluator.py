from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_b0_policy as evaluator


def test_select_manifest_task_is_exact_and_suite_qualified() -> None:
    manifest = {
        "tasks": [
            {"suite": "libero_spatial", "runtime_task_index": 0, "name": "spatial"},
            {"suite": "libero_goal", "runtime_task_index": 0, "name": "goal"},
        ]
    }

    selected = evaluator.select_manifest_task(
        manifest,
        suite="libero_goal",
        runtime_task_index=0,
    )

    assert selected["name"] == "goal"
    with pytest.raises(ValueError, match="cannot be negative"):
        evaluator.select_manifest_task(
            manifest,
            suite="libero_goal",
            runtime_task_index=-1,
        )
    with pytest.raises(ValueError, match="found 0"):
        evaluator.select_manifest_task(
            manifest,
            suite="libero_object",
            runtime_task_index=0,
        )

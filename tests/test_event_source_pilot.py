"""Pilot must reject held-out collisions before opening any episode payload."""

import json

import h5py
import numpy as np
import pytest

from scripts.run_event_source_pilot import (
    heldout_collision,
    inspect_one,
    numeric_digest,
    select_source_demos,
)
from scripts.run_event_source_recovery import eligible


def test_matching_length_in_another_task_is_not_a_collision():
    candidate = {"task_key": "task-a", "frame_count": 100}
    assert not heldout_collision(
        candidate,
        [
            {"task_key": "task-b", "episode_length": 100, "split": "test"},
            {"task_key": "task-a", "episode_length": 101, "split": "validation"},
            {"task_key": "task-a", "episode_length": 100, "split": "train"},
        ],
    )


def test_heldout_collision_returns_before_network_and_payload_access(tmp_path):
    candidate = {
        "queue_rank": 0,
        "task_key": "task-a",
        "frame_count": 100,
        "group_digest": "a",
        "source_episode_index": 1,
    }
    groups = [
        {
            "group_digest": "a",
            "split": "train",
            "episode_indices": [1],
            "task_key": "task-a",
            "episode_length": 100,
        },
        {
            "group_digest": "b",
            "split": "test",
            "episode_indices": [2],
            "task_key": "task-a",
            "episode_length": 100,
        },
    ]
    # Deliberately omit URL/snapshot/resource settings: reading them would fail.
    result = inspect_one((candidate, {"digest": "unit-test"}, groups, str(tmp_path)))
    assert result["status"] == "excluded_heldout_task_length_collision"
    assert result["target_payload_reads"] == result["original_demo_payload_reads"] == 0
    assert not (tmp_path / "candidate-00/http-ranges.jsonl").exists()
    saved = json.loads((tmp_path / "candidate-00/result.json").read_text())
    assert saved["status"] == result["status"]


def test_numeric_digest_includes_shape_and_normalizes_float_dtype():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert numeric_digest(values) == numeric_digest(values.astype(np.float32))
    assert numeric_digest(values) != numeric_digest(values.reshape(4, 3))


def test_attribute_selection_agrees_with_shapes_and_rejects_false_length(tmp_path):
    with h5py.File(tmp_path / "source.h5", "w") as handle:
        for i, length in enumerate([5, 10, 15]):
            demo = handle.create_group(f"demo_{i}")
            demo.attrs["num_samples"] = length
            demo.create_dataset("actions", shape=(length, 7), dtype="f8")
            demo.create_dataset("states", shape=(length, 40), dtype="f8")
        names = list(handle)
        assert select_source_demos(handle, names, 10, "dataset_shapes")[1] == ["demo_1"]
        assert select_source_demos(handle, names, 10, "num_samples_attribute")[1] == ["demo_1"]
        handle["demo_0"].attrs["num_samples"] = 10
        with pytest.raises(ValueError, match="disagrees"):
            select_source_demos(handle, names, 10, "num_samples_attribute")


def test_recovery_rejects_any_prior_payload_access_or_scientific_result(tmp_path):
    ranges = tmp_path / "http-ranges.jsonl"
    ranges.write_text(json.dumps({"phase": "metadata"}) + "\n")
    row = {
        "status": "access_or_integrity_failure",
        "error": "TimeoutError: Per-file wall limit reached",
        "target_payload_reads": 0,
        "original_demo_payload_reads": 0,
        "full_simulator_state_reads": 0,
        "comparisons": [],
    }
    assert eligible(row, tmp_path)
    for key in ["target_payload_reads", "original_demo_payload_reads", "full_simulator_state_reads"]:
        assert not eligible({**row, key: 1}, tmp_path)
    assert not eligible({**row, "status": "numeric_mismatch"}, tmp_path)
    assert not eligible({**row, "error": "ValueError: Numerical identity"}, tmp_path)
    ranges.write_text(json.dumps({"phase": "numeric:demo_1"}) + "\n")
    assert not eligible(row, tmp_path)

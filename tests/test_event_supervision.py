"""Leakage and false-admission regressions for Week 10 preparation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from eventground_wam.event_supervision import build_review_queue, compare_recorded_trajectory

ROOT = Path(__file__).resolve().parents[1]


def groups():
    return [
        json.loads(line)
        for line in (ROOT / "configs/datasets/libero-plus-trajectory-v2/groups.jsonl")
        .read_text()
        .splitlines()
    ]


def test_real_selection_is_disjoint_train_only_and_covers_tasks_lengths():
    source = groups()
    queue = build_review_queue(source)
    rows = queue["rows"]
    train = {g["group_digest"]: g for g in source if g["split"] == "train"}
    assert len(rows) == len({r["group_digest"] for r in rows}) == 300
    assert len({r["source_episode_index"] for r in rows}) == 300
    assert all(r["group_digest"] in train for r in rows)
    assert all(
        r["source_episode_index"] in train[r["group_digest"]]["episode_indices"] for r in rows
    )
    assert set(queue["initial_task_counts"].values()) == {5}
    for task in queue["initial_task_counts"]:
        assert {r["length_stratum"] for r in rows[:200] if r["task_key"] == task} == {
            "short",
            "middle",
            "long",
        }
    assert len({r["task_key"] for r in rows[:40]}) == 40
    assert queue["labels_generated"] == queue["payloads_opened"] == 0
    assert all(r["review_status"] == "not_ready_physical_evidence_pending" for r in rows)


def test_selection_ignores_order_and_visual_variant_order():
    source = groups()
    reverse = copy.deepcopy(source[::-1])
    for row in reverse:
        row["episode_indices"].reverse()
    assert build_review_queue(source) == build_review_queue(reverse)


def test_cross_split_duplicate_episode_rejected():
    source = groups()
    train = next(g for g in source if g["split"] == "train")
    held_out = next(g for g in source if g["split"] == "test")
    held_out["episode_indices"].append(train["episode_indices"][0])
    with pytest.raises(ValueError, match="multiple groups"):
        build_review_queue(source)


def aligned():
    state = np.arange(80).reshape(10, 8).astype(float) / 100
    action = np.zeros((10, 7))
    action[:, 6] = -1
    return {
        "states": state,
        "actions": action,
        "source_states": state.copy(),
        "source_actions": action.copy(),
        "source_frames": np.arange(10),
        "evidence_origin": "recorded_simulator_states",
    }


def test_numeric_match_never_admits_physical_labels():
    result = compare_recorded_trajectory(**aligned())
    assert result["numeric_alignment_passed"]
    assert not result["physical_scene_identity_verified"]
    assert not result["contact_grasp_labels_admitted"]


@pytest.mark.parametrize("change", ["state", "action", "gripper", "frame_shift"])
def test_mismatch_cannot_pass(change):
    data = aligned()
    if change == "state":
        data["source_states"][5, 2] += 0.001
    elif change == "action":
        data["source_actions"][5, 2] += 0.001
    elif change == "gripper":
        data["source_actions"][5, 6] = 1
    else:
        data["source_states"] = np.roll(data["source_states"], 1, axis=0)
    assert not compare_recorded_trajectory(**data)["numeric_alignment_passed"]


@pytest.mark.parametrize(
    "change", ["nan", "short", "repeat", "skip", "fraction", "overflow", "replay"]
)
def test_incomplete_or_untrusted_evidence_rejected(change):
    data = aligned()
    if change == "nan":
        data["source_states"][0, 0] = np.nan
    elif change == "short":
        data["source_states"] = data["source_states"][:-1]
    elif change == "repeat":
        data["source_frames"][5] = 4
    elif change == "skip":
        data["source_frames"][5:] += 1
    elif change == "fraction":
        data["source_frames"] = np.arange(10) + 0.1
    elif change == "overflow":
        data["source_frames"] = np.arange(10, dtype=np.uint64) + np.uint64(2**63)
    else:
        data["evidence_origin"] = "approximate_action_replay"
    with pytest.raises(ValueError):
        compare_recorded_trajectory(**data)

import copy

import numpy as np
import pytest

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.instrumented_labeling import (
    RecordingEvidence,
    extract_events,
    prompt_previews,
    rotation_distance,
)


def evidence(n=50):
    objects = [
        {"object_id": "bowl", "body_id": 1, "geom_ids": [2], "movable_free_object": True},
        {"object_id": "fixture", "body_id": 2, "geom_ids": [3], "movable_free_object": False},
    ]
    geoms = [
        {
            "geom_id": i,
            "object_id": ("bowl" if i == 2 else "fixture" if i == 3 else None),
            "class": "gripper"
            if i < 2
            else "movable_object"
            if i == 2
            else "excluded_fixed_object",
            "collision_enabled": True,
            "finger_sides": ["left_finger"] if i == 0 else ["right_finger"] if i == 1 else [],
        }
        for i in range(4)
    ]
    state = np.zeros((n, 8))
    state[:, -2:] = [0.04, -0.04]
    actions = np.zeros((n, 7))
    actions[:, -1] = -1
    return RecordingEvidence(
        "synthetic",
        "a" * 64,
        actions,
        state,
        np.zeros((n, 2, 3)),
        np.tile(np.eye(3), (n, 2, 1, 1)),
        np.zeros((n, 3)),
        np.tile(np.eye(3), (n, 1, 1)),
        {
            "objects": objects,
            "geoms": geoms,
            "finger_geom_ids": {"left_finger": [0], "right_finger": [1]},
        },
        [{"frame_index": i, "contacts": []} for i in range(n)],
    )


def touch(
    e, start, end, *, sides=(0, 1), object_geom=2, distance=-0.0001, active=True, reverse=False
):
    for t in range(start, end):
        for side in sides:
            a, b = (object_geom, side) if reverse else (side, object_geom)
            e.contacts[t]["contacts"].append(
                {
                    "geom1": a,
                    "geom2": b,
                    "distance_m": distance,
                    "efc_address": 0 if active else -1,
                    "position": [0, 0, 0],
                    "frame": np.eye(3).ravel().tolist(),
                }
            )


def grasp_evidence():
    e = evidence(50)
    e.actions[10:, -1] = 1
    touch(e, 10, 25)
    e.gripper_position[10:15, 0] = np.arange(5) * 0.003
    e.gripper_position[15:, 0] = 0.012
    e.object_position[:, 0] = e.gripper_position + [0, 0, 0.02]
    return e


def typed(result, kind):
    return [r for r in result["events"] if r["event_type"] == kind]


def test_contact_persistence_no_gap_fill_and_exclusions():
    e = evidence()
    touch(e, 3, 5, reverse=True)
    touch(e, 6, 7)
    touch(e, 8, 10)
    touch(e, 12, 15, object_geom=3)
    touch(e, 17, 20, distance=0.001)
    touch(e, 21, 24, active=False)
    r = extract_events(e)
    assert [(x["start"], x["end"]) for x in typed(r, "contact")] == [(3, 5), (8, 10)]
    assert any(x["reason"] == "single_frame_contact" and x["start"] == 6 for x in r["ambiguities"])
    assert not r["labels_admitted"]


def test_grasp_confirm_retention_then_contact_loss():
    r = extract_events(grasp_evidence())
    g = typed(r, "grasp")
    assert len(g) == 1 and (g[0]["start"], g[0]["end"], g[0]["confirmation_frame"]) == (10, 25, 14)
    assert g[0]["review_status"] == "unreviewed" and not g[0]["training_admitted"]


def test_stationary_contact_is_ambiguous_not_grasp_or_none():
    e = evidence()
    e.actions[10:, -1] = 1
    touch(e, 10, 25)
    r = extract_events(e)
    assert not typed(r, "grasp")
    assert any("insufficient_gripper_motion" in x["reason"] for x in r["ambiguities"])
    assert prompt_previews(r)["rows"][0]["reason"] == "unresolved_candidate_in_window"


@pytest.mark.parametrize("change", ["single_finger", "sliding", "rotation_slip", "release_command"])
def test_grasp_requires_each_frozen_condition(change):
    e = grasp_evidence()
    if change == "single_finger":
        for row in e.contacts:
            row["contacts"] = [c for c in row["contacts"] if c["geom1"] == 0]
    elif change == "sliding":
        e.object_position[:, 0] = [0, 0, 0.02]
    elif change == "rotation_slip":
        for t in range(11, 50):
            a = np.deg2rad((t - 10) * 15)
            e.object_rotation[t, 0] = [
                [np.cos(a), -np.sin(a), 0],
                [np.sin(a), np.cos(a), 0],
                [0, 0, 1],
            ]
    else:
        e.actions[12:, -1] = -1
    assert not typed(extract_events(e), "grasp")


def test_grasp_can_confirm_from_rotation_and_end_on_relative_drift():
    e = evidence()
    e.actions[10:, -1] = 1
    touch(e, 10, 30)
    for t in range(10, 50):
        a = np.deg2rad(min(t - 10, 4) * 3)
        r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        e.gripper_rotation[t] = r
        e.object_rotation[t, 0] = r
    e.object_position[20:, 0, 0] = 0.011
    g = typed(extract_events(e), "grasp")
    assert len(g) == 1 and (g[0]["start"], g[0]["end"]) == (10, 20)


def test_close_rule_boundaries_and_weak_response():
    e = evidence()
    e.actions[4:12, -1] = 1
    e.state[9:, -2:] = [0.039, -0.039]
    close = typed(extract_events(e), "gripper_close")
    assert [(x["start"], x["end"]) for x in close] == [(4, 10)]
    e.state[:, -2:] = [0.04, -0.04]
    r = extract_events(e)
    assert not typed(r, "gripper_close")
    assert any("weak_aperture_response" in x["reason"] for x in r["ambiguities"])


def test_initial_and_final_contact_censoring_and_incomplete_grasp():
    e = evidence()
    touch(e, 0, 3)
    touch(e, 48, 50)
    e.actions[48:, -1] = 1
    r = extract_events(e)
    c = typed(r, "contact")
    assert c[0]["left_censored"] and c[1]["right_censored"]
    assert any("incomplete_confirmation_window" in x["reason"] for x in r["ambiguities"])
    assert prompt_previews(r)["rows"][0]["reason"] == "inside_ongoing_event"


def test_full_negative_window_required_and_future_not_exported():
    rows = prompt_previews(extract_events(evidence()))["rows"]
    assert [r["anchor"] for r in rows if r["status"] == "unreviewed_preview"] == list(range(18))
    assert all(r["event_type"] == "none" and r["interval"] is None for r in rows[:18])
    assert all(r["reason"] == "incomplete_future_window" for r in rows[18:])
    assert all(
        set(r)
        <= {
            "anchor",
            "model_input_frame",
            "status",
            "training_admitted",
            "event_type",
            "event_id",
            "interval",
            "horizon",
            "end_censored",
            "reason",
        }
        for r in rows
    )


def test_ties_earliest_onset_and_horizon_clip():
    e = evidence(60)
    touch(e, 10, 60)
    e.actions[10:, -1] = 1
    e.state[15:, -2:] = [0.039, -0.039]
    e.gripper_position[10:15, 0] = np.arange(5) * 0.003
    e.gripper_position[15:, 0] = 0.012
    e.object_position[:, 0] = e.gripper_position + [0, 0, 0.02]
    r = extract_events(e)
    row = prompt_previews(r)["rows"][0]
    assert (
        row["event_type"] == "grasp"
        and row["interval"] == [10, 33]
        and row["horizon"] == 10
        and row["end_censored"]
    )
    assert row["training_admitted"] is False


@pytest.mark.parametrize(
    "corrupt",
    [
        "missing_frame",
        "unknown_geom",
        "duplicate_geom",
        "nonfinite",
        "bad_rotation",
        "unknown_command",
    ],
)
def test_invalid_evidence_never_becomes_negative(corrupt):
    e = evidence()
    if corrupt == "missing_frame":
        e.contacts.pop()
    elif corrupt == "unknown_geom":
        touch(e, 3, 4, object_geom=99)
    elif corrupt == "duplicate_geom":
        e.geometry["geoms"].append(copy.deepcopy(e.geometry["geoms"][0]))
    elif corrupt == "nonfinite":
        e.object_position[3, 0, 0] = np.nan
    elif corrupt == "bad_rotation":
        e.gripper_rotation[0, 0, 0] = 2
    else:
        e.actions[0, -1] = 0
    with pytest.raises(ValueError):
        extract_events(e)


def test_modified_label_thresholds_rejected():
    from eventground_wam.instrumented_supervision import labeler_contract

    rules = labeler_contract()
    rules.pop("digest")
    rules["contact"]["minimum_consecutive_frames"] = 1
    with pytest.raises(ValueError):
        extract_events(evidence(), seal_contract(rules))


def test_so3_geodesic_not_euler_subtraction():
    a = np.deg2rad(359)
    r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    assert float(rotation_distance(np.eye(3), r)) == pytest.approx(np.deg2rad(1))


@pytest.mark.parametrize(
    "corrupt", ["constraint_nan", "constraint_fraction", "position", "owned_geom", "finger_geom"]
)
def test_malformed_contact_identity_is_quarantined(corrupt):
    e = evidence()
    touch(e, 4, 6)
    if corrupt == "constraint_nan":
        e.contacts[4]["contacts"][0]["efc_address"] = float("nan")
    elif corrupt == "constraint_fraction":
        e.contacts[4]["contacts"][0]["efc_address"] = 0.5
    elif corrupt == "position":
        e.contacts[4]["contacts"][0]["position"] = [0, 0]
    elif corrupt == "owned_geom":
        e.geometry["objects"][0]["geom_ids"].append(99)
    else:
        e.geometry["finger_geom_ids"]["left_finger"].append(99)
    with pytest.raises(ValueError):
        extract_events(e)


def test_grasp_invariant_to_global_rigid_transform():
    e = grasp_evidence()
    before = extract_events(e)
    r = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    shift = np.array([3.0, -2.0, 0.5])
    e.gripper_position = e.gripper_position @ r.T + shift
    e.object_position = e.object_position @ r.T + shift
    e.gripper_rotation = r @ e.gripper_rotation
    e.object_rotation = r @ e.object_rotation
    after = extract_events(e)
    assert before["events"] == after["events"]
    assert before["ambiguities"] == after["ambiguities"]


def test_event_at_frame_32_is_in_window_33_is_not():
    e = evidence(60)
    touch(e, 32, 34)
    row = prompt_previews(extract_events(e))["rows"][0]
    assert row["event_type"] == "contact" and row["interval"] == [32, 33]
    e = evidence(60)
    touch(e, 33, 35)
    assert prompt_previews(extract_events(e))["rows"][0]["event_type"] == "none"


def test_all_movable_objects_retained_without_task_success_filter():
    e = evidence()
    e.geometry["objects"][1]["movable_free_object"] = True
    e.geometry["geoms"][3]["class"] = "movable_object"
    touch(e, 5, 8, object_geom=2)
    touch(e, 5, 8, object_geom=3)
    contact = typed(extract_events(e), "contact")
    assert [(r["object_id"], r["start"], r["end"]) for r in contact] == [
        ("bowl", 5, 8),
        ("fixture", 5, 8),
    ]

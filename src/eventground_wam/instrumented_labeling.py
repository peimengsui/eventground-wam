"""Frozen Week 10 physical-event rules; outputs remain unreviewed, never admitted.

Only offline evidence enters this module. Prompt previews identify an anchor
and target; they never export future states or contact features as model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.instrumented_supervision import labeler_contract


@dataclass
class RecordingEvidence:
    recording_id: str
    source_id: str
    actions: np.ndarray
    state: np.ndarray
    object_position: np.ndarray
    object_rotation: np.ndarray
    gripper_position: np.ndarray
    gripper_rotation: np.ndarray
    geometry: dict
    contacts: list[dict]


def runs(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    return list(
        zip(
            np.flatnonzero(np.diff(padded) == 1).tolist(),
            np.flatnonzero(np.diff(padded) == -1).tolist(),
            strict=True,
        )
    )


def rotation_distance(reference, matrices):
    """SO(3) geodesic angles in radians, invariant to quaternion sign."""
    relative = np.einsum("ij,...jk->...ik", reference.T, matrices)
    return np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1, 1))


def _validate(e):
    n = len(e.actions)
    if n < 1 or len(e.source_id) != 64 or any(c not in "0123456789abcdef" for c in e.source_id):
        raise ValueError("Invalid source identity or empty recording")
    objects = e.geometry["objects"]
    if len({o["object_id"] for o in objects}) != len(objects):
        raise ValueError("Duplicate object identity")
    for name, shape in [
        ("actions", (n, 7)),
        ("state", (n, 8)),
        ("object_position", (n, len(objects), 3)),
        ("object_rotation", (n, len(objects), 3, 3)),
        ("gripper_position", (n, 3)),
        ("gripper_rotation", (n, 3, 3)),
    ]:
        array = np.asarray(getattr(e, name))
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Missing, malformed or nonfinite {name}")
    if not np.isin(e.actions[:, -1], [-1.0, 1.0]).all():
        raise ValueError("Unknown gripper command encoding")
    for matrices in (e.object_rotation, e.gripper_rotation):
        gram = np.swapaxes(matrices, -1, -2) @ matrices
        if not np.allclose(gram, np.eye(3), atol=1e-8, rtol=0) or not np.allclose(
            np.linalg.det(matrices), 1, atol=1e-8, rtol=0
        ):
            raise ValueError("Invalid object/gripper rotation evidence")
    geoms = {g["geom_id"]: g for g in e.geometry["geoms"]}
    if len(geoms) != len(e.geometry["geoms"]):
        raise ValueError("Ambiguous geometry identity")
    ownership = {}
    for obj in objects:
        for geom in obj["geom_ids"]:
            if geom not in geoms:
                raise ValueError("Unknown owned object geometry")
            if geom in ownership:
                raise ValueError("Multiply owned object geometry")
            ownership[geom] = obj["object_id"]
    for side in ("left_finger", "right_finger"):
        ids = e.geometry["finger_geom_ids"][side]
        if len(set(ids)) != len(ids) or any(g not in geoms for g in ids):
            raise ValueError("Invalid finger geometry inventory")
    for gid, g in geoms.items():
        if g["object_id"] != ownership.get(gid):
            raise ValueError("Unmapped object geometry")
        if g["class"] == "gripper" and g["object_id"] is not None:
            raise ValueError("Gripper/object ownership conflict")
        if set(g["finger_sides"]) - {"left_finger", "right_finger"}:
            raise ValueError("Unknown finger side")
        for side in ("left_finger", "right_finger"):
            if (side in g["finger_sides"]) != (gid in e.geometry["finger_geom_ids"][side]):
                raise ValueError("Inconsistent finger-side mapping")
    if len(e.contacts) != n:
        raise ValueError("Missing contact frames; cannot infer negatives")
    for frame, record in enumerate(e.contacts):
        if record["frame_index"] != frame:
            raise ValueError("Missing or repeated contact frame")
        for contact in record["contacts"]:
            if any(
                not isinstance(contact[key], (int, np.integer))
                or isinstance(contact[key], (bool, np.bool_))
                for key in ("geom1", "geom2", "efc_address")
            ):
                raise ValueError("Noninteger contact geometry/constraint identity")
            if contact["efc_address"] < -1:
                raise ValueError("Invalid contact constraint address")
            if contact["geom1"] not in geoms or contact["geom2"] not in geoms:
                raise ValueError("Unknown contact geometry")
            if np.asarray(contact["position"]).shape != (3,) or np.asarray(
                contact["frame"]
            ).shape != (9,):
                raise ValueError("Malformed contact frame/position")
            values = np.r_[contact["distance_m"], contact["position"], contact["frame"]]
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite contact evidence")
    return n, objects, geoms


def extract_events(e: RecordingEvidence, rules=None):
    rules = rules or labeler_contract()
    if (
        contract_digest(rules) != rules.get("digest")
        or rules["digest"] != labeler_contract()["digest"]
    ):
        raise ValueError("Only the frozen labeler contract is supported")
    n, objects, geoms = _validate(e)
    events = []
    ambiguous = []
    decisions = []

    def event(kind, obj, start, end, confirmation):
        events.append(
            {
                "event_type": kind,
                "object_id": obj,
                "start": start,
                "end": end,
                "confirmation_frame": confirmation,
                "left_censored": start == 0,
                "right_censored": end == n,
                "review_status": "unreviewed",
                "training_admitted": False,
            }
        )

    def uncertain(kind, obj, start, end, reason):
        ambiguous.append(
            {"event_type": kind, "object_id": obj, "start": start, "end": end, "reason": reason}
        )

    commands = e.actions[:, -1]
    for oi, obj in enumerate(objects):
        if not obj["movable_free_object"]:
            continue
        name = obj["object_id"]
        contact = np.zeros(n, dtype=bool)
        left = contact.copy()
        right = contact.copy()
        for t, record in enumerate(e.contacts):
            for pair in record["contacts"]:
                if pair["distance_m"] > 0 or pair["efc_address"] < 0:
                    continue
                a, b = geoms[pair["geom1"]], geoms[pair["geom2"]]
                if a["class"] != "gripper":
                    a, b = b, a
                if a["class"] != "gripper" or b["object_id"] != name:
                    continue
                if not a["collision_enabled"] or not b["collision_enabled"]:
                    continue
                contact[t] = True
                left[t] |= "left_finger" in a["finger_sides"]
                right[t] |= "right_finger" in a["finger_sides"]
        for start, end in runs(contact):
            if end - start >= rules["contact"]["minimum_consecutive_frames"]:
                event("contact", name, start, end, start + 1)
            else:
                uncertain("contact", name, start, end, "single_frame_contact")
        # Relative object pose in the gripper frame, not world-frame proximity.
        inverse = np.swapaxes(e.gripper_rotation, -1, -2)
        relative_pos = np.einsum(
            "tij,tj->ti", inverse, e.object_position[:, oi] - e.gripper_position
        )
        relative_rot = inverse @ e.object_rotation[:, oi]
        finger = left | right
        bilateral = left & right
        t = 0
        while t < n:
            # A prospective grasp window starts only with a closing command and finger contact.
            if not (commands[t] == 1 and finger[t]):
                t += 1
                continue
            end = min(t + 5, n)
            reasons = []
            if end - t < 5:
                reasons.append("incomplete_confirmation_window")
            if not np.all(commands[t:end] == 1):
                reasons.append("close_not_persistent")
            if not np.all(finger[t:end]):
                reasons.append("finger_contact_not_persistent")
            if np.count_nonzero(bilateral[t:end]) < 3:
                reasons.append("insufficient_bilateral_contact")
            if np.max(np.linalg.norm(relative_pos[t:end] - relative_pos[t], axis=-1)) > 0.005:
                reasons.append("relative_translation_unstable")
            if np.max(rotation_distance(relative_rot[t], relative_rot[t:end])) > np.deg2rad(10):
                reasons.append("relative_rotation_unstable")
            displacement = np.max(
                np.linalg.norm(e.gripper_position[t:end] - e.gripper_position[t], axis=-1)
            )
            rotation = np.max(rotation_distance(e.gripper_rotation[t], e.gripper_rotation[t:end]))
            if displacement < 0.010 and rotation < np.deg2rad(10):
                reasons.append("insufficient_gripper_motion")
            decisions.append(
                {
                    "event_type": "grasp",
                    "object_id": name,
                    "candidate_start": t,
                    "confirmation_end": end,
                    "status": "ambiguous" if reasons else "confirmed_unreviewed",
                    "reasons": reasons,
                }
            )
            if reasons:
                uncertain("grasp", name, t, end, ";".join(reasons))
                t += 1
                continue
            grasp_end = end
            while grasp_end < n:
                if not finger[grasp_end]:
                    break
                if np.linalg.norm(relative_pos[grasp_end] - relative_pos[t]) > 0.010:
                    break
                if float(rotation_distance(relative_rot[t], relative_rot[grasp_end])) > np.deg2rad(
                    20
                ):
                    break
                grasp_end += 1
            event("grasp", name, t, grasp_end, t + 4)
            t = grasp_end
    # Preserve the raw Week 2 command-transition/aperture-response rule exactly.
    aperture = np.abs(e.state[:, -2:]).sum(axis=1)
    for start, end in runs(commands == 1):
        if start == 0:
            uncertain("gripper_close", None, start, end, "initial_close_command_left_censored")
            continue
        response = start + 5
        reasons = []
        if end - start < 5:
            reasons.append("short_close_command")
        if response >= n:
            reasons.append("incomplete_aperture_response")
        elif aperture[start] - aperture[response] < 0.001:
            reasons.append("weak_aperture_response")
        if reasons:
            uncertain("gripper_close", None, start, min(n, response + 1), ";".join(reasons))
        else:
            event("gripper_close", None, start, response + 1, response)
    order = {"grasp": 0, "contact": 1, "gripper_close": 2}
    events.sort(key=lambda r: (r["start"], order[r["event_type"]], r["object_id"] or ""))
    for i, row in enumerate(events):
        row["event_id"] = f"{e.recording_id}:event-{i:04d}"
    return seal_contract(
        {
            "artifact_type": "eventground_wam.instrumented_event_candidates",
            "recording_id": e.recording_id,
            "source_id": e.source_id,
            "frames": n,
            "labeler_digest": rules["digest"],
            "events": events,
            "ambiguities": ambiguous,
            "grasp_window_decisions": decisions,
            "labels_admitted": False,
            "quality_validated": False,
            "future_evidence_is_model_input": False,
        }
    )


def prompt_previews(result):
    """All eligible anchor previews; no balancing/subsampling or admission here."""
    if contract_digest(result) != result.get("digest"):
        raise ValueError("Changed event candidate record")
    n = result["frames"]
    events = result["events"]
    ambiguous = result["ambiguities"]
    rows = []
    for anchor in range(n):
        stop = anchor + 33
        row = {
            "anchor": anchor,
            "status": "excluded",
            "model_input_frame": anchor,
            "training_admitted": False,
        }
        if stop > n:
            row["reason"] = "incomplete_future_window"
        elif any(r["start"] <= anchor < r["end"] for r in events):
            row["reason"] = "inside_ongoing_event"
        elif any(r["start"] < stop and r["end"] > anchor for r in ambiguous):
            row["reason"] = "unresolved_candidate_in_window"
        else:
            future = [r for r in events if anchor < r["start"] < stop and not r["left_censored"]]
            if future:
                first = future[0]
                row.update(
                    {
                        "status": "unreviewed_preview",
                        "event_type": first["event_type"],
                        "event_id": first["event_id"],
                        "interval": [first["start"] - anchor, min(33, first["end"] - anchor)],
                        "horizon": first["start"] - anchor,
                        "end_censored": first["right_censored"] or first["end"] > stop,
                    }
                )
            else:
                row.update(
                    {
                        "status": "unreviewed_preview",
                        "event_type": "none",
                        "event_id": None,
                        "interval": None,
                        "horizon": None,
                        "end_censored": False,
                    }
                )
        rows.append(row)
    return seal_contract(
        {
            "artifact_type": "eventground_wam.instrumented_prompt_previews",
            "candidate_digest": result["digest"],
            "rows": rows,
            "labels_admitted": False,
            "sampling_applied": False,
            "maximum_anchors_per_event_applies_at_later_sampling": 4,
            "maximum_prompts_per_group_applies_at_later_sampling": 16,
        }
    )

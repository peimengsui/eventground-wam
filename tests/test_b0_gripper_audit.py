from __future__ import annotations

import numpy as np
import pytest

from eventground_wam.b0_data import B0Batch, B0SampleIdentity
from eventground_wam.b0_gripper_audit import B0GripperAuditError, audit_b0_gripper_errors


def _cohort() -> B0Batch:
    identities = tuple(
        B0SampleIdentity(
            split="train",
            episode_index=0,
            task_index=3,
            anchor_frame_index=anchor,
            action_frame_end_exclusive=anchor + 8,
            split_assignment_digest="a" * 64,
        )
        for anchor in range(3)
    )
    timeline = np.asarray([-1, -1, -1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    actions = np.zeros((3, 8, 7), dtype=np.float32)
    for anchor in range(3):
        actions[anchor, :, 6] = timeline[anchor : anchor + 8]
    return B0Batch(
        identities=identities,
        instructions=("do the task",) * 3,
        front_rgb=np.arange(3 * 3 * 2 * 2, dtype=np.uint8).reshape(3, 3, 2, 2),
        wrist_rgb=np.arange(3 * 3 * 2 * 2, dtype=np.uint8).reshape(3, 3, 2, 2),
        proprioception=np.arange(24, dtype=np.float32).reshape(3, 8),
        actions=actions,
    )


def test_audit_collapses_overlapping_errors_to_source_frames() -> None:
    cohort = _cohort()
    logits = cohort.actions[..., 6].copy()
    logits[0, 3] = -0.2
    logits[1, 2] = -0.8

    result = audit_b0_gripper_errors(cohort, logits)

    assert result["cohort"]["target_slot_count"] == 24
    assert result["cohort"]["source_action_frame_count"] == 10
    assert result["cohort"]["gripper_transition_count"] == 1
    assert result["label_integrity"]["overlapping_chunk_label_contradiction_count"] == 0
    assert result["label_integrity"]["exact_conditioning_conflicting_label_group_count"] == 0
    residuals = result["residual_errors"]
    assert residuals["error_slot_count"] == 2
    assert residuals["unique_source_action_frame_count"] == 1
    assert residuals["repeated_overlap_error_slot_count"] == 1
    assert residuals["nearest_transition_distance_counts"] == {"0_transition_frame": 2}
    assert residuals["horizon_offset_counts"] == {"2": 1, "3": 1}
    assert residuals["accuracy"] == pytest.approx(22 / 24)


def test_audit_rejects_nonbinary_targets_and_wrong_logit_shape() -> None:
    cohort = _cohort()
    logits = cohort.actions[..., 6].copy()
    with pytest.raises(B0GripperAuditError, match="shape"):
        audit_b0_gripper_errors(cohort, logits[:, :-1])

    cohort.actions[0, 0, 6] = 0.0
    with pytest.raises(B0GripperAuditError, match="not exactly"):
        audit_b0_gripper_errors(cohort, logits)

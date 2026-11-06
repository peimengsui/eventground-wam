"""Deterministic residual-error audit for the B0 gripper classifier."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from eventground_wam.b0_data import B0Batch
from eventground_wam.demonstration_contracts import canonical_json_bytes


class B0GripperAuditError(ValueError):
    """Raised when gripper predictions cannot be audited against a B0 cohort."""


def _conditioning_signature(cohort: B0Batch, sample_index: int) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"instruction": cohort.instructions[sample_index]}))
    for name, array in (
        ("front_rgb", cohort.front_rgb[sample_index]),
        ("wrist_rgb", cohort.wrist_rgb[sample_index]),
        ("proprioception", cohort.proprioception[sample_index]),
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(
            canonical_json_bytes(
                {"name": name, "dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
            )
        )
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _transition_distance_bucket(distance: int | None) -> str:
    if distance is None or distance > 7:
        return "more_than_7_or_no_transition"
    if distance == 0:
        return "0_transition_frame"
    if distance == 1:
        return "1_frame"
    if distance == 2:
        return "2_frames"
    if distance <= 4:
        return "3_to_4_frames"
    return "5_to_7_frames"


def _margin_bucket(absolute_logit: float) -> str:
    if absolute_logit < 0.25:
        return "less_than_0.25"
    if absolute_logit < 0.5:
        return "0.25_to_less_than_0.5"
    if absolute_logit < 1.0:
        return "0.5_to_less_than_1"
    if absolute_logit < 2.0:
        return "1_to_less_than_2"
    return "at_least_2"


def _sorted_counts(counts: Mapping[Any, int]) -> dict[str, int]:
    return {str(key): counts[key] for key in sorted(counts)}


def audit_b0_gripper_errors(
    cohort: B0Batch,
    gripper_logits: NDArray[np.floating[Any]],
) -> dict[str, Any]:
    """Audit every gripper slot and return JSON-native residual diagnostics.

    The audit distinguishes an individual model output slot from its underlying
    source action frame. Because eight-action chunks overlap, one source command
    can appear at several horizon offsets.
    """

    sample_count = len(cohort.identities)
    targets = np.asarray(cohort.actions[..., 6])
    logits = np.asarray(gripper_logits)
    if targets.shape != (sample_count, 8):
        raise B0GripperAuditError(
            f"cohort gripper targets have shape {targets.shape}, expected {(sample_count, 8)}"
        )
    if logits.shape != targets.shape:
        raise B0GripperAuditError(
            f"gripper logits have shape {logits.shape}, expected {targets.shape}"
        )
    if not np.isfinite(logits).all():
        raise B0GripperAuditError("gripper logits contain non-finite values")
    nonbinary_mask = ~np.isin(targets, (-1.0, 1.0))
    if nonbinary_mask.any():
        raise B0GripperAuditError("gripper targets are not exactly -1 or +1")

    target_signs = targets.astype(np.int8, copy=False)
    predicted_signs = np.where(logits >= 0.0, 1, -1).astype(np.int8)
    error_mask = predicted_signs != target_signs

    frame_labels: dict[tuple[int, int], set[int]] = defaultdict(set)
    frame_occurrences: Counter[tuple[int, int]] = Counter()
    for sample_index, identity in enumerate(cohort.identities):
        for horizon_offset in range(8):
            frame_key = (
                identity.episode_index,
                identity.anchor_frame_index + horizon_offset,
            )
            frame_labels[frame_key].add(int(target_signs[sample_index, horizon_offset]))
            frame_occurrences[frame_key] += 1
    contradictory_frames = sorted(key for key, labels in frame_labels.items() if len(labels) != 1)

    episode_timelines: dict[int, dict[int, int]] = defaultdict(dict)
    for (episode_index, frame_index), labels in frame_labels.items():
        if len(labels) == 1:
            episode_timelines[episode_index][frame_index] = next(iter(labels))
    transition_frames: dict[int, tuple[int, ...]] = {}
    for episode_index, timeline in episode_timelines.items():
        ordered_frames = sorted(timeline)
        if ordered_frames != list(range(ordered_frames[-1] + 1)):
            raise B0GripperAuditError(
                f"episode {episode_index} does not have a complete gripper timeline"
            )
        transition_frames[episode_index] = tuple(
            frame_index
            for frame_index in ordered_frames[1:]
            if timeline[frame_index] != timeline[frame_index - 1]
        )

    signatures = [
        _conditioning_signature(cohort, sample_index) for sample_index in range(sample_count)
    ]
    signature_members: dict[str, list[int]] = defaultdict(list)
    for sample_index, signature in enumerate(signatures):
        signature_members[signature].append(sample_index)
    duplicate_signature_groups = {
        signature: members for signature, members in signature_members.items() if len(members) > 1
    }
    conflicting_condition_groups: list[dict[str, Any]] = []
    for signature, members in duplicate_signature_groups.items():
        for horizon_offset in range(8):
            labels = {int(target_signs[index, horizon_offset]) for index in members}
            if len(labels) > 1:
                conflicting_condition_groups.append(
                    {
                        "conditioning_signature": signature,
                        "horizon_offset": horizon_offset,
                        "labels": sorted(labels),
                        "sample_indices": members,
                    }
                )

    error_records: list[dict[str, Any]] = []
    all_transition_distance_counts: Counter[str] = Counter()
    error_transition_distance_counts: Counter[str] = Counter()
    horizon_error_counts: Counter[int] = Counter()
    episode_error_counts: Counter[int] = Counter()
    task_error_counts: Counter[int] = Counter()
    target_error_counts: Counter[int] = Counter()
    margin_bucket_counts: Counter[str] = Counter()
    crossed_transition_errors = 0
    unique_error_frames: set[tuple[int, int]] = set()

    for sample_index, identity in enumerate(cohort.identities):
        timeline = episode_timelines[identity.episode_index]
        transitions = transition_frames[identity.episode_index]
        anchor_sign = timeline[identity.anchor_frame_index]
        for horizon_offset in range(8):
            target_frame_index = identity.anchor_frame_index + horizon_offset
            distance = (
                min(abs(target_frame_index - frame) for frame in transitions)
                if transitions
                else None
            )
            distance_bucket = _transition_distance_bucket(distance)
            all_transition_distance_counts[distance_bucket] += 1
            if not error_mask[sample_index, horizon_offset]:
                continue
            target_sign = int(target_signs[sample_index, horizon_offset])
            predicted_sign = int(predicted_signs[sample_index, horizon_offset])
            logit = float(logits[sample_index, horizon_offset])
            crossed_transition = target_sign != anchor_sign
            crossed_transition_errors += int(crossed_transition)
            unique_error_frames.add((identity.episode_index, target_frame_index))
            horizon_error_counts[horizon_offset] += 1
            episode_error_counts[identity.episode_index] += 1
            task_error_counts[identity.task_index] += 1
            target_error_counts[target_sign] += 1
            error_transition_distance_counts[distance_bucket] += 1
            margin_bucket_counts[_margin_bucket(abs(logit))] += 1
            error_records.append(
                {
                    "sample_index": sample_index,
                    "sample_identity_digest": identity.digest,
                    "episode_index": identity.episode_index,
                    "task_index": identity.task_index,
                    "anchor_frame_index": identity.anchor_frame_index,
                    "horizon_offset": horizon_offset,
                    "target_frame_index": target_frame_index,
                    "target_sign": target_sign,
                    "predicted_sign": predicted_sign,
                    "gripper_logit": logit,
                    "signed_logit_margin": target_sign * logit,
                    "anchor_gripper_sign": anchor_sign,
                    "target_differs_from_anchor": crossed_transition,
                    "nearest_transition_distance_frames": distance,
                    "source_frame_overlap_occurrences": frame_occurrences[
                        (identity.episode_index, target_frame_index)
                    ],
                }
            )

    error_count = len(error_records)
    target_count = int(targets.size)
    error_absolute_logits = np.asarray(
        [abs(record["gripper_logit"]) for record in error_records], dtype=np.float64
    )
    error_quantiles = (
        {}
        if not error_count
        else {
            "minimum": float(error_absolute_logits.min()),
            "q25": float(np.quantile(error_absolute_logits, 0.25)),
            "median": float(np.quantile(error_absolute_logits, 0.5)),
            "q75": float(np.quantile(error_absolute_logits, 0.75)),
            "maximum": float(error_absolute_logits.max()),
        }
    )
    source_frame_records = []
    for episode_index, frame_index in sorted(unique_error_frames):
        matching = [
            record
            for record in error_records
            if record["episode_index"] == episode_index
            and record["target_frame_index"] == frame_index
        ]
        source_frame_records.append(
            {
                "episode_index": episode_index,
                "task_index": matching[0]["task_index"],
                "target_frame_index": frame_index,
                "target_sign": matching[0]["target_sign"],
                "error_slot_count": len(matching),
                "horizon_offsets": sorted(record["horizon_offset"] for record in matching),
                "nearest_transition_distance_frames": matching[0][
                    "nearest_transition_distance_frames"
                ],
            }
        )

    return {
        "cohort": {
            "sample_count": sample_count,
            "target_slot_count": target_count,
            "source_action_frame_count": len(frame_labels),
            "episode_count": len(episode_timelines),
            "gripper_transition_count": sum(len(value) for value in transition_frames.values()),
        },
        "label_integrity": {
            "target_encoding": "exactly--1-open-or-+1-close",
            "nonbinary_target_count": int(nonbinary_mask.sum()),
            "overlapping_chunk_source_frame_count": sum(
                count > 1 for count in frame_occurrences.values()
            ),
            "overlapping_chunk_label_contradiction_count": len(contradictory_frames),
            "contradictory_source_frames": [
                {"episode_index": episode, "frame_index": frame}
                for episode, frame in contradictory_frames
            ],
            "unique_conditioning_signature_count": len(signature_members),
            "duplicate_conditioning_signature_group_count": len(duplicate_signature_groups),
            "exact_conditioning_conflicting_label_group_count": len(
                conflicting_condition_groups
            ),
            "exact_conditioning_conflicting_label_groups": conflicting_condition_groups,
        },
        "residual_errors": {
            "error_slot_count": error_count,
            "correct_slot_count": target_count - error_count,
            "accuracy": (target_count - error_count) / target_count,
            "error_rate": error_count / target_count,
            "unique_source_action_frame_count": len(unique_error_frames),
            "repeated_overlap_error_slot_count": error_count - len(unique_error_frames),
            "target_sign_counts": _sorted_counts(target_error_counts),
            "horizon_offset_counts": _sorted_counts(horizon_error_counts),
            "episode_counts": _sorted_counts(episode_error_counts),
            "task_counts": _sorted_counts(task_error_counts),
            "target_differs_from_anchor_count": crossed_transition_errors,
            "nearest_transition_distance_counts": dict(
                sorted(error_transition_distance_counts.items())
            ),
            "absolute_logit_bucket_counts": dict(sorted(margin_bucket_counts.items())),
            "absolute_logit_quantiles": error_quantiles,
            "records": error_records,
            "source_action_frames": source_frame_records,
        },
        "comparison": {
            "all_target_nearest_transition_distance_counts": dict(
                sorted(all_transition_distance_counts.items())
            ),
        },
    }

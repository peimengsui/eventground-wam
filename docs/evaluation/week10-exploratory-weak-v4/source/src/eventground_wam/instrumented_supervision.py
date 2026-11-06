"""Week 10 v2 recording identities and conservative partition admission.

No event extraction or training is performed here. Unmapped source recordings
and cross-partition aliases cannot enter a production dataset.
"""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

import numpy as np

from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract


def numeric_digest(*arrays):
    result = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f4")
        result.update(str(value.shape).encode() + b"\0")
        result.update(value.tobytes())
    return result.hexdigest()


def physical_source_id(repository, revision, relative_path, demo_id):
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or not relative_path.endswith(".hdf5"):
        raise ValueError("Require a repository-relative HDF5 path")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Require pinned source revision")
    if not demo_id.startswith("demo_") or not demo_id[5:].isdigit():
        raise ValueError("Invalid demonstration ID")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "namespace": "week10-physical-source-v1",
                "repository": repository,
                "revision": revision,
                "path": relative_path,
                "demo_id": demo_id,
            }
        )
    ).hexdigest()


def resolve_partition(bindings):
    """Resolve known exact full-sequence aliases; never decide by majority."""
    if not bindings:
        return {"partition": "quarantine", "reason": "no_verified_source_binding"}
    if any(not row["exact_action_identity"] for row in bindings):
        return {"partition": "quarantine", "reason": "approximate_or_ambiguous_action_identity"}
    allowed = {"train", "event-dev", "event-report", "test"}
    partitions = {row["partition"] for row in bindings}
    if not partitions <= allowed:
        raise ValueError("Unknown inherited partition")
    if len(partitions) != 1:
        return {"partition": "quarantine", "reason": "source_aliases_cross_partitions"}
    return {"partition": next(iter(partitions)), "reason": "all_known_exact_aliases_agree"}


def labeler_contract():
    """Engineering rules frozen before label extraction, not validated quality."""
    return seal_contract(
        {
            "artifact_type": "eventground_wam.instrumented_event_labeler_rules",
            "version": "week10-instrumented-labeler-v1",
            "fps": 20,
            "status": "rules_frozen_quality_unvalidated_extractor_not_yet_implemented",
            "evidence": "new synchronized recordings only; no labels transferred to Sylvest videos",
            "objects": {
                "include": "objects_dict entries with a free joint on their root body",
                "exclude": [
                    "fixtures",
                    "table",
                    "robot",
                    "self-contact",
                    "fixed/articulated fixture parts",
                ],
                "all_movable_objects_included": True,
                "wrong_object_contacts_retained": True,
                "identity": "source_id plus exact model object root body; geom ownership by ancestry",
                "unmapped_or_multiply_owned_geom": "quarantine affected frames; never infer by proximity",
                "gripper": "collision geoms under the exact gripper root body",
                "finger_sides": "Panda important_geoms left_finger and right_finger sets",
            },
            "contact": {
                "raw_presence": "gripper-object geom pair with dist <= 0 m and efc_address >= 0",
                "distance_maximum_m": 0.0,
                "minimum_consecutive_frames": 2,
                "onset": "first frame of each qualifying contiguous contact run, per object",
                "end_exclusive": "first following frame without raw presence for that object",
                "gap_filling_frames": 0,
                "single_frame_run": "ambiguous candidate, not a negative",
                "positive_margin_contacts": "proximity only, not physical contact",
                "forces_required": False,
            },
            "grasp": {
                "candidate_onset": "earliest frame of a five-frame confirmation window",
                "confirmation_window_frames": 5,
                "window_conditions": [
                    "close command +1 at every frame",
                    "at least one finger contacts the same movable object at every frame",
                    "both finger sides contact that object on at least three of five frames",
                    "object position relative to grip-site frame stays within 5 mm of window start",
                    "object orientation relative to grip-site frame stays within 10 degrees of window start",
                    "grip site moves at least 10 mm OR rotates at least 10 degrees from window start",
                ],
                "minimum_bilateral_frames": 3,
                "relative_position_maximum_m": 0.005,
                "relative_rotation_maximum_degrees": 10.0,
                "minimum_gripper_displacement_m": 0.010,
                "minimum_gripper_rotation_degrees": 10.0,
                "motion_measure": "maximum displacement from window start over frames t..t+4",
                "rotation_measure": "SO(3) geodesic angle; no quaternion-sign dependence",
                "confirmation_frame": "onset + 4; offline target metadata only",
                "continuation": "after confirmation, stationary holds remain grasp while one finger contacts object and reference-relative pose stays within exit limits",
                "end_exclusive": "first frame losing all finger contact OR exceeding either exit pose limit; no bridging; reference is accepted onset",
                "exit_relative_position_maximum_m": 0.010,
                "exit_relative_rotation_maximum_degrees": 20.0,
                "failed_or_incomplete_confirmation": "ambiguous grasp candidate; not automatically none",
                "candidate_overlap": "earliest confirmed onset wins per object until grasp ends; retain diagnostics",
                "closure_alone_is_grasp": False,
            },
            "gripper_close": {
                "semantics": "retain existing command-transition plus aperture-response rule",
                "response_window_frames": 5,
                "minimum_close_command_frames": 5,
                "minimum_aperture_drop_m": 0.001,
                "old_human_QA_transfers_to_new_recordings": False,
            },
            "boundaries": {
                "intervals": "integer half-open [start,end) at 20 Hz",
                "present_at_first_frame": "left-censored, not a newly observed onset",
                "ongoing_at_last_frame": "right-censored, end=episode length",
                "tail_without_confirmation": "ambiguous; never convert to none",
                "single_point_or_corrupted_evidence": "exclude affected anchor windows with reasons",
            },
            "prompt_rules": {
                "window_frames": 32,
                "next_event": "earliest admitted onset strictly after anchor",
                "tie_order": ["grasp", "contact", "gripper_close", "stable_object_id"],
                "none": "complete 32-frame future window with no admitted onset and no unresolved candidate",
                "inside_ongoing_event": "exclude anchor",
                "incomplete_tail": "exclude anchor",
                "maximum_anchors_per_event": 4,
                "maximum_prompts_per_source_group": 16,
                "physical_evidence_is_model_input": False,
            },
            "quality_gate": {
                "independent_full_episode_review": True,
                "missed_events_and_negatives_reviewed": True,
                "minimum_train_episodes": 200,
                "coverage_only_maximum": 300,
                "all_tasks": 40,
                "minimum_positives_per_new_type": 100,
                "per_type_precision_minimum": 0.95,
                "per_type_recall_minimum": 0.95,
                "mean_start_and_end_boundary_error_maximum_frames": 2,
                "failure": "stop; changes require a new predeclared version, preserve failed QA",
            },
            "label_extraction_performed": False,
            "labels_admitted": False,
        }
    )

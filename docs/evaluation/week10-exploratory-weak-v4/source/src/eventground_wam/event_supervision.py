"""Week 10 supervision preparation; neither selection nor alignment admits labels.

Keep this separate from the immutable Week 2 extractor and Week 10 study pins.
All selection uses group metadata only. Physical scene identity, a sealed labeler
and independent full-episode review remain separate prerequisites.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract

STUDY_DIGEST = "91447b7ab551231953791290545634c548b926b566be0bab9751541c76f33361"
SPLIT_DIGEST = "8607f32f0ed0ef6cd27ea99449a8de0237e23a4eee973ee970e06bbf3fca115a"
STRATA = ("short", "middle", "long")


def _key(*parts: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(["week10-supervision-v1", 10, *parts])).hexdigest()


def preparation_contract() -> dict[str, Any]:
    return seal_contract(
        {
            "artifact_type": "eventground_wam.event_supervision_preparation",
            "schema_version": 1,
            "parent_study_digest": STUDY_DIGEST,
            "split_digest": SPLIT_DIGEST,
            "scope": "source_metadata_audit_and_prospective_train_only_QA_selection",
            "selection": {
                "seed": 10,
                "initial_episodes": 200,
                "reserve_episodes": 100,
                "qualification_episodes": 40,
                "canonical_tasks": 40,
                "groups_per_episode": 1,
                "episodes_per_group": 1,
                "order": "task round robin; least represented within-task length tercile; hash ties",
                "representative": "minimum hash(seed,group_digest,episode_index)",
                "uses_events_or_outcomes": False,
                "reserve_use": "coverage_only_not_quality_gate_tuning",
                "missing_event_type_or_negative_coverage": "block_or_use_bounded_reserve",
            },
            "numeric_alignment_screen": {
                "actions_atol": 1e-6,
                "projected_proprioception_atol": 1e-5,
                "rtol": 0,
                "exact_gripper_commands": True,
                "requires_full_episode": True,
                "source_frames": "strictly_increasing_contiguous_integers_no_time_warp",
                "approximate_action_replay_accepted": False,
                "matching_numbers_prove_scene_identity": False,
            },
            "resources": {
                "metadata_download_limit_bytes": 32 * 1024**2,
                "cpu_wall_hours_maximum": 2,
                "active_gpu_hours": 0,
                "simulation_steps": 0,
                "human_annotation_hours": 0,
                "episode_payload_downloads": 0,
                "test_and_validation_payloads_opened": 0,
                "minimum_free_bytes": 50 * 1024**3,
            },
            "not_authorized_by_this_child": [
                "label extraction before physical-source and predicate contracts",
                "human review before source/labeler qualification and annotation budget",
                "held-out payload access",
                "training",
                "automatic label promotion",
            ],
            "remaining_gates": [
                "original episode/frame mapping and physical scene identity",
                "pinned full simulator states/model/task/assets and observation phase",
                "sealed contact/grasp labeler and reconstruction/annotation budgets",
                "independent full-episode review including misses and negative cases",
                ">=200 episodes and >=100 reviewed positives per new type",
                "per-type precision/recall >=.95; mean start/end errors <=2 frames",
            ],
        }
    )


def build_review_queue(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Prospective 200 + 100 training episodes, without opening any payload."""
    if not groups:
        raise ValueError("Empty groups")
    episode_ids: set[int] = set()
    group_ids: set[str] = set()
    tasks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        gid = group["group_digest"]
        if (
            not isinstance(gid, str)
            or len(gid) != 64
            or any(c not in "0123456789abcdef" for c in gid)
            or gid in group_ids
        ):
            raise ValueError("Invalid or duplicate group digest")
        group_ids.add(gid)
        if group["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        ids = group["episode_indices"]
        if not ids or any(type(i) is not int or i < 0 for i in ids):
            raise ValueError("Invalid episode IDs")
        if len(set(ids)) != len(ids) or episode_ids.intersection(ids):
            raise ValueError("Episode occurs in multiple groups")
        episode_ids.update(ids)
        if type(group["episode_length"]) is not int or group["episode_length"] < 33:
            raise ValueError("Invalid group length")
        if group["split"] == "train":
            tasks[group["task_key"]].append(group)
    if len(tasks) != 40:
        raise ValueError("Require all 40 training tasks")
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task, values in tasks.items():
        if len(values) < 8:
            raise ValueError("Too few distinct training groups for review and reserve")
        bins: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATA}
        ordered = sorted(values, key=lambda g: (g["episode_length"], g["group_digest"]))
        for rank, group in enumerate(ordered):
            bins[STRATA[min(2, 3 * rank // len(ordered))]].append(group)
        for values in bins.values():
            values.sort(key=lambda g: (_key("group", g["group_digest"]), g["group_digest"]))
        pools[task] = bins
    counts: dict[str, Counter] = defaultdict(Counter)
    rows = []
    task_order = sorted(tasks, key=lambda t: (_key("task", t), t))
    while len(rows) < 300:
        for task in task_order:
            available = [s for s in STRATA if pools[task][s]]
            if not available:
                raise ValueError("Insufficient distinct groups")
            stratum = min(available, key=lambda s: (counts[task][s], _key("stratum", task, s)))
            group = pools[task][stratum].pop(0)
            gid = group["group_digest"]
            episode = min(group["episode_indices"], key=lambda e: (_key("episode", gid, e), e))
            rows.append(
                {
                    "queue_rank": len(rows),
                    "source_episode_index": episode,
                    "group_digest": gid,
                    "split": "train",
                    "task_key": task,
                    "frame_count": group["episode_length"],
                    "length_stratum": stratum,
                    "stage": "initial" if len(rows) < 200 else "coverage_reserve",
                    "qualification_candidate": len(rows) < 40,
                    "review_status": "not_ready_physical_evidence_pending",
                }
            )
            counts[task][stratum] += 1
            if len(rows) == 300:
                break
    return seal_contract(
        {
            "artifact_type": "eventground_wam.event_supervision_review_queue",
            "schema_version": 1,
            "preparation_digest": preparation_contract()["digest"],
            "parent_study_digest": STUDY_DIGEST,
            "parent_split_digest": SPLIT_DIGEST,
            "rows": rows,
            "initial_episodes": 200,
            "reserve_episodes": 100,
            "payloads_opened": 0,
            "labels_generated": 0,
            "initial_task_counts": dict(sorted(Counter(r["task_key"] for r in rows[:200]).items())),
            "initial_length_counts": dict(
                sorted(Counter(r["length_stratum"] for r in rows[:200]).items())
            ),
            "events_and_negative_case_coverage": "unknown_until_qualified_independent_review",
        }
    )


def compare_recorded_trajectory(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    source_states: np.ndarray,
    source_actions: np.ndarray,
    source_frames: np.ndarray,
    evidence_origin: str,
) -> dict[str, Any]:
    """Full-episode numeric screen only; passing never establishes physical truth.

    source_states must be the explicitly documented projection into the pinned
    8-D proprioceptive schema. No resampling, learned mapping, replay or tolerance
    fitting is performed here. Input identity must be checked by the caller.
    """
    if evidence_origin != "recorded_simulator_states":
        raise ValueError("Require original recorded states; replay is not ground truth")
    states, actions, source_states, source_actions = (
        np.asarray(x, dtype=np.float64) for x in (states, actions, source_states, source_actions)
    )
    n = len(states)
    if (
        n < 1
        or states.shape != (n, 8)
        or actions.shape != (n, 7)
        or source_states.shape != states.shape
        or source_actions.shape != actions.shape
    ):
        raise ValueError("Require aligned complete [T,8] / [T,7] trajectories")
    if not all(np.isfinite(x).all() for x in (states, actions, source_states, source_actions)):
        raise ValueError("Nonfinite evidence")
    frames = np.asarray(source_frames)
    if (
        frames.shape != (n,)
        or frames.dtype.kind not in "iu"
        or np.any(frames < 0)
        or np.any(frames > np.iinfo(np.int64).max)
        or not np.all(np.diff(frames.astype(np.int64)) == 1)
    ):
        raise ValueError("Frame omissions, repeats or time warps require a new mapping audit")
    if not set(np.unique(actions[:, 6])) <= {-1.0, 1.0}:
        raise ValueError("Invalid pinned gripper commands")
    action_error = np.abs(actions - source_actions)
    state_error = np.abs(states - source_states)
    gripper_equal = bool(np.array_equal(actions[:, 6], source_actions[:, 6]))
    passed = bool(np.all(action_error <= 1e-6) and np.all(state_error <= 1e-5) and gripper_equal)
    return seal_contract(
        {
            "artifact_type": "eventground_wam.event_numeric_alignment_screen",
            "schema_version": 1,
            "preparation_digest": preparation_contract()["digest"],
            "frame_count": n,
            "source_first_frame": int(frames[0]),
            "source_last_frame": int(frames[-1]),
            "max_action_absolute_error": float(action_error.max()),
            "max_state_absolute_error": float(state_error.max()),
            "gripper_commands_exact": gripper_equal,
            "numeric_alignment_passed": passed,
            "physical_scene_identity_verified": False,
            "contact_grasp_labels_admitted": False,
        }
    )

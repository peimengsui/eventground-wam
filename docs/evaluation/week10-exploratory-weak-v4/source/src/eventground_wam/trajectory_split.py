"""Outcome-independent numeric trajectory groups for future fresh training.

This is a separate v2 split format. It never substitutes for the pinned Week 2
split accepted by the existing B0 training loader.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from eventground_wam.b0_diagnostics import numeric_digest
from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract

SPLITS = ("train", "validation", "test")
SPLIT_ID = "libero-plus-trajectory-v2"
NEAR_ATOL = 1e-5


@dataclass(frozen=True)
class NumericTrajectory:
    episode_index: int
    task_key: str
    states: np.ndarray
    actions: np.ndarray

    def __post_init__(self) -> None:
        if type(self.episode_index) is not int or self.episode_index < 0:
            raise ValueError("episode index must be a nonnegative integer")
        if not self.task_key:
            raise ValueError("task key must be nonempty")
        if (self.states.ndim != 2 or self.states.shape[1] != 8
                or self.actions.shape != (len(self.states), 7) or len(self.states) < 8):
            raise ValueError("require aligned [frames,8] state and [frames,7] actions")
        if self.states.dtype != np.float32 or self.actions.dtype != np.float32:
            raise ValueError("source numeric arrays must be float32")
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError("nonfinite source array")
        if not set(np.unique(self.actions[:, 6])) <= {-1., 1.}:
            raise ValueError("source gripper must be exactly -1/+1")


def split_rule() -> dict[str, Any]:
    return seal_contract({
        "artifact_type": "eventground_wam.trajectory_split_rule", "schema_version": 1,
        "split_id": SPLIT_ID, "namespace": SPLIT_ID,
        "grouping": {
            "exact_identity": "SHA256 of canonical full float32 state/action arrays, with shapes",
            "additional_conservative_edge": "Identical complete action sequence, regardless of state",
            "near_identity": "Equal-length sequences; all raw state and action components differ "
                             "by at most absolute tolerance, rtol=0; identical gripper sequence",
            "near_absolute_tolerance": NEAR_ATOL,
            "closure": "Connected components of exact, action-only and near edges; transitive",
            "images_used": False, "outcomes_used": False,
            "cross_task_component": "Abort if any component spans canonical tasks",
            "group_digest": "SHA256 of split_id, task_key and sorted distinct state_action digests",
        },
        "assignment": {
            "stratum": "canonical task", "ratios": {"train": .8, "validation": .1, "test": .1},
            "unit": "trajectory component, not episode, frame or visual variant",
            "order": "Ascending SHA256 of canonical JSON {namespace,task_key,group_digest}; "
                     "group_digest tie-break",
            "quotas": "Largest remainder with integer weights 8,1,1; ties train,validation,test",
            "slice_order": list(SPLITS), "minimum_groups_per_task": 10,
            "rerolls": 0,
        },
        "source_scope": "All 14347 episodes at the pinned Sylvest snapshot; old test arrays "
                        "read only for structural grouping, not policy evaluation",
        "usage": {
            "initialization": "Pinned base Qwen3.5-2B plus fresh trainable modules/LoRA and optimizer; "
                              "no B0-derived policy, optimizer, teacher or distilled targets",
            "statistics": "Fit all action and state normalization on v2 train only",
            "validation": "Descriptive supervised diagnostics; report task macro averages, then "
                          "equal trajectory-group means within task, equal variant means within "
                          "group and equal valid-chunk means within episode; also report micro totals",
            "selection": "Future experiment must predeclare checkpoints and rollout-based decisions; "
                         "this split does not define a training recipe or acceptance rule",
            "test": "Lock after structural verification; no model scoring or tuning until final "
                    "predeclared evaluation",
            "old_models": "v2 validation/test are not unseen data for existing B0 or its continuations",
        },
        "limitations": [
            "Same 40 tasks; tests trajectory generalization, not unseen tasks",
            "No original demonstration/reset IDs available",
            "Near check covers equal-length aligned sequences only, not trims, shifts or time warps",
            "Grouping does not prove independent simulator scenes or eliminate all semantic duplicates",
        ],
    })


def split_quotas(count: int) -> dict[str, int]:
    if count < 10:
        raise ValueError("need at least ten trajectory groups per task")
    weights = (8, 1, 1)
    values = [count * weight // 10 for weight in weights]
    remainder_order = sorted(range(3), key=lambda i: (-(count * weights[i] % 10), i))
    for i in remainder_order[:count - sum(values)]:
        values[i] += 1
    return dict(zip(SPLITS, values, strict=True))


def build_trajectory_groups(
    episodes: list[NumericTrajectory],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered = sorted(episodes, key=lambda e: e.episode_index)
    if not ordered or len({e.episode_index for e in ordered}) != len(ordered):
        raise ValueError("require nonempty unique episode indices")
    parents = list(range(len(ordered)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    exact: dict[str, int] = {}
    action_only: dict[str, int] = {}
    digests = []
    for i, episode in enumerate(ordered):
        full = numeric_digest(episode.states, episode.actions)
        action = numeric_digest(episode.actions)
        digests.append(full)
        union(i, exact.setdefault(full, i))
        union(i, action_only.setdefault(action, i))

    # Compare every distinct exact sequence of equal length, even across tasks.
    # Do not skip already-connected representatives: retain all direct near edges.
    buckets = defaultdict(list)
    for i in exact.values():
        buckets[len(ordered[i].states)].append(i)
    near_edges = []
    tested = 0
    for indices in buckets.values():
        for offset, i in enumerate(indices):
            a = ordered[i]
            for j in indices[offset + 1:]:
                b = ordered[j]
                tested += 1
                if (np.array_equal(a.actions[:, 6], b.actions[:, 6])
                        and np.allclose(a.states, b.states, atol=NEAR_ATOL, rtol=0)
                        and np.allclose(a.actions, b.actions, atol=NEAR_ATOL, rtol=0)):
                    union(i, j)
                    near_edges.append([a.episode_index, b.episode_index])

    components = defaultdict(list)
    for i in range(len(ordered)):
        components[find(i)].append(i)
    groups = []
    for indices in components.values():
        tasks = {ordered[i].task_key for i in indices}
        if len(tasks) != 1:
            raise ValueError(f"trajectory component crosses tasks: {sorted(tasks)}")
        task = next(iter(tasks))
        lengths = {len(ordered[i].states) for i in indices}
        if len(lengths) != 1:
            raise ValueError("component contains unequal sequence lengths")
        payload = {"split_id": SPLIT_ID, "task_key": task,
                   "numeric_digests": sorted({digests[i] for i in indices})}
        group_digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        groups.append({"group_digest": group_digest, "task_key": task,
                       "episode_indices": [ordered[i].episode_index for i in indices],
                       "episode_length": next(iter(lengths)),
                       "numeric_digests": payload["numeric_digests"]})
    by_task = defaultdict(list)
    for group in groups:
        by_task[group["task_key"]].append(group)
    for task, selected in by_task.items():
        def rank(group, task=task):
            payload = {"namespace": SPLIT_ID, "task_key": task,
                       "group_digest": group["group_digest"]}
            return hashlib.sha256(canonical_json_bytes(payload)).hexdigest(), group["group_digest"]
        selected.sort(key=rank)
        start = 0
        for split, count in split_quotas(len(selected)).items():
            for group in selected[start:start + count]:
                group["split"] = split
            start += count
    groups.sort(key=lambda g: (g["task_key"], g["group_digest"]))
    return groups, {
        "exact_state_action_sequences": len(exact), "exact_action_sequences": len(action_only),
        "connected_components": len(groups), "near_candidate_pairs_tested": tested,
        "near_edges_between_distinct_exact_sequences": near_edges,
    }


def expand_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assignments = []
    seen = set()
    for group in groups:
        if group["split"] not in SPLITS:
            raise ValueError("invalid split")
        if group["episode_length"] < 8 or not group["episode_indices"]:
            raise ValueError("empty or too short group")
        for episode in group["episode_indices"]:
            if episode in seen:
                raise ValueError("episode assigned more than once")
            seen.add(episode)
            assignments.append({
                "episode_index": episode, "task_key": group["task_key"],
                "split": group["split"], "group_digest": group["group_digest"],
                "episode_length": group["episode_length"],
                "valid_action_chunks": group["episode_length"] - 7,
            })
    return sorted(assignments, key=lambda r: r["episode_index"])

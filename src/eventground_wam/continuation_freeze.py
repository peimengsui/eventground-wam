"""Pure planning and decision helpers for the frozen Week 4 continuation pilot.

No model loading, data decoding, optimizer updates or rollout execution occurs here.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract

EXPERIMENT_ID = "week4-oft-continuation-v1"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SCREEN_UPDATES = 15_000
MAX_UPDATES = 60_000
EVALUATION_UPDATES = (15_000, 30_000, 45_000, 60_000)
MIX_WINDOW_UPDATES = 128


def build_development_panel(parent: dict[str, Any]) -> dict[str, Any]:
    """Select five states per task without consulting policy outcomes."""
    rows = []
    for suite in SUITES:
        for task in range(10):
            candidates = [p for p in parent["pairs"] if p["suite"] == suite
                          and p["runtime_task_index"] == task]
            if (len(candidates) != 50
                    or {p["initial_state_index"] for p in candidates} != set(range(50))):
                raise ValueError("parent must contain all 50 states for every task")
            ranked = sorted(candidates, key=lambda p: hashlib.sha256(
                f"{EXPERIMENT_ID}|{p['pair_id']}".encode()).hexdigest())[:5]
            rows.extend(sorted(ranked, key=lambda p: p["initial_state_index"]))
    return seal_contract({
        "artifact_type": "eventground_wam.continuation_development_panel",
        "schema_version": 1, "experiment_id": EXPERIMENT_ID,
        "parent_panel_digest": parent["digest"], "pair_count": 200,
        "selection": "Lowest five SHA256(experiment_id|pair_id) per task; numeric state order",
        "interpretation": "Previously inspected development states; not an unseen holdout",
        "pairs": rows,
    })


def mix_batches(batches: list[list[tuple[int, int, int]]]) -> list[list[tuple[int, int, int]]]:
    """Regroup identical (epoch, episode, anchor) tokens, retaining every batch size.

    Windows never cross a 15k-update evaluation boundary. This keeps exposure
    identical at each decision point; sorting by SHA256 avoids RNG-version drift.
    """
    if len(batches) > MAX_UPDATES or any(not b or len(b) > 32 for b in batches):
        raise ValueError("invalid continuation batch plan")
    mixed = []
    for stage_start in range(0, len(batches), SCREEN_UPDATES):
        stage_end = min(stage_start + SCREEN_UPDATES, len(batches))
        for start in range(stage_start, stage_end, MIX_WINDOW_UPDATES):
            block = batches[start:min(start + MIX_WINDOW_UPDATES, stage_end)]
            tokens = [t for batch in block for t in batch]
            if len(set(tokens)) != len(tokens):
                raise ValueError("duplicate training presentation within a mixing window")
            tokens.sort(key=lambda t: (hashlib.sha256(
                f"{EXPERIMENT_ID}|{start}|{t[0]}|{t[1]}|{t[2]}".encode()).digest(), t))
            offset = 0
            for batch in block:
                mixed.append(tokens[offset:offset + len(batch)])
                offset += len(batch)
    return mixed


def batches_digest(batches: list[list[tuple[int, int, int]]]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(canonical_json_bytes(batch) + b"\n")
    return digest.hexdigest()


def exposure_digest(batches: list[list[tuple[int, int, int]]]) -> str:
    """Order-independent multiset identity, including repeated epoch presentations."""
    digest = hashlib.sha256()
    for token, count in sorted(Counter(t for b in batches for t in b).items()):
        digest.update(canonical_json_bytes([token, count]) + b"\n")
    return digest.hexdigest()


def outcome_counts(panel: dict[str, Any], outcomes: dict[str, bool]) -> dict[str, int]:
    expected = {p["pair_id"] for p in panel["pairs"]}
    if (set(outcomes) != expected or len(expected) != 200
            or any(type(x) is not bool for x in outcomes.values())):
        raise ValueError("require exactly 200 identified boolean outcomes")
    counts = {suite: 0 for suite in SUITES}
    for pair in panel["pairs"]:
        counts[pair["suite"]] += outcomes[pair["pair_id"]]
    counts["overall"] = sum(counts.values())
    return counts


def assess_screen(
    panel: dict[str, Any], parent: dict[str, bool], candidate: dict[str, bool],
    *, valid: bool,
) -> dict[str, Any]:
    """Engineering extension criterion, not the Week 4 acceptance test."""
    old, new = outcome_counts(panel, parent), outcome_counts(panel, candidate)
    delta = {key: new[key] - old[key] for key in old}
    goal_task_deltas = [sum(
        int(candidate[p["pair_id"]]) - int(parent[p["pair_id"]])
        for p in panel["pairs"]
        if p["suite"] == "libero_goal" and p["runtime_task_index"] == task
    ) for task in range(10)]
    checks = {
        "valid_complete_evidence": valid is True,
        "goal_at_least_five_additional_successes": delta["libero_goal"] >= 5,
        "goal_improves_on_at_least_three_tasks": sum(d > 0 for d in goal_task_deltas) >= 3,
        "overall_does_not_decline": delta["overall"] >= 0,
        "object_loses_at_most_two": delta["libero_object"] >= -2,
        "long_loses_at_most_two": delta["libero_10"] >= -2,
        "spatial_loses_at_most_three": delta["libero_spatial"] >= -3,
    }
    return {"eligible_to_extend": all(checks.values()), "checks": checks,
            "counts": new, "delta_counts": delta, "goal_task_deltas": goal_task_deltas}


def select_arm(assessments: dict[str, dict[str, Any]]) -> str | None:
    """Prefer total success, then Goal, then LIBERO-10; exact ties prefer A."""
    if set(assessments) != {"A", "B"}:
        raise ValueError("both fixed arms must be assessed before selecting")
    eligible = [arm for arm in ("A", "B") if assessments[arm]["eligible_to_extend"]]
    if not eligible:
        return None
    return max(eligible, key=lambda arm: (
        assessments[arm]["counts"]["overall"],
        assessments[arm]["counts"]["libero_goal"],
        assessments[arm]["counts"]["libero_10"], arm == "A",
    ))

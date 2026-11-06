"""Frozen paired clean-comparison panel and statistical decision calculation.

This module does not launch policies or certify rollout provenance. A statistical
decision becomes an acceptance decision only after the rule's artifact checks.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
COMPARISON_ID = "week4-clean-comparison-v1"
RULE_ID = "week4-clean-acceptance-v1"
TRIALS = 50
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 20_260_917
MARGIN_PP = -5.0


def read_sealed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("digest") != contract_digest(value):
        raise ValueError(f"invalid sealed record: {path}")
    return value


def pair_seed(suite: str, task_index: int, trial_index: int) -> int:
    if suite not in SUITES or not 0 <= task_index < 10 or not 0 <= trial_index < TRIALS:
        raise ValueError("pair identity is outside the frozen panel")
    payload = f"7|{COMPARISON_ID}|{suite}|{task_index}|{trial_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647 or 1


def build_panel(inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("digest") != contract_digest(inventory):
        raise ValueError("initial-state inventory digest mismatch")
    expected_tasks = [(suite, index) for suite in SUITES for index in range(10)]
    actual_tasks = [(x["suite"], x["runtime_task_index"]) for x in inventory["tasks"]]
    if actual_tasks != expected_tasks:
        raise ValueError("initial-state inventory must contain the ordered canonical 40 tasks")
    pairs = []
    for task in inventory["tasks"]:
        hashes = task["initial_state_sha256_by_index"]
        if (len(hashes) != TRIALS or len(set(hashes)) != TRIALS
                or task["initial_state_count"] != TRIALS):
            raise ValueError("every task must have exactly 50 distinct canonical initial states")
        for trial, state_hash in enumerate(hashes):
            suite, task_index = task["suite"], task["runtime_task_index"]
            pairs.append({
                "pair_id": f"{suite}/task-{task_index:02d}/state-{trial:02d}",
                "suite": suite, "runtime_task_index": task_index,
                "classification_task_id": task["classification_task_id"],
                "task_name": task["task_name"], "instruction": task["instruction"],
                "trial_index": trial, "initial_state_index": trial,
                "initial_state_sha256": state_hash,
                "initial_states_file_sha256": task["initial_states_file_sha256"],
                "bddl_sha256": task["bddl_sha256"],
                "episode_seed": pair_seed(suite, task_index, trial),
            })
    return seal_contract({
        "artifact_type": "eventground_wam.clean_comparison_panel", "schema_version": 1,
        "comparison_id": COMPARISON_ID, "acceptance_rule_id": RULE_ID,
        "source_manifest_digest": inventory["source_manifest_digest"],
        "initial_state_inventory_digest": inventory["digest"],
        "benchmark_revision": inventory["benchmark_revision"],
        "suite_order": list(SUITES), "task_count": 40, "trials_per_task": TRIALS,
        "pair_count": len(pairs), "episode_count_per_policy": len(pairs),
        "environment_seed": 0, "base_episode_seed": 7,
        "seed_payload": "7|week4-clean-comparison-v1|suite|runtime_task_index|trial_index",
        "reuse_clean40_rollouts": False, "previously_observed_initial_state_pairs": 40,
        "pairs": pairs,
    })


def classify_interval(lower_pp: float, upper_pp: float) -> str:
    if not (math.isfinite(lower_pp) and math.isfinite(upper_pp)
            and -100 <= lower_pp <= upper_pp <= 100):
        raise ValueError("invalid success-gap interval")
    if lower_pp >= MARGIN_PP:
        return "PASS"
    if upper_pp < MARGIN_PP:
        return "FAIL"
    return "INCONCLUSIVE"


def analyze_paired_outcomes(b0: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Compute the locked statistic on 4 suites × 10 tasks × 50 paired booleans.

    Resample ten task clusters within each suite with replacement, retaining all
    fifty outcome pairs within each selected task. Keep suite weights equal.
    This approximate interval measures task-cluster variation, not training-seed
    uncertainty or a guarantee about unseen tasks. Never use it on a partial run.
    """
    b0, reference = np.asarray(b0), np.asarray(reference)
    if (b0.shape != (4, 10, TRIALS) or reference.shape != b0.shape
            or b0.dtype != np.bool_ or reference.dtype != np.bool_):
        raise ValueError("expected complete paired boolean arrays with shape (4,10,50)")
    task_gaps = b0.sum(axis=2, dtype=np.int64) - reference.sum(axis=2, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    suite_draws = []
    for suite_index in range(4):
        indices = rng.integers(0, 10, size=(BOOTSTRAP_REPLICATES, 10), dtype=np.int64)
        suite_draws.append(task_gaps[suite_index][indices].sum(axis=1))
    draws = np.stack(suite_draws, axis=1)
    # Quantile on integer discordant-success counts first avoids accumulating
    # per-task floating errors at the exact -5 percentage point boundary.
    overall_interval = np.quantile(draws.sum(axis=1), [0.025, 0.975], method="linear") / 20
    lower, upper = map(float, overall_interval)
    return {
        "acceptance_rule_id": RULE_ID, "pair_count": int(b0.size),
        "b0_success_count": int(b0.sum()), "reference_success_count": int(reference.sum()),
        "b0_success_rate": float(b0.mean()), "reference_success_rate": float(reference.mean()),
        "gap_percentage_points": 100 * (int(b0.sum()) - int(reference.sum())) / int(b0.size),
        "paired_cluster_bootstrap_95_percent_pp": [lower, upper],
        "statistical_decision": classify_interval(lower, upper),
        "artifact_validity_checks_required_before_acceptance": True,
        "suite_results": {
            suite: {
                "b0_success_count": int(b0[index].sum()),
                "reference_success_count": int(reference[index].sum()),
                "gap_percentage_points": 100 * (
                    int(b0[index].sum()) - int(reference[index].sum())) / 500,
                "paired_cluster_bootstrap_95_percent_pp": [
                    float(x) for x in np.quantile(
                        draws[:, index], [0.025, 0.975], method="linear",
                    ) / 5
                ],
            } for index, suite in enumerate(SUITES)
        },
    }

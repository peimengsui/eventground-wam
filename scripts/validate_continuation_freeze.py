#!/usr/bin/env python3
"""Validate the next B0 experiment's frozen inputs without training or rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eventground_wam.clean_acceptance import read_sealed
from eventground_wam.continuation_freeze import build_development_panel, outcome_counts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/training/week4_b0_oft_continuation_v1.json"
FREEZE_DIGEST = "745875e937f96c39b6297334276eec3eee47bfebd4630abb077fb9d45657fa86"


def validate_freeze(root: Path = ROOT) -> dict[str, object]:
    rule = read_sealed(root / CONFIG)
    if rule["digest"] != FREEZE_DIGEST:
        raise ValueError("continuation freeze changed; record a new version")
    documents = {}
    for label, pin in rule["sources"].items():
        path = root / pin["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin["sha256"]:
            raise ValueError(f"frozen source bytes changed: {label}")
        if "digest" in pin:
            documents[label] = read_sealed(path)
            if documents[label]["digest"] != pin["digest"]:
                raise ValueError(f"frozen source digest changed: {label}")
    panel = documents["development_panel"]
    if build_development_panel(documents["clean_parent_panel"]) != panel:
        raise ValueError("development selection differs from the outcome-independent rule")
    baselines = documents["cached_baselines"]
    if (baselines["panel_digest"] != panel["digest"]
            or baselines["parent_new_b0_report_digest"] != documents[
                "clean_completion"]["new_policy_report_digest"]
            or baselines["previous_comparison_report_digest"] != documents[
                "previous_comparison"]["digest"]):
        raise ValueError("cached baselines do not bind the frozen evaluations")
    for name in ("new_b0", "previous_b0", "openvla"):
        outcomes = {r["pair_id"]: r[f"{name}_success"] for r in baselines["pairs"]}
        if len(baselines["pairs"]) != 200 or outcome_counts(panel, outcomes) != (
            baselines["counts"][name]):
            raise ValueError("cached baseline counts differ from identified outcomes")
    previous = {r["pair_id"]: r for r in documents["previous_comparison"]["pairs"]}
    for row in baselines["pairs"]:
        old = previous[row["pair_id"]]
        if (row["previous_b0_success"] != old["b0_success"]
                or row["openvla_success"] != old["reference_success"]
                or row["previous_b0_receipt_digest"] != old["b0_receipt_digest"]
                or row["openvla_receipt_digest"] != old["reference_receipt_digest"]):
            raise ValueError("cached original baseline outcome changed")
    gradient = documents["gradient_panel"]
    validation = documents["validation_panel"]
    first_by_task = {}
    for row in validation["episodes"]:
        first_by_task.setdefault(row["task_key"], row)
        if (len(row["anchors"]) != 16 or len(set(row["anchors"])) != 16
                or not all(0 <= a <= row["length"] - 8 for a in row["anchors"])):
            raise ValueError("invalid fixed validation anchors")
    if (gradient["source_validation_selection_digest"] != validation["digest"]
            or gradient["episodes"] != list(first_by_task.values())
            or gradient["batch_count"] != 40 or gradient["sample_count"] != 640
            or sum(len(r["anchors"]) for r in validation["episodes"]) != 2560):
        raise ValueError("gradient selection changed")
    prompts = documents["prompt_panel"]
    identities = {(r["pair"]["runtime_task_index"], r["pair"]["initial_state_index"],
                   r["phase"]) for r in prompts["inputs"]}
    expected = {(task, state, phase) for task in range(10) for state in (0, 25, 49)
                for phase in ("first", "middle", "last")}
    if identities != expected or len(prompts["inputs"]) != 90:
        raise ValueError("fixed prompt observations changed")
    audit = documents["parent_recovery_audit"]
    parent = documents["parent_training_report"]
    if (audit["training_report_digest"] != parent["digest"]
            or audit["selected_policy_sha256"] != parent["checkpoint_sha256"]
            or audit["planning_helper_sha256"] != rule["sources"][
                "planning_and_decisions"]["sha256"]
            or audit["saved_learning_rates"] != [1e-5, 1e-5]
            or not audit["recovery_weights_equal_selected_policy"]):
        raise ValueError("parent recovery or planning implementation changed")
    budgets = rule["training_budget"]
    checkpoints = audit["checkpoints"]
    if ([r["additional_updates"] for r in checkpoints] != [15000, 30000, 45000, 60000]
            or checkpoints[0]["additional_presentations"] != 435945
            or checkpoints[-1]["additional_presentations"] != 1747065
            or budgets["maximum_total_optimizer_updates_across_arms"] != 75000
            or budgets["maximum_total_example_presentations_across_arms"] != 2183010):
        raise ValueError("matched exposure or resource accounting changed")
    return {
        "experiment_id": rule["experiment_id"], "freeze_digest": rule["digest"],
        "status": "freeze-validated-not-launched", "development_pairs": 200,
        "validation_examples": 2560, "gradient_examples": 640,
        "fixed_prompt_observations": 90, "source_pins_verified": len(rule["sources"]),
        "launch_ready": False, "training_launched": False,
        "launch_readiness_note": "Runtime implementation and interruption pilot remain required.",
        "week4_acceptance_status": "FAIL; historical rule and outcomes unchanged",
    }


if __name__ == "__main__":
    print(json.dumps(validate_freeze(), indent=2))

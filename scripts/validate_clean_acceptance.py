#!/usr/bin/env python3
"""Validate the frozen Week 4 acceptance rule and 2,000-pair panel without rollouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from eventground_wam.clean_acceptance import build_panel, read_sealed
from eventground_wam.evaluation_protocol import load_evaluation_protocol, protocol_sha256

ROOT = Path(__file__).resolve().parents[1]
RULE_DIGEST = "0e3404fbe44d16ad5d6742948d8aeaa8a388cfd04ac6527d21454f1dd0e82f03"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_freeze(root: Path = ROOT) -> dict[str, object]:
    rule = read_sealed(root / "configs/evaluation/week4_clean_acceptance_v1.json")
    if rule["digest"] != RULE_DIGEST:
        raise ValueError("frozen acceptance rule changed; create a new version")
    inventory = read_sealed(root / rule["panel"]["initial_state_inventory_path"])
    panel_path = root / rule["panel"]["path"]
    panel = read_sealed(panel_path)
    if (inventory["digest"] != rule["panel"]["initial_state_inventory_digest"]
            or panel["digest"] != rule["panel"]["digest"]
            or sha256(panel_path) != rule["panel"]["file_sha256"]
            or build_panel(inventory) != panel):
        raise ValueError("frozen task/state panel no longer matches the rule")
    clean_manifest = read_sealed(root / "evaluation/splits/libero-clean-v1/clean-40.json")
    if clean_manifest["digest"] != inventory["source_manifest_digest"]:
        raise ValueError("source clean manifest changed")
    for task, frozen_task in zip(inventory["tasks"], clean_manifest["tasks"], strict=True):
        if any(task[key] != value for key, value in frozen_task.items()):
            raise ValueError("initial-state audit changed a frozen canonical task")
    inherited = rule["rollout_contract"]
    if protocol_sha256(load_evaluation_protocol(root / inherited["inherited_profile_path"])) != (
        inherited["inherited_profile_sha256"]
    ):
        raise ValueError("inherited clean rollout settings changed")
    for policy in rule["policies"].values():
        if sha256(root / policy["runtime_lock_path"]) != policy["runtime_lock_sha256"]:
            raise ValueError("pinned policy runtime lock changed")
    implementation = rule["analysis_implementation"]
    if sha256(root / implementation["path"]) != implementation["sha256"]:
        raise ValueError("frozen acceptance analysis implementation changed")
    b0 = read_sealed(root / "docs/evaluation/week4-b0-clean40-v1.json")
    reference = read_sealed(root / "docs/evaluation/week4-reference-clean40-selection-v1.json")
    if (b0["checkpoint"]["checkpoint_revision"] != rule["policies"]["b0"]["checkpoint_revision"]
            or reference["digest"] != rule["policies"]["reference"][
                "checkpoint_verification_selection_digest"]):
        raise ValueError("selected comparison policies changed")
    return {
        "rule_id": rule["rule_id"], "rule_digest": rule["digest"],
        "panel_digest": panel["digest"], "pair_count": panel["pair_count"],
        "status": "freeze-validated", "this_command_runs_rollouts": False,
        "measurement_decision": "NOT_ASSESSED",
    }


if __name__ == "__main__":
    print(json.dumps(validate_freeze(), indent=2))

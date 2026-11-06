"""Verify approved v3 membership and unchanged scientific/training controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.instrumented_partitions import build_source_split


def sealed(path):
    d = json.loads(path.read_text())
    if d.get("digest") != contract_digest(d):
        raise ValueError(f"Digest mismatch: {path}")
    return d


def verify(root):
    root = Path(root).resolve()
    c = sealed(root / "configs/training/week10_eventground_experiment_v3.json")
    v2 = sealed(root / "configs/training/week10_eventground_experiment_v2.json")
    if (
        c["experiment_id"] != "week10-eventground-v3"
        or c["supersedes_scientific_version"] != v2["digest"]
    ):
        raise ValueError("Wrong study parent")
    for rel, h in c["source_files_sha256"].items():
        p = (root / rel).resolve()
        if not p.is_relative_to(root) or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            raise ValueError(f"Changed pinned source {rel}")
    for key in (
        "parent",
        "prerequisites",
        "io",
        "event_scope",
        "training",
        "reward",
        "promotion",
        "resources",
        "readiness",
    ):
        if c[key] != v2[key]:
            raise ValueError(f"Unapproved scientific change: {key}")
    if c["evaluation"] != {**v2["evaluation"], "bootstrap_unit": "physical_source_id"}:
        raise ValueError("Unapproved evaluation change")
    folder = root / "docs/evaluation/week10-source-split-v3"
    split = sealed(folder / "partitions.json")
    approval = sealed(folder / "approval.json")
    proposal = sealed(
        root / "docs/evaluation/week10-production-plan-v2/source-split-v3-proposal.json"
    )
    plan = sealed(root / "docs/evaluation/week10-production-plan-v2/recording-plan-proposal.json")
    if split != build_source_split(proposal, plan, approval):
        raise ValueError("Active split does not reproduce approved memberships")
    if (
        c["source_split_approval_digest"] != approval["digest"]
        or c["data"]["split_digest"] != split["digest"]
    ):
        raise ValueError("Approval/split links changed")
    if (
        c["data"]["historical_source_closure_claimed"]
        or c["data"]["historical_partition_protection_retained"]
    ):
        raise ValueError("Historical closure incorrectly claimed")
    for key in (
        "sft_unique_prompts",
        "main_rl_unique_prompts",
        "rl_pilot_unique_prompts",
        "maximum_prompts_per_group",
        "maximum_anchors_per_physical_event",
        "event_dev_maximum_prompts",
        "event_report_maximum_prompts",
        "event_report_stride",
        "per_class_prompts",
    ):
        if c["data"][key] != v2["data"][key]:
            raise ValueError(f"Unapproved data exposure: {key}")
    return {
        "status": "passed",
        "study_digest": c["digest"],
        "partition_digest": split["digest"],
        "partition_counts": split["partition_counts"],
        "source_id_overlap": 0,
        "source_pins_verified": len(c["source_files_sha256"]),
        "M4_parent_and_training_controls_unchanged": True,
        "source_split_active": True,
        "historical_source_closure_claimed": False,
        "production_capture_authorized": False,
        "training_authorized": False,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    print(json.dumps(verify(args.root), indent=2))

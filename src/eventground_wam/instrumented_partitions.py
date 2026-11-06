"""Approved v3 canonical source partitions; historical alias claims stay limited."""

from __future__ import annotations

from collections import Counter

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.instrumented_supervision import physical_source_id


def check_sealed(value):
    if value.get("digest") != contract_digest(value):
        raise ValueError("Changed sealed record")


def build_source_split(proposal, plan, approval):
    for record in (proposal, plan, approval):
        check_sealed(record)
    if not approval.get("approved") or approval.get("proposal_digest") != proposal["digest"]:
        raise ValueError("Exact source-split proposal approval required")
    if proposal["training_candidate_plan_digest"] != plan["digest"]:
        raise ValueError("Recording plan differs from approved proposal")
    train = plan["rows"]
    if sorted(r["source_id"] for r in train) != proposal["train_source_ids"]:
        raise ValueError("Approved training memberships changed")
    rows = []
    for rank, r in enumerate(train):
        rows.append(
            {
                k: r[k]
                for k in (
                    "source_id",
                    "repository",
                    "revision",
                    "hdf5_path",
                    "demo_id",
                    "task_key",
                    "frames",
                    "queue_role",
                )
            }
            | {"partition": "train", "queue_rank": rank}
        )
    for r in proposal["new_event_panel_assignments"]:
        rows.append(
            {
                k: r[k]
                for k in (
                    "source_id",
                    "repository",
                    "revision",
                    "hdf5_path",
                    "demo_id",
                    "task_key",
                    "frames",
                )
            }
            | {"partition": r["proposed_partition"], "queue_role": "heldout_locked"}
        )
    ids = [r["source_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate or cross-partition source")
    for r in rows:
        if r["source_id"] != physical_source_id(
            r["repository"], r["revision"], r["hdf5_path"], r["demo_id"]
        ):
            raise ValueError("Source identity mismatch")
    counts = dict(Counter(r["partition"] for r in rows))
    if counts != {"train": 300, "event-dev": 103, "event-report": 77}:
        raise ValueError("Approved partition counts changed")
    protected = set(
        proposal["excluded_from_new_event_panels_previously_accessed_or_diagnostic_sources"]
    )
    if any(r["source_id"] in protected for r in rows if r["partition"] != "train"):
        raise ValueError("Previously inspected source entered new event panel")
    tasks = {r["task_key"] for r in train}
    initial = Counter(r["task_key"] for r in train if r["queue_role"] == "initial_review")
    if len(tasks) != 40 or set(initial) != tasks or set(initial.values()) != {5}:
        raise ValueError("Initial review must contain five sources per task")
    if Counter(r["queue_role"] for r in train) != {"initial_review": 200, "coverage_reserve": 100}:
        raise ValueError("Initial/reserve roles changed")
    if any(
        {r["task_key"] for r in rows if r["partition"] == p} != tasks
        for p in ("event-dev", "event-report")
    ):
        raise ValueError("Event panel task coverage changed")
    return seal_contract(
        {
            "artifact_type": "eventground_wam.canonical_source_partition_contract",
            "version": "week10-canonical-source-split-v3",
            "status": "active",
            "approval_digest": approval["digest"],
            "approved_proposal_digest": proposal["digest"],
            "recording_plan_digest": plan["digest"],
            "rows": rows,
            "partition_counts": counts,
            "source_id_overlap": 0,
            "all_new_variants_share_source_partition": True,
            "unknown_sources": "deny",
            "historical_source_closure_claimed": False,
            "inherited_M4_source_novelty_claimed": False,
            "historical_partition_files_preserved": True,
            "heldout_payloads_locked": True,
            "production_capture_authorized": False,
            "training_authorized": False,
            "claim_limits": proposal["claim_limits"],
        }
    )


def require_train_source(split, source_id, *, allow_reserve=False):
    check_sealed(split)
    found = [r for r in split["rows"] if r["source_id"] == source_id]
    if len(found) != 1 or found[0]["partition"] != "train":
        raise ValueError("Source is unknown or held out")
    if found[0]["queue_role"] == "coverage_reserve" and not allow_reserve:
        raise ValueError("Coverage reserve requires its separate gate")
    return found[0]

import copy
import json
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.instrumented_partitions import build_source_split, require_train_source

ROOT = Path(__file__).resolve().parents[1]


def inputs():
    b = ROOT / "docs/evaluation/week10-production-plan-v2"
    p = json.loads((b / "source-split-v3-proposal.json").read_text())
    plan = json.loads((b / "recording-plan-proposal.json").read_text())
    approval = seal_contract({"approved": True, "proposal_digest": p["digest"]})
    return p, plan, approval


def test_exact_approved_memberships_and_reserve_gate():
    p, plan, a = inputs()
    s = build_source_split(p, plan, a)
    assert s["partition_counts"] == {"train": 300, "event-dev": 103, "event-report": 77}
    assert not s["production_capture_authorized"] and not s["historical_source_closure_claimed"]
    assert require_train_source(s, plan["rows"][0]["source_id"])["queue_role"] == "initial_review"
    reserve = next(r for r in s["rows"] if r["queue_role"] == "coverage_reserve")
    with pytest.raises(ValueError, match="reserve"):
        require_train_source(s, reserve["source_id"])
    assert require_train_source(s, reserve["source_id"], allow_reserve=True) == reserve


def test_no_heldout_or_unknown_source_can_enter_training():
    p, plan, a = inputs()
    s = build_source_split(p, plan, a)
    for sid in [r["source_id"] for r in s["rows"] if r["partition"] != "train"] + ["0" * 64]:
        with pytest.raises(ValueError, match="unknown or held out"):
            require_train_source(s, sid, allow_reserve=True)


@pytest.mark.parametrize(
    "corruption",
    ["approval", "plan", "membership", "overlap", "identity", "prior_access", "coverage"],
)
def test_partition_tampering_rejected(corruption):
    p, plan, a = copy.deepcopy(inputs())
    if corruption == "approval":
        a["approved"] = False
        a = seal_contract({k: v for k, v in a.items() if k != "digest"})
    elif corruption == "plan":
        plan["rows"].pop()
        plan = seal_contract({k: v for k, v in plan.items() if k != "digest"})
    else:
        if corruption == "membership":
            p["train_source_ids"][0] = "0" * 64
        elif corruption == "overlap":
            p["new_event_panel_assignments"][0] = dict(p["new_event_panel_assignments"][1])
        elif corruption == "identity":
            p["new_event_panel_assignments"][0]["demo_id"] = "demo_999"
        elif corruption == "prior_access":
            p["excluded_from_new_event_panels_previously_accessed_or_diagnostic_sources"].append(
                p["new_event_panel_assignments"][0]["source_id"]
            )
        else:
            p["new_event_panel_assignments"][0]["task_key"] = "unapproved/task"
        p = seal_contract({k: v for k, v in p.items() if k != "digest"})
        a = seal_contract({"approved": True, "proposal_digest": p["digest"]})
    with pytest.raises(ValueError):
        build_source_split(p, plan, a)

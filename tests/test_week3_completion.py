from __future__ import annotations

import json
from pathlib import Path

from eventground_wam.demonstration_contracts import contract_digest

PROJECT_ROOT = Path(__file__).parents[1]


def test_week3_completion_evidence_closes_every_step_and_verifies_ledger() -> None:
    path = PROJECT_ROOT / "docs" / "evaluation" / "week3-completion-v1.json"
    completion = json.loads(path.read_text(encoding="utf-8"))

    assert completion["digest"] == contract_digest(completion)
    assert completion["status"] == "complete-with-documented-acceptance-revision"
    assert len(completion["implementation_sequence"]) == 7
    assert all(
        step["status"].startswith("complete")
        for step in completion["implementation_sequence"]
    )
    for record in completion["evidence_ledger"]:
        evidence = json.loads(
            (PROJECT_ROOT / record["relative_path"]).read_text(encoding="utf-8")
        )
        assert evidence["digest"] == contract_digest(evidence)
        assert evidence["digest"] == record["digest"]
        assert evidence["status"] == record["status"]

    cohort = completion["acceptance"]["ten_demonstration_overfit"]
    assert cohort["original_exact_gripper_gate_passed"] is False
    assert cohort["reviewer_authorized_engineering_acceptance"] is True
    assert completion["acceptance"]["closed_loop"]["accepted_hdf5_deep_validation_passed"] is True
    assert completion["week4_handoff"]["unblocked"] is True
    assert completion["known_risks_carried_into_week4"]

import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("damage", [None, "hash", "action_digest", "heldout", "changed_task"])
def test_staging_boundary_checks_before_network(tmp_path, monkeypatch, damage):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    m = importlib.import_module("stage_week10_capture_sources")
    split = json.loads(
        (ROOT / "docs/evaluation/week10-source-split-v3/partitions.json").read_text()
    )
    plan = json.loads(
        (
            ROOT / "docs/evaluation/week10-production-plan-v2/recording-plan-proposal.json"
        ).read_text()
    )
    row = copy.deepcopy(plan["rows"][0])
    source = ROOT / Path(row["evidence_path"]).parent / f"{row['demo_id']}-actions.npz"
    (tmp_path / "actions.npz").write_bytes(source.read_bytes())
    row.update(
        actions_file="actions.npz", actions_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
    )
    if damage == "hash":
        row["actions_sha256"] = "0" * 64
    elif damage == "action_digest":
        row["action_digest"] = "0" * 64
    elif damage == "heldout":
        row["source_id"] = next(
            r["source_id"] for r in split["rows"] if r["partition"] == "event-report"
        )
    elif damage == "changed_task":
        row["task_key"] = "unknown/task"
    if damage:
        with pytest.raises(ValueError):
            m.check_job({"sources": [row]}, split, tmp_path)
    else:
        m.check_job({"sources": [row]}, split, tmp_path)

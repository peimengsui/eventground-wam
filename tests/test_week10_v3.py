import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("v3_verifier", ROOT / "scripts/verify_week10_v3.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_active_v3_matches_approved_partition_and_preserves_controls():
    assert module.verify(ROOT)["partition_counts"] == {
        "train": 300,
        "event-dev": 103,
        "event-report": 77,
    }


@pytest.mark.parametrize("change", ["parent", "reward", "launch", "exposure", "overlap"])
def test_v3_verifier_rejects_unapproved_changes(tmp_path, change):
    cp = Path("configs/training/week10_eventground_experiment_v3.json")
    c = json.loads((ROOT / cp).read_text())
    for name in [*c["source_files_sha256"], str(cp)]:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    if change == "parent":
        c["parent"]["policy_sha256"] = "0" * 64
    elif change == "reward":
        c["reward"]["weights"] = [1, 0, 0, 0]
    elif change == "launch":
        c["readiness"]["training_launched"] = True
    elif change == "exposure":
        c["data"]["sft_unique_prompts"] = 4000
    else:
        p = tmp_path / "docs/evaluation/week10-source-split-v3/partitions.json"
        d = json.loads(p.read_text())
        d["rows"][-1]["source_id"] = d["rows"][0]["source_id"]
        p.write_text(json.dumps(d))
    c.pop("digest")
    (tmp_path / cp).write_text(json.dumps(seal_contract(c)))
    with pytest.raises(ValueError):
        module.verify(tmp_path)

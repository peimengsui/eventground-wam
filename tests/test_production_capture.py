import copy
import json
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.production_capture import initial_capture_targets


def setup():
    root = Path(__file__).resolve().parents[1]
    split = json.loads(
        (root / "docs/evaluation/week10-source-split-v3/partitions.json").read_text()
    )
    rows = sorted(
        (r for r in split["rows"] if r["partition"] == "train"), key=lambda r: r["queue_rank"]
    )
    initial = [r for r in rows if r["queue_role"] == "initial_review"]
    c = seal_contract(
        {
            "partition_digest": split["digest"],
            "targets": rows,
            "training_authorized": False,
            "resources": {
                "control_steps": sum(r["frames"] + 10 for r in initial),
                "wall_seconds": 7200,
            },
        }
    )
    a = seal_contract(
        {
            "approved": True,
            "scope": "initial_200_production_capture",
            "capture_contract_digest": c["digest"],
            "source_ids": [r["source_id"] for r in initial],
            "training_authorized": False,
        }
    )
    return c, split, a


def reseal(x):
    return seal_contract({k: v for k, v in x.items() if k != "digest"})


def test_launch_selects_only_initial_two_hundred():
    c, s, a = setup()
    rows = initial_capture_targets(c, s, a)
    assert len(rows) == 200 and all(r["queue_role"] == "initial_review" for r in rows)


@pytest.mark.parametrize(
    "damage",
    [
        "no_approval",
        "split_approval_only",
        "wrong_contract",
        "reserve_added",
        "new_source",
        "budget",
        "training",
        "source_order",
    ],
)
def test_launch_fails_closed_before_any_recording(damage):
    c, s, a = copy.deepcopy(setup())
    if damage == "no_approval":
        a["approved"] = False
    elif damage == "split_approval_only":
        a["scope"] = "activate_v3_source_split"
    elif damage == "wrong_contract":
        a["capture_contract_digest"] = "0" * 64
    elif damage == "reserve_added":
        a["source_ids"].append(c["targets"][-1]["source_id"])
    elif damage == "new_source":
        c["targets"][0]["source_id"] = "0" * 64
    elif damage == "budget":
        c["resources"]["wall_seconds"] = 7201
    elif damage == "training":
        a["training_authorized"] = True
    elif damage == "source_order":
        c["targets"][0], c["targets"][1] = c["targets"][1], c["targets"][0]
    if damage in ("new_source", "budget", "source_order"):
        c = reseal(c)
        a["capture_contract_digest"] = c["digest"]
    a = reseal(a)
    with pytest.raises(ValueError):
        initial_capture_targets(c, s, a)

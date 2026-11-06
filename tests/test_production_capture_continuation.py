import copy
import json
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.production_capture import continuation_capture_targets


def reseal(value):
    return seal_contract({k: v for k, v in value.items() if k != "digest"})


def fixture():
    root = Path(__file__).resolve().parents[1] / "docs/evaluation"
    prior = json.loads(
        (root / "week10-production-capture-startup-recovery-v1/run/completion.json").read_text()
    )
    c = json.loads((root / "week10-production-capture-v1/recording-contract.json").read_text())
    split = json.loads((root / "week10-source-split-v3/partitions.json").read_text())
    initial = [t for t in c["targets"] if t["queue_role"] == "initial_review"]
    c["continuation"] = {
        "prior_completion_digest": prior["digest"],
        "retained_recording_ids": [r["recording_id"] for r in initial[:15]],
    }
    c["resources"].update(
        {
            "prior_control_steps": 4209,
            "remaining_control_steps": 31372,
            "wall_seconds": 7000,
            "prior_seconds_charged": 200,
            "output_bytes": 96 * 1024**3,
            "prior_output_bytes_charged": 4 * 1024**3,
        }
    )
    c = reseal(c)
    a = seal_contract(
        {
            "approved": True,
            "scope": "initial_200_production_capture",
            "capture_contract_digest": c["digest"],
            "training_authorized": False,
            "source_ids": [t["source_id"] for t in initial],
            "retry_failed_zero_step_source": initial[15]["source_id"],
            "repair_completed_images_from_saved_states": True,
        }
    )
    return c, split, a, prior


def test_continuation_preserves_prefix_and_remaining_control_budget():
    c, s, a, p = fixture()
    assert len(continuation_capture_targets(c, s, a, p)) == 200


@pytest.mark.parametrize(
    "damage",
    [
        "retry_missing",
        "repair_missing",
        "drop_completed",
        "completed_reordered",
        "failed_stepped",
        "step_budget",
        "time_budget",
        "storage_budget",
    ],
)
def test_continuation_rejects_retries_drops_or_resource_expansion(damage):
    c, s, a, p = copy.deepcopy(fixture())
    if damage == "retry_missing":
        a["retry_failed_zero_step_source"] = None
    if damage == "repair_missing":
        a["repair_completed_images_from_saved_states"] = False
    if damage == "drop_completed":
        c["continuation"]["retained_recording_ids"].pop()
    if damage == "completed_reordered":
        p["results"][0], p["results"][1] = p["results"][1], p["results"][0]
    if damage == "failed_stepped":
        p["results"][-1]["control_steps"] = 1
    if damage == "step_budget":
        c["resources"]["remaining_control_steps"] += 1
    if damage == "time_budget":
        c["resources"]["prior_seconds_charged"] += 1
    if damage == "storage_budget":
        c["resources"]["prior_output_bytes_charged"] += 1
    p = reseal(p)
    c["continuation"]["prior_completion_digest"] = p["digest"]
    c = reseal(c)
    a["capture_contract_digest"] = c["digest"]
    with pytest.raises(ValueError):
        continuation_capture_targets(c, s, reseal(a), p)

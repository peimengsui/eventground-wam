import numpy as np
import pytest

from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.instrumented_supervision import (
    labeler_contract,
    numeric_digest,
    physical_source_id,
    resolve_partition,
)


def test_source_identity_does_not_depend_on_visual_variant():
    args = ("owner/repo", "a" * 40, "suite/task_demo.hdf5", "demo_3")
    assert physical_source_id(*args) == physical_source_id(*args)
    assert physical_source_id(*args) != physical_source_id(*args[:-1], "demo_4")
    with pytest.raises(ValueError):
        physical_source_id("owner/repo", "main", "suite/task_demo.hdf5", "demo_3")
    with pytest.raises(ValueError):
        physical_source_id("owner/repo", "a" * 40, "../task_demo.hdf5", "demo_3")


def test_protected_alias_overrides_train_majority():
    rows = [{"partition": "train", "exact_action_identity": True}] * 20
    for protected in ("event-dev", "event-report", "test"):
        assert (
            resolve_partition(rows + [{"partition": protected, "exact_action_identity": True}])[
                "partition"
            ]
            == "quarantine"
        )
    assert resolve_partition(rows)["partition"] == "train"
    assert resolve_partition([])["partition"] == "quarantine"
    assert (
        resolve_partition([{"partition": "train", "exact_action_identity": False}])["partition"]
        == "quarantine"
    )


def test_action_identity_is_canonical_and_order_sensitive():
    actions = np.arange(21).reshape(3, 7)
    assert numeric_digest(actions) == numeric_digest(actions.astype("f4"))
    assert numeric_digest(actions) != numeric_digest(actions[::-1])
    assert numeric_digest(actions) != numeric_digest(actions[:2])


def test_label_rules_preserve_ambiguity_and_independent_quality_gate():
    rules = labeler_contract()
    assert rules["digest"] == contract_digest(rules)
    assert rules["labels_admitted"] is False
    assert rules["contact"]["gap_filling_frames"] == 0
    assert rules["grasp"]["closure_alone_is_grasp"] is False
    assert rules["quality_gate"]["independent_full_episode_review"]
    assert rules["quality_gate"]["minimum_train_episodes"] == 200
    assert rules["prompt_rules"]["physical_evidence_is_model_input"] is False

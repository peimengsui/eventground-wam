from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from eventground_wam.evaluation_protocol import load_evaluation_protocol
from eventground_wam.evaluation_splits import (
    CLASSIFICATION_SHA256,
    SplitValidationError,
    load_manifest_family,
    validate_manifest,
)

MANIFEST_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "evaluation" / "splits" / "libero-plus-v1"
)
EXPECTED_TASK_RECORD_DIGESTS = {
    "sanity-35": "67be0fd659e93fa87a589fa8644d0b43284051d42b54ec112468f67005ff9c1c",
    "dev-210": "6069c171f4e1f983bd3f83dcbd4cc49d7cef9b83f4f6fb89f66e17e2f31d8132",
    "stratified-1K": "5563dc85cc70efb5eca1d1aeaf62213f6aa6a660ccb897ad58b15c2f30e4dc49",
    "full-10K": "7238f79a53c3411baa77ec7ef9f04265a981bf5df193c000a2e9ac1d6837d173",
}


@pytest.fixture(scope="module")
def protocol() -> dict:
    return load_evaluation_protocol()


@pytest.fixture(scope="module")
def manifests(protocol: dict) -> dict:
    return load_manifest_family(MANIFEST_DIRECTORY, protocol)


def test_committed_manifest_identities(manifests: dict) -> None:
    assert {
        split_id: manifest["task_records_sha256"]
        for split_id, manifest in manifests.items()
    } == EXPECTED_TASK_RECORD_DIGESTS
    assert {
        manifest["source"]["classification_sha256"] for manifest in manifests.values()
    } == {CLASSIFICATION_SHA256}


def test_unlabeled_difficulty_policy(manifests: dict) -> None:
    unlabeled_counts = {
        split_id: sum(task["difficulty_level"] is None for task in manifest["tasks"])
        for split_id, manifest in manifests.items()
    }

    assert unlabeled_counts == {
        "sanity-35": 0,
        "dev-210": 0,
        "stratified-1K": 12,
        "full-10K": 121,
    }


def test_small_splits_balance_suites(manifests: dict) -> None:
    suite_counts = {
        split_id: dict(Counter(task["suite"] for task in manifest["tasks"]))
        for split_id, manifest in manifests.items()
        if split_id in {"sanity-35", "dev-210"}
    }

    assert suite_counts == {
        "sanity-35": {
            "libero_spatial": 8,
            "libero_object": 9,
            "libero_goal": 9,
            "libero_10": 9,
        },
        "dev-210": {
            "libero_spatial": 52,
            "libero_object": 52,
            "libero_goal": 53,
            "libero_10": 53,
        },
    }


def test_manifest_validation_rejects_task_drift(protocol: dict, manifests: dict) -> None:
    changed_manifest = deepcopy(manifests["sanity-35"])
    changed_manifest["tasks"][0]["runtime_task_index"] += 1

    with pytest.raises(SplitValidationError, match="Task record digest mismatch"):
        validate_manifest(changed_manifest, protocol)

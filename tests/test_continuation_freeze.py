from __future__ import annotations

import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

from eventground_wam.clean_acceptance import read_sealed
from eventground_wam.continuation_freeze import (
    assess_screen,
    build_development_panel,
    exposure_digest,
    mix_batches,
    outcome_counts,
    select_arm,
)
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_continuation_freeze import CONFIG, validate_freeze


def test_panel_is_reproducible_from_identities_and_rejects_missing_states():
    parent = read_sealed(ROOT / "evaluation/splits/week4-clean-comparison-v1/paired-2000.json")
    panel = build_development_panel(parent)
    assert panel == build_development_panel({**parent, "irrelevant_outcomes": [False] * 2000})
    assert set(Counter((p["suite"], p["runtime_task_index"])
                       for p in panel["pairs"]).values()) == {5}
    parent["pairs"].pop()
    with pytest.raises(ValueError, match="all 50"):
        build_development_panel(parent)


def test_mixing_preserves_sizes_and_exposure_at_every_window_and_decision_boundary():
    batches = [[(1, i, j) for j in range(1 + i % 3)] for i in range(15002)]
    mixed = mix_batches(batches)
    assert mixed == mix_batches(batches)
    assert mixed != batches
    assert [len(b) for b in mixed] == [len(b) for b in batches]
    for end in (128, 14976, 15000, 15002):
        assert exposure_digest(mixed[:end]) == exposure_digest(batches[:end])
    assert all(token[1] < 15000 for b in mixed[:15000] for token in b)


@pytest.mark.parametrize("batches", [[[]], [[(1, 0, 0)] * 33], [[(1, 0, 0), (1, 0, 0)]]])
def test_invalid_or_duplicated_presentations_are_rejected(batches):
    with pytest.raises(ValueError):
        mix_batches(batches)


@pytest.fixture
def screen_inputs():
    panel = read_sealed(ROOT / "evaluation/splits/week4-oft-continuation-v1/development-200.json")
    baseline = read_sealed(ROOT / "docs/evaluation/week4-oft-continuation-baselines-v1.json")
    parent = {r["pair_id"]: r["new_b0_success"] for r in baseline["pairs"]}
    candidate = dict(parent)
    # One gain on each of five different Goal tasks meets the exact minimum.
    for task in range(5):
        pair = next(p for p in panel["pairs"] if p["suite"] == "libero_goal"
                    and p["runtime_task_index"] == task and not parent[p["pair_id"]])
        candidate[pair["pair_id"]] = True
    return panel, parent, candidate


def test_exact_goal_boundary_validity_and_other_suite_guard(screen_inputs):
    panel, parent, candidate = screen_inputs
    assert assess_screen(panel, parent, candidate, valid=True)["eligible_to_extend"]
    assert not assess_screen(panel, parent, candidate, valid=False)["eligible_to_extend"]
    fewer = dict(candidate)
    fewer[next(k for k in candidate if candidate[k] and not parent[k])] = False
    assert not assess_screen(panel, parent, fewer, valid=True)["eligible_to_extend"]
    object_successes = [p["pair_id"] for p in panel["pairs"]
                        if p["suite"] == "libero_object" and parent[p["pair_id"]]]
    for key in object_successes[:2]:
        candidate[key] = False
    assert assess_screen(panel, parent, candidate, valid=True)["eligible_to_extend"]
    candidate[object_successes[2]] = False
    assert not assess_screen(panel, parent, candidate, valid=True)["eligible_to_extend"]


def test_concentrated_goal_gains_cannot_pass_task_breadth_guard(screen_inputs):
    panel, parent, _ = screen_inputs
    candidate = dict(parent)
    failed = [p["pair_id"] for p in panel["pairs"] if p["suite"] == "libero_goal"
              and p["runtime_task_index"] < 2 and not parent[p["pair_id"]]]
    assert len(failed) >= 5
    for key in failed[:5]:
        candidate[key] = True
    result = assess_screen(panel, parent, candidate, valid=True)
    assert result["delta_counts"]["libero_goal"] == 5
    assert not result["eligible_to_extend"]


def test_missing_and_nonboolean_outcomes_cannot_be_used(screen_inputs):
    panel, parent, _ = screen_inputs
    incomplete = dict(parent)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match="200 identified boolean"):
        outcome_counts(panel, incomplete)
    bad = dict(parent)
    bad[next(iter(bad))] = 1
    with pytest.raises(ValueError, match="200 identified boolean"):
        outcome_counts(panel, bad)


def test_arm_selection_requires_both_arms_and_prefers_a_on_exact_tie(screen_inputs):
    panel, parent, candidate = screen_inputs
    result = assess_screen(panel, parent, candidate, valid=True)
    assert select_arm({"A": result, "B": deepcopy(result)}) == "A"
    with pytest.raises(ValueError, match="both fixed arms"):
        select_arm({"A": result})
    fail = assess_screen(panel, parent, parent, valid=True)
    assert select_arm({"A": fail, "B": fail}) is None


def test_real_freeze_validates_without_launching():
    result = validate_freeze()
    assert result["source_pins_verified"] == 20
    assert result["training_launched"] is False


def test_resealed_budget_change_is_not_silently_accepted(tmp_path):
    rule = read_sealed(ROOT / CONFIG)
    rule.pop("digest")
    rule["training_budget"]["shared_total_active_hours_cap"] = 360
    path = tmp_path / CONFIG
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(seal_contract(rule)))
    with pytest.raises(ValueError, match="record a new version"):
        validate_freeze(tmp_path)

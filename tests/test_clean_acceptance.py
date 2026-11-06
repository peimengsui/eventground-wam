from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from eventground_wam.clean_acceptance import (
    analyze_paired_outcomes,
    build_panel,
    classify_interval,
    pair_seed,
    read_sealed,
)
from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_clean_acceptance import validate_freeze


@pytest.mark.parametrize("lower,upper,expected", [
    (-4.9, 0, "PASS"), (-5, -5, "PASS"), (-20, -5.01, "FAIL"),
    (-5.01, -5, "INCONCLUSIVE"), (-7, -3, "INCONCLUSIVE"),
])
def test_decision_margin_and_exact_boundaries(lower, upper, expected):
    assert classify_interval(lower, upper) == expected


@pytest.mark.parametrize("bounds", [(float("nan"), 0), (0, -1), (-101, 0)])
def test_invalid_intervals_are_not_accepted(bounds):
    with pytest.raises(ValueError, match="invalid"):
        classify_interval(*bounds)


def test_frozen_panel_rebuilds_with_distinct_states_and_stable_pair_seeds():
    inventory = read_sealed(ROOT / "docs/evaluation/week4-clean-initial-states-v1.json")
    frozen = read_sealed(ROOT / "evaluation/splits/week4-clean-comparison-v1/paired-2000.json")
    assert build_panel(inventory) == frozen
    assert len({p["pair_id"] for p in frozen["pairs"]}) == 2000
    assert len({p["episode_seed"] for p in frozen["pairs"]}) == 2000
    for pair in frozen["pairs"]:
        assert pair["trial_index"] == pair["initial_state_index"]
        assert pair["episode_seed"] == pair_seed(
            pair["suite"], pair["runtime_task_index"], pair["trial_index"],
        )


def test_panel_rejects_duplicated_initial_states_even_if_record_is_resealed():
    inventory = read_sealed(ROOT / "docs/evaluation/week4-clean-initial-states-v1.json")
    changed = deepcopy(inventory)
    changed.pop("digest")
    hashes = changed["tasks"][0]["initial_state_sha256_by_index"]
    hashes[1] = hashes[0]
    with pytest.raises(ValueError, match="50 distinct"):
        build_panel(seal_contract(changed))


def test_complete_paired_analysis_and_cluster_uncertainty():
    reference = np.ones((4, 10, 50), dtype=bool)
    identical = analyze_paired_outcomes(reference, reference)
    assert identical["statistical_decision"] == "PASS"
    assert identical["gap_percentage_points"] == 0
    assert identical["paired_cluster_bootstrap_95_percent_pp"] == [0, 0]
    b0 = reference.copy()
    b0[:, 0, :25] = False
    # One weak task per suite gives a -5 pp point estimate with cluster uncertainty.
    result = analyze_paired_outcomes(b0, reference)
    assert result["gap_percentage_points"] == -5
    assert result["statistical_decision"] == "INCONCLUSIVE"
    assert result == analyze_paired_outcomes(b0, reference)
    assert result["paired_cluster_bootstrap_95_percent_pp"][0] < -5
    b0[:] = False
    failure = analyze_paired_outcomes(b0, reference)
    assert failure["statistical_decision"] == "FAIL"
    assert failure["gap_percentage_points"] == -100


@pytest.mark.parametrize("shape,dtype", [((4, 10, 49), bool), ((4, 10, 50), int)])
def test_analysis_rejects_missing_trials_or_nonboolean_outcomes(shape, dtype):
    with pytest.raises(ValueError, match="complete paired boolean"):
        analyze_paired_outcomes(np.ones(shape, dtype=dtype), np.ones(shape, dtype=dtype))


def test_complete_freeze_keeps_rule_panel_runtime_and_policy_identities():
    result = validate_freeze()
    assert result["pair_count"] == 2000
    assert result["status"] == "freeze-validated"
    assert result["measurement_decision"] == "NOT_ASSESSED"


def test_resealed_rule_cannot_silently_relax_the_margin(tmp_path):
    rule = read_sealed(ROOT / "configs/evaluation/week4_clean_acceptance_v1.json")
    rule.pop("digest")
    rule["analysis"]["maximum_allowed_deficit_percentage_points"] = 35.0
    path = tmp_path / "configs/evaluation/week4_clean_acceptance_v1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(seal_contract(rule)))
    with pytest.raises(ValueError, match="frozen acceptance rule changed"):
        validate_freeze(tmp_path)

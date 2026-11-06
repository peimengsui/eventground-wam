import pytest

from eventground_wam.week11_multihorizon_analysis import CONDITIONS, screen_decision


def matrix(counts):
    cases=[{'id':str(i)} for i in range(27)]
    return cases,{c:{str(i):i<counts[c] for i in range(27)} for c in CONDITIONS}


def test_common_training_gain_does_not_satisfy_adaptation_interaction():
    counts=dict(zip(CONDITIONS,[15,15,16,16,18,18],strict=True))
    cases,rows=matrix(counts)
    r=screen_decision(cases,rows,integrity_valid=True)
    assert r['checks']['adaptive_gain_over_h8_at_least_two']
    assert not r['eligible_to_propose_untouched_panel']
    assert r['interaction_gain']==0


def test_passing_counts_still_require_valid_inputs():
    cases,rows=matrix(dict(zip(CONDITIONS,[15,15,15,15,15,18],strict=True)))
    assert screen_decision(cases,rows,integrity_valid=True)['eligible_to_propose_untouched_panel']
    assert not screen_decision(cases,rows,integrity_valid=False)['eligible_to_propose_untouched_panel']


def test_fixed8_regression_stops_progression_even_if_adaptive_improves():
    cases,rows=matrix(dict(zip(CONDITIONS,[15,15,15,15,13,20],strict=True)))
    r=screen_decision(cases,rows,integrity_valid=True)
    assert not r['eligible_to_propose_untouched_panel']
    assert not r['checks']['fixed8_loss_vs_original_at_most_one']


def test_missing_outcome_cannot_be_dropped():
    cases,rows=matrix(dict.fromkeys(CONDITIONS,15))
    rows[CONDITIONS[0]].pop('26')
    with pytest.raises(ValueError):screen_decision(cases,rows,integrity_valid=True)

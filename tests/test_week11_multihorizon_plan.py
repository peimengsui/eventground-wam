import random
from collections import Counter

import pytest

from eventground_wam.week11_multihorizon_plan import (
    HORIZONS,
    candidate_horizons,
    projected_seconds,
)


def test_short_batches_and_resume_preserve_the_exact_horizon_stream():
    expected = candidate_horizons(0, 137, mixed=True)
    chunks, cursor = [], 0
    for count in (32, 1, 23, 32, 17, 32):
        chunks.extend(candidate_horizons(cursor, count, mixed=True))
        cursor += count
    assert chunks == expected
    assert candidate_horizons(56, 81, mixed=True) == expected[56:]
    assert Counter(expected).keys() == set(HORIZONS)
    assert max(Counter(expected).values()) - min(Counter(expected).values()) <= 1
    for start in range(0, 135, 5):
        assert sorted(expected[start:start + 5]) == list(HORIZONS)


def test_schedule_does_not_consume_random_state_or_change_control_horizon():
    before = random.getstate()
    first = candidate_horizons(13, 51, mixed=True)
    assert random.getstate() == before
    assert candidate_horizons(13, 51, mixed=False) == [8] * 51
    assert candidate_horizons(13, 51, mixed=True) == first
    assert candidate_horizons(13, 51, mixed=True, seed=18) != first


@pytest.mark.parametrize('start,count,mixed', [(-1, 1, True), (True, 1, True),
                                               (0, 0, True), (0, 1, 'mixed')])
def test_invalid_cursor_or_arm_fails(start, count, mixed):
    with pytest.raises(ValueError):
        candidate_horizons(start, count, mixed=mixed)


def test_cost_includes_checkpoint_time_and_conservative_margin():
    plain = projected_seconds(2000, 5, 18)
    conservative = projected_seconds(2000, 11, 18, margin=1.2)
    assert plain['total_seconds'] > 10000
    assert plain['checkpoint_saves_upper_bound'] >= 35
    assert conservative['total_seconds'] > plain['total_seconds']
    assert projected_seconds(2000, 5, 0)['total_seconds'] == 10000
    with pytest.raises(ValueError):
        projected_seconds(2000, 5, 300)

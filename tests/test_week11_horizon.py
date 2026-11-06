"""Finite-support, boundary and failure tests for the event-to-horizon interface."""
import copy
import math
from dataclasses import FrozenInstanceError

import pytest

from eventground_wam.eventground_horizon import (
    EVENT_TYPES,
    SUPPORTED_HORIZONS,
    InvalidEventPrediction,
    nearest_supported_horizon,
    select_horizon,
)

# Explicit bins supply a second representation of the contract, independent of argmin.
EXPECTED = (2,) * 3 + (4,) * 3 + (8,) * 6 + (16,) * 12 + (32,) * 8


def positive(start, end=None, event="contact"):
    return {"next_event": event, "interval": [start, start + 1 if end is None else end],
            "horizon": start}


def test_exhaustive_positive_intervals_and_classes():
    checked = 0
    for event in EVENT_TYPES[1:]:
        for start in range(1, 33):
            for end in range(start + 1, 34):
                prediction = positive(start, end, event)
                original = copy.deepcopy(prediction)
                decision = select_horizon(prediction)
                assert prediction == original
                assert decision.event_type == event
                assert (decision.raw_start, decision.raw_end) == (start, end)
                assert decision.selected_horizon == EXPECTED[start - 1]
                assert decision.quantization_delta_frames == EXPECTED[start - 1] - start
                assert decision.fallback_reason is None
                checked += 1
    assert checked == 1584


def test_supported_fixed_points_monotonicity_and_maximum_displacement():
    selected = [nearest_supported_horizon(s) for s in range(1, 33)]
    assert selected == sorted(selected)
    assert set(selected) == set(SUPPORTED_HORIZONS)
    assert all(nearest_supported_horizon(h) == h for h in SUPPORTED_HORIZONS)
    assert max(abs(h - s) for s, h in enumerate(selected, 1)) == 8
    assert [nearest_supported_horizon(s) for s in (3, 6, 12, 24)] == [2, 4, 8, 16]


def test_none_fallback_is_distinct_from_positive_eight():
    decision = select_horizon({"next_event": "none", "interval": None, "horizon": None})
    assert decision.selected_horizon == 8
    assert decision.fallback_reason == "no_new_onset"
    assert decision.raw_start is decision.raw_end is decision.quantization_delta_frames is None
    assert select_horizon(positive(8)).fallback_reason is None
    with pytest.raises(FrozenInstanceError):
        decision.selected_horizon = 16


@pytest.mark.parametrize("value", [0, -1, 33, 100, True, False, 3.0, "3", None,
                                        math.nan, math.inf, -math.inf, [], {}])
def test_out_of_domain_onset_never_clamps_or_coerces(value):
    with pytest.raises(InvalidEventPrediction):
        nearest_supported_horizon(value)


@pytest.mark.parametrize("prediction", [
    None, "contact", [], {},
    {"next_event": "contact", "interval": [2, 4]},
    {"next_event": "contact", "interval": [2, 4], "horizon": 2, "target": "grasp"},
    {"next_event": "contact", "interval": [2, 4], "horizon": 2, "confidence": 0.99},
    {"next_event": "release", "interval": [2, 4], "horizon": 2},
    {"next_event": ["contact"], "interval": [2, 4], "horizon": 2},
    {"next_event": True, "interval": [2, 4], "horizon": 2},
    {"next_event": "none", "interval": [0, 0], "horizon": 0},
    {"next_event": "none", "interval": None, "horizon": 8},
    {"next_event": "contact", "interval": None, "horizon": 2},
    {"next_event": "contact", "interval": (2, 4), "horizon": 2},
    {"next_event": "contact", "interval": [2], "horizon": 2},
    {"next_event": "contact", "interval": [2, 4, 5], "horizon": 2},
    {"next_event": "contact", "interval": [2, 4], "horizon": 3},
    {"next_event": "contact", "interval": [2, 4], "horizon": 2.0},
    {"next_event": "contact", "interval": [True, 4], "horizon": 1},
    {"next_event": "contact", "interval": [2, False], "horizon": 2},
    {"next_event": "contact", "interval": [2.0, 4], "horizon": 2},
    {"next_event": "contact", "interval": [2, math.nan], "horizon": 2},
    {"next_event": "contact", "interval": [2, math.inf], "horizon": 2},
    positive(0, 4), positive(-2, 4), positive(33, 34),
    positive(4, 4), positive(4, 3), positive(32, 34),
])
def test_malformed_prediction_is_failure_not_fallback(prediction):
    with pytest.raises(InvalidEventPrediction):
        select_horizon(prediction)


def test_canonical_serialized_wire_examples_without_model_execution():
    # These are the established serialize_tokens JSON representations; no model is loaded.
    examples = [
        ({"next_event": "none", "interval": None, "horizon": None}, 8),
        ({"next_event": "gripper_close", "interval": [1, 33], "horizon": 1}, 2),
        ({"next_event": "contact", "interval": [12, 18], "horizon": 12}, 8),
        ({"next_event": "grasp", "interval": [30, 33], "horizon": 30}, 32),
    ]
    for prediction, expected in examples:
        assert select_horizon(prediction).selected_horizon == expected

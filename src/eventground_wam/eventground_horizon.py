"""Pure observation-prediction boundary for the frozen Week 11 horizon rule.

No model, labels, probabilities, simulator state, or action execution is accepted.
The caller must pass the canonical Week 10 serialize_tokens output. Invalid
predictions fail the query; only a valid none prediction falls back to eight.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

RULE_ID = "week11-nearest-onset-tie-shorter-v1"
SUPPORTED_HORIZONS = (2, 4, 8, 16, 32)
EVENT_TYPES = ("none", "gripper_close", "contact", "grasp")
NONE_HORIZON = 8
PREDICTION_KEYS = frozenset({"next_event", "interval", "horizon"})


class InvalidEventPrediction(ValueError):
    """A policy/interface failure, never a request for a fallback action."""


@dataclass(frozen=True)
class HorizonDecision:
    rule_id: str
    event_type: str
    raw_start: int | None
    raw_end: int | None
    selected_horizon: int
    quantization_delta_frames: int | None
    fallback_reason: str | None


def nearest_supported_horizon(start: int) -> int:
    """Minimize absolute frame distance; exact ties choose the smaller horizon."""
    if type(start) is not int or not 1 <= start <= 32:
        raise InvalidEventPrediction("start must be an integer control-frame offset in [1,32]")
    return min(SUPPORTED_HORIZONS, key=lambda horizon: (abs(horizon - start), horizon))


def select_horizon(prediction: Mapping) -> HorizonDecision:
    """Map one valid canonical prediction; no coercion, repair, or confidence gate."""
    if not isinstance(prediction, Mapping) or set(prediction) != PREDICTION_KEYS:
        raise InvalidEventPrediction("prediction requires exactly next_event, interval and horizon")
    event = prediction["next_event"]
    interval = prediction["interval"]
    raw_horizon = prediction["horizon"]
    if type(event) is not str or event not in EVENT_TYPES:
        raise InvalidEventPrediction("unknown event type")
    if event == "none":
        if interval is not None or raw_horizon is not None:
            raise InvalidEventPrediction("none requires null interval and null horizon")
        return HorizonDecision(RULE_ID, event, None, None, NONE_HORIZON, None, "no_new_onset")
    if (not isinstance(interval, list) or len(interval) != 2
            or any(type(offset) is not int for offset in interval)
            or not 1 <= interval[0] < interval[1] <= 33
            or type(raw_horizon) is not int or raw_horizon != interval[0]):
        raise InvalidEventPrediction("positive prediction requires [start,end), horizon=start, "
                                     "and integer 1 <= start < end <= 33")
    start, end = interval
    selected = nearest_supported_horizon(start)
    return HorizonDecision(RULE_ID, event, start, end, selected, selected - start, None)

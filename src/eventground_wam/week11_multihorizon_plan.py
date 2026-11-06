"""Pure scheduling and cost helpers for the bounded Week 11 continuation plan.

These helpers cannot load a model, train a policy, or launch an evaluation.
"""
from __future__ import annotations

import hashlib
import math

HORIZONS = (2, 4, 8, 16, 32)
SCHEDULE_ID = 'week11-balanced-five-presentation-blocks-v1'


def candidate_horizons(start: int, count: int, *, mixed: bool, seed: int = 17) -> list[int]:
    """Assign one horizon per presentation independently of training RNG or targets.

    Every five consecutive presentations contain one of each horizon in an
    independently hash-ordered permutation. The committed sample count is the
    only cursor, so short batches and interruption cannot change assignment.
    """
    if (type(start) is not int or start < 0 or type(count) is not int or count < 1
            or type(seed) is not int or seed < 0 or type(mixed) is not bool):
        raise ValueError('invalid horizon schedule cursor, count, seed or arm')
    if not mixed:
        return [8] * count
    result = []
    permutations = {}
    for ordinal in range(start, start + count):
        block, offset = divmod(ordinal, len(HORIZONS))
        if block not in permutations:
            permutations[block] = sorted(HORIZONS, key=lambda h: (
                hashlib.sha256(f'{SCHEDULE_ID}|{seed}|{block}|{h}'.encode()).digest(), h))
        result.append(permutations[block][offset])
    return result


def projected_seconds(updates: int, step_seconds: float, checkpoint_seconds: float,
                      *, margin: float = 1.0, checkpoint_interval_seconds: int = 300,
                      checkpoint_interval_updates: int = 2000) -> dict:
    """Planning proxy including periodic saves and a final save, excluding startup.

    The fixed-point upper bound also charges save time against the wall-clock
    save interval. It deliberately overcounts if the runtime resets that timer
    after each completed save. No new hardware measurement is implied.
    """
    if (type(updates) is not int or updates < 1 or not math.isfinite(step_seconds)
            or step_seconds <= 0 or not math.isfinite(checkpoint_seconds)
            or checkpoint_seconds < 0 or not math.isfinite(margin) or margin < 1
            or type(checkpoint_interval_seconds) is not int or checkpoint_interval_seconds < 1
            or type(checkpoint_interval_updates) is not int or checkpoint_interval_updates < 1
            or checkpoint_seconds >= checkpoint_interval_seconds):
        raise ValueError('invalid cost projection inputs')
    work = updates * step_seconds * margin
    saves = max(math.ceil(work / checkpoint_interval_seconds),
                math.ceil(updates / checkpoint_interval_updates)) + 1
    for _ in range(10000):
        total = work + saves * checkpoint_seconds
        required = max(math.ceil(total / checkpoint_interval_seconds),
                       math.ceil(updates / checkpoint_interval_updates)) + 1
        if required <= saves:
            return {'training_seconds': work, 'checkpoint_saves_upper_bound': saves,
                    'checkpoint_seconds': saves * checkpoint_seconds, 'total_seconds': total}
        saves = required
    raise ValueError('checkpoint projection did not converge')

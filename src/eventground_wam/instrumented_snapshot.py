"""Strict snapshot-forward audit allowing only MuJoCo quaternion normalization."""

import numpy as np


def verify_snapshot_forward(before, after, normalized_qpos, quaternion_indices):
    """Require exact canonical physics and bound quaternion-only roundoff.

    `normalized_qpos` must be computed by mj_normalizeQuat on a copy of the
    pre-forward qpos. Full-physics layout is time, qpos, qvel, activation, plugin.
    This function never changes either simulator or weakens contact predicates.
    """
    before = np.asarray(before)
    after = np.asarray(after)
    normalized_qpos = np.asarray(normalized_qpos)
    nq = len(normalized_qpos)
    if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Invalid physics state")
    expected = before.copy()
    expected[1 : 1 + nq] = normalized_qpos
    allowed = np.zeros(before.shape, dtype=bool)
    allowed[1 + np.asarray(quaternion_indices, dtype=int)] = True
    if not np.array_equal(expected[~allowed], before[~allowed]):
        raise ValueError("Normalization altered a non-quaternion physics coordinate")
    adjustment = float(np.max(np.abs(expected - before)))
    if adjustment > 1e-12:
        raise ValueError("Quaternion adjustment exceeds the roundoff ceiling")
    if not np.array_equal(after, expected):
        raise ValueError("Forward physics differs from exact MuJoCo canonicalization")
    return adjustment

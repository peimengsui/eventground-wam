import numpy as np
import pytest

from eventground_wam.instrumented_snapshot import verify_snapshot_forward


def test_allows_only_exact_bounded_canonicalization():
    before = np.array([0.5, 3.0, 1.0 - 2e-16, 0.0, 0.0, 0.0, 0.1])
    canonical = before[1:6].copy()
    canonical[1] = 1.0
    after = before.copy()
    after[1:6] = canonical
    assert 0 < verify_snapshot_forward(before, after, canonical, [1, 2, 3, 4]) < 1e-12
    for index in (0, 1, 2, 6):
        changed = after.copy()
        changed[index] += 1e-14
        with pytest.raises(ValueError):
            verify_snapshot_forward(before, changed, canonical, [1, 2, 3, 4])


def test_rejects_large_normalization_and_nonquaternion_changes():
    before = np.array([0.5, 3.0, 0.99, 0.0, 0.0, 0.0, 0.1])
    canonical = before[1:6].copy()
    canonical[1] = 1.0
    after = before.copy()
    after[1:6] = canonical
    with pytest.raises(ValueError, match="roundoff"):
        verify_snapshot_forward(before, after, canonical, [1, 2, 3, 4])
    canonical = before[1:6].copy()
    canonical[0] += 1e-14
    after[1:6] = canonical
    with pytest.raises(ValueError, match="non-quaternion"):
        verify_snapshot_forward(before, after, canonical, [1, 2, 3, 4])

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from validate_demonstration_stream import evenly_spaced_indices


def test_evenly_spaced_indices_include_dataset_endpoints() -> None:
    assert evenly_spaced_indices(14_347, 3) == [0, 7_173, 14_346]
    assert evenly_spaced_indices(3, 5) == [0, 1, 2]


@pytest.mark.parametrize(("total", "count"), [(0, 1), (3, 0), (3, -1)])
def test_evenly_spaced_indices_reject_invalid_counts(total: int, count: int) -> None:
    with pytest.raises(ValueError):
        evenly_spaced_indices(total, count)

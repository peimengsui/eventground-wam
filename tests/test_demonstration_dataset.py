from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from check_demonstrations import evenly_spaced_indices
from demonstration_dataset import SourceFile, canonical_source_files, validate_relative_path


def test_canonical_source_files_are_sorted() -> None:
    files = [SourceFile("z", 2, ""), SourceFile("a", 1, "f" * 64)]
    assert canonical_source_files(files) == f"a\t1\t{'f' * 64}\nz\t2\t\n".encode()


@pytest.mark.parametrize("path", ["", ".", "../escape", "a/../../escape", "/absolute"])
def test_repository_paths_must_be_relative_and_contained(path: str) -> None:
    with pytest.raises(ValueError, match="unsafe repository path"):
        validate_relative_path(path)


def test_evenly_spaced_indices_include_ends() -> None:
    assert evenly_spaced_indices(14_347, 3) == [0, 7_173, 14_346]

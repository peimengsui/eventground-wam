from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from check_demonstrations import evenly_spaced_indices
from demonstration_dataset import SourceFile, canonical_source_files, validate_relative_path
from download_demonstrations import cached_xet_connection_refresher


def test_canonical_source_files_are_sorted() -> None:
    files = [SourceFile("z", 2, ""), SourceFile("a", 1, "f" * 64)]
    assert canonical_source_files(files) == f"a\t1\t{'f' * 64}\nz\t2\t\n".encode()


@pytest.mark.parametrize("path", ["", ".", "../escape", "a/../../escape", "/absolute"])
def test_repository_paths_must_be_relative_and_contained(path: str) -> None:
    with pytest.raises(ValueError, match="unsafe repository path"):
        validate_relative_path(path)


def test_evenly_spaced_indices_include_ends() -> None:
    assert evenly_spaced_indices(14_347, 3) == [0, 7_173, 14_346]


def test_xet_connection_info_is_cached_by_refresh_route_until_near_expiry() -> None:
    now = [1_000.0]
    calls: list[str] = []

    def refresh_connection(*, file_data: object, headers: dict[str, str]) -> object:
        calls.append(file_data.refresh_route)  # type: ignore[attr-defined]
        return SimpleNamespace(expiration_unix_epoch=2_000, headers=headers)

    cached_refresh = cached_xet_connection_refresher(
        refresh_connection,
        clock=lambda: now[0],
    )
    file_data = SimpleNamespace(refresh_route="https://hub.example/xet/repository/revision")

    first = cached_refresh(file_data=file_data, headers={"authorization": "token"})
    second = cached_refresh(file_data=file_data, headers={"authorization": "token"})

    assert second is first
    assert calls == [file_data.refresh_route]

    now[0] = 1_950.0
    third = cached_refresh(file_data=file_data, headers={"authorization": "token"})

    assert third is not first
    assert calls == [file_data.refresh_route, file_data.refresh_route]


def test_xet_connection_info_is_not_shared_between_refresh_routes() -> None:
    calls: list[str] = []

    def refresh_connection(*, file_data: object, headers: dict[str, str]) -> object:
        calls.append(file_data.refresh_route)  # type: ignore[attr-defined]
        return SimpleNamespace(expiration_unix_epoch=2_000, headers=headers)

    cached_refresh = cached_xet_connection_refresher(
        refresh_connection,
        clock=lambda: 1_000.0,
    )

    cached_refresh(file_data=SimpleNamespace(refresh_route="route-a"), headers={})
    cached_refresh(file_data=SimpleNamespace(refresh_route="route-b"), headers={})

    assert calls == ["route-a", "route-b"]

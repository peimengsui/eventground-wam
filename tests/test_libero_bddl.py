from pathlib import Path

import pytest

from eventground_wam.libero_bddl import resolve_libero_bddl_paths


def test_exact_bddl_path_is_both_runtime_and_source(tmp_path: Path) -> None:
    source = tmp_path / "pick_up_the_bowl.bddl"
    source.write_text("(define (problem test))\n", encoding="utf-8")

    paths = resolve_libero_bddl_paths(source)

    assert paths.runtime == source.resolve()
    assert paths.source == source.resolve()


def test_synthetic_perturbation_path_resolves_to_source_bddl(tmp_path: Path) -> None:
    source = tmp_path / "pick_up_the_bowl_language_7.bddl"
    source.write_text("(define (problem test))\n", encoding="utf-8")
    runtime = tmp_path / (
        "pick_up_the_bowl_language_7_view_0_0_100_4_354_"
        "initstate_12_noise_3.bddl"
    )

    paths = resolve_libero_bddl_paths(runtime)

    assert paths.runtime == runtime.resolve()
    assert not paths.runtime.exists()
    assert paths.source == source.resolve()


def test_missing_source_bddl_is_rejected(tmp_path: Path) -> None:
    runtime = tmp_path / "missing_view_0_0_100_0_0_initstate_0.bddl"

    with pytest.raises(FileNotFoundError, match="expected_source"):
        resolve_libero_bddl_paths(runtime)

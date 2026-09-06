"""Resolve LIBERO-Plus task-map paths to runtime and source BDDL files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LiberoBDDLPaths:
    """Paths required by LIBERO-Plus and by artifact provenance."""

    runtime: Path
    source: Path


def resolve_libero_bddl_paths(value: str | Path) -> LiberoBDDLPaths:
    """Return the synthetic runtime path and the BDDL file it resolves to.

    LIBERO-Plus encodes camera, robot-initialization, and observation-noise
    perturbations in a synthetic ``*_view_*_initstate_*`` filename. Its
    ``ControlEnv`` parses that filename before opening the shorter source BDDL.
    The synthetic path therefore need not exist, while the source path must.
    """

    runtime = Path(value).expanduser().resolve()
    runtime_text = str(runtime)
    if "_view_" in runtime_text and "_initstate_" in runtime_text:
        source = Path(runtime_text.split("_view_", 1)[0] + ".bddl")
    else:
        source = runtime

    if not source.is_file():
        raise FileNotFoundError(
            "LIBERO-Plus source BDDL does not exist: "
            f"runtime={runtime}, expected_source={source}"
        )
    return LiberoBDDLPaths(runtime=runtime, source=source.resolve())

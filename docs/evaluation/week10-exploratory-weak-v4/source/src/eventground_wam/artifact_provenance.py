"""Helpers for binding generated artifacts to the current source tree."""

from __future__ import annotations

import subprocess
from pathlib import Path


def project_state(project_root: Path) -> tuple[str, bool]:
    """Return the complete Git revision and whether the worktree is clean."""

    revision = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(project_root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return revision, not bool(status.strip())

"""Generate or verify the locked LIBERO-Plus evaluation split manifests."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from eventground_wam.evaluation_protocol import DEFAULT_PROTOCOL_PATH, load_evaluation_protocol
from eventground_wam.evaluation_splits import (
    SPLIT_IDS,
    SplitValidationError,
    generate_split_manifests,
    manifest_text,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "splits" / "libero-plus-v1"


def default_classification_path() -> Path:
    upstream_root = Path(
        os.environ.get("EVENTGROUND_LIBERO_ROOT", "~/code/vendor/LIBERO-plus")
    ).expanduser()
    return upstream_root / "libero" / "libero" / "benchmark" / "task_classification.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--classification", type=Path, default=default_classification_path())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate in memory and fail if committed manifests differ",
    )
    parser.add_argument(
        "--verify-runtime",
        action="store_true",
        help="Also compare all task names with the importable LIBERO runtime",
    )
    return parser.parse_args()


def verify_runtime(
    manifests: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    """Verify the classification ID-to-runtime-index mapping against LIBERO."""

    try:
        from libero.libero import benchmark
    except ImportError as error:
        raise SplitValidationError(
            "--verify-runtime requires the LIBERO environment; use scripts/run_libero.sh"
        ) from error

    full_tasks = manifests["full-10K"]["tasks"]
    tasks_by_suite: dict[str, list[Mapping[str, Any]]] = {
        suite: [] for suite in protocol["benchmark"]["suites"]
    }
    for task in full_tasks:
        tasks_by_suite[task["suite"]].append(task)

    benchmark_classes = benchmark.get_benchmark_dict()
    for suite, expected_tasks in tasks_by_suite.items():
        with contextlib.redirect_stdout(io.StringIO()):
            runtime_suite = benchmark_classes[suite]()
        if runtime_suite.n_tasks != len(expected_tasks):
            raise SplitValidationError(f"Runtime task count differs for {suite}")
        for task in expected_tasks:
            runtime_task = runtime_suite.get_task(task["runtime_task_index"])
            if runtime_task.name != task["task_name"]:
                raise SplitValidationError(
                    f"Runtime name mismatch for {suite} task {task['runtime_task_index']}"
                )


def main() -> None:
    args = parse_args()
    protocol = load_evaluation_protocol(args.protocol)
    manifests = generate_split_manifests(args.classification, protocol)
    output_directory = args.output_dir.expanduser().resolve()

    if args.verify_runtime:
        verify_runtime(manifests, protocol)

    if args.check:
        mismatches: list[str] = []
        for split_id in SPLIT_IDS:
            path = output_directory / f"{split_id}.json"
            try:
                committed_text = path.read_text(encoding="utf-8")
            except OSError:
                mismatches.append(str(path))
                continue
            if committed_text != manifest_text(manifests[split_id]):
                mismatches.append(str(path))
        if mismatches:
            raise SystemExit("Split manifests are missing or stale:\n" + "\n".join(mismatches))
        action = "verified"
    else:
        output_directory.mkdir(parents=True, exist_ok=True)
        for split_id in SPLIT_IDS:
            path = output_directory / f"{split_id}.json"
            temporary_path = path.with_suffix(".json.tmp")
            temporary_path.write_text(manifest_text(manifests[split_id]), encoding="utf-8")
            temporary_path.replace(path)
        action = "generated"

    for split_id in SPLIT_IDS:
        manifest = manifests[split_id]
        unlabeled = sum(task["difficulty_level"] is None for task in manifest["tasks"])
        print(
            f"{action}: {split_id} tasks={manifest['task_count']} "
            f"unlabeled={unlabeled} records_sha256={manifest['task_records_sha256']}"
        )


if __name__ == "__main__":
    main()

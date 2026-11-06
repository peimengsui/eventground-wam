"""Deterministically generate and validate LIBERO-Plus evaluation splits."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Hashable, Mapping, Sequence
from fractions import Fraction
from functools import cache
from itertools import pairwise
from pathlib import Path
from typing import Any, TypeVar

from eventground_wam.evaluation_protocol import (
    derive_episode_seed,
    protocol_sha256,
    validate_evaluation_protocol,
)

CLASSIFICATION_SHA256 = "faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6"
SPLIT_IDS = ("sanity-35", "dev-210", "stratified-1K", "full-10K")
MANIFEST_SCHEMA_VERSION = 1
ALGORITHM_VERSION = "eventground-wam-splits-v1"

_Key = TypeVar("_Key", bound=Hashable)


class SplitValidationError(ValueError):
    """Raised when classification data or generated split manifests are invalid."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selection_rank(protocol: Mapping[str, Any], task: Mapping[str, Any]) -> str:
    payload = "|".join(
        [
            str(protocol["seeding"]["split_sampling_seed"]),
            str(protocol["protocol_id"]),
            str(task["suite"]),
            str(task["classification_task_id"]),
            str(task["task_name"]),
        ]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ranked_tasks(
    protocol: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    suite_order = {name: index for index, name in enumerate(protocol["benchmark"]["suites"])}
    return sorted(
        (dict(task) for task in tasks),
        key=lambda task: (
            _selection_rank(protocol, task),
            suite_order[task["suite"]],
            task["runtime_task_index"],
        ),
    )


def _apportion(
    total: int,
    populations: Mapping[_Key, int],
    ordered_keys: Sequence[_Key],
    lower_bounds: Mapping[_Key, int] | None = None,
) -> dict[_Key, int]:
    """Allocate *total* proportionally with deterministic lower bounds and ties."""

    if total < 0 or not populations:
        raise SplitValidationError("Apportionment requires a nonnegative total and populations")
    if set(populations) != set(ordered_keys):
        raise SplitValidationError("Apportionment order must contain every population key once")
    bounds = {key: 0 for key in ordered_keys}
    if lower_bounds is not None:
        unknown = set(lower_bounds) - set(populations)
        if unknown:
            raise SplitValidationError(f"Lower bounds contain unknown strata: {unknown!r}")
        bounds.update(lower_bounds)
    if any(populations[key] < bounds[key] or bounds[key] < 0 for key in ordered_keys):
        raise SplitValidationError("Apportionment lower bounds exceed a population")
    if sum(bounds.values()) > total or total > sum(populations.values()):
        raise SplitValidationError("Apportionment total is incompatible with its bounds")

    population_total = sum(populations.values())
    ideals = {
        key: Fraction(total * populations[key], population_total) for key in ordered_keys
    }
    allocation = {
        key: max(bounds[key], ideals[key].numerator // ideals[key].denominator)
        for key in ordered_keys
    }

    while sum(allocation.values()) < total:
        candidates = [key for key in ordered_keys if allocation[key] < populations[key]]
        if not candidates:
            raise SplitValidationError("No capacity remains while apportioning tasks")
        best_key = max(
            candidates,
            key=lambda key: (ideals[key] - allocation[key], -ordered_keys.index(key)),
        )
        allocation[best_key] += 1

    while sum(allocation.values()) > total:
        candidates = [key for key in ordered_keys if allocation[key] > bounds[key]]
        if not candidates:
            raise SplitValidationError("Lower bounds prevent the requested apportionment")
        worst_key = min(
            candidates,
            key=lambda key: (ideals[key] - allocation[key], ordered_keys.index(key)),
        )
        allocation[worst_key] -= 1

    return allocation


def _task_key(task: Mapping[str, Any]) -> tuple[str, int]:
    return str(task["suite"]), int(task["runtime_task_index"])


def _difficulty_stratum(task: Mapping[str, Any]) -> int | str:
    difficulty = task["difficulty_level"]
    return "unlabeled" if difficulty is None else int(difficulty)


def _canonical_task_order(
    protocol: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    suite_order = {name: index for index, name in enumerate(protocol["benchmark"]["suites"])}
    return sorted(
        (dict(task) for task in tasks),
        key=lambda task: (suite_order[task["suite"]], task["runtime_task_index"]),
    )


def _select_balanced_sanity_tasks(
    protocol: Mapping[str, Any],
    dev_tasks_by_cell: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    all_tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Choose one dev task per cell while meeting a global proportional suite quota."""

    suites = list(protocol["benchmark"]["suites"])
    categories = list(protocol["benchmark"]["classification"]["categories"])
    difficulties = list(protocol["benchmark"]["classification"]["difficulty_levels"])
    suite_populations = Counter(str(task["suite"]) for task in all_tasks)
    target_counts = _apportion(
        protocol["splits"]["sanity-35"]["task_count"],
        suite_populations,
        suites,
    )

    candidate_by_cell_and_suite: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for cell, cell_tasks in dev_tasks_by_cell.items():
        candidates: dict[str, dict[str, Any]] = {}
        for suite in suites:
            suite_tasks = [task for task in cell_tasks if task["suite"] == suite]
            if suite_tasks:
                candidates[suite] = _ranked_tasks(protocol, suite_tasks)[0]
        if not candidates:
            raise SplitValidationError(f"No development task is available for sanity cell {cell!r}")
        candidate_by_cell_and_suite[cell] = candidates

    canonical_cells = [(category, difficulty) for category in categories for difficulty in difficulties]
    canonical_cell_index = {cell: index for index, cell in enumerate(canonical_cells)}
    search_cells = sorted(
        canonical_cells,
        key=lambda cell: (len(candidate_by_cell_and_suite[cell]), canonical_cell_index[cell]),
    )

    @cache
    def solve(position: int, remaining: tuple[int, ...]) -> tuple[str, ...] | None:
        if position == len(search_cells):
            return () if all(count == 0 for count in remaining) else None
        cell = search_cells[position]
        candidates = candidate_by_cell_and_suite[cell]
        suite_options = sorted(
            candidates,
            key=lambda suite: (
                _selection_rank(protocol, candidates[suite]),
                suites.index(suite),
            ),
        )
        for suite in suite_options:
            suite_position = suites.index(suite)
            if remaining[suite_position] == 0:
                continue
            updated = list(remaining)
            updated[suite_position] -= 1
            future_cells = search_cells[position + 1 :]
            if any(
                updated[index]
                > sum(
                    candidate_suite in candidate_by_cell_and_suite[future_cell]
                    for future_cell in future_cells
                )
                for index, candidate_suite in enumerate(suites)
            ):
                continue
            tail = solve(position + 1, tuple(updated))
            if tail is not None:
                return (suite, *tail)
        return None

    initial_remaining = tuple(target_counts[suite] for suite in suites)
    assignment = solve(0, initial_remaining)
    if assignment is None:
        raise SplitValidationError("Could not satisfy sanity split suite quotas within dev-210")
    suite_by_cell = dict(zip(search_cells, assignment))
    return [
        candidate_by_cell_and_suite[cell][suite_by_cell[cell]] for cell in canonical_cells
    ]


def load_classification(
    path: str | Path, protocol: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], str]:
    """Load, checksum, normalize, and validate the pinned classification JSON."""

    validate_evaluation_protocol(protocol)
    classification_path = Path(path).expanduser().resolve()
    try:
        source_bytes = classification_path.read_bytes()
        source = json.loads(source_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise SplitValidationError(
            f"Could not load task classification {classification_path}: {error}"
        ) from error
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != CLASSIFICATION_SHA256:
        raise SplitValidationError(
            "Classification checksum does not match the locked LIBERO-Plus revision: "
            f"expected {CLASSIFICATION_SHA256}, found {source_sha256}"
        )
    if not isinstance(source, dict):
        raise SplitValidationError("Classification root must be a JSON object")

    expected_suites = protocol["benchmark"]["suites"]
    if list(source) != list(expected_suites):
        raise SplitValidationError("Classification suites or their order changed")

    categories = set(protocol["benchmark"]["classification"]["categories"])
    difficulties = set(protocol["benchmark"]["classification"]["difficulty_levels"])
    normalized: list[dict[str, Any]] = []
    for suite, suite_specification in expected_suites.items():
        source_tasks = source[suite]
        if len(source_tasks) != suite_specification["task_count"]:
            raise SplitValidationError(f"Unexpected task count for {suite}")
        expected_ids = list(range(1, len(source_tasks) + 1))
        actual_ids = [task.get("id") for task in source_tasks]
        if actual_ids != expected_ids:
            raise SplitValidationError(f"Classification IDs for {suite} are not contiguous")
        for source_task in source_tasks:
            category = source_task.get("category")
            difficulty = source_task.get("difficulty_level")
            task_name = source_task.get("name")
            if category not in categories:
                raise SplitValidationError(f"Unknown perturbation category: {category!r}")
            if difficulty is not None and difficulty not in difficulties:
                raise SplitValidationError(f"Unknown difficulty level: {difficulty!r}")
            if not isinstance(task_name, str) or not task_name:
                raise SplitValidationError("Every classified task must have a nonempty name")
            classification_task_id = int(source_task["id"])
            runtime_task_index = classification_task_id - 1
            normalized.append(
                {
                    "suite": suite,
                    "runtime_task_index": runtime_task_index,
                    "classification_task_id": classification_task_id,
                    "task_name": task_name,
                    "perturbation_category": category,
                    "difficulty_level": difficulty,
                    "difficulty_stratum": "unlabeled" if difficulty is None else difficulty,
                    "episode_seed": derive_episode_seed(
                        protocol, suite, runtime_task_index, trial_index=0
                    ),
                }
            )

    if len(normalized) != protocol["benchmark"]["total_task_count"]:
        raise SplitValidationError("Normalized classification task count changed")
    missing_count = sum(task["difficulty_level"] is None for task in normalized)
    expected_missing = protocol["benchmark"]["classification"][
        "missing_difficulty_task_count"
    ]
    if missing_count != expected_missing:
        raise SplitValidationError(
            f"Expected {expected_missing} missing difficulty labels, found {missing_count}"
        )
    return normalized, source_sha256


def _make_manifest(
    split_id: str,
    tasks: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    classification_sha256: str,
) -> dict[str, Any]:
    canonical_tasks = _canonical_task_order(protocol, tasks)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split_id": split_id,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256(protocol),
        "source": {
            "benchmark_revision": protocol["benchmark"]["revision"],
            "classification_relative_path": protocol["benchmark"]["classification"][
                "relative_path"
            ],
            "classification_sha256": classification_sha256,
        },
        "generation": {
            "algorithm_version": ALGORITHM_VERSION,
            "sampling_method": protocol["splits"]["sampling_method"],
            "split_sampling_seed": protocol["seeding"]["split_sampling_seed"],
            "rank_payload": (
                "split_sampling_seed|protocol_id|suite|classification_task_id|task_name"
            ),
            "apportionment": "constrained_largest_remainder_with_protocol_order_ties",
        },
        "task_count": len(canonical_tasks),
        "task_records_sha256": _json_sha256(canonical_tasks),
        "tasks": canonical_tasks,
    }


def generate_split_manifests(
    classification_path: str | Path, protocol: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Generate all nested split manifests from the pinned classification file."""

    tasks, classification_sha256 = load_classification(classification_path, protocol)
    suites = list(protocol["benchmark"]["suites"])
    categories = list(protocol["benchmark"]["classification"]["categories"])
    difficulties = list(protocol["benchmark"]["classification"]["difficulty_levels"])

    by_cell_and_suite: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        if task["difficulty_level"] is not None:
            by_cell_and_suite[
                (
                    task["perturbation_category"],
                    task["difficulty_level"],
                    task["suite"],
                )
            ].append(task)

    dev_tasks: list[dict[str, Any]] = []
    dev_tasks_by_cell: dict[tuple[str, int], list[dict[str, Any]]] = {}
    dev_per_cell = protocol["splits"]["dev-210"]["task_count"] // (
        len(categories) * len(difficulties)
    )
    for category in categories:
        for difficulty in difficulties:
            populations = {
                suite: len(by_cell_and_suite[(category, difficulty, suite)])
                for suite in suites
            }
            quotas = _apportion(dev_per_cell, populations, suites)
            cell_dev_tasks: list[dict[str, Any]] = []
            for suite in suites:
                candidates = _ranked_tasks(
                    protocol, by_cell_and_suite[(category, difficulty, suite)]
                )
                cell_dev_tasks.extend(candidates[: quotas[suite]])
            if len(cell_dev_tasks) != dev_per_cell:
                raise SplitValidationError("Development cell allocation has the wrong size")
            dev_tasks.extend(cell_dev_tasks)
            dev_tasks_by_cell[(category, difficulty)] = cell_dev_tasks

    sanity_tasks = _select_balanced_sanity_tasks(protocol, dev_tasks_by_cell, tasks)

    suite_index = {suite: index for index, suite in enumerate(suites)}
    category_index = {category: index for index, category in enumerate(categories)}

    def atomic_key(task: Mapping[str, Any]) -> tuple[str, str, int | str]:
        return (
            str(task["suite"]),
            str(task["perturbation_category"]),
            _difficulty_stratum(task),
        )

    atomic_populations = Counter(atomic_key(task) for task in tasks)
    atomic_order = sorted(
        atomic_populations,
        key=lambda key: (
            suite_index[key[0]],
            category_index[key[1]],
            6 if key[2] == "unlabeled" else int(key[2]),
        ),
    )
    dev_counts = Counter(atomic_key(task) for task in dev_tasks)
    stratified_quotas = _apportion(
        protocol["splits"]["stratified-1K"]["task_count"],
        atomic_populations,
        atomic_order,
        lower_bounds=dev_counts,
    )
    dev_keys = {_task_key(task) for task in dev_tasks}
    remaining_by_stratum: dict[tuple[str, str, int | str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for task in tasks:
        if _task_key(task) not in dev_keys:
            remaining_by_stratum[atomic_key(task)].append(task)

    stratified_tasks = list(dev_tasks)
    for stratum in atomic_order:
        additional_count = stratified_quotas[stratum] - dev_counts[stratum]
        candidates = _ranked_tasks(protocol, remaining_by_stratum[stratum])
        stratified_tasks.extend(candidates[:additional_count])

    manifests = {
        "sanity-35": _make_manifest(
            "sanity-35", sanity_tasks, protocol, classification_sha256
        ),
        "dev-210": _make_manifest("dev-210", dev_tasks, protocol, classification_sha256),
        "stratified-1K": _make_manifest(
            "stratified-1K", stratified_tasks, protocol, classification_sha256
        ),
        "full-10K": _make_manifest("full-10K", tasks, protocol, classification_sha256),
    }
    validate_manifest_family(manifests, protocol)
    return manifests


def manifest_text(manifest: Mapping[str, Any]) -> str:
    """Serialize a manifest deterministically for storage and check mode."""

    return json.dumps(manifest, ensure_ascii=True, indent=2) + "\n"


def validate_manifest(manifest: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    """Validate one split manifest and all of its task records."""

    validate_evaluation_protocol(protocol)
    split_id = manifest.get("split_id")
    if split_id not in SPLIT_IDS:
        raise SplitValidationError(f"Unknown split ID: {split_id!r}")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SplitValidationError(f"Unexpected schema version in {split_id}")
    if manifest.get("protocol_id") != protocol["protocol_id"]:
        raise SplitValidationError(f"Protocol ID mismatch in {split_id}")
    if manifest.get("protocol_sha256") != protocol_sha256(protocol):
        raise SplitValidationError(f"Protocol digest mismatch in {split_id}")
    if manifest.get("task_count") != protocol["splits"][split_id]["task_count"]:
        raise SplitValidationError(f"Task count metadata mismatch in {split_id}")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != manifest["task_count"]:
        raise SplitValidationError(f"Task records mismatch in {split_id}")
    if manifest.get("task_records_sha256") != _json_sha256(tasks):
        raise SplitValidationError(f"Task record digest mismatch in {split_id}")
    if manifest.get("source", {}).get("classification_sha256") != CLASSIFICATION_SHA256:
        raise SplitValidationError(f"Classification digest mismatch in {split_id}")

    suites = protocol["benchmark"]["suites"]
    categories = set(protocol["benchmark"]["classification"]["categories"])
    difficulties = set(protocol["benchmark"]["classification"]["difficulty_levels"])
    suite_order = {suite: index for index, suite in enumerate(suites)}
    task_keys: set[tuple[str, int]] = set()
    previous_order: tuple[int, int] | None = None
    for task in tasks:
        suite = task.get("suite")
        task_index = task.get("runtime_task_index")
        classification_id = task.get("classification_task_id")
        if suite not in suites or not isinstance(task_index, int):
            raise SplitValidationError(f"Invalid suite or task index in {split_id}")
        if not 0 <= task_index < suites[suite]["task_count"]:
            raise SplitValidationError(f"Out-of-range task index in {split_id}")
        if classification_id != task_index + 1:
            raise SplitValidationError(f"One-based task ID mismatch in {split_id}")
        if task.get("perturbation_category") not in categories:
            raise SplitValidationError(f"Unknown category in {split_id}")
        difficulty = task.get("difficulty_level")
        if difficulty is not None and difficulty not in difficulties:
            raise SplitValidationError(f"Unknown difficulty in {split_id}")
        expected_stratum = "unlabeled" if difficulty is None else difficulty
        if task.get("difficulty_stratum") != expected_stratum:
            raise SplitValidationError(f"Difficulty stratum mismatch in {split_id}")
        if task.get("episode_seed") != derive_episode_seed(protocol, suite, task_index):
            raise SplitValidationError(f"Episode seed mismatch in {split_id}")
        key = (suite, task_index)
        if key in task_keys:
            raise SplitValidationError(f"Duplicate task in {split_id}: {key!r}")
        task_keys.add(key)
        current_order = (suite_order[suite], task_index)
        if previous_order is not None and current_order <= previous_order:
            raise SplitValidationError(f"Tasks are not in canonical order in {split_id}")
        previous_order = current_order


def validate_manifest_family(
    manifests: Mapping[str, Mapping[str, Any]], protocol: Mapping[str, Any]
) -> None:
    """Validate all manifests, nesting, and the exact stratification guarantees."""

    if set(manifests) != set(SPLIT_IDS):
        raise SplitValidationError("Manifest family does not contain all protocol splits")
    for split_id in SPLIT_IDS:
        validate_manifest(manifests[split_id], protocol)

    task_sets = {
        split_id: {_task_key(task) for task in manifests[split_id]["tasks"]}
        for split_id in SPLIT_IDS
    }
    for smaller, larger in pairwise(SPLIT_IDS):
        if not task_sets[smaller] < task_sets[larger]:
            raise SplitValidationError(f"{smaller} is not a strict subset of {larger}")

    for split_id, expected_per_cell in (("sanity-35", 1), ("dev-210", 6)):
        cell_counts = Counter(
            (task["perturbation_category"], task["difficulty_level"])
            for task in manifests[split_id]["tasks"]
        )
        if len(cell_counts) != 35 or set(cell_counts.values()) != {expected_per_cell}:
            raise SplitValidationError(f"{split_id} does not cover the labeled 7x5 grid")
        if any(task["difficulty_level"] is None for task in manifests[split_id]["tasks"]):
            raise SplitValidationError(f"{split_id} unexpectedly contains unlabeled tasks")

    stratified_unlabeled = sum(
        task["difficulty_level"] is None for task in manifests["stratified-1K"]["tasks"]
    )
    if stratified_unlabeled == 0:
        raise SplitValidationError("stratified-1K did not preserve unlabeled tasks")
    full_unlabeled = sum(
        task["difficulty_level"] is None for task in manifests["full-10K"]["tasks"]
    )
    expected_unlabeled = protocol["benchmark"]["classification"][
        "missing_difficulty_task_count"
    ]
    if full_unlabeled != expected_unlabeled:
        raise SplitValidationError("full-10K does not preserve all unlabeled tasks")


def load_manifest_family(
    directory: str | Path, protocol: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Load and validate the four committed split manifests."""

    manifest_directory = Path(directory).expanduser().resolve()
    manifests: dict[str, dict[str, Any]] = {}
    for split_id in SPLIT_IDS:
        path = manifest_directory / f"{split_id}.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SplitValidationError(f"Could not load split manifest {path}: {error}") from error
        if not isinstance(manifest, dict):
            raise SplitValidationError(f"Split manifest {path} must contain a JSON object")
        manifests[split_id] = manifest
    validate_manifest_family(manifests, protocol)
    return manifests

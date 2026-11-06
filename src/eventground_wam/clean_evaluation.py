"""Canonical clean-40 inventory shared by B0 and external reference evaluators."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from eventground_wam.b0_training_rollout import (
    BASE_SUITES,
    BaseTaskSelection,
    BaseTrainingTaskSuite,
    _base_task_paths,
)
from eventground_wam.demonstration_contracts import canonical_json_bytes, seal_contract
from eventground_wam.episode_recorder import sha256_file
from eventground_wam.evaluation_protocol import derive_episode_seed, protocol_sha256


def clean_instruction(task_name: str) -> str:
    # Same canonical filename rule as the pinned upstream grab_language_from_filename.
    if task_name[0].isupper():
        task_name = re.sub(r"^.*SCENE\d+_", "", task_name)
    return task_name.replace("_", " ")


def clean_manifest(libero_root: Path, protocol: Any) -> tuple[Any, dict[Any, Any]]:
    import hashlib

    tasks = []
    suites = {}
    for suite in BASE_SUITES:
        inventory = libero_root / "libero/libero/bddl_files" / suite / "tasks_info.txt"
        for index, (bddl, states) in enumerate(_base_task_paths(libero_root, suite)):
            selection = BaseTaskSelection(
                suite, index, bddl.stem, clean_instruction(bddl.stem), bddl, states, inventory,
            )
            suites[(suite, index)] = BaseTrainingTaskSuite(selection)
            tasks.append({
                "suite": suite, "runtime_task_index": index, "classification_task_id": index + 1,
                "task_name": bddl.stem, "instruction": selection.instruction,
                "perturbation_category": "Clean", "difficulty_level": None,
                "episode_seed": derive_episode_seed(protocol, suite, index),
                "bddl_sha256": sha256_file(bddl), "initial_states_file_sha256": sha256_file(states),
                "tasks_info_sha256": sha256_file(inventory),
            })
    return seal_contract({
        "artifact_type": "eventground_wam.clean_evaluation_manifest", "schema_version": 1,
        "split_id": "clean-40", "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256(protocol), "task_count": 40,
        "task_records_sha256": hashlib.sha256(canonical_json_bytes(tasks)).hexdigest(),
        "tasks": tasks,
    }), suites


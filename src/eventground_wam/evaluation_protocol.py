"""Load and validate locked evaluation protocols."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "evaluation"
    / "libero_plus_v1.json"
)
B0_TRAINING_ROLLOUT_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "evaluation"
    / "week3_b0_training_rollout_v1.json"
)

_EXPECTED_SUITES = {
    "libero_spatial": {"task_count": 2402, "max_policy_steps": 220},
    "libero_object": {"task_count": 2518, "max_policy_steps": 280},
    "libero_goal": {"task_count": 2591, "max_policy_steps": 300},
    "libero_10": {"task_count": 2519, "max_policy_steps": 520},
}
_EXPECTED_CATEGORIES = [
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
]
_EXPECTED_B0_TRAINING_SUITES = {
    "libero_spatial": {"task_count": 10, "max_policy_steps": 220},
    "libero_object": {"task_count": 10, "max_policy_steps": 280},
    "libero_goal": {"task_count": 10, "max_policy_steps": 300},
    "libero_10": {"task_count": 10, "max_policy_steps": 520},
}


CLEAN_COMPARISON_SHA256 = "5e247bf423facc73e3b98f44d9e66c6862860a4218924c2b60f4768f99fce9c2"


class ProtocolValidationError(ValueError):
    """Raised when a protocol does not satisfy its locked invariants."""


def _at(protocol: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = protocol
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise ProtocolValidationError(f"Missing protocol field: {dotted_path}")
        value = value[key]
    return value


def _require_equal(protocol: Mapping[str, Any], dotted_path: str, expected: Any) -> None:
    actual = _at(protocol, dotted_path)
    if actual != expected:
        raise ProtocolValidationError(
            f"Expected {dotted_path}={expected!r}, found {actual!r}"
        )


def validate_evaluation_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate a supported locked evaluation or diagnostic protocol."""

    protocol_id = _at(protocol, "protocol_id")
    if protocol_id == "libero-plus-v1":
        _validate_libero_plus_protocol(protocol)
    elif protocol_id in {"week3-b0-training-rollout-v1", "libero-clean-v1"}:
        _validate_b0_training_rollout_protocol(protocol, clean=protocol_id == "libero-clean-v1")
    elif protocol_id == "week4-clean-comparison-v1":
        encoded = json.dumps(protocol, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        if hashlib.sha256(encoded.encode()).hexdigest() != CLEAN_COMPARISON_SHA256:
            raise ProtocolValidationError("Frozen clean comparison protocol changed")
    else:
        raise ProtocolValidationError(f"Unsupported protocol_id: {protocol_id!r}")


def _validate_libero_plus_protocol(protocol: Mapping[str, Any]) -> None:
    """Validate immutable LIBERO-Plus v1 scientific choices."""

    required_values = {
        "schema_version": 1,
        "protocol_id": "libero-plus-v1",
        "status": "locked",
        "benchmark.revision": "4976dc30028e805ff8094b55501d532c48fec182",
        "benchmark.total_task_count": 10030,
        "benchmark.trials_per_task": 1,
        "benchmark.classification.classification_task_id_base": 1,
        "benchmark.classification.runtime_task_index_base": 0,
        "benchmark.classification.categories": _EXPECTED_CATEGORIES,
        "benchmark.classification.difficulty_levels": [1, 2, 3, 4, 5],
        "benchmark.classification.missing_difficulty_task_count": 121,
        "simulation.control_frequency_hz": 20,
        "simulation.settling_steps": 10,
        "simulation.action.dimension": 7,
        "simulation.action.chunk_length": 8,
        "simulation.action.replan_interval_steps": 8,
        "observation.capture_resolution": [256, 256],
        "observation.image_transform": "rotate_180",
        "observation.cameras.agent": "agentview_image",
        "observation.cameras.wrist": "robot0_eye_in_hand_image",
        "observation.proprioception.dimension": 8,
        "seeding.environment_seed": 0,
        "seeding.base_episode_seed": 7,
        "seeding.split_sampling_seed": 7,
        "splits.sanity-35.task_count": 35,
        "splits.dev-210.task_count": 210,
        "splits.stratified-1K.task_count": 1000,
        "splits.full-10K.task_count": 10030,
        "episode_outcomes.timeout_counts_as_failure": True,
        "episode_outcomes.policy_exception_counts_as_failure": True,
        "metrics.primary": "micro_success_rate_over_tasks",
        "metrics.unlabeled_difficulty_is_never_imputed": True,
        "recording.container": "HDF5",
        "recording.record_every_control_step": True,
        "reference_policy_profile.checkpoint_revision": (
            "a85655ec941bae6644c9fbdf62db02b9726d7cf5"
        ),
        "reference_policy_profile.evaluator_revision": (
            "e4287e94541f459edc4feabc4e181f537cd569a8"
        ),
    }
    for dotted_path, expected in required_values.items():
        _require_equal(protocol, dotted_path, expected)

    suites = _at(protocol, "benchmark.suites")
    if suites != _EXPECTED_SUITES:
        raise ProtocolValidationError(
            f"Benchmark suite inventory changed: expected {_EXPECTED_SUITES!r}, found {suites!r}"
        )
    task_count = sum(spec["task_count"] for spec in suites.values())
    if task_count != _at(protocol, "benchmark.total_task_count"):
        raise ProtocolValidationError(
            f"Suite task counts sum to {task_count}, not benchmark.total_task_count"
        )

    split_sizes = [
        _at(protocol, "splits.sanity-35.task_count"),
        _at(protocol, "splits.dev-210.task_count"),
        _at(protocol, "splits.stratified-1K.task_count"),
        _at(protocol, "splits.full-10K.task_count"),
    ]
    if split_sizes != sorted(split_sizes) or not _at(protocol, "splits.nested"):
        raise ProtocolValidationError("Evaluation splits must be nested in increasing size")

    if _at(protocol, "simulation.action.chunk_length") != _at(
        protocol, "reference_policy_profile.action_chunk_length"
    ):
        raise ProtocolValidationError("Shared and reference-policy action chunks disagree")
    if _at(protocol, "simulation.action.replan_interval_steps") != _at(
        protocol, "reference_policy_profile.open_loop_steps"
    ):
        raise ProtocolValidationError("Shared and reference-policy replanning intervals disagree")


def _validate_b0_training_rollout_protocol(
    protocol: Mapping[str, Any], *, clean: bool = False,
) -> None:
    """Validate the Week 3 unperturbed training-task smoke protocol."""

    required_values = {
        "schema_version": 1,
        "protocol_id": "libero-clean-v1" if clean else "week3-b0-training-rollout-v1",
        "status": "locked",
        "benchmark.revision": "4976dc30028e805ff8094b55501d532c48fec182",
        "benchmark.total_task_count": 40,
        "benchmark.trials_per_task": 1,
        "benchmark.classification.classification_task_id_base": 1,
        "benchmark.classification.runtime_task_index_base": 0,
        "benchmark.classification.categories": ["Clean"] if clean else ["Training Cohort"],
        "benchmark.initial_state.state_index": 0,
        "benchmark.classification.difficulty_levels": [],
        "benchmark.classification.missing_difficulty_task_count": 40,
        "simulation.control_frequency_hz": 20,
        "simulation.settling_steps": 10,
        "simulation.action.dimension": 7,
        "simulation.action.chunk_length": 8,
        "simulation.action.replan_interval_steps": 8,
        "observation.capture_resolution": [256, 256],
        "observation.image_transform": "rotate_180",
        "observation.cameras.agent": "agentview_image",
        "observation.cameras.wrist": "robot0_eye_in_hand_image",
        "observation.proprioception.dimension": 8,
        "seeding.environment_seed": 0,
        "seeding.base_episode_seed": 7,
        "episode_outcomes.timeout_counts_as_failure": True,
        "episode_outcomes.policy_exception_counts_as_failure": True,
        "recording.container": "HDF5",
        "recording.record_every_control_step": True,
        "b0_policy_profile.objective": "continuous-l1-plus-gripper-bce-v1",
        "b0_policy_profile.gripper_bce_weight": 1.0,
        "b0_policy_profile.action_chunk_length": 8,
        "b0_policy_profile.open_loop_steps": 8,
    }
    for dotted_path, expected in required_values.items():
        _require_equal(protocol, dotted_path, expected)

    suites = _at(protocol, "benchmark.suites")
    if suites != _EXPECTED_B0_TRAINING_SUITES:
        raise ProtocolValidationError(
            "B0 training-task suite inventory changed: "
            f"expected {_EXPECTED_B0_TRAINING_SUITES!r}, found {suites!r}"
        )
    if sum(spec["task_count"] for spec in suites.values()) != 40:
        raise ProtocolValidationError("B0 training-task suite counts must sum to 40")
    if _at(protocol, "simulation.action.chunk_length") != _at(
        protocol, "b0_policy_profile.action_chunk_length"
    ):
        raise ProtocolValidationError("Shared and B0 action chunks disagree")
    if _at(protocol, "simulation.action.replan_interval_steps") != _at(
        protocol, "b0_policy_profile.open_loop_steps"
    ):
        raise ProtocolValidationError("Shared and B0 replanning intervals disagree")


def load_evaluation_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    """Load JSON from *path*, validate it, and return the decoded protocol."""

    protocol_path = Path(path).expanduser().resolve()
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolValidationError(
            f"Could not load evaluation protocol {protocol_path}: {error}"
        ) from error
    if not isinstance(protocol, dict):
        raise ProtocolValidationError("Evaluation protocol root must be a JSON object")
    validate_evaluation_protocol(protocol)
    return protocol


def canonical_protocol_bytes(protocol: Mapping[str, Any]) -> bytes:
    """Return a stable JSON representation used for protocol identity."""

    validate_evaluation_protocol(protocol)
    return json.dumps(
        protocol,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    """Return the SHA-256 identity of a validated protocol."""

    return hashlib.sha256(canonical_protocol_bytes(protocol)).hexdigest()


def derive_episode_seed(
    protocol: Mapping[str, Any],
    suite: str,
    runtime_task_index: int,
    trial_index: int = 0,
) -> int:
    """Derive an order-independent seed for one evaluation episode."""

    validate_evaluation_protocol(protocol)
    suites = _at(protocol, "benchmark.suites")
    if suite not in suites:
        raise ProtocolValidationError(f"Unknown suite for episode seed: {suite!r}")
    if not 0 <= runtime_task_index < suites[suite]["task_count"]:
        raise ProtocolValidationError(
            f"runtime_task_index {runtime_task_index} is outside suite {suite!r}"
        )
    if not 0 <= trial_index < _at(protocol, "benchmark.trials_per_task"):
        raise ProtocolValidationError(f"trial_index {trial_index} is outside the protocol")

    payload = "|".join(
        [
            str(_at(protocol, "seeding.base_episode_seed")),
            str(_at(protocol, "protocol_id")),
            suite,
            str(runtime_task_index),
            str(trial_index),
        ]
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2_147_483_647
    return seed or 1

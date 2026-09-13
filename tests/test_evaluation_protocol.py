from __future__ import annotations

from copy import deepcopy

import pytest

from eventground_wam.evaluation_protocol import (
    B0_TRAINING_ROLLOUT_PROTOCOL_PATH,
    DEFAULT_PROTOCOL_PATH,
    ProtocolValidationError,
    derive_episode_seed,
    load_evaluation_protocol,
    protocol_sha256,
    validate_evaluation_protocol,
)

EXPECTED_V1_SHA256 = "e22ef918b7703e7e3d2cdab475558217ce900a839bf2bf36f6ce92395c738ebd"


def test_locked_protocol_identity() -> None:
    protocol = load_evaluation_protocol()

    assert DEFAULT_PROTOCOL_PATH.is_file()
    assert protocol["protocol_id"] == "libero-plus-v1"
    assert protocol_sha256(protocol) == EXPECTED_V1_SHA256


def test_episode_seed_is_stable_and_order_independent() -> None:
    protocol = load_evaluation_protocol()

    assert derive_episode_seed(protocol, "libero_spatial", 0) == 1_518_511_290
    assert derive_episode_seed(protocol, "libero_spatial", 1) == 1_282_635_642
    assert derive_episode_seed(protocol, "libero_object", 0) == 79_062_796
    assert derive_episode_seed(protocol, "libero_spatial", 0) == 1_518_511_290


def test_protocol_rejects_scientific_drift() -> None:
    protocol = load_evaluation_protocol()
    changed_protocol = deepcopy(protocol)
    changed_protocol["simulation"]["control_frequency_hz"] = 10

    with pytest.raises(ProtocolValidationError, match="control_frequency_hz"):
        validate_evaluation_protocol(changed_protocol)


def test_episode_seed_rejects_invalid_task_index() -> None:
    protocol = load_evaluation_protocol()

    with pytest.raises(ProtocolValidationError, match="outside suite"):
        derive_episode_seed(protocol, "libero_10", 2519)


def test_locked_b0_training_rollout_protocol() -> None:
    protocol = load_evaluation_protocol(B0_TRAINING_ROLLOUT_PROTOCOL_PATH)

    assert protocol["protocol_id"] == "week3-b0-training-rollout-v1"
    assert protocol["benchmark"]["total_task_count"] == 40
    assert protocol["benchmark"]["classification"]["categories"] == [
        "Training Cohort"
    ]
    assert derive_episode_seed(protocol, "libero_goal", 4) > 0


def test_protocol_rejects_unknown_profile() -> None:
    protocol = load_evaluation_protocol()
    protocol["protocol_id"] = "unknown"

    with pytest.raises(ProtocolValidationError, match="Unsupported protocol_id"):
        validate_evaluation_protocol(protocol)

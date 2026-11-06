"""Verify the immutable mapping child and its complete Step 1 parent package."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from eventground_wam.eventground_horizon import (
    EVENT_TYPES,
    NONE_HORIZON,
    RULE_ID,
    SUPPORTED_HORIZONS,
    InvalidEventPrediction,
    select_horizon,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = "configs/evaluation/week11_horizon_mapping_v1.json"
CONTRACT_DIGEST = "9a804bf6d7a0d4afacd7e770279ce666040832bc5584a5a6840ab2007ba5ae7f"


def parent_verifier(root):
    spec = importlib.util.spec_from_file_location(
        "week11_parent_verify", Path(root) / "scripts/verify_week11_protocol.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(root=ROOT):
    root = Path(root)
    parent = parent_verifier(root)
    require = parent.require
    child = parent.read(root / CONTRACT_PATH, sealed=True)
    require(child["digest"] == CONTRACT_DIGEST,
            "Week 11 mapping freeze changed; use a versioned amendment")
    for name, expected in child["source_files_sha256"].items():
        require(parent.sha(root / name) == expected, f"Frozen mapping input changed: {name}")
    binding = child["parent"]
    require(parent.sha(root / binding["path"]) == binding["sha256"], "Parent bytes changed")
    original = parent.read(root / binding["path"], sealed=True)
    require(original["digest"] == binding["digest"] == parent.CONTRACT_DIGEST,
            "Wrong experiment parent")
    package = parent.read(root / binding["package_path"], sealed=True)
    require(package["digest"] == binding["package_digest"]
            and parent.sha(root / binding["package_path"]) == binding["package_sha256"],
            "Original Step 1 package index changed")
    for name, expected in package["files_sha256"].items():
        require(parent.sha(root / name) == expected, f"Original Step 1 package changed: {name}")
    previous = parent.verify(root)
    require(previous["passed"], "Step 1 verification did not pass")

    rule = child["selection"]
    require(RULE_ID == child["rule_id"] and list(SUPPORTED_HORIZONS) == rule["supported_horizons"]
            == original["mapping"]["supported_horizons"]
            and list(EVENT_TYPES) == child["prediction"]["event_type_order"]
            and NONE_HORIZON == rule["none_horizon"], "Rule constants differ from child")
    checked = 0
    for event in EVENT_TYPES[1:]:
        for start in range(1, 33):
            for end in range(start + 1, 34):
                output = select_horizon({"next_event": event, "interval": [start, end],
                                         "horizon": start})
                expected = rule["lookup_by_start_1_through_32"][start - 1]
                require(output.selected_horizon == expected
                        and output.quantization_delta_frames == expected - start
                        and output.fallback_reason is None, "Exhaustive mapping mismatch")
                checked += 1
    none = select_horizon(child["prediction"]["none"])
    require(none.selected_horizon == 8 and none.fallback_reason == rule["none_fallback_reason"]
            and none.raw_start is none.quantization_delta_frames is None, "None rule mismatch")
    try:
        select_horizon({"next_event": "contact", "interval": [33, 34], "horizon": 33})
    except InvalidEventPrediction:
        pass
    else:
        raise ValueError("Invalid prediction silently mapped to a horizon")

    selection = parent.read(root / "docs/evaluation/week11-horizon-v1/selection.json", sealed=True)
    require(selection["contract_digest"] == child["digest"]
            and selection["parent_digest"] == original["digest"]
            and selection["sealed_at_utc"] == child["created_at_utc"]
            and not selection["new_recorded_dev_diagnostic_started"],
            "Pre-diagnostic selection record mismatch")
    result = {
        "artifact_type": "eventground_wam.week11_horizon_verification", "passed": True,
        "contract_digest": child["digest"], "parent_digest": original["digest"],
        "parent_package_digest": package["digest"],
        "parent_package_files_unchanged": len(package["files_sha256"]),
        "child_source_files_verified": len(child["source_files_sha256"]),
        "valid_positive_intervals_checked": checked, "mapping_frozen": True,
        "none_horizon": none.selected_horizon, "confidence_gate": False,
        "model_calls": 0, "robot_rollouts": 0, "remote_bytes_rehashed": False,
        "integration_implemented": False, "hardware_qualified": False,
    }
    result["digest"] = parent.digest(result)
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))

"""Test preservation boundaries and matched-denominator diagnostic statistics."""
import importlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from eventground_wam.eventground_horizon import select_horizon

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return (importlib.import_module("verify_week11_horizon"),
            importlib.import_module("analyze_week11_horizon"))


@pytest.fixture
def frozen_copy(tmp_path, modules):
    verify, _ = modules
    parent = verify.parent_verifier(ROOT)
    child = parent.read(ROOT / verify.CONTRACT_PATH)
    original = parent.read(ROOT / child["parent"]["path"])
    package = parent.read(ROOT / child["parent"]["package_path"])
    names = {verify.CONTRACT_PATH, "docs/evaluation/week11-horizon-v1/selection.json"}
    names.update(child["source_files_sha256"])
    names.update(original["source_files_sha256"])
    names.update(package["files_sha256"])
    for name in names:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    return tmp_path


def test_child_and_original_package_verify(frozen_copy, modules):
    result = modules[0].verify(frozen_copy)
    assert result["passed"]
    assert result["parent_package_files_unchanged"] == 20
    assert result["valid_positive_intervals_checked"] == 1584
    assert not result["integration_implemented"]


def test_resealed_mapping_change_rejected(frozen_copy, modules):
    verify, _ = modules
    parent = verify.parent_verifier(ROOT)
    path = frozen_copy / verify.CONTRACT_PATH
    child = parent.read(path)
    child["selection"]["none_horizon"] = 16
    child["digest"] = parent.digest(child)
    path.write_text(json.dumps(child))
    with pytest.raises(ValueError, match="versioned amendment"):
        verify.verify(frozen_copy)


def test_changed_mapping_helper_rejected(frozen_copy, modules):
    path = frozen_copy / "src/eventground_wam/eventground_horizon.py"
    path.write_text(path.read_text().replace("NONE_HORIZON = 8", "NONE_HORIZON = 16"))
    with pytest.raises(ValueError, match="Frozen mapping input changed"):
        modules[0].verify(frozen_copy)


def test_original_step1_package_remains_immutable(frozen_copy, modules):
    path = frozen_copy / "docs/evaluation/week11-freeze-v1/after/docs/WEEK11_STATUS.md"
    path.write_text("Changed historical snapshot\n")
    with pytest.raises(ValueError, match="Original Step 1 package changed"):
        modules[0].verify(frozen_copy)


def test_timing_uses_same_matched_positive_rows(modules):
    _, analysis = modules
    records = []
    for index, (event, start, target_event, target_start) in enumerate([
        ("contact", 3, "contact", 3), ("grasp", 6, "grasp", 4),
        ("contact", 8, "grasp", 16), ("none", None, "none", None),
    ]):
        prediction = {"next_event": event, "interval": [start, start + 1] if start else None,
                      "horizon": start}
        decision = select_horizon(prediction)
        correct = event == target_event
        matched = correct and target_start is not None
        records.append({"source_id": str(index), "target": {"next_event": target_event},
                        "correct_event": correct, "lead_time_bin": "1-4",
                        "decision": asdict(decision),
                        "raw_signed_error": start - target_start if matched else None,
                        "mapped_signed_error": decision.selected_horizon - target_start
                        if matched else None})
    result = analysis.summarize(records)
    assert result["target_positive"] == 3
    assert result["correctly_typed_positive"] == 2
    assert result["valid_none_fallbacks"] == 1
    assert result["conditional_onset_errors"]["raw"]["n"] == 2
    assert result["conditional_onset_errors"]["mapped"]["n"] == 2
    assert result["conditional_onset_errors"]["raw"]["mae_frames"] == 1
    assert result["conditional_onset_errors"]["mapped"]["mae_frames"] == .5
    assert result["positive_type_coverage"] == pytest.approx(2 / 3)


def test_empty_and_tail_statistics(modules):
    _, analysis = modules
    assert analysis.timing([])["mae_frames"] is None
    summary = analysis.timing([-10, 0, 1])
    assert summary["median_absolute_frames"] == 1
    assert summary["p90_absolute_frames"] == pytest.approx(8.2)
    assert summary["mean_signed_frames"] == -3
    assert summary["within_two_frames"] == 2
    assert summary["more_than_eight_frames"] == 1

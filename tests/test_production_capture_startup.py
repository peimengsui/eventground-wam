import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import seal_contract


def test_sealed_approval_is_preserved_before_simulator_import(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    bundle = root / "docs/evaluation/week10-production-capture-v1"
    spec = importlib.util.spec_from_file_location(
        "record_production_startup", root / "scripts/record_week10_production.py"
    )
    recorder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recorder)
    contract = json.loads((bundle / "recording-contract.json").read_text())
    contract.pop("digest")
    contract["bundle_files_sha256"] = {}
    contract["input_sha256"] = {}
    contract["runtime"]["versions"] = {}
    contract["resources"]["minimum_free_bytes"] = 0
    contract = seal_contract(contract)
    approval = seal_contract(
        {
            "approved": True,
            "scope": "initial_200_production_capture",
            "capture_contract_digest": contract["digest"],
            "source_ids": [
                r["source_id"] for r in contract["targets"] if r["queue_role"] == "initial_review"
            ],
            "training_authorized": False,
        }
    )
    for name, value in [("recording-contract.json", contract), ("approval.json", approval)]:
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "partitions.json").write_bytes((bundle / "partitions.json").read_bytes())
    original_import = builtins.__import__

    def stop_before_simulator(name, *args, **kwargs):
        if name == "mujoco":
            raise RuntimeError("test stops before simulator")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stop_before_simulator)
    monkeypatch.setattr(
        sys,
        "argv",
        ["record", "--base", str(tmp_path), "--authorization", str(tmp_path / "approval.json")],
    )
    with pytest.raises(RuntimeError, match="test stops before simulator"):
        recorder.main()
    assert recorder.load(tmp_path / "run/authorization.json") == approval
    assert recorder.load(tmp_path / "run/launch.json")["authorization_digest"] == approval["digest"]
    completion = recorder.load(tmp_path / "run/completion.json")
    assert completion["control_steps"] == 0 and completion["results"] == []
    assert completion["status"] == "incomplete"

"""Exercise the immutable experiment boundary without a GPU or remote access."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "week11_verify", ROOT / "scripts/verify_week11_protocol.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


@pytest.fixture
def frozen_copy(tmp_path):
    contract = verifier.read(ROOT / verifier.CONTRACT_PATH)
    names = {verifier.CONTRACT_PATH, "docs/evaluation/week11-freeze-v1/entry.json"}
    names.update(contract["source_files_sha256"])
    names.update(panel["path"] for panel in contract["panels"].values())
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    return tmp_path


def test_real_freeze_and_evidence_accounting():
    result = verifier.verify()
    assert result["passed"]
    assert result["formal_episodes_all_arms"] == 2001
    assert not result["models_loaded"]
    assert not result["remote_bytes_freshly_rehashed"]


def test_copied_bundle_verifies(frozen_copy):
    assert verifier.verify(frozen_copy)["passed"]


def test_resealed_scope_change_cannot_replace_freeze(frozen_copy):
    path = frozen_copy / verifier.CONTRACT_PATH
    record = verifier.read(path)
    record["decision"]["targeted_gain_pp_at_least"] = 0
    record["digest"] = verifier.digest(record)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="versioned amendment"):
        verifier.verify(frozen_copy)


def test_resealed_episode_seed_change_rejected(frozen_copy):
    path = frozen_copy / "evaluation/splits/week11-integration-v1/targeted-467.json"
    record = verifier.read(path)
    record["tasks"][0]["episode_seed"] += 1
    record["digest"] = verifier.digest(record)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Panel bytes changed"):
        verifier.verify(frozen_copy)


def test_changed_parent_endpoint_rejected(frozen_copy):
    path = frozen_copy / "docs/evaluation/week10-closeout-v1/handoff.json"
    record = verifier.read(path)
    record["endpoint_sha256"] = "0" * 64
    record["digest"] = verifier.digest(record)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Frozen input changed"):
        verifier.verify(frozen_copy)


def test_false_claim_of_remote_verification_rejected(frozen_copy):
    path = frozen_copy / "docs/evaluation/week11-freeze-v1/entry.json"
    record = verifier.read(path)
    record["remote_bytes_freshly_rehashed"] = True
    record["digest"] = verifier.digest(record)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Entry scope mismatch"):
        verifier.verify(frozen_copy)

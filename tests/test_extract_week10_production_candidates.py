import copy
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from eventground_wam.demonstration_contracts import seal_contract

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "production_extractor", ROOT / "scripts/extract_week10_production_candidates.py"
)
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)


def reseal(value):
    return seal_contract({k: v for k, v in value.items() if k != "digest"})


def setup():
    base = ROOT / "docs/evaluation/week10-production-capture-visual-v2"
    metadata = {key: extractor.sealed(base / name) for key, name in extractor.METADATA.items()}
    implementation = extractor.sealed(
        ROOT / "docs/evaluation/week10-labeler-implementation-v1/implementation-contract.json"
    )
    targets = [
        t
        for t in metadata["capture_contract_digest"]["targets"]
        if t["queue_role"] == "initial_review"
    ]
    execution = seal_contract(
        {
            "artifact_type": "eventground_wam.production_event_candidate_extraction",
            "authorized_operation": "extract_unreviewed_candidates_and_diagnostic_manifest",
            **{key: value["digest"] for key, value in metadata.items()},
            "implementation_contract_digest": implementation["digest"],
            "source_ids": [t["source_id"] for t in targets],
            "recording_ids": [t["recording_id"] for t in targets],
            "resources": {
                "max_recordings": 200,
                "max_frames": 33581,
                "wall_seconds": 1800,
                "output_bytes": 1024**3,
                "minimum_free_bytes": 150 * 1024**3,
            },
            "labels_admitted": False,
            "training_authorized": False,
            "control_steps": 0,
            "heldout_payload_reads": 0,
        }
    )
    return execution, metadata, implementation


def test_real_capture_metadata_passes_new_child_gate():
    contract, metadata, implementation = setup()
    rows = extractor.validate_metadata(contract, metadata, implementation)
    assert len(rows) == 200 and sum(r["frames"] for r in rows) == 33581
    assert len({r["task_key"] for r in rows}) == 40


@pytest.mark.parametrize(
    "damage",
    [
        "old_pilot_scope",
        "reserve",
        "order",
        "training",
        "pixel",
        "missing_verification",
        "incomplete",
        "budget",
        "omitted_case",
        "contact_error",
    ],
)
def test_rejects_lineage_scope_qualification_or_coverage_damage(damage):
    contract, metadata, implementation = copy.deepcopy(setup())
    changed_key = None
    if damage == "old_pilot_scope":
        contract["artifact_type"] = "eventground_wam.instrumented_labeler_implementation"
    elif damage == "reserve":
        contract["source_ids"].append("a" * 64)
    elif damage == "order":
        contract["source_ids"].reverse()
    elif damage == "training":
        contract["training_authorized"] = True
    elif damage == "pixel":
        changed_key = "camera_review_digest"
        metadata[changed_key]["pixel_tolerance_changed"] = True
    elif damage == "missing_verification":
        changed_key = "verification_digest"
        metadata[changed_key]["all_200_visual_states_reconstructed"] = False
    elif damage == "incomplete":
        changed_key = "capture_completion_digest"
        metadata[changed_key]["status"] = "incomplete"
    elif damage == "budget":
        contract["resources"]["wall_seconds"] = 1801
    elif damage == "omitted_case":
        changed_key = "verification_digest"
        metadata[changed_key]["cases"].pop()
    elif damage == "contact_error":
        changed_key = "verification_digest"
        metadata[changed_key]["cases"][0]["maximum_contact_error"] = 0.1
    if changed_key:
        metadata[changed_key] = reseal(metadata[changed_key])
        contract[changed_key] = metadata[changed_key]["digest"]
        if changed_key == "verification_digest":
            metadata["camera_review_digest"]["verification_digest"] = metadata[changed_key][
                "digest"
            ]
            metadata["camera_review_digest"] = reseal(metadata["camera_review_digest"])
            contract["camera_review_digest"] = metadata["camera_review_digest"]["digest"]
    with pytest.raises(ValueError):
        extractor.validate_metadata(reseal(contract), metadata, implementation)


def numeric_fixture(tmp_path):
    n = 40
    source = "a" * 64
    target = {"recording_id": "synthetic", "source_id": source, "frames": n}
    geometry = seal_contract({"objects": [{"object_id": "bowl"}]})
    extractor.write(tmp_path / "geometry-map.json", geometry)
    with (tmp_path / "contacts.jsonl").open("w") as file:
        for frame in range(n):
            file.write(
                json.dumps({"frame_index": frame, "time": float(frame * 0.05), "contacts": []})
                + "\n"
            )
    with h5py.File(tmp_path / "recording.h5", "w") as handle:
        handle.attrs["complete"] = True
        handle.attrs["source_id"] = source
        handle.attrs["recording_contract_digest"] = "b" * 64
        handle.attrs["observation_phase"] = "fresh_snapshot_before_source_action"
        arrays = {
            "time": np.arange(n) * 0.05,
            "frame_index": np.arange(n),
            "source_action_index": np.arange(n),
            "action": np.zeros((n, 7)),
            "observation/state": np.zeros((n, 8)),
            "object_position": np.zeros((n, 1, 3)),
            "object_rotation_matrix": np.tile(np.eye(3).ravel(), (n, 1, 1)),
            "gripper_position": np.zeros((n, 3)),
            "gripper_rotation_matrix": np.tile(np.eye(3).ravel(), (n, 1)),
        }
        for name, array in arrays.items():
            handle.create_dataset(name, data=array)
    for name in extractor.EVIDENCE_FILES - {"recording.h5", "geometry-map.json", "contacts.jsonl"}:
        (tmp_path / name).write_text("unused immutable fixture\n")
    result = {
        "recording_id": "synthetic",
        "source_id": source,
        "frames": n,
        "status": "completed",
        "evidence_sha256": {
            name: extractor.sha(tmp_path / name) for name in extractor.EVIDENCE_FILES
        },
    }
    extractor.write(tmp_path / "result.json", seal_contract(result))
    return target, result


def test_numeric_loader_reads_only_current_capture_numeric_fields(tmp_path):
    target, result = numeric_fixture(tmp_path)
    evidence = extractor.load_evidence(tmp_path, target, result, "b" * 64)
    assert evidence.actions.shape == (40, 7)
    assert evidence.object_rotation.shape == (40, 1, 3, 3)
    assert len(evidence.contacts) == 40
    assert not hasattr(evidence, "images")


def test_all_evidence_hashes_checked_before_numeric_use(tmp_path):
    target, result = numeric_fixture(tmp_path)
    (tmp_path / "capture-model.mjb").write_text("tampered")
    with pytest.raises(ValueError, match="Changed recording evidence"):
        extractor.load_evidence(tmp_path, target, result, "b" * 64)


def test_frame_order_cannot_be_repaired_by_loader(tmp_path):
    target, result = numeric_fixture(tmp_path)
    with h5py.File(tmp_path / "recording.h5", "r+") as handle:
        handle["source_action_index"][3] = 2
    result["evidence_sha256"]["recording.h5"] = extractor.sha(tmp_path / "recording.h5")
    (tmp_path / "result.json").unlink()
    extractor.write(tmp_path / "result.json", seal_contract(result))
    with pytest.raises(ValueError, match="frame order"):
        extractor.load_evidence(tmp_path, target, result, "b" * 64)


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        extractor.safe_child(tmp_path, "../heldout/recording.h5")

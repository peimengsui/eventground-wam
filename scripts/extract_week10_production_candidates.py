"""Bounded, read-only numeric extraction on the completed initial 200 recordings.

A new sealed execution contract authorizes this application of the unchanged
pilot-qualified physical labeler. Outputs are candidates/coverage, not admitted
labels. No simulator, renderer, network client or optimizer is imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_manifest import build_diagnostic_manifest, require_sealed
from eventground_wam.instrumented_labeling import RecordingEvidence, extract_events, prompt_previews
from eventground_wam.instrumented_partitions import require_train_source

METADATA = {
    "study_digest": "study-contract.json",
    "capture_contract_digest": "recording-contract.json",
    "capture_completion_digest": "run/completion.json",
    "verification_digest": "verification.json",
    "camera_review_digest": "camera-qualification-review.json",
    "partition_digest": "partitions.json",
}
REQUIRED_CODE = {
    "scripts/extract_week10_production_candidates.py",
    "src/eventground_wam/eventground_manifest.py",
    "src/eventground_wam/demonstration_contracts.py",
    "src/eventground_wam/instrumented_labeling.py",
    "src/eventground_wam/instrumented_partitions.py",
    "src/eventground_wam/instrumented_supervision.py",
}
EVIDENCE_FILES = {
    "canonical-reset.xml",
    "capture-model.mjb",
    "contacts.jsonl",
    "geometry-map.json",
    "model.xml",
    "recording.h5",
    "scene.json",
}


def sha(path, check_budget=lambda: None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            check_budget()
            digest.update(block)
    return digest.hexdigest()


def sealed(path):
    return require_sealed(json.loads(Path(path).read_text()), str(path))


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def safe_child(base, name):
    path = Path(base) / name
    if Path(name).is_absolute() or not path.resolve().is_relative_to(Path(base).resolve()):
        raise ValueError("Input path escapes its frozen base")
    return path


def validate_metadata(execution, metadata, implementation):
    """Validate sealed lineage before any recording payload is read."""
    require_sealed(execution, "execution contract")
    require_sealed(implementation, "frozen labeler implementation")
    if execution.get("artifact_type") != "eventground_wam.production_event_candidate_extraction":
        raise ValueError("New production extraction child contract required")
    if (
        execution.get("authorized_operation")
        != "extract_unreviewed_candidates_and_diagnostic_manifest"
    ):
        raise ValueError("Wrong authorized operation")
    for key in ("labels_admitted", "training_authorized"):
        if execution.get(key) is not False:
            raise ValueError("Extraction cannot authorize labels or training")
    for key in ("control_steps", "heldout_payload_reads"):
        if execution.get(key) != 0:
            raise ValueError("Extraction cannot run simulation or read held-out payloads")
    for key in METADATA:
        record = require_sealed(metadata[key], key)
        if record["digest"] != execution[key]:
            raise ValueError(f"Wrong frozen {key}")
    if execution["implementation_contract_digest"] != implementation["digest"]:
        raise ValueError("Frozen labeler implementation changed")
    study = metadata["study_digest"]
    capture = metadata["capture_contract_digest"]
    completion = metadata["capture_completion_digest"]
    verification = metadata["verification_digest"]
    camera = metadata["camera_review_digest"]
    partition = metadata["partition_digest"]
    if (
        capture["study_digest"] != study["digest"]
        or capture["partition_digest"] != partition["digest"]
        or study["data"]["partition_contract_digest"] != partition["digest"]
    ):
        raise ValueError("Capture/study/source partition lineage mismatch")
    if (
        completion["status"] != "complete"
        or completion["contract_digest"] != capture["digest"]
        or completion["error"] is not None
        or completion["requested_sources"] != 200
    ):
        raise ValueError("Capture must be complete with exact initial 200")
    if (
        verification["completion_digest"] != completion["digest"]
        or verification["recording_contract_digest"] != capture["digest"]
        or verification["status"] != "passed_state_contact_checks_with_pixel_equality_qualification"
        or verification["all_200_visual_states_reconstructed"] is not True
    ):
        raise ValueError("Independent physical and visual verification required")
    if (
        camera["verification_digest"] != verification["digest"]
        or camera["status"] != "reviewed_with_explicit_pixel_equality_qualification"
        or camera["pixel_tolerance_changed"] is not False
        or camera["strict_pixel_equality_claimed"] is not False
    ):
        raise ValueError("Explicit unchanged camera qualification required")
    targets = [t for t in capture["targets"] if t["queue_role"] == "initial_review"]
    if len(targets) != 200 or len({t["source_id"] for t in targets}) != 200:
        raise ValueError("Exact unique initial 200 required")
    if execution["source_ids"] != [t["source_id"] for t in targets]:
        raise ValueError("Source selection/order changed or reserve requested")
    if execution["recording_ids"] != [t["recording_id"] for t in targets]:
        raise ValueError("Recording order changed")
    if [r["source_id"] for r in completion["results"]] != execution["source_ids"]:
        raise ValueError("Completed source set/order changed")
    if [r["recording_id"] for r in verification["cases"]] != execution["recording_ids"]:
        raise ValueError("Independent verification omitted/reordered a source")
    counts = Counter(t["task_key"] for t in targets)
    if len(counts) != 40 or set(counts.values()) != {5}:
        raise ValueError("Initial coverage must be five sources per task")
    for target, result, check in zip(
        targets, completion["results"], verification["cases"], strict=True
    ):
        source = require_train_source(partition, target["source_id"])
        if any(target[k] != source[k] for k in ("frames", "task_key", "queue_role", "demo_id")):
            raise ValueError("Frozen training source metadata mismatch")
        if (
            result["status"] != "completed"
            or result["recording_id"] != target["recording_id"]
            or result["frames"] != target["frames"]
            or check["frames"] != target["frames"]
            or check["all_visual_states_independently_reconstructed"] is not True
            or any(
                check[k] != 0
                for k in (
                    "maximum_contact_error",
                    "maximum_object_gripper_pose_error",
                    "maximum_proprio_error",
                )
            )
        ):
            raise ValueError("Incomplete or physically inconsistent recording")
        if set(result["evidence_sha256"]) != EVIDENCE_FILES:
            raise ValueError("Recording evidence inventory changed")
    resources = execution["resources"]
    frames = sum(t["frames"] for t in targets)
    if resources["max_recordings"] != 200 or resources["max_frames"] != frames or frames != 33581:
        raise ValueError("Only the completed 200/33581-frame batch is authorized")
    if verification["frames_verified"] != frames:
        raise ValueError("Independent verification frame coverage mismatch")
    if not (
        0 < resources["wall_seconds"] <= 1800
        and 0 < resources["output_bytes"] <= 1024**3
        and resources["minimum_free_bytes"] >= 150 * 1024**3
    ):
        raise ValueError("Extraction resource limits missing or expanded")
    return targets


def load_evidence(folder, target, result, capture_digest, check_budget=lambda: None):
    """Verify every stored artifact before loading only the frozen numeric fields."""
    saved = sealed(folder / "result.json")
    if {k: v for k, v in saved.items() if k != "digest"} != result:
        raise ValueError("Per-recording result differs from capture completion")
    for name, expected in result["evidence_sha256"].items():
        if sha(safe_child(folder, name), check_budget) != expected:
            raise ValueError(f"Changed recording evidence: {target['recording_id']}/{name}")
    geometry = sealed(folder / "geometry-map.json")
    contacts = [json.loads(line) for line in (folder / "contacts.jsonl").read_text().splitlines()]
    with h5py.File(folder / "recording.h5", "r") as handle:
        n = target["frames"]
        if (
            not bool(handle.attrs.get("complete", False))
            or handle.attrs.get("source_id") != target["source_id"]
            or handle.attrs.get("recording_contract_digest") != capture_digest
            or handle.attrs.get("observation_phase") != "fresh_snapshot_before_source_action"
        ):
            raise ValueError("Recording identity/phase/completion mismatch")
        if (
            len(handle["time"]) != n
            or len(contacts) != n
            or not np.array_equal(handle["frame_index"][:], np.arange(n))
            or not np.array_equal(handle["source_action_index"][:], np.arange(n))
            or not np.allclose(np.diff(handle["time"][:]), 0.05, rtol=0, atol=1e-9)
            or not np.array_equal(handle["time"][:], np.asarray([r["time"] for r in contacts]))
        ):
            raise ValueError("Recording frame order/contact timestamps mismatch")
        objects = len(geometry["objects"])
        return RecordingEvidence(
            target["recording_id"],
            target["source_id"],
            handle["action"][:],
            handle["observation/state"][:],
            handle["object_position"][:],
            handle["object_rotation_matrix"][:].reshape(n, objects, 3, 3),
            handle["gripper_position"][:],
            handle["gripper_rotation_matrix"][:].reshape(n, 3, 3),
            geometry,
            contacts,
        )


def run(capture_base, execution_path, implementation_path, repository_root, output):
    execution = sealed(execution_path)
    start = time.monotonic()
    resources = execution["resources"]
    output = Path(output)
    capture_base = Path(capture_base)
    if output.resolve().is_relative_to(capture_base.resolve()):
        raise ValueError("Outputs cannot mutate the immutable capture bundle")

    def budget():
        if time.monotonic() - start > resources["wall_seconds"]:
            raise TimeoutError("Bounded extraction active-time budget exceeded")
        storage = output if output.exists() else output.parent
        if shutil.disk_usage(storage).free < resources["minimum_free_bytes"]:
            raise RuntimeError("Minimum free space gate failed")

    if set(execution["input_files_sha256"]) != set(METADATA.values()):
        raise ValueError("Exact frozen metadata inventory required")
    for name, expected in execution["input_files_sha256"].items():
        if sha(safe_child(capture_base, name), budget) != expected:
            raise ValueError(f"Changed metadata: {name}")
    metadata = {key: sealed(capture_base / name) for key, name in METADATA.items()}
    implementation = sealed(implementation_path)
    targets = validate_metadata(execution, metadata, implementation)
    root = Path(repository_root)
    if not REQUIRED_CODE <= set(execution["code_sha256"]):
        raise ValueError("Required execution code pins missing")
    for mapping in (
        execution["code_sha256"],
        implementation["code_sha256"],
        metadata["study_digest"]["source_files_sha256"],
    ):
        for name, expected in mapping.items():
            if sha(safe_child(root, name), budget) != expected:
                raise ValueError(f"Pinned implementation/source changed: {name}")
    output.mkdir(parents=True, exist_ok=False)
    cases, records = [], []
    diagnostic, coverage = None, None
    status, error = "incomplete", None
    try:
        for target, result in zip(
            targets, metadata["capture_completion_digest"]["results"], strict=True
        ):
            budget()
            folder = safe_child(capture_base / "run", target["recording_id"])
            evidence = load_evidence(
                folder, target, result, execution["capture_contract_digest"], budget
            )
            candidates = extract_events(evidence)
            previews = prompt_previews(candidates)
            out = safe_child(output, target["recording_id"])
            out.mkdir()
            write(out / "event-candidates.json", candidates)
            write(out / "prompt-previews.json", previews)
            records.append(
                {
                    "candidate": candidates,
                    "task_key": target["task_key"],
                    "instruction": target["instruction"],
                    "recording_path": str(folder / "recording.h5"),
                    "recording_sha256": result["evidence_sha256"]["recording.h5"],
                }
            )
            cases.append(
                {
                    "recording_id": target["recording_id"],
                    "source_id": target["source_id"],
                    "task_key": target["task_key"],
                    "frames": target["frames"],
                    "candidate_digest": candidates["digest"],
                    "preview_digest": previews["digest"],
                    "candidate_file_sha256": sha(out / "event-candidates.json", budget),
                    "preview_file_sha256": sha(out / "prompt-previews.json", budget),
                    "events_by_type": dict(Counter(e["event_type"] for e in candidates["events"])),
                    "ambiguities_by_type": dict(
                        Counter(e["event_type"] for e in candidates["ambiguities"])
                    ),
                }
            )
            if (
                sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
                > resources["output_bytes"]
            ):
                raise RuntimeError("Output budget exceeded")
            print(
                json.dumps(
                    {"completed_recordings": len(cases), "recording_id": target["recording_id"]}
                ),
                flush=True,
            )
        diagnostic = build_diagnostic_manifest(
            records, metadata["study_digest"], metadata["partition_digest"]
        )
        diagnostic = seal_contract(
            {k: v for k, v in diagnostic.items() if k != "digest"}
            | {
                "capture_completion_digest": execution["capture_completion_digest"],
                "extraction_contract_digest": execution["digest"],
                "camera_qualification_digest": execution["camera_review_digest"],
            }
        )
        write(output / "diagnostic-manifest.json", diagnostic)
        coverage = seal_contract(
            {
                "diagnostic_manifest_digest": diagnostic["digest"],
                **diagnostic["coverage"],
                "labels_admitted": False,
                "quality_validated": False,
            }
        )
        write(output / "coverage.json", coverage)
        budget()
        if (
            sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            > resources["output_bytes"]
        ):
            raise RuntimeError("Output budget exceeded")
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        completion = seal_contract(
            {
                "artifact_type": "eventground_wam.production_event_candidate_completion",
                "status": status,
                "error": error,
                "execution_contract_digest": execution["digest"],
                "capture_completion_digest": execution["capture_completion_digest"],
                "implementation_digest": implementation["digest"],
                "diagnostic_manifest_digest": diagnostic["digest"] if diagnostic else None,
                "coverage_digest": coverage["digest"] if coverage else None,
                "output_files_sha256": {
                    str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*.json"))
                },
                "elapsed_seconds": time.monotonic() - start,
                "cases": cases,
                "labels_admitted": False,
                "quality_validated": False,
                "training_updates": 0,
                "heldout_payload_reads": 0,
                "control_steps": 0,
                "new_recordings": 0,
                "camera_qualification_digest": execution["camera_review_digest"],
                "note": "Unreviewed candidates and deterministic coverage only; not independent label QA.",
            }
        )
        write(output / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-base", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--implementation-contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.capture_base,
        args.execution_contract,
        args.implementation_contract,
        args.repository_root,
        args.output,
    )


if __name__ == "__main__":
    main()

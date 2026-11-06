"""Apply the sealed labeler to the three existing pilot recordings, unreviewed."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.demonstration_contracts import contract_digest, seal_contract
from eventground_wam.instrumented_labeling import RecordingEvidence, extract_events, prompt_previews


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def sealed(path):
    d = json.loads(Path(path).read_text())
    if contract_digest(d) != d["digest"]:
        raise ValueError(f"Changed record {path}")
    return d


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--implementation-contract", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    impl = sealed(args.implementation_contract)
    for rel, h in impl["code_sha256"].items():
        if sha(root / rel) != h:
            raise ValueError(f"Implementation changed: {rel}")
    pilot = args.pilot_base
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    finalization = sealed(pilot / "finalization.json")
    done = sealed(pilot / "run/completion.json")
    if finalization["digest"] != impl["pilot_finalization_digest"] or done["status"] != "complete":
        raise ValueError("Wrong/incomplete pilot")
    results = []
    for case in done["results"]:
        folder = pilot / "run" / case["recording_id"]
        for name in ("recording.h5", "contacts.jsonl", "geometry-map.json"):
            if sha(folder / name) != case["evidence_sha256"][name]:
                raise ValueError(f"Changed input {name}")
        geometry = sealed(folder / "geometry-map.json")
        contacts = [json.loads(x) for x in (folder / "contacts.jsonl").read_text().splitlines()]
        with h5py.File(folder / "recording.h5", "r") as h:
            if not h.attrs["complete"] or h.attrs["source_id"] != case["source_id"]:
                raise ValueError("Recording identity failure")
            n = len(h["time"])
            objects = len(geometry["objects"])
            if not np.array_equal(h["frame_index"][:], np.arange(n)):
                raise ValueError("Bad frame order")
            if not np.allclose(np.diff(h["time"][:]), 0.05, rtol=0, atol=1e-9):
                raise ValueError("Wrong sampling phase/period")
            if not np.array_equal(h["time"][:], np.array([r["time"] for r in contacts])):
                raise ValueError("Contact timestamp mismatch")
            e = RecordingEvidence(
                case["recording_id"],
                case["source_id"],
                h["action"][:],
                h["observation/state"][:],
                h["object_position"][:],
                h["object_rotation_matrix"][:].reshape(n, objects, 3, 3),
                h["gripper_position"][:],
                h["gripper_rotation_matrix"][:].reshape(n, 3, 3),
                geometry,
                contacts,
            )
        events = extract_events(e)
        previews = prompt_previews(events)
        target = out / case["recording_id"]
        target.mkdir()
        write(target / "event-candidates.json", events)
        write(target / "prompt-previews.json", previews)
        results.append(
            {
                "recording_id": case["recording_id"],
                "frames": n,
                "events_by_type": dict(Counter(r["event_type"] for r in events["events"])),
                "ambiguities_by_type": dict(
                    Counter(r["event_type"] for r in events["ambiguities"])
                ),
                "preview_counts": dict(
                    Counter(r.get("event_type", r.get("reason")) for r in previews["rows"])
                ),
                "candidate_digest": events["digest"],
                "preview_digest": previews["digest"],
                "event_file_sha256": sha(target / "event-candidates.json"),
                "preview_file_sha256": sha(target / "prompt-previews.json"),
            }
        )
    report = seal_contract(
        {
            "artifact_type": "eventground_wam.instrumented_labeler_pilot",
            "implementation_digest": impl["digest"],
            "source_finalization_digest": finalization["digest"],
            "cases": results,
            "elapsed_seconds": time.monotonic() - start,
            "new_recordings": 0,
            "new_downloads": 0,
            "heldout_payload_reads": 0,
            "candidate_extraction_performed": True,
            "labels_admitted": False,
            "quality_validated": False,
            "training_updates": 0,
            "note": "Candidate counts are implementation diagnostics, not label precision/recall or a training manifest.",
        }
    )
    write(out / "completion.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

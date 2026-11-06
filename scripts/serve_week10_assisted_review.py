"""Separate candidate-assisted review; never an independent reference or label admission.

Reuses only the blind server's verified evidence reader, bounded HTTP handler and
atomic file-writing utilities. It does not alter the blind server or draft chain.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import re
import secrets
import time
from collections import Counter
from datetime import datetime
from http.server import HTTPServer
from pathlib import Path

from eventground_wam.demonstration_contracts import contract_digest, seal_contract

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "week10_evidence_reader", ROOT / "scripts/serve_week10_label_review.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
Error = base.ReviewError
TYPES = {"contact", "grasp", "gripper_close"}
DECISIONS = {"pending", "correct", "incorrect", "needs_correction", "unsure"}


def build_manifest(blind_manifest, completion, candidate_root):
    """Preserve every proposal/ambiguity and bind it to the verified capture identity."""
    for value in (blind_manifest, completion):
        if contract_digest(value) != value.get("digest"):
            raise Error("Unsealed preparation input")
    if (completion.get("status") != "complete" or completion.get("labels_admitted") is not False
            or completion["capture_completion_digest"] != blind_manifest["capture_completion_digest"]):
        raise Error("Candidate extraction does not match this capture")
    cases = {row["recording_id"]: row for row in completion["cases"]}
    if len(cases) != len(completion["cases"]) or set(cases) != {
        row["recording_id"] for row in blind_manifest["episodes"]
    }:
        raise Error("Candidate population differs from the exact review population")
    episodes = []
    for row in blind_manifest["episodes"]:
        rid, case = row["recording_id"], cases[row["recording_id"]]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
            raise Error("Unsafe recording identity")
        relative = f"{rid}/event-candidates.json"
        checksum = case["candidate_file_sha256"]
        if completion["output_files_sha256"].get(relative) != checksum:
            raise Error("Candidate checksum links disagree")
        verified = base.VerifiedFile(Path(candidate_root) / relative, checksum)
        candidate = base.parse_json(verified.read())
        if (contract_digest(candidate) != candidate.get("digest")
                or candidate["digest"] != case["candidate_digest"]
                or candidate.get("labels_admitted") is not False):
            raise Error("Candidate integrity or admission mismatch")
        for key in ("recording_id", "source_id", "frames"):
            if candidate[key] != row[key] or case[key] != row[key]:
                raise Error("Candidate source/frame mismatch")
        proposals = []
        for kind, key in (("event", "events"), ("ambiguity", "ambiguities")):
            for index, item in enumerate(candidate[key]):
                if (item["event_type"] not in TYPES or type(item["start"]) is not int
                        or type(item["end"]) is not int
                        or not 0 <= item["start"] < item["end"] <= row["frames"]):
                    raise Error("Invalid proposal interval")
                proposals.append({"proposal_id": f"{rid}:{kind}:{index:04d}",
                                  "kind": kind, "original": copy.deepcopy(item)})
        episodes.append({**copy.deepcopy(row), "candidate_digest": candidate["digest"],
                         "candidate_file_sha256": checksum, "proposals": proposals})
    return seal_contract({
        "artifact_type": "eventground_wam.week10_assisted_review_manifest", "version": 1,
        "annotation_origin": "frozen_extractor_proposals_for_human_verification",
        "candidate_exposure": True, "independent_reference": False,
        "labels_admitted": False, "quality_validated": False,
        "study_digest": blind_manifest["study_digest"],
        "capture_completion_digest": blind_manifest["capture_completion_digest"],
        "blind_manifest_digest": blind_manifest["digest"],
        "candidate_completion_digest": completion["digest"], "control_hz": 20,
        "episodes": episodes,
    })


def make_draft(manifest):
    return {"artifact_type": "eventground_wam.week10_assisted_review_decisions", "version": 1,
            "manifest_digest": manifest["digest"], "independent_reference": False,
            "candidate_exposure_before_review": True, "labels_admitted": False,
            "quality_validated": False, "status": "unsigned_assisted_review",
            "reviewer_id": None, "started_at": None,
            "episodes": [{"recording_id": row["recording_id"], "notes": "",
                          "whole_episode_checked_for_missed_events": False,
                          "decisions": [{"proposal_id": p["proposal_id"], "decision": "pending",
                                         "notes": "", "decided_at": None}
                                        for p in row["proposals"]]}
                         for row in manifest["episodes"]]}


def timestamp(value):
    if not isinstance(value, str):
        raise Error("A real review timestamp is required")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("Missing timezone")
        return result
    except ValueError as exc:
        raise Error("Invalid review timestamp") from exc


def validate_draft(draft, manifest):
    blank = make_draft(manifest)
    if not isinstance(draft, dict) or set(draft) != set(blank):
        raise Error("Unexpected assisted-review schema")
    for key in blank.keys() - {"reviewer_id", "started_at", "episodes"}:
        if draft[key] != blank[key] or type(draft[key]) is not type(blank[key]):
            raise Error("Assisted decisions cannot be promoted to independent QA or admitted labels")
    identity = draft["reviewer_id"]
    if identity is None:
        if draft != blank:
            raise Error("Begin with your own reviewer identity before recording decisions")
        return draft
    if not isinstance(identity, str) or not identity.strip() or len(identity) > 200:
        raise Error("Invalid reviewer identity")
    started = timestamp(draft["started_at"])
    if not isinstance(draft["episodes"], list) or len(draft["episodes"]) != len(blank["episodes"]):
        raise Error("Every source episode must remain in the review")
    for row, expected in zip(draft["episodes"], blank["episodes"], strict=True):
        if set(row) != set(expected) or row["recording_id"] != expected["recording_id"]:
            raise Error("Episode identity or fields changed")
        if (not isinstance(row["notes"], str) or len(row["notes"]) > 10000
                or type(row["whole_episode_checked_for_missed_events"]) is not bool):
            raise Error("Invalid episode notes/check")
        if len(row["decisions"]) != len(expected["decisions"]):
            raise Error("No event or ambiguity may be silently removed")
        for decision, proposed in zip(row["decisions"], expected["decisions"], strict=True):
            if (set(decision) != set(proposed) or decision["proposal_id"] != proposed["proposal_id"]
                    or decision["decision"] not in DECISIONS
                    or not isinstance(decision["notes"], str) or len(decision["notes"]) > 10000):
                raise Error("Invalid decision or changed proposal identity")
            if decision["decision"] == "pending":
                if decision["decided_at"] is not None:
                    raise Error("Pending entries cannot claim a decision timestamp")
            elif timestamp(decision["decided_at"]) < started:
                raise Error("Decision precedes human review start")
    return draft


class DecisionStore(base.DraftStore):
    """Distinct schema and directory, using the existing atomic append-only primitive."""
    def __init__(self, directory, manifest, maximum_bytes, maximum_human_seconds=14400):
        self.directory, self.manifest = Path(directory), manifest
        self.maximum_bytes, self.maximum_human_seconds = maximum_bytes, maximum_human_seconds
        self.reported_human_minutes = 0.0
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise Error("Draft directory cannot be a symlink")
        self.metadata = {"artifact_type": "eventground_wam.week10_assisted_decision_store",
                         "manifest_digest": manifest["digest"], "labels_admitted": False}
        meta = self.directory / "store.json"
        if not meta.exists():
            if any(self.directory.iterdir()):
                raise Error("Cannot initialize a nonempty store")
            self._create(meta, base.encoded(self.metadata))
        elif base.parse_json(meta.read_bytes()) != self.metadata:
            raise Error("Store belongs to a different workflow or manifest")
        self.revision, self.previous_sha = 0, None
        self.current = make_draft(manifest)
        files = sorted(self.directory.glob("draft-*.json"))
        allowed = {"store.json", ".writer.lock"} | {p.name for p in files}
        if any(p.name not in allowed for p in self.directory.iterdir()):
            raise Error("Unexpected store entry; preserve and inspect")
        if self._size() > maximum_bytes:
            raise Error("Draft budget exceeded")
        for revision, path in enumerate(files, 1):
            if path.name != f"draft-{revision:06d}.json" or path.stat().st_size > 16 * 1024**2:
                raise Error("Invalid draft history")
            raw = path.read_bytes()
            record = base.parse_json(raw)
            if (record.get("artifact_type") != "eventground_wam.week10_assisted_decision_version"
                    or record.get("revision") != revision
                    or record.get("manifest_digest") != manifest["digest"]
                    or record.get("previous_file_sha256") != self.previous_sha
                    or record.get("draft_sha256") != hashlib.sha256(base.encoded(record["draft"])).hexdigest()):
                raise Error("Broken assisted decision history")
            self._validate_identity(record["draft"])
            self._validate_minutes(record["reported_human_minutes"])
            self.reported_human_minutes = record["reported_human_minutes"]
            self.current, self.revision = record["draft"], revision
            self.previous_sha = hashlib.sha256(raw).hexdigest()

    def _validate_identity(self, draft):
        validate_draft(draft, self.manifest)
        for key in ("reviewer_id", "started_at"):
            if self.current[key] is not None and draft[key] != self.current[key]:
                raise Error("Existing reviewer and start time are immutable")

    def save(self, draft, base_revision, reported_human_minutes=0):
        with (self.directory / ".writer.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            latest = sorted(self.directory.glob("draft-*.json"))
            if (type(base_revision) is not int or base_revision != self.revision
                    or len(latest) != self.revision):
                raise base.Conflict("Another save exists; export your work before reloading")
            if base.parse_json((self.directory / "store.json").read_bytes()) != self.metadata:
                raise Error("Store identity changed")
            if latest and hashlib.sha256(latest[-1].read_bytes()).hexdigest() != self.previous_sha:
                raise Error("Previous draft changed")
            self._validate_identity(draft)
            self._validate_minutes(reported_human_minutes)
            record = {"artifact_type": "eventground_wam.week10_assisted_decision_version",
                      "manifest_digest": self.manifest["digest"], "revision": self.revision + 1,
                      "previous_file_sha256": self.previous_sha, "saved_at": base.stamp(),
                      "reported_human_minutes": reported_human_minutes,
                      "draft_sha256": hashlib.sha256(base.encoded(draft)).hexdigest(), "draft": draft}
            raw = base.encoded(record)
            self._create(self.directory / f"draft-{record['revision']:06d}.json", raw)
            self.current = copy.deepcopy(draft)
            self.revision += 1
            self.previous_sha = hashlib.sha256(raw).hexdigest()
            self.reported_human_minutes = reported_human_minutes
            return {"revision": self.revision, "saved_at": record["saved_at"], "unsigned": True}


class AssistedApp:
    def __init__(self, args):
        started = time.monotonic()
        self.contract, contract_file = base.sealed_file(args.session_contract)
        c = self.contract
        if (c.get("artifact_type") != "eventground_wam.week10_assisted_review_session"
                or c.get("labels_admitted") is not False or c.get("independent_reference") is not False
                or c.get("candidate_exposure") is not True or c.get("optimizer_updates") != 0):
            raise Error("Wrong assisted session contract")
        if (not 0 < c["maximum_duration_seconds"] <= 14400
                or not 0 < c["maximum_request_bytes"] <= 16 * 1024**2
                or not 0 < c["maximum_output_bytes"] <= 1024**3):
            raise Error("Excessive session limits")
        self.deadline = started + c["maximum_duration_seconds"]
        self.inputs = [contract_file]
        self.manifest, manifest_file = base.sealed_file(args.manifest)
        self.inputs.append(manifest_file)
        if (self.manifest["digest"] != c["manifest_digest"]
                or self.manifest["artifact_type"] != "eventground_wam.week10_assisted_review_manifest"
                or self.manifest["labels_admitted"] is not False
                or self.manifest["independent_reference"] is not False):
            raise Error("Assisted manifest mismatch")
        counts = Counter(row["task_key"] for row in self.manifest["episodes"])
        if len(counts) != 40 or set(counts.values()) != {5}:
            raise Error("Exact initial population required")
        for name, checksum in c["code_sha256"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise Error("Unsafe code path")
            self.inputs.append(base.VerifiedFile(ROOT / name, checksum, self.deadline))
        ui = ROOT / "src/eventground_wam/assisted_review_ui.html"
        if "src/eventground_wam/assisted_review_ui.html" not in c["code_sha256"]:
            raise Error("UI must be pinned")
        self.ui = ui.read_bytes()
        self.evidence = base.Evidence(args.capture_base, self.manifest, self.deadline)
        for row in self.manifest["episodes"]:
            for proposal in row["proposals"]:
                original = proposal["original"]
                if (original["event_type"] != "gripper_close"
                        and original["object_id"] not in self.evidence.objects[row["recording_id"]]):
                    raise Error("Candidate object not present in original geometry")
        self.store = DecisionStore(args.drafts, self.manifest, c["maximum_output_bytes"])
        self.csrf, self.port = secrets.token_urlsafe(32), args.port
        if self.port != c["port"]:
            raise Error("Port differs from session contract")

    def live(self):
        if time.monotonic() >= self.deadline:
            raise Error("Session ended; saved decisions remain available")
        for file in self.inputs:
            file.check()

    def get(self, path):
        self.live()
        if path == "/":
            return self.ui, "text/html; charset=utf-8"
        if path == "/health":
            return base.encoded({"status": "ready", "mode": "assisted",
                                 "session_digest": self.contract["digest"], "labels_admitted": False,
                                 "remaining_seconds": int(self.deadline-time.monotonic())}), "application/json"
        if path == "/api/session":
            return base.encoded({"manifest": self.manifest, "draft": self.store.current,
                                 "csrf": self.csrf, "revision": self.store.revision,
                                 "reported_human_minutes": self.store.reported_human_minutes,
                                 "remaining_seconds": int(self.deadline-time.monotonic())}), "application/json"
        if path == "/api/export":
            return base.encoded(self.store.current), "application/json"
        match = re.fullmatch(r"/frame/([A-Za-z0-9_-]+)/([0-9]+)/(front|wrist)\.png", path)
        if match and match[1] in self.evidence.rows:
            return self.evidence.read("frame", match[1], int(match[2]), match[3]), "image/png"
        match = re.fullmatch(r"/numeric/([A-Za-z0-9_-]+)/([0-9]+)\.json", path)
        if match and match[1] in self.evidence.rows:
            return self.evidence.read("numeric", match[1], int(match[2])), "application/json"
        raise KeyError("Unknown assisted review route")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "capture-base", "session-contract", "drafts"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    app = AssistedApp(args)
    server = HTTPServer(("127.0.0.1", args.port), base.make_handler(app))
    server.timeout = 1
    print(f"Assisted review ready; {app.contract['digest']}", flush=True)
    try:
        while time.monotonic() < app.deadline:
            server.handle_request()
    finally:
        server.server_close()
        print("Assisted session ended; decisions preserved", flush=True)


if __name__ == "__main__":
    main()

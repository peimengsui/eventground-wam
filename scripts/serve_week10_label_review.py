"""Bounded loopback blind annotation UI; unsigned human drafts, never QA admission.

The immutable v1 preparation is an input. This child never opens extractor candidates,
never imports a simulator, and never changes the prepared evidence or reference form.
"""
from __future__ import annotations

import argparse
import binascii
import copy
import hashlib
import json
import os
import re
import secrets
import stat
import struct
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.eventground_qa import (
    ACTIVE_STUDY_DIGEST,
    EVENT_TYPES,
    MATCHING_RULE,
    LabelReviewError,
    _interval,
    _overlap,
    _ranges,
    _same_identity,
    make_review_template,
)

REPO = Path(__file__).resolve().parents[1]
UI_PATH = REPO / "src/eventground_wam/review_ui.html"
MAX_JSON = 16 * 1024**2
CHECKS = (
    "both_camera_views_reviewed", "physical_evidence_reviewed",
    "missed_event_search_completed", "negative_cases_reviewed",
)
EVIDENCE_NAMES = {"recording.h5", "contacts.jsonl", "geometry-map.json"}


class ReviewError(ValueError):
    """Fail closed without changing source evidence or declaring QA complete."""


class Conflict(ReviewError):
    pass


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def parse_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReviewError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ReviewError("Nonfinite JSON")))


def stamp():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class VerifiedFile:
    def __init__(self, path, expected_sha, deadline=None):
        self.path = Path(path)
        before = self.path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ReviewError("Evidence must be a regular file, never a symlink")
        self.identity = fingerprint(before)
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            self.check_fd(stream.fileno())
            for data in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(data)
                if deadline is not None and time.monotonic() >= deadline:
                    raise ReviewError("Session budget exhausted during input verification")
            self.check_fd(stream.fileno())
        self.check()
        self.sha256 = digest.hexdigest()
        if self.sha256 != expected_sha:
            raise ReviewError(f"Evidence checksum mismatch: {self.path.name}")

    def check(self):
        if fingerprint(self.path.lstat()) != self.identity:
            raise ReviewError("Previously verified evidence changed; stop and investigate")

    def check_fd(self, fd):
        if fingerprint(os.fstat(fd)) != self.identity:
            raise ReviewError("Opened evidence differs from verified input")

    def read(self, maximum=MAX_JSON):
        self.check()
        if self.identity[2] > maximum:
            raise ReviewError("Metadata exceeds bounded read size")
        with self.path.open("rb") as stream:
            self.check_fd(stream.fileno())
            raw = stream.read(maximum + 1)
            self.check_fd(stream.fileno())
        self.check()
        return raw


def sealed_file(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON:
        raise ReviewError("Sealed input is too large")
    raw = path.read_bytes()
    value = parse_json(raw)
    if contract_digest(value) != value.get("digest"):
        raise ReviewError("Changed or unsealed input")
    verified = VerifiedFile(path, hashlib.sha256(raw).hexdigest())
    return value, verified


def png(array):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF))
    if array.shape != (256, 256, 3) or str(array.dtype) != "uint8":
        raise ReviewError("Original camera evidence must be 256x256 uint8 RGB")
    raw = b"".join(b"\x00" + row.tobytes() for row in array)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def validate_draft(draft, manifest, objects=None):
    """Validate partial human entries, keeping the existing unsigned schema intact."""
    template = make_review_template(manifest)
    if set(draft) != set(template):
        raise ReviewError("Draft schema changed; no signatures, seals, or admission fields allowed")
    for key in ("artifact_type", "version", "status", "review_manifest_digest",
                "completed_at", "attestation", "labels_admitted"):
        if draft[key] != template[key]:
            raise ReviewError("Draft must remain unsigned and unreviewed; QA completion is separate")
    reviewer = draft["reviewer_id"]
    if reviewer is None:
        if any(draft[key] is not None for key in (
            "reviewer_kind", "annotation_origin", "created_by", "started_at",
            "candidate_exposure_before_reference_sealed", "independent_of_labeler_and_training",
        )):
            raise ReviewError("Reviewer identity must be entered explicitly")
    else:
        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 200:
            raise ReviewError("Enter a nonempty reviewer identity")
        if (draft["reviewer_kind"] != "human" or draft["created_by"] != reviewer
                or draft["annotation_origin"] != "human_blind_full_episode"):
            raise ReviewError("Explicit human reviewer self-identification is required")
        try:
            started = datetime.fromisoformat(draft["started_at"].replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ReviewError("Reviewer start time must be entered on explicit begin") from exc
        if started.tzinfo is None:
            raise ReviewError("Reviewer start time requires a timezone")
        for key in ("candidate_exposure_before_reference_sealed",
                    "independent_of_labeler_and_training"):
            if draft[key] is not None and type(draft[key]) is not bool:
                raise ReviewError("Reviewer disclosure must be explicit yes/no or unanswered")
    if not isinstance(draft["episodes"], list) or len(draft["episodes"]) != len(template["episodes"]):
        raise ReviewError("Draft must retain every source episode")
    for reference, blank, meta in zip(draft["episodes"], template["episodes"], manifest["episodes"],
                                      strict=True):
        if set(reference) != set(blank):
            raise ReviewError("Episode schema changed")
        for key in ("recording_id", "source_id", "evidence_sha256"):
            if reference[key] != blank[key]:
                raise ReviewError("Episode order, physical source, or evidence identity changed")
        frames = meta["frames"]
        try:
            reviewed = _ranges(reference["reviewed_frame_ranges"], frames)
            for key in CHECKS:
                if reference[key] is not None and type(reference[key]) is not bool:
                    raise ReviewError("Episode review check must be explicit true/false or unanswered")
            for key, ambiguous in (("events", False), ("ambiguities", True)):
                if not isinstance(reference[key], list) or len(reference[key]) > frames * 3:
                    raise ReviewError("Unbounded or invalid event list")
                for event in reference[key]:
                    required = {"event_type", "object_id", "start", "end"}
                    required |= ({"reason"} if ambiguous else {
                        "reference_id", "left_censored", "right_censored"})
                    if set(event) != required:
                        raise ReviewError("Unexpected event fields")
                    _interval(event, frames, ambiguity=ambiguous)
                    if (objects is not None and event["object_id"] is not None
                            and event["object_id"] not in objects[meta["recording_id"]]):
                        raise ReviewError("Object identity is absent from raw source geometry")
            ids = [event["reference_id"] for event in reference["events"]]
            if any(not isinstance(value, str) or not value or len(value) > 200 for value in ids):
                raise ReviewError("Events require explicit unique reference identities")
            if len(set(ids)) != len(ids):
                raise ReviewError("Duplicate reference event identity")
            for index, event in enumerate(reference["events"]):
                others = reference["events"][index + 1:] + reference["ambiguities"]
                if any(_same_identity(event, other) and _overlap(event, other) for other in others):
                    raise ReviewError("Same-object/type positive intervals cannot overlap events/ambiguity")
            negatives = reference["negative_frame_ranges_by_type"]
            if set(negatives) != set(EVENT_TYPES):
                raise ReviewError("Every event type needs its explicit negative-range list")
            for kind in EVENT_TYPES:
                mask = _ranges(negatives[kind], frames)
                occupied = [False] * frames
                for event in reference["events"] + reference["ambiguities"]:
                    if event["event_type"] == kind:
                        occupied[event["start"]:event["end"]] = [True] * (event["end"]-event["start"])
                if any(a and b for a, b in zip(mask, occupied, strict=True)):
                    raise ReviewError("Negative ranges conflict with human events/ambiguities")
                if reference["negative_cases_reviewed"] is True and any(
                    a == b for a, b in zip(mask, occupied, strict=True)
                ):
                    raise ReviewError("Confirmed negatives must cover exactly the clear complement")
            if all(reference[key] is True for key in CHECKS) and not all(reviewed):
                raise ReviewError("Completed episode checks require explicit full-frame review coverage")
        except LabelReviewError as exc:
            raise ReviewError(str(exc)) from exc
        if not isinstance(reference["notes"], str) or len(reference["notes"]) > 100000:
            raise ReviewError("Invalid or excessive episode notes")
        if reviewer is None and reference != blank:
            raise ReviewError("Begin review with your own identity before saving annotations")
    return draft


class DraftStore:
    """Append-only version chain. Atomic creation + fsync; no silently overwritten draft."""
    def __init__(self, directory, manifest, maximum_bytes, objects=None, maximum_human_seconds=14400):
        self.directory = Path(directory)
        self.manifest, self.maximum_bytes, self.objects = manifest, maximum_bytes, objects
        self.maximum_human_seconds = maximum_human_seconds
        self.reported_human_minutes = 0.0
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise ReviewError("Draft directory cannot be a symlink")
        self.metadata = {"artifact_type": "eventground_wam.week10_unsigned_draft_store",
                         "review_manifest_digest": manifest["digest"], "labels_admitted": False}
        meta = self.directory / "store.json"
        if not meta.exists():
            if any(self.directory.iterdir()):
                raise ReviewError("Refuse to initialize over an unrelated nonempty directory")
            self._create(meta, encoded(self.metadata))
        elif parse_json(meta.read_bytes()) != self.metadata:
            raise ReviewError("Draft store belongs to another review manifest")
        self.revision, self.previous_sha = 0, None
        self.current = make_review_template(manifest)
        self._load()

    def _size(self):
        total = 0
        for path in self.directory.iterdir():
            if path.is_symlink() or not path.is_file():
                raise ReviewError("Unexpected nonregular path in draft store")
            total += path.stat().st_size
        return total

    def _create(self, path, payload):
        if self._size() + len(payload) > self.maximum_bytes:
            raise ReviewError("Draft output budget exhausted; existing versions preserved")
        temporary = self.directory / f".pending-{secrets.token_hex(12)}"
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # link is atomic and refuses an existing final version; never replace one.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _load(self):
        if self._size() > self.maximum_bytes:
            raise ReviewError("Existing drafts exceed this session output budget")
        files = sorted(self.directory.glob("draft-*.json"))
        allowed = {"store.json", ".writer.lock"} | {path.name for path in files}
        if any(path.name not in allowed for path in self.directory.iterdir()):
            raise ReviewError("Unexpected draft-store entry; preserve and inspect")
        for revision, path in enumerate(files, 1):
            if path.name != f"draft-{revision:06d}.json" or path.is_symlink():
                raise ReviewError("Draft history is incomplete or reordered")
            if path.stat().st_size > MAX_JSON * 4:
                raise ReviewError("Excessive draft version size")
            raw = path.read_bytes()
            record = parse_json(raw)
            if (record.get("revision") != revision or record.get("previous_file_sha256") != self.previous_sha
                    or record.get("review_manifest_digest") != self.manifest["digest"]
                    or record.get("unsigned") is not True
                    or record.get("draft_sha256") != hashlib.sha256(encoded(record["draft"])).hexdigest()):
                raise ReviewError("Draft history integrity failure")
            validate_draft(record["draft"], self.manifest, self.objects)
            minutes = record.get("reported_human_minutes")
            self._validate_minutes(minutes)
            self.reported_human_minutes = minutes
            self.current, self.revision = record["draft"], revision
            self.previous_sha = hashlib.sha256(raw).hexdigest()
        if self._size() > self.maximum_bytes:
            raise ReviewError("Existing drafts exceed this session's output budget")

    def _validate_minutes(self, minutes):
        import math
        if (type(minutes) not in (int, float) or not math.isfinite(minutes)
                or not self.reported_human_minutes <= minutes <= self.maximum_human_seconds / 60):
            raise ReviewError("Reported human minutes must be cumulative and within the child budget")

    def save(self, draft, base_revision, reported_human_minutes=0):
        import fcntl
        if type(base_revision) is not int:
            raise Conflict("A valid base revision is required")
        self._size()
        with (self.directory / ".writer.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            latest = sorted(self.directory.glob("draft-*.json"))
            if len(latest) != self.revision or base_revision != self.revision:
                raise Conflict("Another save exists; reload before editing to avoid lost work")
            if parse_json((self.directory / "store.json").read_bytes()) != self.metadata:
                raise ReviewError("Draft-store identity changed")
            if latest and hashlib.sha256(latest[-1].read_bytes()).hexdigest() != self.previous_sha:
                raise ReviewError("Previously saved draft changed")
            self._validate_minutes(reported_human_minutes)
            validate_draft(draft, self.manifest, self.objects)
            if self.current["reviewer_id"] is not None and draft["reviewer_id"] != self.current["reviewer_id"]:
                raise ReviewError("Changing reviewer requires a new independent draft store")
            if self.current["started_at"] is not None and draft["started_at"] != self.current["started_at"]:
                raise ReviewError("The saved review start time is immutable")
            record = {"artifact_type": "eventground_wam.week10_unsigned_draft_version",
                      "revision": self.revision + 1, "saved_at": stamp(), "unsigned": True,
                      "reported_human_minutes": reported_human_minutes,
                      "review_manifest_digest": self.manifest["digest"],
                      "previous_file_sha256": self.previous_sha,
                      "draft_sha256": hashlib.sha256(encoded(draft)).hexdigest(), "draft": draft}
            payload = encoded(record)
            self._create(self.directory / f"draft-{record['revision']:06d}.json", payload)
            self.current = copy.deepcopy(draft)
            self.reported_human_minutes = reported_human_minutes
            self.revision += 1
            self.previous_sha = hashlib.sha256(payload).hexdigest()
            return {"revision": self.revision, "saved_at": record["saved_at"],
                    "draft_sha256": record["draft_sha256"], "unsigned": True}


class Evidence:
    def __init__(self, capture_base, manifest, deadline):
        self.rows = {row["recording_id"]: row for row in manifest["episodes"]}
        self.files, self.objects = {}, {}
        self.cache = {}
        for rid, row in self.rows.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", rid) or set(row["evidence_sha256"]) != EVIDENCE_NAMES:
                raise ReviewError("Unsafe or unexpected evidence identity")
            self.files[rid] = {}
            for name, checksum in row["evidence_sha256"].items():
                path = Path(capture_base) / "run" / rid / name
                self.files[rid][name] = VerifiedFile(path, checksum, deadline)
            geometry = parse_json(self.files[rid]["geometry-map.json"].read())
            self.objects[rid] = {item["object_id"] for item in geometry["objects"]
                                 if item["movable_free_object"] is True}

    def read(self, mode, rid, frame, view=None):
        import h5py
        row = self.rows[rid]
        if type(frame) is not int or not 0 <= frame < row["frames"]:
            raise ReviewError("Frame outside this episode")
        files = self.files[rid]
        for file in files.values():
            file.check()
        h5 = files["recording.h5"]
        with h5py.File(h5.path, "r") as handle:
            fd = handle.id.get_vfd_handle()
            h5.check_fd(fd)
            if mode == "frame" and view in ("front", "wrist"):
                payload = png(handle[f"observation/{view}"][frame])
            elif mode == "numeric":
                if rid not in self.cache:
                    contacts = [parse_json(line) for line in files["contacts.jsonl"].read().splitlines()]
                    if len(contacts) != row["frames"]:
                        raise ReviewError("Contact evidence/frame count mismatch")
                    self.cache = {rid: (contacts, parse_json(files["geometry-map.json"].read()))}
                contacts, geometry = self.cache[rid]
                # Deliberate numeric allowlist; no outcomes or candidate arrays are exposed.
                payload = encoded({"frame": frame, "contacts": contacts[frame], "geometry": geometry,
                                   **{key: handle[key][frame].tolist() for key in (
                                       "time", "action", "observation/state", "object_position",
                                       "object_rotation_matrix", "gripper_position",
                                       "gripper_rotation_matrix")}})
            else:
                raise ReviewError("Unknown evidence route")
            h5.check_fd(fd)
        for file in files.values():
            file.check()
        return payload


def check_request(headers, port, csrf=None):
    """Exact Host prevents DNS rebinding; mutation additionally requires Origin + nonce."""
    expected_host = f"127.0.0.1:{port}"
    if headers.get("Host") != expected_host:
        raise ReviewError("Use the exact 127.0.0.1 loopback URL")
    origin = headers.get("Origin")
    if origin is not None and origin != f"http://{expected_host}":
        raise ReviewError("Foreign Origin refused")
    if headers.get("Sec-Fetch-Site") not in (None, "none", "same-origin"):
        raise ReviewError("Cross-site requests refused")
    if csrf is not None:
        if origin != f"http://{expected_host}" or not secrets.compare_digest(
            headers.get("X-Review-CSRF", ""), csrf
        ):
            raise ReviewError("Saving requires this session's same-origin CSRF token")
        if headers.get("Content-Type") != "application/json":
            raise ReviewError("Only explicit JSON saves are accepted")


class ReviewApp:
    def __init__(self, args):
        started = time.monotonic()
        self.contract, contract_file = sealed_file(args.session_contract)
        c = self.contract
        if (c.get("artifact_type") != "eventground_wam.week10_independent_review_session"
                or c.get("bind_host") != "127.0.0.1" or c.get("port") != args.port
                or c.get("human_annotations_only") is not True
                or c.get("labels_admitted") is not False or c.get("training_authorized") is not False):
            raise ReviewError("Wrong or unsafe review-session contract")
        for key, maximum in (("maximum_duration_seconds", 14400),
                             ("maximum_output_bytes", 1024**3),
                             ("maximum_human_review_seconds", 14400),
                             ("maximum_request_bytes", MAX_JSON)):
            if type(c.get(key)) is not int or not 0 < c[key] <= maximum:
                raise ReviewError(f"Missing or excessive session budget: {key}")
        if type(args.port) is not int or not 1024 <= args.port <= 65535:
            raise ReviewError("Use a fixed unprivileged loopback port")
        self.deadline = started + c["maximum_duration_seconds"]
        self.inputs = [contract_file]
        for key, path in (
            ("script_sha256", Path(__file__).resolve()), ("ui_sha256", UI_PATH),
            ("qa_sha256", REPO / "src/eventground_wam/eventground_qa.py"),
            ("contracts_sha256", REPO / "src/eventground_wam/demonstration_contracts.py"),
        ):
            self.inputs.append(VerifiedFile(path, c.get(key), self.deadline))
        self.ui = self.inputs[2].read()
        self.manifest, manifest_file = sealed_file(args.materials / "manifest.json")
        prepared, prepared_file = sealed_file(args.materials / "completion.json")
        self.inputs.extend((manifest_file, prepared_file))
        if (self.manifest.get("artifact_type") != "eventground_wam.week10_blind_review_manifest"
                or self.manifest.get("study_digest") != ACTIVE_STUDY_DIGEST
                or self.manifest.get("matching_rule") != MATCHING_RULE
                or self.manifest.get("candidate_data_included") is not False
                or self.manifest.get("labels_admitted") is not False
                or self.manifest.get("control_hz") != 20
                or c.get("review_manifest_digest") != self.manifest["digest"]
                or prepared.get("manifest_digest") != self.manifest["digest"]):
            raise ReviewError("Review does not bind the immutable blind source manifest")
        if set(prepared["files_sha256"]) != {"manifest.json", "review-template.json", "index.html"}:
            raise ReviewError("Unexpected original preparation artifacts")
        for name, checksum in prepared["files_sha256"].items():
            self.inputs.append(VerifiedFile(args.materials / name, checksum, self.deadline))
        template = parse_json(next(file for file in self.inputs
                                   if file.path.name == "review-template.json").read())
        if template != make_review_template(self.manifest):
            raise ReviewError("Original template is not the blank unsigned review form")
        rows = self.manifest["episodes"]
        if (len(rows) != 200 or len({row["recording_id"] for row in rows}) != 200
                or len({row["source_id"] for row in rows}) != 200
                or len(Counter(row["task_key"] for row in rows)) != 40
                or set(Counter(row["task_key"] for row in rows).values()) != {5}):
            raise ReviewError("Only the exact initial 200 sources across all 40 tasks may be reviewed")
        self.evidence = Evidence(args.capture_base, self.manifest, self.deadline)
        self.store = DraftStore(args.drafts, self.manifest, c["maximum_output_bytes"],
                                self.evidence.objects, c["maximum_human_review_seconds"])
        self.csrf = secrets.token_urlsafe(32)
        self.port = args.port

    def live(self):
        if time.monotonic() >= self.deadline:
            raise ReviewError("Review session expired; saved versions remain available")
        for file in self.inputs:
            file.check()

    def get(self, path):
        self.live()
        if path == "/":
            return self.ui, "text/html; charset=utf-8"
        if path == "/health":
            return encoded({"status": "ready", "session_digest": self.contract["digest"],
                            "remaining_seconds": int(self.deadline-time.monotonic()),
                            "labels_admitted": False}), "application/json"
        if path == "/api/session":
            return encoded({"manifest": self.manifest, "csrf": self.csrf,
                            "revision": self.store.revision, "draft": self.store.current,
                            "reported_human_minutes": self.store.reported_human_minutes,
                            "maximum_human_review_seconds": self.contract["maximum_human_review_seconds"],
                            "session_digest": self.contract["digest"],
                            "remaining_seconds": int(self.deadline-time.monotonic())}), "application/json"
        if path == "/api/export":
            return encoded(self.store.current), "application/json"
        match = re.fullmatch(r"/frame/([A-Za-z0-9_-]+)/([0-9]+)/(front|wrist)\.png", path)
        if match and match[1] in self.evidence.rows:
            return self.evidence.read("frame", match[1], int(match[2]), match[3]), "image/png"
        match = re.fullmatch(r"/numeric/([A-Za-z0-9_-]+)/([0-9]+)\.json", path)
        if match and match[1] in self.evidence.rows:
            return self.evidence.read("numeric", match[1], int(match[2])), "application/json"
        raise KeyError("Unknown allowlisted review route")


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def send(self, code, payload, mime="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; img-src 'self' blob:; "
                             "script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                             "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
            if self.path == "/api/export":
                self.send_header("Content-Disposition", 'attachment; filename="unsigned-review-draft.json"')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            try:
                check_request(self.headers, app.port)
                url = urlsplit(self.path)
                if url.query or url.fragment or url.scheme or url.netloc:
                    raise KeyError("No query or absolute routes")
                payload, mime = app.get(url.path)
                self.send(200, payload, mime)
            except KeyError:
                self.send(404, encoded({"error": "Unknown review route"}))
            except (ReviewError, OSError, ValueError) as exc:
                self.send(403, encoded({"error": str(exc)}))

        def do_POST(self):
            try:
                check_request(self.headers, app.port, app.csrf)
                app.live()
                if self.path != "/api/draft" or self.headers.get("Transfer-Encoding"):
                    raise ReviewError("Only the bounded draft-save endpoint accepts POST")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= app.contract["maximum_request_bytes"]:
                    raise ReviewError("Missing or excessive save length")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ReviewError("Incomplete request")
                body = parse_json(raw)
                if set(body) != {"base_revision", "draft", "reported_human_minutes"}:
                    raise ReviewError("Unexpected save payload")
                # Retain complete source identity; frame inputs themselves remain read-only.
                for files in app.evidence.files.values():
                    for file in files.values():
                        file.check()
                app.live()
                result = app.store.save(body["draft"], body["base_revision"],
                                        body["reported_human_minutes"])
                self.send(200, encoded(result))
            except Conflict as exc:
                self.send(409, encoded({"error": str(exc)}))
            except (ReviewError, OSError, ValueError, KeyError, TypeError) as exc:
                self.send(400, encoded({"error": str(exc)}))

        def log_message(self, *_args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("materials", "capture-base", "session-contract", "drafts"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    app = ReviewApp(args)
    server = HTTPServer(("127.0.0.1", args.port), make_handler(app))
    server.timeout = 1
    print(f"Blind human review ready at http://127.0.0.1:{args.port}; "
          f"session {app.contract['digest']}; unsigned drafts {args.drafts}", flush=True)
    try:
        while time.monotonic() < app.deadline:
            server.handle_request()
    finally:
        server.server_close()
        print("Review session ended. Saved unsigned versions are preserved.", flush=True)


if __name__ == "__main__":
    main()

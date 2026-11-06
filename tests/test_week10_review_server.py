"""Synthetic-only tests for blind review serving and durable unsigned drafts."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_qa import (
    ACTIVE_STUDY_DIGEST,
    MATCHING_RULE,
    make_review_template,
)

SCRIPT = Path(__file__).parents[1] / "scripts/serve_week10_label_review.py"
spec = importlib.util.spec_from_file_location("week10_review_server", SCRIPT)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def manifest():
    return seal_contract({
        "artifact_type": "eventground_wam.week10_blind_review_manifest",
        "study_digest": ACTIVE_STUDY_DIGEST, "matching_rule": MATCHING_RULE,
        "control_hz": 20, "candidate_data_included": False, "labels_admitted": False,
        "episodes": [{"recording_id": "source-00", "source_id": "a"*64,
                      "task_key": "task-00", "instruction": "Synthetic task", "frames": 10,
                      "evidence_sha256": {name: "b"*64 for name in server.EVIDENCE_NAMES}}],
    })


def human_draft(m):
    draft = make_review_template(m)
    draft.update(reviewer_id="Synthetic test reviewer", reviewer_kind="human",
                 created_by="Synthetic test reviewer", annotation_origin="human_blind_full_episode",
                 started_at="2026-10-18T20:00:00+00:00")
    return draft


def event(start=2, end=5, kind="contact", obj="cup"):
    return {"reference_id": "manual-1", "event_type": kind, "object_id": obj,
            "start": start, "end": end, "left_censored": start == 0,
            "right_censored": end == 10}


def test_blank_template_stays_unsigned_and_no_identity_is_invented(tmp_path):
    m = manifest()
    store = server.DraftStore(tmp_path / "drafts", m, 100000)
    assert store.revision == 0
    assert store.current == make_review_template(m)
    assert store.current["reviewer_id"] is None
    assert store.current["status"] == "unreviewed"
    assert list((tmp_path / "drafts").iterdir()) == [tmp_path / "drafts/store.json"]


def test_save_resume_version_chain_and_cumulative_time(tmp_path):
    m = manifest()
    path = tmp_path / "drafts"
    store = server.DraftStore(path, m, 100000)
    d = human_draft(m)
    d["episodes"][0]["events"].append(event())
    first = store.save(d, 0, 2.5)
    before = (path / "draft-000001.json").read_bytes()
    d["episodes"][0]["notes"] = "Synthetic observation only"
    store.save(d, 1, 4.0)
    assert (path / "draft-000001.json").read_bytes() == before
    resumed = server.DraftStore(path, m, 100000)
    assert resumed.current == d
    assert resumed.revision == 2 and resumed.reported_human_minutes == 4
    assert resumed.current["attestation"]["signed_at"] is None
    assert resumed.current["status"] == "unreviewed" and first["unsigned"]
    assert not list(path.glob(".pending-*"))
    with pytest.raises(server.ReviewError, match="cumulative"):
        resumed.save(d, 2, 3)


def test_stale_revision_conflicts_and_review_identity_cannot_change(tmp_path):
    m = manifest()
    store = server.DraftStore(tmp_path, m, 100000)
    d = human_draft(m)
    store.save(d, 0)
    with pytest.raises(server.Conflict):
        store.save(d, 0)
    d.update(reviewer_id="Another", created_by="Another")
    with pytest.raises(server.ReviewError, match="Changing reviewer"):
        store.save(d, 1)
    assert len(list(tmp_path.glob("draft-*.json"))) == 1


def test_second_writer_cannot_overwrite_first(tmp_path):
    m = manifest()
    one = server.DraftStore(tmp_path, m, 100000)
    two = server.DraftStore(tmp_path, m, 100000)
    one.save(human_draft(m), 0)
    with pytest.raises(server.Conflict):
        two.save(human_draft(m), 0)


@pytest.mark.parametrize("field,value", [
    ("status", "complete"), ("labels_admitted", True),
    ("completed_at", "2026-10-19T00:00:00Z"),
    ("attestation", {"statement": "I certify", "reviewer_id": "someone", "signed_at": "now"}),
    ("digest", "f"*64),
])
def test_save_never_accepts_completion_or_signature(tmp_path, field, value):
    m = manifest()
    store = server.DraftStore(tmp_path, m, 100000)
    d = human_draft(m)
    d[field] = value
    with pytest.raises(server.ReviewError):
        store.save(d, 0)
    assert store.revision == 0


def test_source_links_and_geometry_object_are_immutable():
    m = manifest()
    d = human_draft(m)
    d["episodes"][0]["source_id"] = "f"*64
    with pytest.raises(server.ReviewError, match="identity changed"):
        server.validate_draft(d, m)
    d = human_draft(m)
    d["episodes"][0]["events"] = [event(obj="fixed-fixture")]
    with pytest.raises(server.ReviewError, match="raw source geometry"):
        server.validate_draft(d, m, {"source-00": {"cup"}})


def test_annotations_cannot_be_saved_without_manual_human_identity():
    m = manifest()
    d = make_review_template(m)
    d["episodes"][0]["events"].append(event())
    with pytest.raises(server.ReviewError, match="your own identity"):
        server.validate_draft(d, m)


def test_same_type_overlap_rejected_but_simultaneous_different_types_allowed():
    m = manifest()
    d = human_draft(m)
    d["episodes"][0]["events"] = [event(), {**event(kind="grasp"), "reference_id": "manual-2"}]
    server.validate_draft(d, m)
    d["episodes"][0]["ambiguities"] = [{"event_type": "contact", "object_id": "cup",
                                         "start": 4, "end": 6, "reason": "Synthetic uncertainty"}]
    with pytest.raises(server.ReviewError, match="cannot overlap"):
        server.validate_draft(d, m)


def test_negative_complement_and_full_review_checks():
    m = manifest()
    d = human_draft(m)
    episode = d["episodes"][0]
    episode["events"] = [event()]
    episode["negative_cases_reviewed"] = True
    with pytest.raises(server.ReviewError, match="clear complement"):
        server.validate_draft(d, m)
    episode["negative_frame_ranges_by_type"] = {
        "contact": [[0, 2], [5, 10]], "grasp": [[0, 10]], "gripper_close": [[0, 10]],
    }
    for check in server.CHECKS:
        episode[check] = True
    with pytest.raises(server.ReviewError, match="full-frame"):
        server.validate_draft(d, m)
    episode["reviewed_frame_ranges"] = [[0, 10]]
    server.validate_draft(d, m)
    assert d["status"] == "unreviewed"  # Complete episode checks are not a signature.


def test_budget_blocks_before_creating_a_version(tmp_path):
    m = manifest()
    store = server.DraftStore(tmp_path, m, 1000)
    with pytest.raises(server.ReviewError, match="budget"):
        store.save(human_draft(m), 0)
    assert store.revision == 0
    assert not list(tmp_path.glob("draft-*.json"))
    assert not list(tmp_path.glob(".pending-*"))


def test_incomplete_or_tampered_version_fails_closed_on_resume(tmp_path):
    m = manifest()
    store = server.DraftStore(tmp_path, m, 100000)
    store.save(human_draft(m), 0)
    path = tmp_path / "draft-000001.json"
    record = json.loads(path.read_text())
    record["draft"]["episodes"][0]["notes"] = "Changed"
    path.write_text(json.dumps(record))
    with pytest.raises(server.ReviewError, match="integrity"):
        server.DraftStore(tmp_path, m, 100000)


def test_atomic_save_failure_leaves_prior_versions_and_no_partial_final(tmp_path, monkeypatch):
    m = manifest()
    store = server.DraftStore(tmp_path, m, 100000)
    store.save(human_draft(m), 0)
    before = (tmp_path / "draft-000001.json").read_bytes()
    monkeypatch.setattr(os, "link", lambda *_: (_ for _ in ()).throw(OSError("Synthetic disk error")))
    with pytest.raises(OSError):
        store.save(human_draft(m), 1)
    assert (tmp_path / "draft-000001.json").read_bytes() == before
    assert not (tmp_path / "draft-000002.json").exists()
    assert not list(tmp_path.glob(".pending-*"))


def test_file_mutation_same_size_detected(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"abcd")
    verified = server.VerifiedFile(path, hashlib.sha256(b"abcd").hexdigest())
    original = path.stat()
    path.write_bytes(b"abce")
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(server.ReviewError, match="changed"):
        verified.read()


def test_symlink_evidence_rejected(tmp_path):
    real = tmp_path / "real"
    real.write_bytes(b"data")
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(server.ReviewError, match="symlink"):
        server.VerifiedFile(link, hashlib.sha256(b"data").hexdigest())


def valid_headers():
    return {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765",
            "Sec-Fetch-Site": "same-origin", "X-Review-CSRF": "test-token",
            "Content-Type": "application/json"}


@pytest.mark.parametrize("changed", [
    {"Host": "evil.example:8765"}, {"Origin": "https://evil.example"},
    {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"},
    {"X-Review-CSRF": "wrong"}, {"Origin": None}, {"Content-Type": "text/plain"},
])
def test_origin_host_csrf_guards(changed):
    headers = {**valid_headers(), **changed}
    with pytest.raises(server.ReviewError):
        server.check_request(headers, 8765, "test-token")


def test_same_origin_get_and_post_allowed():
    server.check_request({"Host": "127.0.0.1:8765", "Sec-Fetch-Site": "none"}, 8765)
    server.check_request(valid_headers(), 8765, "test-token")


def request(app, method, path, raw=b"", headers=None):
    handler = object.__new__(server.make_handler(app))
    handler.path, handler.rfile = path, io.BytesIO(raw)
    handler.headers = {**valid_headers(), "Content-Length": str(len(raw)), **(headers or {})}
    result = {}
    handler.send = lambda code, payload, mime="application/json": result.update(
        code=code, payload=json.loads(payload) if mime == "application/json" else payload, mime=mime)
    getattr(handler, "do_"+method)()
    return result


def app_fixture(tmp_path):
    app = object.__new__(server.ReviewApp)
    app.deadline = time.monotonic()+1000
    app.contract = {"digest": "c"*64, "maximum_request_bytes": 100000,
                    "maximum_human_review_seconds": 14400}
    app.inputs, app.port, app.csrf = [], 8765, "test-token"
    app.manifest = manifest()
    app.store = server.DraftStore(tmp_path, app.manifest, 1000000)
    app.evidence = SimpleNamespace(rows={}, files={})
    app.ui = b"synthetic UI"
    return app


def test_http_body_read_validation_and_stale_revision(tmp_path):
    app = app_fixture(tmp_path)
    body = server.encoded({"base_revision": 0, "draft": human_draft(app.manifest),
                           "reported_human_minutes": 1})
    assert request(app, "POST", "/api/draft", body)["code"] == 200
    assert request(app, "POST", "/api/draft", body)["code"] == 409
    assert request(app, "POST", "/api/draft", body[:-1],
                   {"Content-Length": str(len(body))})["code"] == 400
    assert request(app, "POST", "/api/draft", b"{}").get("code") == 400
    assert request(app, "POST", "/api/draft", body,
                   {"Content-Length": "100001"})["code"] == 400
    assert request(app, "POST", "/api/draft", body,
                   {"Transfer-Encoding": "chunked"})["code"] == 400
    assert app.store.revision == 1


@pytest.mark.parametrize("path", ["/../event-candidates.json", "/api/export?file=secret",
                                  "/frame/../../secret", "/candidate/source-00/0.json",
                                  "http://evil.example/api/session", "/%2e%2e/manifest.json"])
def test_http_unknown_routes_never_read_arbitrary_files(tmp_path, path):
    app = app_fixture(tmp_path)
    assert request(app, "GET", path)["code"] == 404


def test_health_has_no_token_export_is_unsigned_and_expiry_blocks_save(tmp_path):
    app = app_fixture(tmp_path)
    health = request(app, "GET", "/health")["payload"]
    assert "csrf" not in health and "draft" not in health
    assert health["labels_admitted"] is False
    exported = request(app, "GET", "/api/export")["payload"]
    assert exported == make_review_template(app.manifest)
    assert "digest" not in exported and exported["attestation"]["signed_at"] is None
    app.deadline = time.monotonic()-1
    assert request(app, "GET", "/api/session")["code"] == 403
    assert request(app, "POST", "/api/draft", b"{}")["code"] == 400


def test_original_media_and_numeric_allowlist_no_outcome_leak(tmp_path):
    folder = tmp_path / "run/source-00"
    folder.mkdir(parents=True)
    with h5py.File(folder / "recording.h5", "w") as h:
        for view in ("front", "wrist"):
            h.create_dataset(f"observation/{view}", data=np.zeros((10, 256, 256, 3), np.uint8))
        for key in ("time", "action", "observation/state", "object_position",
                    "object_rotation_matrix", "gripper_position", "gripper_rotation_matrix"):
            h.create_dataset(key, data=np.zeros((10, 3), np.float32))
        h.create_dataset("task_success", data=np.ones(10))
        h.create_dataset("event_candidates", data=np.ones(10))
    (folder / "contacts.jsonl").write_text("\n".join(
        json.dumps({"frame_index": i, "contacts": []}) for i in range(10)))
    (folder / "geometry-map.json").write_text(json.dumps({
        "objects": [{"object_id": "cup", "movable_free_object": True},
                    {"object_id": "table", "movable_free_object": False}], "geoms": []}))
    m = manifest()
    m["episodes"][0]["evidence_sha256"] = {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in server.EVIDENCE_NAMES}
    e = server.Evidence(tmp_path, m, time.monotonic()+60)
    assert e.objects == {"source-00": {"cup"}}
    data = json.loads(e.read("numeric", "source-00", 0))
    assert "task_success" not in data and "event_candidates" not in data
    assert e.read("frame", "source-00", 0, "front").startswith(b"\x89PNG")
    with pytest.raises(server.ReviewError):
        e.read("frame", "source-00", 10, "front")
    (folder / "geometry-map.json").write_text("{}")
    with pytest.raises(server.ReviewError, match="changed"):
        e.read("numeric", "source-00", 0)


def test_duplicate_json_keys_and_nonfinite_values_rejected():
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        with pytest.raises(server.ReviewError):
            server.parse_json(raw)

"""Synthetic decisions only: no production reviewer identity or approval is invented."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from eventground_wam.demonstration_contracts import seal_contract

SCRIPT = Path(__file__).parents[1] / "scripts/serve_week10_assisted_review.py"
spec = importlib.util.spec_from_file_location("assisted_review_test", SCRIPT)
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def manifest():
    return seal_contract({"artifact_type": "eventground_wam.week10_assisted_review_manifest",
                          "episodes": [{"recording_id": "synthetic-00", "frames": 12,
                                        "proposals": [
                                            {"proposal_id": "synthetic-00:event:0000", "kind": "event",
                                             "original": {"event_type": "grasp", "start": 2, "end": 9}},
                                            {"proposal_id": "synthetic-00:ambiguity:0000", "kind": "ambiguity",
                                             "original": {"event_type": "grasp", "start": 1, "end": 6}}
                                        ]}]})


def human(m):
    draft = server.make_draft(m)
    draft.update(reviewer_id="Synthetic test reviewer", started_at="2026-10-22T04:00:00Z")
    return draft


def test_blank_has_no_human_decisions_and_preserves_ambiguities():
    draft = server.make_draft(manifest())
    assert draft["reviewer_id"] is None and draft["independent_reference"] is False
    assert all(d["decision"] == "pending" for d in draft["episodes"][0]["decisions"])
    assert len(draft["episodes"][0]["decisions"]) == 2
    server.validate_draft(draft, manifest())


@pytest.mark.parametrize("field,value", [("labels_admitted", True), ("quality_validated", True),
                                       ("independent_reference", True), ("status", "complete"),
                                       ("candidate_exposure_before_review", False)])
def test_cannot_promote_assisted_review(field, value):
    m = manifest()
    draft = human(m)
    draft[field] = value
    with pytest.raises(server.Error):
        server.validate_draft(draft, m)


def test_cannot_remove_ambiguity_or_reorder_source():
    m = manifest()
    for change in (lambda d: d["episodes"][0]["decisions"].pop(),
                   lambda d: d["episodes"][0]["decisions"].reverse()):
        draft = human(m)
        change(draft)
        with pytest.raises(server.Error):
            server.validate_draft(draft, m)


def test_cannot_create_decisions_without_identity():
    m = manifest()
    draft = server.make_draft(m)
    draft["episodes"][0]["decisions"][0].update(decision="correct", decided_at="2026-10-22T04:00:10Z")
    with pytest.raises(server.Error):
        server.validate_draft(draft, m)


def test_no_extra_corrected_interval_or_reference_claim():
    m = manifest()
    draft = human(m)
    draft["episodes"][0]["decisions"][0]["start"] = 3
    with pytest.raises(server.Error):
        server.validate_draft(draft, m)


def test_save_resume_keeps_original_and_pending_uncertainty(tmp_path):
    m = manifest()
    store = server.DecisionStore(tmp_path / "drafts", m, 100000)
    draft = human(m)
    draft["episodes"][0]["decisions"][0].update(decision="needs_correction", notes="Starts too early",
                                                decided_at="2026-10-22T04:00:10Z")
    store.save(draft, 0, 1)
    before = (tmp_path / "drafts/draft-000001.json").read_bytes()
    draft["episodes"][0]["notes"] = "Possible missed contact"
    store.save(draft, 1, 2)
    assert (tmp_path / "drafts/draft-000001.json").read_bytes() == before
    resumed = server.DecisionStore(tmp_path / "drafts", m, 100000)
    assert resumed.revision == 2 and resumed.current == draft
    assert resumed.current["episodes"][0]["decisions"][1]["decision"] == "pending"
    assert resumed.current["labels_admitted"] is False


def test_stale_saves_identity_changes_and_time_rollback_refused(tmp_path):
    m = manifest()
    store = server.DecisionStore(tmp_path / "drafts", m, 100000)
    other = server.DecisionStore(tmp_path / "drafts", m, 100000)
    draft = human(m)
    store.save(draft, 0, 3)
    with pytest.raises(server.base.Conflict):
        other.save(draft, 0, 4)
    with pytest.raises(server.Error):
        store.save(draft, 1, 2)
    changed = copy.deepcopy(draft)
    changed["reviewer_id"] = "Another synthetic reviewer"
    with pytest.raises(server.Error):
        store.save(changed, 1, 4)


def test_tampered_history_refused(tmp_path):
    m = manifest()
    store = server.DecisionStore(tmp_path / "drafts", m, 100000)
    store.save(human(m), 0)
    path = tmp_path / "drafts/draft-000001.json"
    record = json.loads(path.read_text())
    record["draft"]["episodes"][0]["notes"] = "Tampered"
    path.write_text(json.dumps(record))
    with pytest.raises(server.Error):
        server.DecisionStore(tmp_path / "drafts", m, 100000)


def test_output_limit_preserves_existing_versions(tmp_path):
    m = manifest()
    store = server.DecisionStore(tmp_path / "drafts", m, 1000)
    with pytest.raises(server.Error):
        store.save(human(m), 0)
    assert not list((tmp_path / "drafts").glob("draft-*.json"))


def extraction_fixture(tmp_path):
    row = {"recording_id": "synthetic-00", "source_id": "a"*64, "frames": 12, "task_key": "task-00"}
    candidate = seal_contract({**row, "labels_admitted": False,
                               "events": [{"event_type": "grasp", "start": 2, "end": 9}],
                               "ambiguities": [{"event_type": "grasp", "start": 1, "end": 6}]})
    path = tmp_path / "synthetic-00/event-candidates.json"
    path.parent.mkdir()
    path.write_bytes(server.base.encoded(candidate))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    blind = seal_contract({"study_digest": "b"*64, "capture_completion_digest": "c"*64,
                           "episodes": [row]})
    completion = seal_contract({"status": "complete", "labels_admitted": False,
                                "capture_completion_digest": "c"*64,
                                "cases": [{**row, "candidate_digest": candidate["digest"],
                                           "candidate_file_sha256": sha}],
                                "output_files_sha256": {"synthetic-00/event-candidates.json": sha}})
    return blind, completion, path


def test_preparation_keeps_all_original_proposals_without_human_decisions(tmp_path):
    blind, completion, path = extraction_fixture(tmp_path)
    result = server.build_manifest(blind, completion, tmp_path)
    assert len(result["episodes"][0]["proposals"]) == 2
    original = json.loads(path.read_text())
    assert result["episodes"][0]["proposals"][0]["original"] == original["events"][0]
    assert result["episodes"][0]["proposals"][1]["original"] == original["ambiguities"][0]
    assert result["independent_reference"] is False


def test_preparation_rejects_tampered_candidate(tmp_path):
    blind, completion, path = extraction_fixture(tmp_path)
    path.write_text(path.read_text() + " ")
    with pytest.raises(server.Error):
        server.build_manifest(blind, completion, tmp_path)


def test_wrong_capture_completion_refused(tmp_path):
    blind, completion, _ = extraction_fixture(tmp_path)
    completion["capture_completion_digest"] = "d"*64
    completion = seal_contract({k:v for k,v in completion.items() if k != "digest"})
    with pytest.raises(server.Error):
        server.build_manifest(blind, completion, tmp_path)

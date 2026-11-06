from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from eventground_wam.future_pairs import CAMERAS, HORIZONS
from eventground_wam.jepa_baselines import (
    FIELDS,
    assess_candidate,
    collapse_diagnostics,
    cosine_errors,
    error_masses,
    grouped_measurements,
    representative_sources,
    retrieval_metrics,
    retrieval_pool,
    retrieval_scores,
    token_motion_weights,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_train_mean_selection_rejects_validation_and_test_before_fitting():
    source = {"split": "validation", "episode_index": 1, "task_key": "a", "group_digest": "g"}
    with pytest.raises(ValueError):
        representative_sources([source], "train", "mean")
    with pytest.raises(ValueError):
        representative_sources([], "test", "mean")
    source["split"] = "train"
    rows = [source, {**source, "episode_index": 2}]
    assert representative_sources(rows, "train", "mean") == representative_sources(rows[::-1], "train", "mean")


def test_cosine_reference_and_nonfinite_rejection():
    left = np.array([[1., 0], [1., 0], [1., 0]], np.float32)
    right = np.array([[1., 0], [0., 1], [-1., 0]], np.float32)
    np.testing.assert_array_equal(cosine_errors(left, right), [0., 1., 2.])
    with pytest.raises(ValueError):
        cosine_errors(left, right*np.nan)


def test_motion_tokens_preserve_pixel_threshold_area_and_mass():
    current = np.zeros((1, 3, 128, 128), np.float32)
    target = current.copy()
    target[:, :, :16, :16] = 10/255
    assert token_motion_weights(current, target).sum() == 0
    target[:, :, :16, :16] = 11/255
    weights = token_motion_weights(current, target)
    assert weights[0, 0] == 1 and weights.sum() == 1
    masses = error_masses(np.ones((1, 64)), weights)
    assert masses["motion_area"] == masses["motion_mass"] == 1/64
    assert masses["static_mass"] == 63/64 and masses["cosine"] == 1


def test_tie_credit_is_chance_and_independent_of_target_slot():
    for slot in range(8):
        result = retrieval_metrics(np.ones(8), slot)
        assert result["recall1"] == 1/8 and result["recall3"] == 3/8
        assert result["mrr"] == pytest.approx(sum(1/i for i in range(1, 9))/8)
    assert retrieval_metrics(np.arange(8), 7)["recall1"] == 1
    assert retrieval_metrics(np.arange(8), 6)["mrr"] == .5


def test_retrieval_scores_use_all_aligned_tokens():
    rng = np.random.default_rng(3)
    query = rng.normal(size=(64, 2048)).astype(np.float32)
    query /= np.linalg.norm(query, axis=-1, keepdims=True)
    candidates = np.stack([query, -query, *[np.zeros_like(query)]*6])
    scores = retrieval_scores(query, candidates)
    np.testing.assert_allclose(scores[:2], [1, -1], atol=1e-7)
    assert retrieval_metrics(scores, 0)["recall1"] == 1


def test_temporal_pool_deduplicates_and_never_shrinks_pool():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(160, 64, 16)).astype(np.float32)
    values /= np.linalg.norm(values, axis=-1, keepdims=True)
    hashes = [str(i) for i in range(160)]
    pool = retrieval_pool(values, hashes, episode=3, camera="front", anchor=40, horizon=16)
    assert pool["eligible"] and len(pool["frames"]) == 8
    assert pool["frames"][pool["target_slot"]] == 56
    assert all(abs(a-b) >= 4 for i, a in enumerate(pool["frames"]) for b in pool["frames"][i+1:])
    duplicate = np.repeat(values[:1], 160, axis=0)
    excluded = retrieval_pool(duplicate, hashes, episode=3, camera="front", anchor=40, horizon=16)
    assert not excluded["eligible"] and excluded["retained_count"] == 1
    rgb_duplicate = retrieval_pool(values, ["same"]*160, episode=3, camera="front", anchor=40, horizon=16)
    assert not rgb_duplicate["eligible"]


def fixture_rows():
    baseline, predicted, queries, candidates = [], [], [], []
    for episode in range(120):
        identity = {"episode_index": episode, "source_digest": str(episode),
                    "task_key": f"task{episode//3:02}", "group_digest": str(episode)}
        for camera in CAMERAS:
            for h in HORIZONS:
                row = {**identity, "camera": camera, "horizon": h, "pairs": 200-h,
                       "copy": dict(zip(FIELDS, [.2, .08, .12, .4, .6])),
                       "mean": dict(zip(FIELDS, [.4, .16, .24, .4, .6]))}
                baseline.append(row)
                predicted.append({**{k: v for k, v in row.items() if k not in ("copy", "mean")},
                                  "prediction": dict(zip(FIELDS, [.1, .04, .06, .4, .6]))})
        q = {**identity, "camera": CAMERAS[0], "horizon": 8, "anchor": 50, "target": 58,
             "pool": {"eligible": True, "frames": list(range(8)), "target_slot": 0},
             "copy": {"recall1": .25, "recall3": .5, "mrr": .4}}
        queries.append(q)
        candidates.append({**{k: v for k, v in q.items() if k != "copy"},
                           "prediction": {"recall1": .5, "recall3": .7, "mrr": .6}})
    return baseline, predicted, queries, candidates


def collapse_reference():
    return {c: {"fixed_position_std_rms": .01, "pooled_effective_rank": 20.,
                "within_task_group_cosine": .5, "nonzero_pooled_vectors": True} for c in CAMERAS}


def test_group_means_do_not_treat_appearance_variants_as_independent_groups():
    baseline, _, _, _ = fixture_rows()
    groups, before = grouped_measurements(baseline, "copy")
    variants = [{**r, "episode_index": 999, "source_digest": "variant"} for r in baseline[:10]]
    after_groups, after = grouped_measurements(baseline+variants, "copy")
    assert groups == after_groups
    np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError):
        grouped_measurements(baseline[:-1], "copy")
    with pytest.raises(ValueError):
        grouped_measurements(baseline+[baseline[0]], "copy")


def test_candidate_gate_requires_improvement_and_review():
    baseline, predicted, queries, candidates = fixture_rows()
    reference = collapse_reference()
    result = assess_candidate(baseline, queries, predicted, candidates, reference, reference, reviewed_examples=480)
    assert result["status"] == "PASS"
    assert assess_candidate(baseline, queries, predicted, candidates, reference, reference)["status"] == "FAIL"
    for b, p in zip(baseline, predicted):
        p["prediction"] = b["copy"]
    result = assess_candidate(baseline, queries, predicted, candidates, reference, reference, reviewed_examples=480)
    assert result["status"] == "FAIL" and not result["checks"]["overall_cosine"]


def test_gate_rejects_mask_and_pool_changes():
    b, p, q, c = fixture_rows()
    ref = collapse_reference()
    p[0]["prediction"]["motion_area"] = .5
    with pytest.raises(ValueError, match="masks"):
        assess_candidate(b, q, p, c, ref, ref)
    p[0]["prediction"]["motion_area"] = .4
    c[0]["pool"] = {**c[0]["pool"], "target_slot": 1}
    with pytest.raises(ValueError, match="pools"):
        assess_candidate(b, q, p, c, ref, ref)


def test_collapse_cannot_be_hidden_by_spatial_texture_or_task_backgrounds():
    rng = np.random.default_rng(8)
    spatial = rng.normal(size=(1, 128, 2048)).astype(np.float32)
    groups = [("same-task", str(i)) for i in range(6)]
    metrics = collapse_diagnostics(np.repeat(spatial, 6, axis=0), groups)
    assert all(v["pooled_effective_rank"] == 0 for v in metrics.values())
    b, p, q, c = fixture_rows()
    ref = collapse_reference()
    collapsed = deepcopy(ref)
    for camera in CAMERAS:
        collapsed[camera]["within_task_group_cosine"] = 1.
    result = assess_candidate(b, q, p, c, ref, collapsed, reviewed_examples=480)
    assert result["status"] == "FAIL" and not result["checks"][f"collapse/{CAMERAS[0]}"]


def test_production_rejects_code_drift_after_pilot(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from run_jepa_baselines import verify_pilot

    identity = {"phase": "pilot", "digest": "pilot-id", "config_digest": "config",
                "pair_manifest_digest": "pairs", "source_sha256": {"dependency.py": "old"}, "versions": {"torch": "pinned"}}
    full = {**identity, "phase": "full"}
    pilot = {"passed": True, "teacher_unchanged": True, "identity_digest": "pilot-id"}
    verify_pilot(pilot, identity, full)
    full["source_sha256"] = {"dependency.py": "changed"}
    with pytest.raises(ValueError, match="identity"):
        verify_pilot(pilot, identity, full)


def test_independent_bootstrap_replays_shared_group_draws(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from verify_jepa_baselines import draws

    from eventground_wam.pixel_baselines import group_weights

    groups = [("a", "g1"), ("a", "g2"), ("b", "g3"), ("b", "g4"), ("b", "g5")]
    values = np.random.default_rng(4).normal(size=(5, 2, 3))
    point, bootstrap = group_weights(groups)
    reference, samples = draws(groups, values)
    np.testing.assert_allclose(reference, np.einsum("g,gmc->mc", point, values), atol=1e-15)
    np.testing.assert_allclose(samples, np.einsum("rg,gmc->rmc", bootstrap, values), atol=1e-15)

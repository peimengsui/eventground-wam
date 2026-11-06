from copy import deepcopy

import numpy as np
import pytest
import torch

from eventground_wam.future_pairs import CAMERAS, HORIZONS
from eventground_wam.pixel_baselines import FIELDS, measure_video
from eventground_wam.pixel_evaluation import EpisodeAccumulator, prediction_summary


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture()
def data():
    source = {"episode_index": 7, "digest": "source", "group_digest": "group",
              "task_key": "task", "split": "validation", "episode_length": 37}
    frames = torch.randint(0, 1021, (37, 2, 3, 16, 16), generator=torch.Generator().manual_seed(29)).float()/1020
    identities, current, target = [], [], []
    for t in range(35):
        for h in HORIZONS:
            if t+h < 37:
                identities.append({"episode_index": 7, "source_digest": "source", "group_digest": "group",
                                   "task_key": "task", "split": "validation", "anchor": t,
                                   "horizon": h, "target": t+h})
                current.append(frames[t]); target.append(frames[t+h])
    return source, frames, identities, torch.stack(current), torch.stack(target)


def accumulate(data, batch, oracle=False):
    source, _, identities, current, target = data
    acc = EpisodeAccumulator(source, "cpu")
    prediction = target if oracle else current
    for i in range(0, len(identities), batch):
        acc.add(identities[i:i+batch], current[i:i+batch], target[i:i+batch], prediction[i:i+batch])
    return acc.finish()


def test_streamed_copy_matches_frozen_all_anchor_baseline(data):
    rows = accumulate(data, 7)
    for c, camera in enumerate(CAMERAS):
        reference = measure_video(data[1][:, c], torch.zeros(3, 16, 16), batch_size=11)
        for row in rows:
            if row["camera"] == camera:
                for field in FIELDS:
                    assert row["prediction"][field] == pytest.approx(reference[row["horizon"]]["copy"][field], abs=1e-10)


def test_metrics_independent_of_microbatch_boundaries_and_order(data):
    expected = accumulate(data, 1)
    assert accumulate(data, 10) == expected
    source, frames, identities, current, target = data
    reverse = (source, frames, list(reversed(identities)), current.flip(0), target.flip(0))
    actual = accumulate(reverse, 13)
    for x, y in zip(expected, actual, strict=True):
        np.testing.assert_allclose(list(x["prediction"].values()), list(y["prediction"].values()), atol=1e-12, rtol=0)


def test_oracle_has_zero_error_and_identical_truth_masks(data):
    copy, oracle = accumulate(data, 8), accumulate(data, 17, oracle=True)
    for x, y in zip(copy, oracle, strict=True):
        assert y["prediction"]["l1"] == y["prediction"]["motion_error_mass"] == y["prediction"]["static_error_mass"] == 0
        assert y["prediction"]["ssim"] == pytest.approx(1, abs=1e-12)
        assert y["prediction"]["motion_area"] == x["prediction"]["motion_area"]
    summary = prediction_summary(oracle)
    assert summary["overall"]["l1"]["value"] == 0
    assert summary["overall"]["ssim"]["value"] == pytest.approx(1)


def test_missing_or_duplicate_pairs_are_rejected(data):
    source, _, ids, current, target = data
    acc = EpisodeAccumulator(source, "cpu")
    acc.add(ids[:2], current[:2], target[:2], current[:2])
    with pytest.raises(ValueError, match="incomplete"):
        acc.finish()
    with pytest.raises(ValueError, match="duplicate"):
        acc.add(ids[:1], current[:1], target[:1], current[:1])
    fresh = EpisodeAccumulator(source, "cpu")
    with pytest.raises(ValueError, match="duplicate"):
        fresh.add([ids[0], ids[0]], current[:2], target[:2], current[:2])


@pytest.mark.parametrize("field,value", [("split", "train"), ("source_digest", "changed"),
                                         ("target", 500), ("anchor", -1), ("horizon", 3)])
def test_invalid_pair_identity_rejected(data, field, value):
    source, _, ids, current, target = data
    bad = deepcopy(ids[:1]); bad[0][field] = value
    with pytest.raises(ValueError):
        EpisodeAccumulator(source, "cpu").add(bad, current[:1], target[:1], current[:1])


def test_forbidden_split_and_nonfinite_prediction_rejected(data):
    source, _, ids, current, target = data
    with pytest.raises(ValueError):
        EpisodeAccumulator({**source, "split": "test"}, "cpu")
    for value in (float("nan"), float("inf"), -.1, 1.1):
        bad = current[:1].clone(); bad[0, 0, 0, 0, 0] = value
        with pytest.raises(ValueError):
            EpisodeAccumulator(source, "cpu").add(ids[:1], current[:1], target[:1], bad)

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from eventground_wam.future_pairs import CAMERAS, HORIZONS
from eventground_wam.jepa_baselines import FIELDS
from eventground_wam.jepa_evaluation import EpisodeAccumulator, score_queries


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    source = {"split": "validation", "episode_index": 3, "digest": "source", "group_digest": "group", "task_key": "task", "episode_length": 33}
    identities = [{"episode_index": 3, "source_digest": "source", "group_digest": "group", "task_key": "task", "split": "validation",
                       "anchor": t, "horizon": h, "target": t+h} for t in range(31) for h in HORIZONS if t+h < 33]
    rng = np.random.default_rng(48)
    target = rng.normal(size=(len(identities), 128, 2048)).astype(np.float32)
    target /= np.linalg.norm(target, axis=-1, keepdims=True)
    prediction = np.roll(target, 1, axis=-1)
    current = np.zeros((len(identities), 2, 3, 128, 128), np.float32)
    future = current.copy()
    future[:, 0, :, :16, :16] = 11/255
    return source, identities, current, future, prediction, target


def test_full_episode_scalar_reference_and_uneven_chunking():
    s, ids, current, future, pred, target = fixture()
    acc = EpisodeAccumulator(s, split="validation")
    for begin in range(0, len(ids), 7):
        acc.add(ids[begin:begin+7], current[begin:begin+7], future[begin:begin+7], pred[begin:begin+7], target[begin:begin+7])
    rows = acc.finish()
    assert len(rows) == 10 and sum(r['pairs'] for r in rows)//2 == len(ids)
    for row in rows:
        ci = CAMERAS.index(row['camera'])
        selected = [i for i, item in enumerate(ids) if item['horizon'] == row['horizon']]
        errors = np.clip(1-(pred[selected, ci*64:(ci+1)*64].astype(np.float64)*target[selected, ci*64:(ci+1)*64]).sum(-1), 0, 2)
        weights = np.zeros_like(errors)
        if ci == 0:
            weights[:, 0] = 1
        expected = [errors.mean(), (errors*weights).mean(), (errors*(1-weights)).mean(), weights.mean(), (1-weights).mean()]
        np.testing.assert_allclose([row['prediction'][f] for f in FIELDS], expected, atol=1e-14, rtol=0)


def test_rejects_duplicates_incomplete_coverage_and_target_misalignment():
    s, ids, current, future, pred, target = fixture()
    acc = EpisodeAccumulator(s, split="validation")
    acc.add(ids[:2], current[:2], future[:2], pred[:2], target[:2])
    with pytest.raises(ValueError, match="incomplete"):
        acc.finish()
    with pytest.raises(ValueError, match="duplicate"):
        acc.add(ids[:1], current[:1], future[:1], pred[:1], target[:1])
    changed = deepcopy(ids[2:3]); changed[0]['target'] += 1
    with pytest.raises(ValueError, match="invalid"):
        acc.add(changed, current[:1], future[:1], pred[:1], target[:1])


def test_rejects_wrong_split_source_and_nonunit_features():
    s, ids, current, future, pred, target = fixture()
    with pytest.raises(ValueError, match="split"):
        EpisodeAccumulator(s, split="train")
    with pytest.raises(ValueError, match="split"):
        EpisodeAccumulator({**s, 'split': 'test'}, split="test")
    acc = EpisodeAccumulator(s, split="validation")
    changed = deepcopy(ids[:1]); changed[0]['source_digest'] = 'wrong'
    with pytest.raises(ValueError, match="source"):
        acc.add(changed, current[:1], future[:1], pred[:1], target[:1])
    with pytest.raises(ValueError, match="unit"):
        acc.add(ids[:1], current[:1], future[:1], pred[:1]*2, target[:1])


def test_frozen_retrieval_pool_ties_and_camera_alignment():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(8, 128, 2048)).astype(np.float32)
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    query = {"episode_index": 3, "camera": CAMERAS[1], "anchor": 0, "horizon": 2, "target": 2,
                 "pool": {"eligible": True, "frames": list(range(8)), "target_slot": 2}}
    original = deepcopy(query)
    result = score_queries([query], {(0, 2): features[2]}, features)[0]
    assert result['prediction']['recall1'] == result['prediction']['mrr'] == 1
    assert query == original and result['pool'] == query['pool']
    repeated = np.repeat(features[:1], 8, axis=0)
    tied = score_queries([query], {(0, 2): repeated[0]}, repeated)[0]
    assert tied['prediction']['recall1'] == 1/8
    assert tied['prediction']['mrr'] == pytest.approx(sum(1/i for i in range(1, 9))/8)
    with pytest.raises(ValueError, match="missing"):
        score_queries([query], {}, features)
    query['pool']['target_slot'] = 1
    with pytest.raises(ValueError, match="pool"):
        score_queries([query], {(0, 2): features[2]}, features)


def test_pilot_requires_same_candidate_code_and_runtime(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from evaluate_jepa_cpt import verify_pilot

    from eventground_wam.demonstration_contracts import seal_contract
    pilot_identity = seal_contract({"phase": 'pilot', "candidate_sha256": 'candidate', "runtime": {'torch':'pinned'}, "source_sha256": {'a':'sha'}})
    full_identity = seal_contract({**{k:v for k,v in pilot_identity.items() if k != 'digest'}, 'phase':'full'})
    pilot = {"status": 'passed', "checks": {'reference':True}, "identity_digest": pilot_identity['digest']}
    verify_pilot(pilot,pilot_identity,full_identity)
    for field, value in [('candidate_sha256','changed'), ('runtime',{}), ('source_sha256',{})]:
        with pytest.raises(ValueError, match="exact"):
            verify_pilot(pilot,pilot_identity,{**full_identity,field:value})
    with pytest.raises(ValueError, match="exact"):
        verify_pilot({**pilot,'checks':{'reference':False}},pilot_identity,full_identity)


def test_projection_charges_pilot_only_work_once(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from evaluate_jepa_cpt import project_runtime

    measurements = [{"seconds": 8., "pilot_only_seconds": 2., "pairs": 100},
                    {"seconds": 21., "pilot_only_seconds": 1., "pairs": 400}]
    # Slowest recurring rate .06 s/pair; 30 s startup + 3 s probes paid once.
    assert project_runtime(measurements, 30., 10000) == (600+33)*1.5+900
